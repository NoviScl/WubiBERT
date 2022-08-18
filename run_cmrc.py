# coding=utf-8
import argparse
from pathlib import Path
import collections
import json
from time import time

import torch
from torch.utils.data import TensorDataset, DataLoader

import modeling
from optimization import get_optimizer
from utils import (
    json_load_by_line, 
    json_save_by_line,
    get_device, 
    set_seed,
    load_tokenizer,
)
from mrc.preprocess.cmrc2018_evaluate import get_eval
from mrc.preprocess.cmrc2018_output import write_predictions
from mrc.preprocess.cmrc2018_preprocess import (
    read_cmrc_examples,
    convert_examples_to_features,
)


def evaluate(
    model, 
    args, 
    file_data: Path, 
    examples: list, 
    features: list, 
    device: str, 
    epoch: int, 
    output_dir: Path,
    ):
    RawResult = collections.namedtuple(
        "RawResult",
        ["unique_id", "start_logits", "end_logits"])
    output_dir.mkdir(exist_ok=True, parents=True)
    file_preds = output_dir / 'preds.json'
    file_nbest = output_dir / 'nbest.json'
    
    dataset = features_to_dataset(features, is_training=False,
                                  two_level_embeddings=args.two_level_embeddings,
                                  avg_char_tokens=args.avg_char_tokens)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model.eval()
    all_results = []
    print("*** Start evaluating ***", flush=True)
    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        tensors = expand_batch(batch, is_training=False,
                               two_level_embeddings=args.two_level_embeddings,
                               avg_char_tokens=args.avg_char_tokens)
        # Some tuple elements are None depending on processing method.
        (input_ids, input_mask, segment_ids, example_indices, token_ids, 
         pos_left, pos_right) = tensors

        with torch.no_grad():
            batch_start_logits, batch_end_logits = model(
                input_ids, segment_ids, input_mask, token_ids=token_ids, 
                pos_left=pos_left, pos_right=pos_right,
                use_token_embeddings=args.two_level_embeddings,
                avg_char_tokens=args.avg_char_tokens)
        
        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            feature = features[example_index.item()]
            unique_id = int(feature['unique_id'])
            all_results.append(RawResult(unique_id=unique_id,
                                         start_logits=start_logits,
                                         end_logits=end_logits))
    print(f'Writing predictions to {file_preds} and {file_nbest}', flush=True)
    write_predictions(
        examples, 
        features, 
        all_results,
        n_best_size=args.n_best, 
        max_answer_length=args.max_ans_length,
        do_lower_case=True, 
        output_prediction_file=file_preds,
        output_nbest_file=file_nbest,
        two_level_embeddings=False,
    )

    # file_truth = os.path.join(args.test_dir, file_data)
    res = get_eval(file_data, file_preds)
    result_file = output_dir / 'result.json'
    json.dump(res, open(result_file, 'w'))
    model.train()
    return res['em'], res['f1']


def parse_args():
    p = argparse.ArgumentParser()

    # Hyperparameters
    p.add_argument('--epochs', type=int, default=6)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--grad_acc_steps', type=int, default=2)
    p.add_argument('--lr', type=float, default=3e-5)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--clip_norm', type=float, default=1.0)
    p.add_argument('--warmup_rate', type=float, default=0.05)
    p.add_argument("--schedule", default='warmup_linear', type=str, help='schedule')
    p.add_argument("--weight_decay_rate", default=0.01, type=float, help='weight_decay_rate')
    p.add_argument('--max_ans_length', type=int, default=50)
    p.add_argument('--n_best', type=int, default=20)
    p.add_argument('--max_seq_length', type=int, default=512)
    p.add_argument('--seed', type=int, default=0)

    # Other arguments
    # p.add_argument('--data_dir', type=str, required=True)
    p.add_argument('--train_dir')
    p.add_argument('--dev_dir')
    p.add_argument('--test_dir')
    p.add_argument('--do_train', action='store_true')
    p.add_argument('--do_test', action='store_true')
    p.add_argument('--init_ckpt')
    p.add_argument('--config_file', type=str, required=True)
    p.add_argument('--tokenizer_type', type=str, required=True)
    p.add_argument('--vocab_file', type=str, required=True)
    p.add_argument('--vocab_model_file', type=str, required=True)
    p.add_argument('--cws_vocab_file', type=str, default=None)
    p.add_argument('--output_dir', type=str, default='logs/temp')
    p.add_argument('--two_level_embeddings', action='store_true')
    p.add_argument('--avg_char_tokens', action='store_true')
    p.add_argument('--debug', action='store_true')
    p.add_argument('--test_name', default='test_clean')
    p.add_argument('--tokenizer_name', default='XXX')
    p.add_argument('--test_ckpt')
    p.add_argument('--log_interval', type=int)

    return p.parse_args()


def features_to_dataset(features: list, is_training: bool, two_level_embeddings, 
                        avg_char_tokens) -> TensorDataset:
    '''
    Turn list of features into Tensor datasets
    '''
    all_input_ids = torch.tensor([f['input_ids'] for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f['input_mask'] for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f['segment_ids'] for f in features], dtype=torch.long)
    if is_training:
        all_start_positions = torch.tensor([f['start_position'] for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f['end_position'] for f in features], dtype=torch.long)
    else:
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    def get_features_tle():
        all_token_ids = torch.tensor([f['token_ids'] for f in features], dtype=torch.long)
        all_pos_left = torch.tensor([f['pos_left'] for f in features], dtype=torch.long)
        all_pos_right = torch.tensor([f['pos_right'] for f in features], dtype=torch.long)
        if is_training:
            return TensorDataset(
                all_input_ids, 
                all_input_mask, 
                all_segment_ids, 
                all_start_positions, 
                all_end_positions, 
                all_token_ids,
                all_pos_left,
                all_pos_right,
            )
        else:
            return TensorDataset(
                all_input_ids, 
                all_input_mask, 
                all_segment_ids, 
                all_example_index, 
                all_token_ids,
                all_pos_left,
                all_pos_right,
            )
        
    def get_features_act():
        all_char_ids = torch.tensor([f['char_ids'] for f in features], dtype=torch.long)
        if is_training:
            return TensorDataset(
                all_input_ids,
                all_input_mask,
                all_segment_ids,
                all_start_positions,
                all_end_positions,
                all_char_ids,
            )
        else:
            return TensorDataset(
                all_input_ids,
                all_input_mask,
                all_segment_ids,
                all_example_index,
                all_char_ids,
            )
        
    if two_level_embeddings:
        return get_features_tle()
    elif avg_char_tokens:
        return get_features_act()
    else:
        if is_training:
            return TensorDataset(
                all_input_ids,
                all_input_mask,
                all_segment_ids,
                all_start_positions,
                all_end_positions,
            )
        else:
            return TensorDataset(
                all_input_ids,
                all_input_mask,
                all_segment_ids,
                all_example_index,
            )


def expand_batch(batch, is_training, two_level_embeddings, avg_char_tokens) -> tuple:
    '''Expand batch into a tuple.
    
    Return `(input_ids, input_mask, segment_ids, start_positions, end_positions,
    token_ids, pos_left, pos_right)` or `(input_ids, input_mask, segment_ids,
    example_index, token_ids, pos_left, pos_right)`, depending on `is_training`. 
    
    The last 3 elements in might be none depending on `two_level_embeddings` and
    `avg_char_tokens`.
    '''
    input_ids = batch[0]
    input_mask = batch[1]
    segment_ids = batch[2]
    token_ids = None
    pos_left = None
    pos_right = None
    if two_level_embeddings:
        token_ids = batch[-3]
        pos_left = batch[-2]
        pos_right = batch[-1]
    elif avg_char_tokens:
        char_ids = batch[-1]
        token_ids = char_ids  # Use `token_ids` to reduce param count
    if is_training:
        start_positions = batch[3]
        end_positions = batch[4]
        return (input_ids, input_mask, segment_ids, start_positions, end_positions,
                token_ids, pos_left, pos_right)
    else:
        example_index = batch[3]
        return (input_ids, input_mask, segment_ids, example_index,
                token_ids, pos_left, pos_right)


def get_filename_examples_and_features(
    data_type: str,
    data_dir: str,
    args, 
    tokenizer_name: str,
    vocab_size: int=22680,
    ) -> tuple:
    '''
    Return:
        example_file: str,
        feature_file: str,
    '''
    # suffix = '_{}_{}_{}'.format(str(args.max_seq_length), tokenizer_name, str(vocab_size))
    suffix_ex = '_examples.json'
    if args.two_level_embeddings:
        suffix = '_{}_{}_{}_twolevel'.format(str(args.max_seq_length), tokenizer_name, str(vocab_size))
    elif args.avg_char_tokens:
        suffix = '_{}_{}_{}_chartokens'.format(str(args.max_seq_length), tokenizer_name, str(vocab_size))
    else:
        suffix = '_{}_{}_{}'.format(str(args.max_seq_length), tokenizer_name, str(vocab_size))
    file_examples = Path(data_dir, data_type + suffix_ex)
    file_features = Path(data_dir, data_type + '_features' + suffix + '.json')
    return file_examples, file_features


def gen_examples_and_features(
    file_data: Path,
    file_examples: Path,
    file_features: Path,
    is_training: bool,
    tokenizer,
    max_seq_length: int,
    # max_query_length=64,
    # doc_stride=128,
    two_level_embeddings: bool=False,
    avg_char_tokens: bool=False,
    ):
    '''
    Return:
        examples: [dict]
        features: [dict]
    '''

    use_example_cache = True
    use_feature_cache = True

    examples, features = None, None
    # Examples
    if use_example_cache and file_examples.exists():
        print('Found example file, loading...')
        examples = json_load_by_line(file_examples)
        print(f'Loaded {len(examples)} examples')
    else:
        print('Example file not found, generating...')
        examples, mismatch = read_cmrc_examples(file_data, is_training)
        print(f'num examples: {len(examples)}')
        print(f'mismatch: {mismatch}')
        print(f'Generated {len(examples)} examples')

        print(f'Saving to "{file_examples}"...')
        json_save_by_line(examples, file_examples)
        print(f'Saved {len(examples)} examples')
        # Somehow, just saving will result in all empty evaluation XD
        print(f'Loading from "{file_examples}"...')
        examples = json_load_by_line(file_examples)
        print(f'Loaded {len(examples)} examples', flush=True)

    # Load or gen features
    if use_feature_cache and file_features.exists():
        print('Found feature file, loading...')
        features = json_load_by_line(file_features)
        print(f'Loaded {len(features)} features')
    else:
        print('Feature file not found, generating...')
        features = convert_examples_to_features(
            examples, tokenizer, max_seq_length=max_seq_length,
            two_level_embeddings=two_level_embeddings,
            avg_char_tokens=avg_char_tokens)
        print(f'Generated {len(features)} features')
        print(f'Saving to "{file_features}"...')
        json_save_by_line(features, file_features)
        print(f'Saved {len(features)} features')
        print('Found feature file, loading...')
        features = json_load_by_line(file_features)
        print(f'Loaded {len(features)} features', flush=True)

    return examples, features


def load_model(config_file: str, init_ckpt: str):
    '''Load model from a checkpoint'''
    print(f'Loading model from checkpoint "{init_ckpt}"')
    config = modeling.BertConfig.from_json_file(config_file)
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)
    model = modeling.BertForQuestionAnswering(config)
    modeling.ACT2FN["bias_gelu"] = modeling.bias_gelu_training
    state_dict = torch.load(init_ckpt, map_location='cpu')
    model.load_state_dict(state_dict['model'], strict=False)
    return model


def get_best_ckpt(output_dir: Path) -> Path:
    max_acc = float('-inf')
    best_ckpt = None
    for ckpt_dir in output_dir.glob('ckpt-*'):
        if not ckpt_dir.is_dir(): continue
        result_file = ckpt_dir / 'result.json'
        result = json.load(open(result_file))
        acc = result['em']
        if acc > max_acc:
            max_acc = acc
            best_ckpt = ckpt_dir / 'ckpt.pt'
    return best_ckpt


def train(args):
    # Prepare files
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save arguments
    print("Arguments:")
    print(json.dumps(vars(args), indent=4))
    args_file = output_dir / 'train_args.json'
    json.dump(vars(args), open(args_file, 'w'), indent=4)

    # Device
    device = get_device()  # Get gpu with most free RAM
    n_gpu = torch.cuda.device_count()
    print("device: {} n_gpu: {}".format(device, n_gpu))
    set_seed(args.seed)

    # Prepare model
    model = load_model(args.config_file, args.init_ckpt).to(device)

    # Save config
    model_to_save = model.module if hasattr(model, 'module') else model
    filename_config = output_dir / modeling.CONFIG_NAME
    with open(filename_config, 'w') as f:
        f.write(model_to_save.config.to_json_string())

    # Tokenizer
    print('Loading tokenizer...')
    tokenizer = load_tokenizer(
        args.tokenizer_type, args.vocab_file, args.vocab_model_file)
    print('Loaded tokenizer')

    # Because tokenizer_type is a part of the feature file name,
    # new features will be generated for every tokenizer type.
    # tokenizer_name = output_dir_to_tokenizer_name(args.output_dir)
    file_train = Path(args.train_dir, 'train.json')
    file_dev = Path(args.dev_dir, 'dev.json')
    file_train_examples, file_train_features = get_filename_examples_and_features(
        'train', args.train_dir, args, tokenizer_name=args.tokenizer_name)
    file_dev_examples, file_dev_features = get_filename_examples_and_features(
        'dev', args.dev_dir, args, tokenizer_name=args.tokenizer_name)
    # Generate train examples and features
    print('Generating train data:')
    print(f'  file_examples: {file_train_examples}')
    print(f'  file_features: {file_train_features}')

    train_examples, train_features = gen_examples_and_features(
        file_train,
        file_train_examples,
        file_train_features,
        is_training=True,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        two_level_embeddings=args.two_level_embeddings,
        avg_char_tokens=args.avg_char_tokens)
    print('Generating dev data:')
    print(f'  file_examples: {file_dev_examples}')
    print(f'  file_features: {file_dev_features}')
    dev_examples, dev_features = gen_examples_and_features(
        file_dev,
        file_dev_examples,
        file_dev_features,
        is_training=False,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        two_level_embeddings=args.two_level_embeddings,
        avg_char_tokens=args.avg_char_tokens)
    del train_examples  # Only need examples for predictions
    print('Done generating data')

    update_size = args.batch_size * args.grad_acc_steps
    steps_per_ep = len(train_features) // args.batch_size
    total_steps = steps_per_ep * args.epochs

    optimizer = get_optimizer(
        model=model,
        float16=False,
        learning_rate=args.lr,
        total_steps=total_steps,
        schedule=args.schedule,
        warmup_rate=args.warmup_rate,
        max_grad_norm=args.clip_norm,
        weight_decay_rate=args.weight_decay_rate)

    # Train and evaluation
    train_data = features_to_dataset(
        train_features, is_training=True,
        two_level_embeddings=args.two_level_embeddings,
        avg_char_tokens=args.avg_char_tokens)
    train_dataloader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True)


    global_step = 1
    train_losses = []
    dev_history = []
    train_start_time = time()
    eval_interval = steps_per_ep // 2
    
    print('*** Start Training ***')
    print(f'num epochs = {args.epochs}')
    print(f'eval interval = {eval_interval}')
    print(f'log interval = {args.log_interval}')
    print(f'steps for epoch = {steps_per_ep}')
    print(f'total steps = {total_steps}')
    print(f'batch size = {args.batch_size}')
    print(f'grad acc = {args.grad_acc_steps}')
    print(f'num train features = {len(train_features)}')
    print(f'num dev features = {len(dev_features)}')

    for ep in range(args.epochs):
        print(f'*** Training Epoch {ep} ***')
        
        model.train()
        model.zero_grad()
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            tensors = expand_batch(
                batch, is_training=True,
                two_level_embeddings=args.two_level_embeddings,
                avg_char_tokens=args.avg_char_tokens)
            (input_ids, input_mask, segment_ids, start_positions, end_positions,
                token_ids, pos_left, pos_right) = tensors

            loss = model(
                input_ids, 
                segment_ids, 
                input_mask, 
                start_positions, 
                end_positions,
                token_ids=token_ids, 
                pos_left=pos_left, 
                pos_right=pos_right,
                use_token_embeddings=args.two_level_embeddings,
                avg_char_tokens=args.avg_char_tokens,
            )
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            train_losses.append(loss.item())

            loss.backward()

            if (step + 1) % args.grad_acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if global_step % args.log_interval == 0:
                train_state = {
                    'step': global_step,
                    'ep': global_step / steps_per_ep,
                    'loss': train_losses[-1],
                    'time_elapsed': time() - train_start_time,
                }
                print(train_state, flush=True)

            if global_step % eval_interval == 0:
                ckpt_dir = output_dir / f'ckpt-{global_step}'
                ckpt_dir.mkdir(exist_ok=True)
                
                print('*** Start evaluation ***', flush=True)
                
                dev_acc, dev_f1 = evaluate(
                    model, args, 
                    file_data=Path(args.dev_dir, 'dev.json'),
                    examples=dev_examples, 
                    features=dev_features, 
                    device=device, 
                    epoch=ep, 
                    output_dir=ckpt_dir,
                )
                result = {
                    'step': global_step,
                    'dev_acc': dev_acc,
                    'dev_f1': dev_f1,
                    'train_loss': train_losses[-1],
                }
                print(f'result: {result}', flush=True)
                dev_history.append(result)

                # Save model
                model_to_save = model.module if hasattr(model, 'module') else model
                ckpt_file = ckpt_dir / f'ckpt.pt'
                print(f'Saving model to {ckpt_file}', flush=True)
                torch.save(
                    {"model": model_to_save.state_dict()},
                    ckpt_file,
                )
                print('*** Evaluation finished ***', flush=True)
                
                # Save train loss
                train_loss_file = output_dir / 'train_loss.json'
                json.dump(train_losses, open(train_loss_file, 'w'), indent=2)
                

            global_step += 1

    # release the memory
    del model
    del optimizer
    torch.cuda.empty_cache()

    print('Training finished')


def test(args):
    print('Test')
    print(json.dumps(vars(args), indent=4))

    # Prepare files
    output_dir = Path(args.output_dir)
    test_dir = output_dir / args.test_name
    data_dir = Path(args.test_dir)
    assert output_dir.exists()
    assert args.batch_size > 0, 'Batch size must be positive'

    # Device
    device = get_device()  # Get gpu with most free RAM
    n_gpu = torch.cuda.device_count()
    print("device: {} n_gpu: {}".format(device, n_gpu))

    print('SEED: ' + str(args.seed))
    set_seed(args.seed)

    # Prepare model
    # best_ckpt = output_dir / modeling.FILENAME_BEST_MODEL
    if args.test_ckpt:
        best_ckpt = args.test_ckpt
    else:
        best_ckpt = get_best_ckpt(output_dir)
    model = load_model(args.config_file, best_ckpt).to(device)

    # Tokenizer
    tokenizer = load_tokenizer(
        args.tokenizer_type, args.vocab_file, args.vocab_model_file)

    # Because tokenizer_type is a part of the feature file name,
    # new features will be generated for every tokenizer type.
    file_data = data_dir / 'test.json'
    file_examples, file_features = get_filename_examples_and_features(
        'test', data_dir, args, tokenizer_name=args.tokenizer_name)
    # Generate train examples and features
    print('Generating data:', flush=True)
    print(f'  file_examples: {file_examples}', flush=True)
    print(f'  file_features: {file_features}', flush=True)
    examples, features = gen_examples_and_features(
        file_data,
        file_examples,
        file_features,
        is_training=False,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        two_level_embeddings=args.two_level_embeddings,
        avg_char_tokens=args.avg_char_tokens,
    )
    print('Done generating data', flush=True)
    print('***** Testing *****', flush=True)
    print(f'batch size = {args.batch_size}', flush=True)
    print(f'num features = {len(features)}', flush=True)

    model.zero_grad()
    acc, f1 = evaluate(
        model, 
        args,
        file_data=file_data,
        examples=examples,
        features=features,
        device=device,
        epoch='test',
        output_dir=test_dir,
    )
    result = {'acc': acc, 'f1': f1}
    print(f'result: {result}', flush=True)

    result_file = test_dir / 'result.json'
    json.dump(result, open(result_file, 'w'))
    print('*** Testing finished ***', flush=True)


def main(args):
    # Sanity check
    assert args.grad_acc_steps > 0
    assert args.batch_size > 0
    assert args.epochs > 0
    
    if args.do_train:
        train(args)
    if args.do_test:
        test(args)
    print('DONE', flush=True)


if __name__ == '__main__':
    main(parse_args())