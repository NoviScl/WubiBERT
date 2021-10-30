'''
Run BERT + CRF on NER tasks
'''
import glob
import logging
import os
import json
import time

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler

import consts
import modeling
import utils
from tokenization import ALL_TOKENIZERS
from optimization import BertAdam, warmup_linear
from schedulers import LinearWarmUpScheduler

from ner.callback.optimizater.adamw import AdamW
from ner.callback.lr_scheduler import get_linear_schedule_with_warmup
from ner.callback.progressbar import ProgressBar
from ner.tools.common import json_to_text
from ner.tools.finetuning_argparse import get_argparse

from ner.models.transformers import BertConfig
from ner.models.bert_for_ner import BertCrfForNer
from ner.processors.utils_ner import get_entities, get_char_labels
from ner.processors.ner_seq import convert_examples_to_features
from ner.processors.ner_seq import ner_processors as processors
from ner.processors.ner_seq import get_collate_fn
from ner.metrics.ner_metrics import SeqEntityScore
from run_pretraining import pretraining_dataset, WorkerInitObj

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# TWO_LEVEL_EMBEDDINGS = False
# USE_TOKEN_EMBEDDINGS = True
AVG_CHAR_TOKENS = True
CLS_TOKEN = '[CLS]'
SEP_TOKEN = '[SEP]'
PAD_TOKEN = '[PAD]'


def collate_fn_from_args(args):
    return get_collate_fn(args.two_level_embeddings, args.avg_char_tokens)


def evaluate(args, model, dataset, id2label, device):
    metric = SeqEntityScore(id2label, markup=args.markup)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size, 
                            collate_fn=collate_fn_from_args(args))

    # Evaluation
    total_eval_loss = 0.0
    n_eval_steps = 0

    for step, batch in enumerate(tqdm(dataloader, mininterval=8.0, desc='Evaluating')):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                'token_type_ids': batch[2],
                "labels": batch[3],
                'input_lens': batch[4],
            }
            if args.avg_char_tokens:
                inputs['char_ids'] = batch[5]
                inputs['avg_char_tokens'] = True
            elif args.two_level_embeddings:
                inputs['token_ids'] = batch[5]
                inputs['pos_right'] = batch[7]
                inputs['pos_left'] = batch[6]
                inputs['use_token_embeddings'] = True
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            tags = model.crf.decode(logits, inputs['attention_mask'])
        total_eval_loss += tmp_eval_loss.item()
        n_eval_steps += 1

        # NOTE: labels of tokens might not match char labels from dataset
        out_label_ids = inputs['labels'].cpu().numpy().tolist()  # Token labels
        input_lens = inputs['input_lens'].cpu().numpy().tolist()
        tags = tags.squeeze(0).cpu().numpy().tolist()
        
        json.dump(out_label_ids, open('labels.json', 'w'))

        for i, label in enumerate(out_label_ids):
            temp_1 = []
            temp_2 = []
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif j == input_lens[i] - 1:
                    wrong_cnt = metric.update(pred_paths=[temp_2], label_paths=[temp_1])
                    break                        
                else:
                    temp_1.append(id2label[out_label_ids[i][j]])
                    temp_2.append(id2label[tags[i][j]])

    logger.info("\n")
    eval_loss = total_eval_loss / n_eval_steps
    eval_info, entity_info = metric.result()
    results = {f'{key}': value for key, value in eval_info.items()}
    results['loss'] = eval_loss
    logger.info("***** Eval results *****")
    info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
    logger.info(info)
    # logger.info("***** Entity results *****")
    # for key in sorted(entity_info.keys()):
    #     logger.info("******* %s results ********" % key)
    #     info = "-".join([f' {key}: {value:.4f} ' for key, value in entity_info[key].items()])
    #     logger.info(info)

    return results


def get_dataset(task, data_dir, tokenizer, tokenizer_name, data_type, max_seq_len,
                two_level_embeddings, avg_char_tokens):
    '''
    Generate dataset from data file.
    This will cache features using torch.save in data_dir.

    two_level_embeddings: Whether features should contain both split char tokens and ordinary tokens.
    '''
    processor = processors[task]()
    use_cache = False  # TODO: Set as True on deployment
    
    # Load data features from cache or dataset file
    cached_features_file = 'cache_{}_{}_{}'.format(data_type, tokenizer_name,
                                                   str(max_seq_len))
    if two_level_embeddings:
        cached_features_file += '_twolevel'
    elif avg_char_tokens:
        cached_features_file += '_chartokens'
    cached_features_file = os.path.join(data_dir, cached_features_file + '.json')

    if use_cache and os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = json.load(open(cached_features_file, 'r'))
    else:
        logger.info("Creating features from dataset file at %s", data_dir)
        label_list = processor.get_labels()
        if data_type == 'train':
            examples = processor.get_train_examples(data_dir)
        elif data_type == 'dev':
            examples = processor.get_dev_examples(data_dir)
        else:
            examples = processor.get_test_examples(data_dir)
        features = convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            label_list=label_list,
            max_seq_length=max_seq_len,
            cls_token=CLS_TOKEN,
            sep_token=SEP_TOKEN,
            cls_token_at_end=False,
            cls_token_segment_id=0,
            # pad on the left for xlnet
            pad_on_left=False,
            pad_token=tokenizer.convert_tokens_to_ids([PAD_TOKEN])[0],
            pad_token_segment_id=0,
            two_level_embeddings=two_level_embeddings,
            avg_char_tokens=avg_char_tokens,
        )
        logger.info("Saving features into cached file %s", cached_features_file)
        json.dump(features, open(cached_features_file, 'w'))
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f['input_ids'] for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f['input_mask'] for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f['segment_ids'] for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f['label_ids'] for f in features], dtype=torch.long)
    all_lens = torch.tensor([f['input_len'] for f in features], dtype=torch.long)

    if two_level_embeddings:
        all_pos_left = torch.tensor([f['pos_left'] for f in features], dtype=torch.long)
        all_pos_right = torch.tensor([f['pos_right'] for f in features], dtype=torch.long)
        all_token_ids = torch.tensor([f['token_ids'] for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, 
                                all_lens, all_label_ids, all_token_ids,
                                all_pos_left, all_pos_right)
    elif avg_char_tokens:
        all_char_ids = [f['char_ids'] for f in features]
        all_char_ids = torch.tensor(all_char_ids, dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_lens, all_label_ids, all_char_ids)
    else:
        raise NotImplementedError
    return dataset


def get_optimizer_and_scheduler(model, lr, lr_crf, weight_decay, 
                                adam_eps, n_train_steps, n_warmup_steps):
    no_decay = ["bias", "LayerNorm.weight"]
    bert_param_optimizer = list(model.bert.named_parameters())
    crf_param_optimizer = list(model.crf.named_parameters())
    linear_param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay, 'lr': lr},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': lr},

        {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay, 'lr': lr_crf},
        {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': lr_crf},

        {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay, 'lr': lr_crf},
        {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': lr_crf}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_eps)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=n_warmup_steps,
                                                num_training_steps=n_train_steps)
    return optimizer, scheduler


def load_model(config_file, model_file, num_labels):
    config = modeling.BertConfig.from_json_file(config_file)
    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)
    model = BertCrfForNer(config, num_labels=num_labels)
    state_dict = torch.load(model_file, map_location='cpu')
    model.load_state_dict(state_dict["model"], strict=False)
    return model


def train(args):
    logger.info('Training arguments:')
    logger.info(json.dumps(vars(args), indent=4))

    # Setup output dir
    tokenizer_name = utils.output_dir_to_tokenizer_name(args.output_dir)
    output_dir = os.path.join(args.output_dir, str(args.seed))
    os.makedirs(output_dir, exist_ok=True)
    filename_params = os.path.join(output_dir, consts.FILENAME_PARAMS)
    json.dump(vars(args), open(filename_params, 'w'), indent=4)

    args.train_batch_size //= args.gradient_accumulation_steps
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set seed
    logger.info(f'device: {device}')
    logger.info(f'Set seed: {args.seed}')
    utils.set_seed(args.seed)

    # Load pretrained model and tokenizer
    tokenizer = ALL_TOKENIZERS[args.tokenizer_type](args.vocab_file, args.vocab_model_file)
    processor = processors[args.task_name]()
    label_list = processor.get_labels()
    id2label = {i: label for i, label in enumerate(label_list)}  # For evaluation
    num_labels = len(label_list)

    logger.info(f'Loading model from "{args.init_checkpoint}"')
    model = load_model(args.config_file, args.init_checkpoint, num_labels)
    logger.info('Loaded model')
    model.to(device)

    # Save config
    filename_config = os.path.join(output_dir, modeling.CONFIG_NAME)
    with open(filename_config, 'w') as f:
        f.write(model.config.to_json_string())

    # Train and dev data
    train_dataset = get_dataset(
        args.task_name, 
        args.train_dir,
        tokenizer, 
        tokenizer_name=tokenizer_name,
        data_type='train',
        max_seq_len=args.train_max_seq_length,
        two_level_embeddings=args.two_level_embeddings,
        avg_char_tokens=args.avg_char_tokens)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, 
                                  sampler=train_sampler, 
                                  batch_size=args.train_batch_size,
                                  collate_fn=collate_fn_from_args(args))

    dev_dataset = get_dataset(
        # args, 
        args.task_name, 
        args.dev_dir,
        tokenizer,
        tokenizer_name=tokenizer_name,
        data_type='dev',
        max_seq_len=args.eval_max_seq_length,
        two_level_embeddings=args.two_level_embeddings,
        avg_char_tokens=args.avg_char_tokens)

    # Optimizer
    n_train_steps = len(train_dataloader) * args.epochs
    n_warmup_steps = int(n_train_steps * args.warmup_proportion)
    optimizer, scheduler = get_optimizer_and_scheduler(
        model, 
        lr=args.learning_rate,
        lr_crf=args.crf_learning_rate,
        weight_decay=args.weight_decay,
        adam_eps=args.adam_epsilon,
        n_train_steps=n_train_steps,
        n_warmup_steps=n_warmup_steps,
    )

    # Training
    logger.info('*** Training ***')
    logger.info(f'  Num train examples = {len(train_dataset)}')
    logger.info(f'  Num dev examples = {len(dev_dataset)}')
    logger.info(f'  Num Epochs = {args.epochs}')
    logger.info(f'  Train batch size = {args.train_batch_size}')
    logger.info(f'  Dev batch size = {args.eval_batch_size}')
    logger.info(f'  Gradient Accumulation Steps = {args.gradient_accumulation_steps}')
    logger.info(f'  Total steps = {n_train_steps}')

    global_steps = 0
    cur_train_steps = 0
    total_train_loss = 0
    train_loss_history = []
    dev_loss_history = []
    dev_acc_history = []
    dev_f1_history = []
    model.zero_grad()
    utils.set_seed(args.seed)

    for ep in range(args.epochs):
        model.train()
        model.zero_grad()
        pbar = tqdm(train_dataloader, mininterval=8.0)
        for step, batch in enumerate(pbar):
            # if step == 0:
                # logger.info('Example:')
                # logger.info('input ids:')
                # logger.info(batch[0])
                # logger.info('labels:')
                # logger.info(batch[3])

            batch = tuple(t.to(device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                'token_type_ids': batch[2],
                "labels": batch[3],
                'input_lens': batch[4],
            }
            if args.avg_char_tokens:
                inputs['char_ids'] = batch[5]
                inputs['avg_char_tokens'] = True
            if args.two_level_embeddings:
                inputs['token_ids'] = batch[5]
                inputs['pos_left'] = batch[6]
                inputs['pos_right'] = batch[7]
                inputs['use_token_embeddings'] = True
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            loss /= args.gradient_accumulation_steps
            loss.backward()

            total_train_loss += loss.item()
            cur_train_steps += 1
            pbar.set_description(f'train_loss = {total_train_loss / cur_train_steps}')

            # Backward
            if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1 == len(train_dataloader)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_steps += 1

        train_loss = total_train_loss / (cur_train_steps + 1e-10)
        train_loss_history.append(train_loss)


        # Evaluation
        logger.info('*** Evaluation ***')
        logger.info(f'  Current epoch = {ep}')
        logger.info(f'  Num examples = {len(dev_dataset)}')
        logger.info(f'  Batch size = {args.eval_batch_size}')

        model.eval()
        total_dev_loss = 0
        cur_dev_steps = 0

        dev_result = evaluate(args, model, dev_dataset, id2label, device)
        dev_acc = dev_result['acc']
        dev_f1 = dev_result['f1']
        dev_loss = dev_result['loss']
        dev_acc_history.append(dev_acc)
        dev_f1_history.append(dev_f1)
        dev_loss_history.append(dev_loss)
    
        logger.info('*** Evaluation result ***')
        logger.info(f'  Current epoch = {ep}')
        logger.info(f'  Dev acc = {dev_acc}')
        logger.info(f'  Dev F1 = {dev_f1}')
        logger.info(f'  Dev loss = {dev_loss}')
        logger.info(f'  Train loss = {train_loss}')

        # Save to scores
        filename_scores = os.path.join(output_dir, consts.FILENAME_SCORES)
        with open(filename_scores, 'w') as f:
            f.write(f'epoch\ttrain_loss\tdev_loss\tdev_acc\tdev_f1\n')
            for i in range(ep + 1):
                train_loss = train_loss_history[i]
                dev_loss = dev_loss_history[i]
                dev_acc = dev_acc_history[i]
                dev_f1 = dev_f1_history[i]
                f.write(f'{i}\t{train_loss}\t{dev_loss}\t{dev_acc}\t{dev_f1}\n')
                

        # Save best model
        is_best = len(dev_f1_history) == 0 or dev_f1 == max(dev_f1_history)
        if is_best:
            model_dir = os.path.join(output_dir, 'models')
            os.makedirs(model_dir, exist_ok=True)
            best_model_filename = os.path.join(output_dir,
                                               consts.FILENAME_BEST_MODEL)
            torch.save({'model': model.state_dict()}, best_model_filename)
    
    logger.info('Training finished')


def test(args):
    logger.info('Testing start')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer_name = utils.output_dir_to_tokenizer_name(args.output_dir)
    output_dir = os.path.join(args.output_dir, str(args.seed))

    processor = processors[args.task_name]()
    tokenizer = ALL_TOKENIZERS[args.tokenizer_type](args.vocab_file, args.vocab_model_file)
    label_list = processor.get_labels()
    id2label = {i: label for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    # Load best model
    best_model_filename = os.path.join(output_dir, consts.FILENAME_BEST_MODEL)
    logger.info(f'Loading model from "{best_model_filename}"')
    model = load_model(args.config_file, best_model_filename, num_labels)
    logger.info(f'Loaded model')
    model.to(device)

    # Test data
    dataset = get_dataset(
        args.task_name,
        args.test_dir,
        tokenizer,
        tokenizer_name=tokenizer_name,
        data_type='test',
        max_seq_len=args.eval_max_seq_length,
        two_level_embeddings=args.two_level_embeddings,
        avg_char_tokens=args.avg_char_tokens)

    # Test
    utils.set_seed(args.seed)
    logger.info('*** Testing ***')
    logger.info(f'  Num examples = {len(dataset)}')
    logger.info(f'  Batch size = {args.eval_batch_size}')
    result = evaluate(args, model, dataset, id2label, device)
    
    acc = result['acc']
    f1 = result['f1']
    loss = result['loss']

    logger.info('*** Test result ***')
    logger.info(f'  acc = {acc}')
    logger.info(f'  f1 = {f1}')
    logger.info(f'  loss = {loss}')

    # Save result
    file_test_result = os.path.join(output_dir, consts.FILENAME_TEST_RESULT)
    with open(file_test_result, 'w') as f:
        f.write(f'test_loss\ttest_acc\ttest_f1\n')
        f.write(f'{loss}\t{acc}\t{f1}\n')

    logger.info('Testing finished')


def main(args):
    assert args.do_train or args.do_test, 'At least one of `do_train` and `do_test` has to be true.'
    assert 1 <= args.gradient_accumulation_steps <= args.train_batch_size

    if args.do_train:
        train(args)
    
    if args.do_test:
        test(args)

    logger.info('DONE')


if __name__ == "__main__":
    main(get_argparse().parse_args())
