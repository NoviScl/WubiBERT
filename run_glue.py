# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

from pathlib import Path
import argparse
# import logging
import os
import json
from time import time
# from shutil import copyfile

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import consts
import modeling
from optimization import BertAdam
from schedulers import LinearWarmUpScheduler
# from apex import amp
from sklearn.metrics import matthews_corrcoef, f1_score
import utils
from utils import is_main_process
from processors.glue import PROCESSORS, convert_examples_to_features

# torch._C._jit_set_profiling_mode(False)
# torch._C._jit_set_profiling_executor(False)

# logging.basicConfig(
#     format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
#     datefmt='%m/%d/%Y %H:%M:%S',
#     level=logging.INFO,
# )
# logger = logging.getLogger(__name__)

FILENAME_BEST_MODEL = 'best_model.bin'
FILENAME_TEST_RESULT = 'result_test.txt'

def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    # elif task_name == "sst-2":
    #     return {"acc": simple_accuracy(preds, labels)}
    # elif task_name == "mrpc":
    #     return acc_and_f1(preds, labels)
    # elif task_name == "sts-b":
        # return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        # use acc for other classification tasks. Add exceptions above.
        return {
            "acc": simple_accuracy(preds, labels),
            "macro_f1": f1_score(labels, preds, average='macro'),
            'micro_f1': f1_score(labels, preds, average='micro'),
        }


def simple_accuracy(preds, labels):
    correct = 0
    for p, l in zip(preds, labels):
        if p == l:
            correct += 1
    return correct / len(preds)


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


# from apex.multi_tensor_apply import multi_tensor_applier


class GradientClipper:
    """
    Clips gradient norm of an iterable of parameters.
    """

    def __init__(self, max_grad_norm):
        self.max_norm = max_grad_norm
        if multi_tensor_applier.available:
            import amp_C
            self._overflow_buf = torch.cuda.IntTensor([0])
            self.multi_tensor_l2norm = amp_C.multi_tensor_l2norm
            self.multi_tensor_scale = amp_C.multi_tensor_scale
        else:
            raise RuntimeError('Gradient clipping requires cuda extensions')

    def step(self, parameters):
        l = [p.grad for p in parameters if p.grad is not None]
        total_norm, _ = multi_tensor_applier(
            self.multi_tensor_l2norm,
            self._overflow_buf,
            [l],
            False,
        )
        total_norm = total_norm.item()
        if (total_norm == float('inf')): return
        clip_coef = self.max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            multi_tensor_applier(
                self.multi_tensor_scale,
                self._overflow_buf,
                [l, l],
                clip_coef,
            )


def parse_args(p=argparse.ArgumentParser()):
    ## Required parameters
    p.add_argument('--train_dir', type=str)
    p.add_argument('--dev_dir', type=str)
    p.add_argument('--test_dir', type=str)
    p.add_argument("--task_name", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True, help="The output directory where the model predictions and checkpoints will be written.")
    p.add_argument("--init_ckpt", type=str, required=True, help="The checkpoint file from pretraining",)
    p.add_argument('--tokenizer_type', type=str, required=True, help="Type of tokenizer")
    p.add_argument('--vocab_file', type=str, required=True, help="Vocabulary mapping/file BERT was pretrainined on")
    p.add_argument('--vocab_model_file', type=str, required=True, help="Model file for sentencepiece")
    p.add_argument("--config_file", type=str, required=True, help="The BERT model config")
    p.add_argument('--test_name', default='test_clean')

    ## Other parameters
    p.add_argument("--max_seq_length", default=128, type=int)
    p.add_argument("--do_train", action='store_true')
    p.add_argument("--do_test", action='store_true')
    p.add_argument("--train_batch_size", default=32, type=int,)
    p.add_argument("--eval_batch_size", default=8, type=int)
    p.add_argument("--lr", default=2e-5, type=float)
    p.add_argument("--epochs", default=-1, type=int)
    p.add_argument("--warmup_prop", default=0.1, type=float, help="Proportion of training to perform linear learning rate warmup ")
    p.add_argument("--no_cuda", action='store_true', help="If true, don't use CUDA")
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--grad_acc_steps', type=int, default=1)
    p.add_argument('--loss_scale', type=float, default=0)
    p.add_argument('--skip_checkpoint', action='store_true', help="Whether to save checkpoints")
    p.add_argument('--two_level_embeddings', action="store_true")
    p.add_argument('--tokenize_char_by_char', action="store_true")
    p.add_argument('--fewshot', type=int, default=0)
    p.add_argument('--test_model', type=str, default=None)
    p.add_argument('--cws_vocab_file', type=str, default=None)
    p.add_argument('--log_interval', type=int, default=40)
    return p.parse_args()


def init_optimizer_and_amp(model, lr, loss_scale, warmup_proportion,
                           num_train_optimization_steps, use_fp16):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [
                p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.01
        },
        {
            'params': [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.0
        },
    ]
    optimizer, scheduler = None, None
    if use_fp16:
        print("using fp16")
        try:
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from "
                              "https://www.github.com/nvidia/apex to use "
                              "distributed and fp16 training.")

        if num_train_optimization_steps is not None:
            optimizer = FusedAdam(
                optimizer_grouped_parameters,
                lr=lr,
                bias_correction=False,
            )
        amp_inits = amp.initialize(
            model,
            optimizers=optimizer,
            opt_level="O2",
            keep_batchnorm_fp32=False,
            loss_scale="dynamic" if loss_scale == 0 else loss_scale,
        )
        model, optimizer = (amp_inits
                            if num_train_optimization_steps is not None else
                            (amp_inits, None))
        if num_train_optimization_steps is not None:
            scheduler = LinearWarmUpScheduler(
                optimizer,
                warmup=warmup_proportion,
                total_steps=num_train_optimization_steps,
            )
    else:
        print("using fp32")
        if num_train_optimization_steps is not None:
            optimizer = BertAdam(
                optimizer_grouped_parameters,
                lr=lr,
                warmup=warmup_proportion,
                t_total=num_train_optimization_steps,
            )
            scheduler = LinearWarmUpScheduler(
                optimizer,
                warmup=warmup_proportion,
                total_steps=num_train_optimization_steps,
            )
    return model, optimizer, scheduler


def gen_tensor_dataset(features, two_level_embeddings):
    all_input_ids = torch.tensor(
        [f.input_ids for f in features],
        dtype=torch.long,
    )
    all_input_mask = torch.tensor(
        [f.input_mask for f in features],
        dtype=torch.long,
    )
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in features],
        dtype=torch.long,
    )
    all_label_ids = torch.tensor(
        [f.label_id for f in features],
        dtype=torch.long,
    )
    if not two_level_embeddings:
        return TensorDataset(
            all_input_ids,
            all_input_mask,
            all_segment_ids,
            all_label_ids,
        )
    else:
        all_token_ids = torch.tensor(
            [f.token_ids for f in features],
            dtype=torch.long,
        )
        all_pos_left = torch.tensor(
            [f.pos_left for f in features],
            dtype=torch.long,
        )
        all_pos_right = torch.tensor(
            [f.pos_right for f in features],
            dtype=torch.long,
        )
        return TensorDataset(
            all_input_ids,
            all_input_mask,
            all_segment_ids,
            all_label_ids,
            all_token_ids,
            all_pos_left,
            all_pos_right,
        )


def dump_predictions(path: str, label_list: list, preds: list, examples: list):
    # label_rmap = {label_idx: label for label, label_idx in label_map.items()}
    preds = {
        example.guid: label_list[preds[i]] for i, example in enumerate(examples)
    }
    json.dump(preds, open(path, 'w'), ensure_ascii=False, indent=2)


def load_model(config_file, filename, num_labels):
    # Prepare model
    config = modeling.BertConfig.from_json_file(config_file)
    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)
    model = modeling.BertForSequenceClassification(config, num_labels=num_labels)
    state_dict = torch.load(filename, map_location='cpu')
    model.load_state_dict(state_dict["model"], strict=False)
    return model


def get_device(args):
    if torch.cuda.is_available():
        return 'cuda'
        # free_gpu = get_freer_gpu()
        # return torch.device('cuda', free_gpu)
    else:
        return torch.device('cpu')


def expand_batch(batch, two_level_embeddings):
    input_ids = batch[0]
    input_mask = batch[1]
    segment_ids = batch[2]
    label_ids = batch[3]

    if two_level_embeddings:
        token_ids = batch[4]
        pos_left = batch[5]
        pos_right = batch[6]
    else:
        token_ids = None
        pos_left = None
        pos_right = None
    return (input_ids, input_mask, segment_ids, label_ids,
            token_ids, pos_left, pos_right)


def evaluate(
    model, 
    dataset, 
    batch_size: int, 
    task_name: str, 
    num_labels: int,
    device='cuda', 
    two_level_embeddings=False,
    ):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size,)
    loss_fct = torch.nn.CrossEntropyLoss()
    
    print('*** Start evaluating ***')
    total_loss = 0
    # Result to gather
    all_label_ids = []
    all_logits = []

    for i, batch in tqdm(enumerate(dataloader), desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)
        (input_ids, input_mask, segment_ids, label_ids,
         token_ids, pos_left, pos_right) = expand_batch(batch, two_level_embeddings)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask,
                           token_ids=token_ids, pos_left=pos_left, pos_right=pos_right,
                           use_token_embeddings=two_level_embeddings)
            total_loss += loss_fct(
                logits.view(-1, num_labels),
                label_ids.view(-1),
            ).mean().item()

        # Get preds and output ids
        all_logits += logits.tolist()
        all_label_ids += label_ids.tolist()
    print("*** Done evaluating ***", flush=True)

    preds = np.argmax(all_logits, axis=1)
    metrics = compute_metrics(task_name, preds, all_label_ids)
    result = {
        'preds': preds,
        'loss': total_loss / len(dataloader),
        'acc': metrics['acc'],
        'f1': float(metrics['macro_f1']),
    }
    return result


def get_features(examples, tokenizer, processor, args):
    features = convert_examples_to_features(
        examples,
        label_list=processor.get_labels(),
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        two_level_embeddings=args.two_level_embeddings,
        char_by_char=args.tokenize_char_by_char,
    )
    # return features[:256]  # TODO: remove on release
    return features


def get_datasets(tokenizer, args, processor):
    train_examples = processor.get_train_examples(args.train_dir)
    train_features = get_features(train_examples, tokenizer, processor, args)
    dev_examples = processor.get_dev_examples(args.dev_dir)
    dev_features = get_features(dev_examples, tokenizer, processor, args)
    train_data = gen_tensor_dataset(train_features, two_level_embeddings=args.two_level_embeddings)
    dev_data = gen_tensor_dataset(dev_features, two_level_embeddings=args.two_level_embeddings)
    return train_data, dev_data


def train(args):
    device = get_device(args)
    n_gpu = torch.cuda.device_count()
    print(f'Device: {device}')
    print(f'Num gpus: {n_gpu}')

    print('Loading processor and tokenizer...')
    processor = PROCESSORS[args.task_name]()
    num_labels = len(processor.get_labels())
    tokenizer = utils.load_tokenizer(args)

    # Setup output files
    # output_dir = os.path.join(args.output_dir, str(args.seed))
    output_dir = Path(args.output_dir)
    json.dump(vars(args), open(output_dir / 'args_train.json', 'w'), indent=2)

    # Load data
    print('Getting training features...')
    train_dataset, dev_dataset = get_datasets(tokenizer, args, processor)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    num_opt_steps = len(train_dataloader) // args.grad_acc_steps * args.epochs


    utils.set_seed(args.seed)
    # Prepare model
    modeling.ACT2FN["bias_gelu"] = modeling.bias_gelu_training
    print('Loading model from "{}"...'.format(args.init_ckpt))
    model = load_model(args.config_file, args.init_ckpt, num_labels)
    model.to(device)

    # Prepare optimizer
    print('Preparint optimizer...')
    model, optimizer, scheduler = init_optimizer_and_amp(
        model,
        args.lr,
        args.loss_scale,
        args.warmup_prop,
        num_opt_steps,
        False,
    )
    loss_fct = torch.nn.CrossEntropyLoss()

    print("*** Start training ***")
    print(f"Batch size = {args.train_batch_size}")
    print(f"# examples = {len(train_dataset)}")
    print(f'# epochs = {args.epochs}')
    print(f"# steps = {len(train_dataloader)}")
    print(f'Total opt. steps = {num_opt_steps}')

    global_step = 0
    train_start_time = time()
    result_history = []
    
    for ep in range(args.epochs):
        print(f'*** Start training epoch {ep} ***', flush=True)
        model.train()
        total_train_loss = 0
        for step, batch in enumerate(train_dataloader):
            # print(step, flush=True)
            batch = tuple(t.to(device) for t in batch)
            (input_ids, input_mask, segment_ids, label_ids,
             token_ids, pos_left, pos_right) = expand_batch(batch, args.two_level_embeddings)
            
            # Forward pass
            logits = model(input_ids, segment_ids, input_mask,
                            token_ids=token_ids, pos_left=pos_left, pos_right=pos_right,
                            use_token_embeddings=args.two_level_embeddings)
            loss = loss_fct(
                logits.view(-1, num_labels),
                label_ids.view(-1),
            )
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
                
            # Backward pass
            loss /= args.grad_acc_steps
            loss.backward()
            if (step + 1) % args.grad_acc_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
            total_train_loss += loss.item()
            global_step += 1
                
            # Log
            if step % args.log_interval == 0:
                print(f'step: {global_step / len(train_dataloader):.2f}, '
                      f'loss: {total_train_loss / (step + 1)} '
                      f'lr: {scheduler.get_lr()[0]} '
                      f'time_elapsed: {time() - train_start_time}',
                      flush=True)
        
        train_loss = total_train_loss / len(train_dataloader)

        # Evaluation
        if is_main_process():
            eval_result = evaluate(
                model,
                dev_dataset,
                args.eval_batch_size,
                task_name=args.task_name,
                num_labels=num_labels,
                two_level_embeddings=args.two_level_embeddings,
            )

            epoch_result = {
                'epoch': ep,
                'train_loss': train_loss,
                'dev_loss': eval_result['loss'],
                'dev_acc': eval_result['acc'],
                'dev_f1': eval_result['f1'],
            }
            result_history.append(epoch_result)
            # Log to stdout and file
            print("*** Results ***")
            print(epoch_result)
            json.dump(result_history, open(output_dir / 'scores.json', 'w'), 
                      indent=2, ensure_ascii=False)

            # Save checkpoint
            if not args.skip_checkpoint:
                # model_to_save = model.module if hasattr(model, 'module') else model
                ckpt_dir = output_dir / f'ckpt-{ep}'
                ckpt_dir.mkdir(exist_ok=True, parents=True)
                ckpt_file = ckpt_dir / f'model.pt'
                print(f'Saving model to {ckpt_file}')
                torch.save({"model": model.state_dict()}, ckpt_file)
                json.dump(
                    {k: eval_result[k] for k in ['acc', 'loss', 'f1']}, 
                    open(ckpt_dir / 'result.json', 'w'), 
                    indent=2, ensure_ascii=False)

    print('*** Training finished ***')
    print(f'time_elapsed: {time() - train_start_time}')


def get_best_ckpt(output_dir: Path) -> Path:
    min_loss = float('inf')
    best_ckpt_dir = None
    for ckpt_dir in output_dir.glob('ckpt-*'):
        if not ckpt_dir.is_dir(): continue
        loss = json.load(open(ckpt_dir / 'result.json'))['loss']
        if loss < min_loss:
            min_loss = loss
            best_ckpt_dir = ckpt_dir
    
    return best_ckpt_dir / 'model.pt'


def test(args):
    # Setup output files
    # output_dir = os.path.join(args.output_dir, str(args.seed))
    output_dir = Path(args.output_dir, args.test_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args)
    utils.set_seed(args.seed)

    # Tokenizer and processor
    print('Loading processor and tokenizer...')
    processor = PROCESSORS[args.task_name]()
    num_labels = len(processor.get_labels())
    tokenizer = utils.load_tokenizer(args)

    # Load best model
    if args.test_model:
        best_model_filename = args.test_model
    else:
        best_model_filename = get_best_ckpt(output_dir.parent)
    print('Loading model from "{}"...'.format(best_model_filename))
    model = load_model(args.config_file, best_model_filename, num_labels)
    print('Loaded model from "{}"'.format(best_model_filename))
    model.to(device)
    
    # Load test data
    print('Loading test data from "{}"'.format(args.test_dir))
    examples = processor.get_test_examples(args.test_dir)
    features = get_features(examples, tokenizer, processor, args)
    examples = examples[:len(features)]
    dataset = gen_tensor_dataset(features, two_level_embeddings=args.two_level_embeddings)

    result = evaluate(
        model,
        dataset,
        batch_size=args.eval_batch_size,
        task_name=args.task_name,
        num_labels=num_labels,
        two_level_embeddings=args.two_level_embeddings)

    # Save result
    json.dump(result['preds'].tolist(), open(output_dir / 'preds.json', 'w'), indent=2, ensure_ascii=False)
    del result['preds']
    json.dump(result, open(output_dir / 'result.json', 'w'), ensure_ascii=False)
    print(result)


    print('Test finished')


def main(args):
    print("Arguments:")
    print(json.dumps(vars(args), indent=4), flush=True)

    # Setup output files
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    filename_params = os.path.join(output_dir, consts.FILENAME_PARAMS)
    json.dump(vars(args), open(filename_params, 'w'), indent=4)


    # Sanity check on arguments
    if not args.do_train and not args.do_test:
        raise ValueError("At least one of `do_train` or `do_test` must be True.")
    if args.grad_acc_steps < 1:
        raise ValueError("Invalid grad_acc_steps parameter: {}, "
                         "should be >= 1".format(
                             args.grad_acc_steps))
    # if args.grad_acc_steps > args.train_batch_size:
    #     raise ValueError("grad_acc_steps ({}) cannot be larger "
    #                      "train_batch_size ({}) - there cannot be a fraction "
    #                      "of one sample.".format(
    #                          args.grad_acc_steps,
    #                          args.train_batch_size,
    #                      ))
    # args.train_batch_size = (args.train_batch_size //
                            #  args.grad_acc_steps)

    # Set seed
    if args.do_train:
        train(args)
    if args.do_test:
        test(args)
    print('DONE', flush=True)


if __name__ == "__main__":
    main(parse_args())
