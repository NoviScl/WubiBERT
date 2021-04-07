"""
@name = 'roberta_wwm_ext_large'
@author = 'zhangxinrui'
@time = '2019/11/15'
roberta_wwm_ext_large 的baseline版本

coding=utf-8
Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import print_function

import argparse
import os
import random
import logging
import json
from shutil import copyfile

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

from file_utils import PYTORCH_PRETRAINED_BERT_CACHE
import modeling
from tokenization import BertTokenizer, ConcatSepTokenizer, WubiZhTokenizer, RawZhTokenizer, BertZhTokenizer
from optimization import BertAdam, warmup_linear, get_optimizer
from schedulers import LinearWarmUpScheduler
from utils import mkdir

# from google_albert_pytorch_modeling import AlbertConfig, AlbertForMultipleChoice
from mrc.preprocess.CHID_preprocess import RawResult, get_final_predictions, write_predictions, generate_input, evaluate
from mrc.pytorch_modeling import ALBertConfig, ALBertForMultipleChoice
from mrc.pytorch_modeling import BertConfig, BertForMultipleChoice
# from tools.official_tokenization import BertTokenizer
# from tools.pytorch_optimization import get_optimization, warmup_linear


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


ALL_TOKENIZERS = {
    "ConcatSep": ConcatSepTokenizer,
    "WubiZh": WubiZhTokenizer,
    "RawZh": RawZhTokenizer,
    "BertZh": BertZhTokenizer,
    "Bert": BertTokenizer,
    "BertHF": BertTokenizer
}


def reset_model(args, bert_config, model_cls):
    # Prepare model
    model = model_cls(bert_config, num_choices=args.max_num_choices)
    if args.init_checkpoint is not None:
        print('load bert weight')
        state_dict = torch.load(args.init_checkpoint, map_location='cpu')
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})

            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(model, prefix='' if hasattr(model, 'bert') else 'bert.')
        print("missing keys:{}".format(missing_keys))
        print('unexpected keys:{}'.format(unexpected_keys))
        print('error msgs:{}'.format(error_msgs))

    if args.fp16:
        model.half()

    return model


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--gpu_ids", default='', required=True, type=str)
    # parser.add_argument("--bert_config_file", required=True,
    #                     default='check_points/pretrain_models/roberta_wwm_ext_large/bert_config.json')
    # parser.add_argument("--vocab_file", required=True,
    #                     default='check_points/pretrain_models/roberta_wwm_ext_large/vocab.txt')
    # parser.add_argument("--init_restore_dir", required=True,
    #                     default='check_points/pretrain_models/roberta_wwm_ext_large/pytorch_model.pth')
    # parser.add_argument("--input_dir", required=True, default='dataset/CHID')
    # parser.add_argument("--output_dir", required=True, default='check_points/CHID')

    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--vocab_file', type=str, required=True)
    parser.add_argument('--vocab_model_file', type=str, required=True)
    parser.add_argument('--init_checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--tokenizer_type', type=str, required=True)

    ## Other parameters
    # parser.add_argument("--train_file", default='./origin_data/CHID/train.json', type=str,
    #                     help="SQuAD json for training. E.g., train-v1.1.json")
    # parser.add_argument("--train_ans_file", default='./origin_data/CHID/train_answer.json', type=str,
    #                     help="SQuAD answer for training. E.g., train-v1.1.json")
    # parser.add_argument("--predict_file", default='./origin_data/CHID/dev.json', type=str,
    #                     help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    # parser.add_argument("--predict_ans_file", default='origin_data/CHID/dev_answer.json', type=str,
    #                     help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--train_ans_file', type=str, required=True)
    parser.add_argument('--predict_file', type=str, required=True)
    parser.add_argument('--predict_ans_file', type=str, required=True)

    parser.add_argument("--max_seq_length", default=64, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--max_num_choices", default=10, type=int,
                        help="The maximum number of cadicate answer,  shorter than this will be padded.")
    parser.add_argument("--train_batch_size", default=20, type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=16, type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--warmup_proportion", default=0.06, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                             "of training.")
    parser.add_argument("--num_train_epochs", type=int, required=True)
    # parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--do_lower_case", default=True,
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument('--fp16', default=False, action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")

    args = parser.parse_args()

    # Manage output files
    output_dir = os.path.join(args.output_dir, str(args.seed))
    filename_scores = os.path.join(args.output_dir, 'scores.txt')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    logger.info("Arguments:")
    logger.info(json.dumps(vars(args), indent=4))
    filename_params = os.path.join(output_dir, 'params.json')
    json.dump(vars(args), open(filename_params, 'w'), indent=4)

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("device: {} n_gpu: {}, 16-bits training: {}".format(device, n_gpu, args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


    # if os.path.exists(args.input_dir) == False:
        # os.makedirs(args.input_dir, exist_ok=True)


    # tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)
    logger.info('Loading tokenizer...')
    tokenizer = ALL_TOKENIZERS[args.tokenizer_type](args.vocab_file, args.vocab_model_file)
    
    # Load train data
    logger.info('Generating train features...')
    suffix = '_{}_{}.pkl'.format(str(args.max_seq_length), args.tokenizer_type)
    train_example_file = os.path.join(args.data_dir, 'train_examples' + suffix)
    train_feature_file = os.path.join(args.data_dir, 'train_features' + suffix)

    train_features = generate_input(
        args.train_file, 
        args.train_ans_file, 
        train_example_file, 
        train_feature_file,
        tokenizer,
        max_seq_length=args.max_seq_length,
        max_num_choices=args.max_num_choices,
        is_training=True)

    # Load dev data
    logger.info("Generating dev features...")
    dev_example_file = os.path.join(args.data_dir, 'dev_examples' + suffix)
    dev_feature_file = os.path.join(args.data_dir, 'dev_features' + suffix)

    eval_features = generate_input(
        args.predict_file, 
        None, 
        dev_example_file, 
        dev_feature_file, 
        tokenizer,
        max_seq_length=args.max_seq_length, 
        max_num_choices=args.max_num_choices,
        is_training=False)

    logger.info("Train features {}".format(len(train_features)))
    num_train_steps = int(
        len(train_features) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    logger.info("Loaded train dataset")

    logger.info("Num generate examples = {}".format(len(train_features)))
    logger.info("Batch size = {}".format(args.train_batch_size))
    logger.info("Num steps for a epoch = {}".format(num_train_steps))

    # Transform data
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_masks = torch.tensor([f.input_masks for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_choice_masks = torch.tensor([f.choice_masks for f in train_features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in train_features], dtype=torch.long)

    train_data = TensorDataset(all_input_ids, all_input_masks, all_segment_ids, all_choice_masks, all_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size,
                                  drop_last=True)

    all_example_ids = [f.example_id for f in eval_features]
    all_tags = [f.tag for f in eval_features]
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_masks = torch.tensor([f.input_masks for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_choice_masks = torch.tensor([f.choice_masks for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_masks, all_segment_ids, all_choice_masks,
                              all_example_index)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)

    # Prepare model
    logger.info('Preparing model from checkpoint {}'.format(args.init_checkpoint))
    config = modeling.BertConfig.from_json_file(args.config_file)
    model = modeling.BertForMultipleChoice(config, args.max_num_choices)
    model.load_state_dict(
        torch.load(args.init_checkpoint, map_location='cpu')["model"],
        strict=False,
    )
    model = model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # optimizer = get_optimization(model,
    #                              float16=args.fp16,
    #                              learning_rate=args.learning_rate,
    #                              total_steps=num_train_steps,
    #                              schedule='warmup_linear',
    #                              warmup_rate=args.warmup_proportion,
    #                              weight_decay_rate=0.01,
    #                              max_grad_norm=1.0,
    #                              opt_pooler=True)
    optimizer = get_optimizer(
        model,
        float16=args.fp16,
        learning_rate=args.learning_rate,
        total_steps=num_train_steps,
        schedule='warmup_linear',
        warmup_rate=args.warmup_proportion,
        weight_decay_rate=0.01,
        max_grad_norm=1.0,
        opt_pooler=True)

    # Save config
    model_to_save = model.module if hasattr(model, 'module') else model
    filename_config = os.path.join(output_dir, modeling.CONFIG_NAME)
    with open(filename_config, 'w') as f:
        f.write(model_to_save.config.to_json_string())
    
    global_step = 0
    best_acc = None
    acc = 0
    
    dev_acc_history = []
    train_loss_history = []

    # Start training and evaluation
    logger.info('***** Training *****')
    logger.info('Number of examples: {}'.format(len(train_data)))
    logger.info('Number of epochs: ' + str(args.num_train_epochs))
    logger.info('Batch size: ' + str(args.train_batch_size))
    for ep in range(int(args.num_train_epochs)):
        num_step = 0
        average_loss = 0
        model.train()
        model.zero_grad()  # 等价于optimizer.zero_grad()
        steps_per_epoch = num_train_steps // args.num_train_epochs
        with tqdm(total=int(steps_per_epoch), desc='Epoch %d' % (ep + 1)) as pbar:
            for step, batch in enumerate(train_dataloader):
                if n_gpu == 1:
                    batch = tuple(t.to(device) for t in batch)  # multi-gpu does scattering it-self
                input_ids, input_masks, segment_ids, choice_masks, labels = batch
                if step == 0 and ep == 0:
                    logger.info('shape of input_ids: {}'.format(input_ids.shape))
                    logger.info('shape of labels: {}'.format(labels.shape))
                loss = model(input_ids=input_ids,
                             token_type_ids=segment_ids,
                             attention_mask=input_masks,
                             labels=labels)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                    # modify learning rate with special warm up BERT uses
                    # if args.fp16 is False, BertAdam is used and handles this automatically
                    lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_steps,
                                                                      args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                else:
                    loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1


                train_loss_history.append(loss.item())
                average_loss += loss.item()
                num_step += 1

                pbar.set_postfix({'loss': '{0:1.5f}'.format(average_loss / (num_step + 1e-5))})
                pbar.update(1)

        logger.info("***** Running predictions *****")
        logger.info("Num split examples = {}".format(len(eval_features)))
        logger.info("Batch size = {}".format(args.predict_batch_size))

        model.eval()
        all_results = []
        logger.info("Start evaluating")
        for input_ids, input_masks, segment_ids, choice_masks, example_indices in tqdm(eval_dataloader,
                                                                                       desc="Evaluating",
                                                                                       disable=None):
            if len(all_results) == 0:
                print('shape of input_ids: {}'.format(input_ids.shape))
            input_ids = input_ids.to(device)
            input_masks = input_masks.to(device)
            segment_ids = segment_ids.to(device)
            with torch.no_grad():
                batch_logits = model(input_ids=input_ids,
                                     token_type_ids=segment_ids,
                                     attention_mask=input_masks,
                                     labels=None)
            for i, example_index in enumerate(example_indices):
                logits = batch_logits[i].detach().cpu().tolist()
                eval_feature = eval_features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                all_results.append(RawResult(unique_id=unique_id,
                                             example_id=all_example_ids[unique_id],
                                             tag=all_tags[unique_id],
                                             logit=logits))

        predict_file = 'dev_predictions.json'
        logger.info('decoder raw results')

        tmp_predict_file = os.path.join(output_dir, "raw_predictions.pkl")
        output_prediction_file = os.path.join(output_dir, predict_file)
        results = get_final_predictions(all_results, tmp_predict_file, g=True)
        write_predictions(results, output_prediction_file)
        
        logger.info('predictions saved to {}'.format(output_prediction_file))

        if args.predict_ans_file:
            acc = evaluate(args.predict_ans_file, output_prediction_file)
            logger.info(f'{args.predict_file} 预测精度：{acc}')
            dev_acc_history.append(acc)
        
         # Save model
        model_to_save = model.module if hasattr(model, 'module') else model
        model_filename = os.path.join(output_dir, modeling.WEIGHTS_NAME + '_' + str(ep))
        torch.save(
            {"model": model_to_save.state_dict()},
            model_filename,
        )

        # Save best model
        if best_acc is None or acc > best_acc:
            best_acc = acc
            best_model_filename = os.path.join(output_dir, modeling.WEIGHTS_NAME + '_best')
            copyfile(model_filename, best_model_filename)
            # model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            # torch.save(model_to_save.state_dict(), output_model_file)
            # logger.info('save trained model from {}'.format(output_model_file))
            logger.info('New best model saved')


if __name__ == "__main__":
    main()
