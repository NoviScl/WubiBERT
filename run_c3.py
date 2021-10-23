# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
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
# """BERT finetuning runner."""

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import argparse
import csv
import json
import logging
import os
import pickle
import random
import time
# from shutil import copyfile

import consts

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import modeling
from optimization import BertAdam, warmup_linear, get_optimizer
from utils import get_freer_gpu, load_tokenizer

from mrc.tools import official_tokenization as tokenization
from mrc.tools import utils

# Contants for C3
NUM_CHOICES = 4  # 数据集里不一定有四个选项，但是会手动加 “无效答案” 至4个
REVERSE_ORDER = False
SA_STEP = False

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, text_c=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class c3Processor(DataProcessor):
    def __init__(self, data_dir, do_train=False, do_eval=False, do_test=False):
        self.D = [[], [], []]
        self.data_dir = data_dir

        for sid in range(3):
        # for sid in range(2):
            # Skip files that are not going to use
            if not do_train:
                if sid == 0:
                    continue
            if not do_eval:
                if sid == 1:
                    continue
            if not do_test:
                if sid == 2:
                    continue
                
            data = []
            for subtask in ["d", "m"]:
                files = ["train.json", "dev.json", "test.json"]
                # files = ['train.json', 'dev.json']
                filename = self.data_dir + "/" + subtask + "-" + files[sid]
                with open(filename, "r", encoding="utf8") as f:
                    data += json.load(f)
            logger.info('Loaded {} examples from "{}"'.format(len(data), filename))
            if sid == 0:
                random.shuffle(data)
            for i in range(len(data)):
                for j in range(len(data[i][1])):
                    d = ['\n'.join(data[i][0]).lower(), data[i][1][j]["question"].lower()]
                    for k in range(len(data[i][1][j]["choice"])):
                        d += [data[i][1][j]["choice"][k].lower()]
                    for k in range(len(data[i][1][j]["choice"]), 4):
                        d += ['无效答案']  # 有些C3数据选项不足4个，添加[无效答案]能够有效增强模型收敛稳定性
                    d += [data[i][1][j]["answer"].lower()]
                    self.D[sid] += [d]

    def get_train_examples(self):
        """See base class."""
        return self._create_examples(self.D[0], "train")

    def get_examples(self):
        """See base class."""
        return self._create_examples(self.D[2], "test")

    def get_dev_examples(self):
        """See base class."""
        return self._create_examples(self.D[1], "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _create_examples(self, data, set_type):
        """Creates examples for the training and dev sets."""
        cache_dir = os.path.join(self.data_dir, set_type + '_examples.pkl')
        # if os.path.exists(cache_dir):
        if False:
            examples = pickle.load(open(cache_dir, 'rb'))
        else:
            examples = []
            for (i, d) in enumerate(data):
                answer = -1
                # 这里data[i]有6个元素，0是context，1是问题，2~5是choice，6是答案
                for k in range(4):
                    if data[i][2 + k] == data[i][6]:
                        answer = str(k)
                label = tokenization.convert_to_unicode(answer)
                for k in range(4):
                    guid = "%s-%s-%s" % (set_type, i, k)
                    text_a = tokenization.convert_to_unicode(data[i][0])
                    text_b = tokenization.convert_to_unicode(data[i][k + 2])
                    text_c = tokenization.convert_to_unicode(data[i][1])
                    examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, text_c=text_c))

            with open(cache_dir, 'wb') as w:
                pickle.dump(examples, w)

        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    # print("#examples", len(examples))

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = [[]]
    for (ex_index, example) in enumerate(tqdm(examples)):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = tokenizer.tokenize(example.text_b)

        tokens_c = tokenizer.tokenize(example.text_c)

        _truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_seq_length - 4)
        tokens_b = tokens_c + ["[SEP]"] + tokens_b

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("label: %s (id = %d)" % (example.label, label_id))

        features[-1].append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id))
        if len(features[-1]) == NUM_CHOICES:
            features.append([])

    if len(features[-1]) == 0:
        features = features[:-1]
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def _truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_length):
    """Truncates a sequence tuple in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_a) >= len(tokens_b) and len(tokens_a) >= len(tokens_c):
            tokens_a.pop()
        elif len(tokens_b) >= len(tokens_a) and len(tokens_b) >= len(tokens_c):
            tokens_b.pop()
        else:
            tokens_c.pop()


def accuracy(logits, labels):
    preds = np.argmax(logits, axis=1)
    return np.sum(preds == labels)


def parse_args():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--tokenizer_type', type=str, required=True)
    parser.add_argument('--vocab_file', type=str, required=True)
    parser.add_argument('--vocab_model_file', type=str, required=True)
    parser.add_argument('--cws_vocab_file', type=str, default='')
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--init_checkpoint', type=str, required=True)

    ## Other parameters
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument('--do_test',  action='store_true')
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--eval_batch_size", default=16, type=int)
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--schedule",
                        default='warmup_linear',
                        type=str,
                        help='schedule')
    parser.add_argument("--weight_decay_rate",
                        default=0.01,
                        type=float,
                        help='weight_decay_rate')
    parser.add_argument('--clip_norm',
                        type=float,
                        default=1.0)
    parser.add_argument("--epochs", default=8, type=int)
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--test_model', type=str, default=None)
    return parser.parse_args()


def get_features(
    examples, 
    data_type, 
    data_dir, 
    max_seq_length,
    tokenizer,
    tokenizer_type,
    vocab_size,
    label_list):

    if data_type == 'eval':
        data_type = 'dev'

    if data_type not in ['train', 'dev', 'test']:
        raise ValueError('Expected "train", "dev" or "test", but got', data_type)

    file_feature = '{}_features_{}_{}_{}.pkl'.format(data_type, max_seq_length, tokenizer_type, vocab_size)
    file_feature = os.path.join(data_dir, file_feature)

    if data_type != 'test' and os.path.exists(file_feature):
    # if False:
        logger.info('Loading features from \"' + file_feature + '\"...')
        features = pickle.load(open(file_feature, 'rb'))
        logger.info('Loaded {} features.'.format(len(features)))
    else:
        logger.info('Converting {} examples into features...'.format(len(examples)))
        features = convert_examples_to_features(examples, label_list, max_seq_length, tokenizer)
        with open(file_feature, 'wb') as w:
            pickle.dump(features, w)
        logger.info('Saved {} features to "{}".'.format(len(features), file_feature))
    return features


def get_device(args):
    if torch.cuda.is_available():
        return torch.device('cuda')  # Only one gpu
        free_gpu = get_freer_gpu()
        return torch.device('cuda', free_gpu)
    else:
        return torch.device('cpu')


def load_model_and_config(config_file, model_file, num_choices):
    config = modeling.BertConfig.from_json_file(config_file)
    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)
    modeling.ACT2FN["bias_gelu"] = modeling.bias_gelu_training
    model = modeling.BertForMultipleChoice(config, num_choices)
    state_dict = torch.load(model_file, map_location='cpu')['model']
    model.load_state_dict(state_dict, strict=False)
    return model, config


def features_to_dataset(features, num_choices):
    input_ids = []
    input_mask = []
    segment_ids = []
    label_id = []
    for f in features:
        input_ids.append([])
        input_mask.append([])
        segment_ids.append([])
        for i in range(num_choices):
            input_ids[-1].append(f[i].input_ids)
            input_mask[-1].append(f[i].input_mask)
            segment_ids[-1].append(f[i].segment_ids)
        label_id.append(f[0].label_id)

    all_input_ids = torch.tensor(input_ids, dtype=torch.long)
    all_input_mask = torch.tensor(input_mask, dtype=torch.long)
    all_segment_ids = torch.tensor(segment_ids, dtype=torch.long)
    all_label_ids = torch.tensor(label_id, dtype=torch.long)
    return TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)


def save_submission(logits, folder):
    # the test submission order can't be changed
    submission_test = os.path.join(folder, "submission_test.json")
    preds = [int(np.argmax(x)) for x in logits]
    with open(submission_test, "w") as f:
        json.dump(preds, f)


def save_logits(logits, filename):
    with open(filename, "w") as f:
        for i in range(len(logits)):
            for j in range(len(logits[i])):
                f.write(str(logits[i][j]))
                if j == len(logits[i]) - 1:
                    f.write("\n")
                else:
                    f.write(" ")


def save_time(start_time, folder):
    stop_time = time.time()
    elapsed = stop_time - start_time
    time_stats = {
        'start_time': str(start_time),
        'stop_time': str(stop_time),
        'elapsed': str(elapsed),
    }
    time_logfile = os.path.join(folder, 'time_log.txt')
    json.dump(time_stats, open(time_logfile, 'w', encoding='utf8'))
      

def evaluate(model, dataloader, device):
    '''Return (logits, acc, loss)'''
    model.eval()
    loss = 0
    acc = 0
    steps, n_examples = 0, 0
    logits_all = []
    for batch in tqdm(dataloader, mininterval=2.0, desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        with torch.no_grad():
            tmp_test_loss, logits = model(input_ids, segment_ids, input_mask,
                                          label_ids, return_logits=True)
        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        for i in range(len(logits)):
            logits_all.append(logits[i])
        tmp_test_accuracy = accuracy(logits, label_ids.reshape(-1))
        loss += tmp_test_loss.mean().item()
        acc += tmp_test_accuracy

        n_examples += input_ids.size(0)
        steps += 1

    loss = loss / (steps + 1e-8)
    acc = acc / (n_examples + 1e-8)
    return logits, acc, loss


def train(args):
    # args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    device = get_device(args)
    n_gpu = torch.cuda.device_count()
    logger.info('Device: ' + str(device))
    logger.info('Num gpus: ' + str(n_gpu))
    
    # Output files
    output_dir = os.path.join(args.output_dir, str(args.seed))
    os.makedirs(output_dir, exist_ok=True)

    filename_scores = os.path.join(output_dir, 'scores.txt')
    filename_params = os.path.join(output_dir, 'params.json')
    logger.info(json.dumps(vars(args), indent=4))
    json.dump(vars(args), open(filename_params, 'w'), indent=4)
    
    # Init file for logging scores
    with open(filename_scores, 'w') as f:
        f.write('\t'.join(['epoch', 'train_loss', 'dev_loss', 'dev_acc']) + '\n')

    # Processor
    logger.info('Loading processor...')
    processor = c3Processor(args.data_dir, do_train=True, do_eval=True)
    label_list = processor.get_labels()

    # Tokenizer
    logger.info('Loading tokenizer...')
    logger.info('vocab file={}, vocab_model_file={}'.format(args.vocab_file, args.vocab_model_file))
    # tokenizer = ALL_TOKENIZERS[args.tokenizer_type](args.vocab_file, args.vocab_model_file)
    tokenizer = load_tokenizer(args)
    real_tokenizer_type = args.output_dir.split(os.path.sep)[-2]

    # Prepare Model
    logger.info('Loading model from checkpoint "{}"...'.format(args.init_checkpoint))
    model, config = load_model_and_config(args.config_file, args.init_checkpoint,
                                          NUM_CHOICES)

    if args.max_seq_length > config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length {} because the BERT model was only trained up to sequence length {}".format(
                args.max_seq_length, config.max_position_embeddings))

    # Save config file
    logger.info('Saving config file...')
    model_to_save = model.module if hasattr(model, 'module') else model
    filename_config = os.path.join(output_dir, modeling.FILENAME_CONFIG)
    with open(filename_config, 'w') as f:
        f.write(model_to_save.config.to_json_string())

    
    # Load training data
    logger.info('Loading training data...')
    train_examples = processor.get_train_examples()
    actual_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    num_train_steps = args.epochs * int(len(train_examples) / NUM_CHOICES / actual_batch_size)

    # Optimizer
    optimizer = get_optimizer(
        model=model,
        float16=False,
        learning_rate=args.learning_rate,
        total_steps=num_train_steps,
        schedule=args.schedule,
        warmup_rate=args.warmup_proportion,
        max_grad_norm=args.clip_norm,
        weight_decay_rate=args.weight_decay_rate,
        opt_pooler=True)  # multi_choice must update pooler

    # Load eval data
    if args.do_eval:
        logger.info('Loading eval data...')
        eval_examples = processor.get_dev_examples()
        eval_features = get_features(
            eval_examples, 
            'eval', 
            args.data_dir, 
            args.max_seq_length,
            tokenizer,
            real_tokenizer_type,
            config.vocab_size,
            label_list)
        eval_data = features_to_dataset(eval_features, NUM_CHOICES)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    train_features = get_features(
        train_examples, 
        'train', 
        args.data_dir,
        args.max_seq_length,
        tokenizer,
        real_tokenizer_type,
        config.vocab_size,
        label_list)
    train_data = features_to_dataset(train_features, NUM_CHOICES)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, 
                                  batch_size=actual_batch_size, drop_last=True)
    model.to(device)
    
    logger.info("***** Running training *****")
    logger.info('  Num epochs = %d', args.epochs)
    logger.info("  Num train examples = %d", len(train_examples))
    logger.info('  Num train features = %d', len(train_features))
    logger.info("  Num eval examples = %d", len(eval_examples))
    logger.info('  Num eval features = %d', len(eval_features))
    logger.info("  Train batch size = %d", args.train_batch_size)
    logger.info("  Eval batch size = %d", args.eval_batch_size)
    logger.info("  Num steps = %d", num_train_steps)
    logger.info("  Grad acc steps = %d", args.gradient_accumulation_steps)
    logger.info("****************************")

    # Start timer
    start_time = time.time()

    train_loss_history = []
    eval_loss_history = []
    eval_acc_history = []

    for ep in range(int(args.epochs)):
        model.train()
        total_loss = 0
        nb_tr_examples, n_train_steps = 0, 0
        for step, batch in tqdm(enumerate(train_dataloader), 
                                desc=f"Training (epoch {ep})",
                                mininterval=2.0,
                                total=len(train_dataloader)):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            loss = model(input_ids, segment_ids, input_mask, label_ids)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            total_loss += loss.item()

            loss.backward()

            nb_tr_examples += input_ids.size(0)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()  # We have accumulated enought gradients
                model.zero_grad()
                n_train_steps += 1
        train_loss = total_loss / (n_train_steps + 1e-8)

        # Evaluation
        if args.do_eval:
            logger.info("***** Running Evaluation *****")
            logits_all, eval_acc, eval_loss = evaluate(model, eval_dataloader,
                                                       device)
            eval_acc_history.append(eval_acc)
            eval_loss_history.append(eval_loss)
            train_loss_history.append(train_loss)

            result = {'eval_loss': eval_loss,
                      'eval_acc': eval_acc,
                      'train_steps': n_train_steps,
                      'train_loss': train_loss}
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
            logger.info('  Epoch = {}'.format(ep))
            logger.info("************************")

            # Log results to scores file
            with open(filename_scores, 'a') as f:
                f.write("{}\t{}\t{}\t{}\n".format(ep, train_loss, eval_loss, eval_acc))

        # Save model if it's best on dev set
        if args.do_eval:
            if len(eval_acc_history) == 0 or eval_acc_history[-1] == max(eval_acc_history):
                model_to_save = model.module if hasattr(model, 'module') else model
                # dir_model = os.path.join(output_dir, 'models')
                # os.makedirs(dir_model, exist_ok=True)
                # filename_model = os.path.join(dir_model, 'model_epoch_' + str(ep) + '.bin')
                # filename_model = os.path.join(output_dir, 'model_epoch_' + str(ep) + '.bin')
                filename_model = os.path.join(output_dir, modeling.FILENAME_BEST_MODEL)
                torch.save(
                    {"model": model_to_save.state_dict()},
                    filename_model,
                )
                # copyfile(filename_model, filename_best_model)
                logger.info('New best model saved')


    logger.info('Training finished')
    
    # Log time to file
    save_time(start_time, output_dir)


def test(args):
    # Output files
    output_dir = os.path.join(args.output_dir, str(args.seed))
    os.makedirs(output_dir, exist_ok=True)
    logger.info(json.dumps(vars(args), indent=4))
    
    device = get_device(args)
    n_gpu = torch.cuda.device_count()
    logger.info('Device: ' + str(device))
    logger.info('Num gpus: ' + str(n_gpu))
    
    batch_size = args.eval_batch_size

    # Tokenizer and processor
    logger.info('Loading tokenizer...')
    tokenizer = load_tokenizer(args)
    
    # Load model
    if args.test_model is not None and len(args.test_model) > 0:
        filename_best_model = args.test_model
    else:
        filename_best_model = os.path.join(output_dir, modeling.FILENAME_BEST_MODEL)
    if args.config_file is not None and len(args.config_file) > 0:
        filename_config = args.config_file
    else:
        filename_config = os.path.join(output_dir, modeling.FILENAME_CONFIG)

    logger.info('Loading model from "{}"...'.format(filename_best_model))
    model, config = load_model_and_config(filename_config, filename_best_model,
                                          NUM_CHOICES)

    # Sanity checks
    if args.max_seq_length > config.max_position_embeddings:
        msg = "Cannot use sequence length {} because the BERT model was only \
trained up to sequence length {}".format(args.max_seq_length, 
                                         config.max_position_embeddings)
        raise ValueError(msg)
    model.to(device)

    real_tokenizer_type = args.output_dir.split(os.path.sep)[-2]
    
    # Load test data
    logger.info('Loading processor...')
    processor = c3Processor(args.data_dir, do_test=True)
    label_list = processor.get_labels()
    logger.info('Loading test data...')
    examples = processor.get_examples()
    examples = processor.get_examples()
    features = get_features(
        examples, 
        'test', 
        args.data_dir, 
        args.max_seq_length,
        tokenizer,
        real_tokenizer_type,
        config.vocab_size,
        label_list)
    test_data = features_to_dataset(features, NUM_CHOICES)
    sampler = SequentialSampler(test_data)
    dataloader = DataLoader(test_data, sampler=sampler, batch_size=batch_size)

    logger.info("***** Running testing *****")
    logger.info('  Num examples = {}'.format(len(examples)))
    logger.info('  Num features = {}'.format(len(features)))
    logger.info('  Batch size   = {}'.format(batch_size))


    # Execute testing
    logits_all, acc, loss = evaluate(model, dataloader, device)

    # Save results
    result = {'test_loss': loss,
              'test_acc': acc}

    result_test_file = os.path.join(output_dir, consts.FILENAME_TEST_RESULT)
    with open(result_test_file, "w") as f:
        logger.info("***** Test results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            f.write("%s = %s\n" % (key, str(result[key])))
        logger.info("************************")

    # Save logits to file
    logger.info('Saving to logits_test.txt')
    file_logits = os.path.join(output_dir, "logits_test.txt")
    save_logits(logits_all, file_logits)

    # Save to submission file
    logger.info('Saving predictions to submission_test.json')
    save_submission(logits_all, output_dir)

    print('Testing finished')


def main(args):
    # Sanity checks
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_eval and not args.do_test:
        raise ValueError("At least one of `do_train` or `do_eval` or `do_test` must be True.")
    
    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.do_train:
        train(args)

    if args.do_test:
        test(args)

    print('DONE')


if __name__ == "__main__":
    main(parse_args())
