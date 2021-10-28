# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
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

import os
import random
import json

import numpy as np
import torch
import torch.distributed as dist

from pathlib import Path
from tokenization import ALL_TOKENIZERS
import consts


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_main_process():
    return get_rank() == 0


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def format_step(step):
    if isinstance(step, str):
        return step
    s = ""
    if len(step) > 0:
        s += "Training Epoch: {} ".format(step[0])
    if len(step) > 1:
        s += "Training Iteration: {} ".format(step[1])
    if len(step) > 2:
        s += "Validation Iteration: {} ".format(step[2])
    return s


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def mkdir_by_main_process(path):
    if is_main_process():
        mkdir(path)
    barrier()


def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
        free_gpu = get_freer_gpu()
        return torch.device('cuda', free_gpu)
    else:
        return torch.device('cpu')


def output_dir_to_tokenizer_name(output_dir):
    tokenizer_types = [
        'pinyin_concat_wubi',
        'pinyin_shuffled',
        'wubi_shuffled',
        'pinyin_no_index',
        'wubi_no_index',

        # New CWS
        'pinyin_cws',
        'wubi_cws',

        # Old CWS
        'cws_raw',
        'cws_wubi',
        'cws_zhuyin',

        # Ordinary
        'cangjie',
        'stroke',
        'pinyin',
        'wubi',
        'zhengma',
        'zhuyin',
        'raw',
        'bert',
        'byte',
        'random_index',
    ]
    out_dir = output_dir.split(os.path.sep)[-2]
    for t in tokenizer_types:
        if t in out_dir:
            return t
    return None


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def json_save_by_line(data, filename):
    with open(filename, 'w', encoding='utf8') as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False))
            f.write('\n')


def json_load_by_line(filename, n_lines=None):
    '''Load `n_lines` json dicts from file `filename`, where each line in the 
    file is a json dict.'''
    data = []
    with open(filename, 'r', encoding='utf8') as f:
        for line in f:
            data.append(json.loads(line))
            # Break if loaded `n_lines` number of examples
            if n_lines is not None:
                assert n_lines > 0
                if len(data) == n_lines:
                    break
    return data


def get_subchar_pos(tokens, subchars):
	'''
	Return starting index of each subchar in tokens.
	NOTE: This assumes that the concatenation of tokens is equal to the 
	concatenation of subchars.

	Example:
	>>> Input:
	>>> subchars  = ['jin+', 'ti', 'an+', 'ti', 'an+', 'qi+', 'hen+', 'hao+']
	>>> tokens    = ['jin', '+', 'tian+', 'tian+qi', '+', 'hen+hao+']
	>>> token_pos = [0, 2, 2, 3, 3, 3, 5, 5]
	'''
	pos = [None] * len(subchars)
	len_t = 0
	len_s = 0
	j = -1  # idx of last token that was added to len_t
	for i, subchar in enumerate(subchars):
		while len_t <= len_s:
			j += 1
			len_t += len(tokens[j])
		pos[i] = j
		len_s += len(subchar)
	return pos


def load_tokenizer(args):
    if args.tokenizer_type == 'CWS':
        tokenizer = ALL_TOKENIZERS[args.tokenizer_type](args.vocab_file, args.vocab_model_file, args.cws_vocab_file)
    else:
        tokenizer = ALL_TOKENIZERS[args.tokenizer_type](args.vocab_file, args.vocab_model_file)
    return tokenizer


def tokenizer_to_tokenizer_type(tokenizer) -> str:
    for tok_type, t in ALL_TOKENIZERS.items():
        if isinstance(tokenizer, t):
            return tok_type
    raise ValueError('Unrecognized tokenizer class')


def name_of_tokenizer(tokenizer):
    tok_type = tokenizer_to_tokenizer_type(tokenizer)
    type_to_unique_name = {'RawZh': 'raw',
                           'BertZh': 'bert',
                           'Byte': 'byte',
                           'RandomIndex': 'random_index'}
    names = ['cangjie', 'stroke', 'pinyin', 'wubi', 'zhengma', 'zhuyin']
    if tok_type in type_to_unique_name:
        return type_to_unique_name[tok_type]
    elif tok_type == 'CommonZh':
        for name in names:
            if name in tokenizer.vocab_file:
                return name
    elif tok_type == 'CommonZhNoIndex':
        for name in names:
            if name in tokenizer.vocab_file:
                return name + '_no_index'
    else:
        raise ValueError('Unrecognized tokenizer type')
    

def normalize_tokenizer_name(name):
    '''Remove tokenizer suffices'''
    names = ['pinyin_concat_wubi',
             'cangjie',
             'pinyin',
             'stroke',
             'wubi',
             'zhengma',
             'zhuyin',
             'raw',
             'bert',
             'byte',
             'random_index']
    for n in names:
        if n in name:
            return n
    raise ValueError("Unrecognized tokenizer name: " + name)
    
    
def auto_tokenizer(name):
    tokenizer_type = consts.TOKENIZER_TYPES[name]
    vocab_file = consts.ALL_VOCAB_FILES[name]
    model_file = vocab_file.replace('.vocab', '.model')
    return ALL_TOKENIZERS[tokenizer_type](vocab_file, model_file)