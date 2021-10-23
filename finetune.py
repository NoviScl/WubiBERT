'''
Runs all finetuning tasks
'''
import argparse
import json
import os
import subprocess

import consts
from job import Job


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--two_level_embeddings')
    p.add_argument('--test', action='store_true')
    p.add_argument('--max_seq_len', type=int)
    return p.parse_args()

########### Other settings ###########

########### Finetuning settings ###########
DEBUG = False
DONT_RUN = False
DO_TRAIN = True
DO_TEST = True

TWO_LEVEL_EMBEDDINGS = False
# USE_BASE = False
# USE_LONG = False
USE_SHUFFLED = False
USE_NO_INDEX = False
USE_CWS = False
USE_500 = False

# `run_glue.py` has max_seq_len = 128 by default
MAX_SEQ_LENS = [16, 24, 32, 48, 64, 80, 96, 128]  # iflytek
# MAX_SEQ_LENS = [8, 12, 16, 24, 32, 64, 128]       # tnews
MAX_SEQ_LENS = [32, 64, 128, 256, 384, 512]  # c3
MAX_SEQ_LENS = [32, 384]
PROD_SEQLEN_BS = 16384  # Correspond to batch_size = 128 when max_seq_len = 128
EPOCHS = 6

NOISE_TYPE = None
# NOISE_TYPE = 'glyph'
# NOISE_TYPE = 'phonetic'

NOISE_TRAIN = [
    0,
]

NOISE_TEST = [
    # 0,
    # 10,
    # 20,
    # 30,
    # 40,
    50,
    100,
]

SEEDS = [
    10,
    11, 
    12, 
    13, 
    14,
    # 15, 
    # 16, 17, 
    # 18, 19,
]
TOKENIZERS = [
    # 'cangjie',
    # 'pinyin',
    # 'stroke',
    # 'wubi',
    # 'zhengma',
    # 'zhuyin',
    # 'raw',
    'bert',
    # 'pinyin_concat_wubi',
    # 'byte',
    # 'random_index',
]


TASKS = [
    # 'tnews',
    # 'iflytek',
    # 'wsc',
    # 'afqmc',
    # 'csl',
    # 'ocnli',
    # 'cmrc',
    # 'drcd',
    # 'chid',
    'c3',
    # 'lcqmc',
    # 'bq',
    # 'thucnews',
    # 'chid',
    # 'cluener',
    # 'chinese_nli' ,  # Hu Hai's ChineseNLIProbing
]

########### Settings end ###########

def sanity_assert():
    '''Sanity check on global variables'''
    if any(task in ['cmrc', 'drcd', 'cluener'] for task in TASKS):
        assert TWO_LEVEL_EMBEDDINGS
    if any(task not in ['cmrc', 'drcd', 'cluener'] for task in TASKS):
        assert not TWO_LEVEL_EMBEDDINGS
    if USE_NO_INDEX or USE_SHUFFLED:
        assert all(t in ['pinyin', 'wubi'] for t in TOKENIZERS)
    if USE_500:
        assert all(t in ['pinyin', 'wubi', 'byte', 'random_index'] for t in TOKENIZERS)
    if NOISE_TYPE == 'glyph':
        assert NOISE_TEST == [50, 100]
    if NOISE_TYPE == 'phonetic':
        assert NOISE_TEST == [10, 20, 30, 40, 50]


def submit_job(config: dict):
    job = Job(config)
    script = job.get_script()
    env = job.get_vars()

    print('********* Job config *****************')
    print(json.dumps(config, indent=2))
    print('********* Job environment ************')
    print('Script =', script)
    print(json.dumps(env, indent=2))
    print('**************************************')

    if DONT_RUN:    # Just for testing
        return

    output_dir = os.path.join(job.output_dir, str(config['seed']))
    os.makedirs(output_dir, exist_ok=True)

    # Make sure all variables in environment is str, 
    # and bool is "0" or "1".
    for k in env:
        if isinstance(env[k], bool):
            env[k] = str(int(env[k]))
        env[k] = str(env[k])
    env.update(os.environ)
    
    # Execute
    process = subprocess.run([script], env=env)


def get_best_ckpt(tokenizer: str) -> [str]:
    if tokenizer == 'pinyin_concat_wubi':
        return 'ckpt_8840'
    elif USE_500:
        return consts.BEST_CKPTS_500[tokenizer]
    elif USE_NO_INDEX:
        return consts.BEST_CKPTS_NO_INDEX[tokenizer]
    elif USE_SHUFFLED:
        return consts.BEST_CKPTS_SHUFFLED[tokenizer]
    elif USE_CWS:
        return consts.BEST_CKPTS_CWS[tokenizer]
    else:
        return consts.BEST_CKPTS[tokenizer]


def finetune(config: dict, tasks: [str], tokenizers: [str], seeds: [int], 
             noise_type: str=None, noise_train: list=None, noise_test: list=None,
             max_seq_lens: [int]=None):
    '''
    Submit finetuning job using given config
    '''
    def _submit_each_seed(config, seeds):
        for seed in seeds:
            config['seed'] = seed
            submit_job(config)
        
    for task in tasks:
        for tokenizer in tokenizers:
            ckpt = get_best_ckpt(tokenizer)
            config.update({'task': task,
                           'tokenizer': tokenizer,
                           'ckpt': ckpt})
            if noise_type is not None:
                # Loop different noise config
                for noise_train in noise_train:
                    for noise_test in noise_test:
                        config.update({'noise_type': noise_type,
                                       'noise_train': noise_train,
                                       'noise_test': noise_test})
                        _submit_each_seed(config, SEEDS)
            elif max_seq_lens is not None:
                # Loop values for max_seq_len
                for max_seq_len in max_seq_lens:
                    config['max_seq_len'] = max_seq_len
                    config['batch_size'] = PROD_SEQLEN_BS // max_seq_len
                    _submit_each_seed(config, SEEDS)
            else:
                # Ordinary
                _submit_each_seed(config, SEEDS)


def main(args):
    sanity_assert()
    # Default glboal settings
    global_config = {
        'debug': DEBUG,
        'two_level_embeddings': TWO_LEVEL_EMBEDDINGS,
        'use_shuffled': USE_SHUFFLED,
        'use_no_index': USE_NO_INDEX,
        'use_cws': USE_CWS,
        'use_500': USE_500,
        'do_train': DO_TRAIN,
        'do_test': DO_TEST,
    }
    
    # TODO: Add support for args
    # Change config according to args
    if args.test:
        global_config['do_train'] = False
        global_config['do_test'] = True
        
    # if args.max_seq_len != None:
    #     global_config['max_seq_len'] = args.max_seq_len
    
    finetune(global_config, TASKS, TOKENIZERS, SEEDS, noise_type=NOISE_TYPE,
             noise_train=NOISE_TRAIN, noise_test=NOISE_TEST,
             max_seq_lens=MAX_SEQ_LENS)


if __name__ == '__main__':
    main(parse_args())
