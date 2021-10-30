'''
For loading testing results from logs/*
'''
import os
import json
from pathlib import Path

import consts
from result_getter import print_scores, get_ckpts, get_accs, get_accs_all_seed


def normalize_model_name(model):
    normalized = ['pinyin_concat_wubi', 'cangjie', 'pinyin', 'stroke', 'wubi', 
                  'zhengma', 'zhuyin', 'raw', 'bert', 'byte', 'random_index']
    for name in normalized:
        if name in model:
            return name
    assert ValueError(f'"{model}" could not be normalized.')


def get_result_all_seqlen():
    tasks = ['tnews', 'iflytek', 
            #  'thucnews', 'wsc', 'afqmc', 'bq', 'csl',
             'ocnli', 
            #  'chid', 
             'c3',
             ]
    models = [
        'pinyin', 
        'bert', 
        'pinyin_no_index'
        ]
    
    max_seq_lens = [8, 12, 16, 24, 32, 48, 64, 76, 80, 96, 128, 256, 384, 512]
    # max_seq_lens = []
    
    all_models = []
    
    if max_seq_lens == []:
        all_models = models
    else:
        for msl in max_seq_lens:
            for m in models:
                all_models.append(m + '_seqlen' + str(msl))

    result = {}
    for task in tasks:
        task_result = {}
        for model in all_models:
            use_no_index = 'no_index' in model
            model_normalized = normalize_model_name(model)
            ckpt = get_ckpts(use_no_index=use_no_index)[model_normalized]
            
            
            path = Path(os.path.join('logs', task, model, ckpt))
            if path.exists():
                accs = get_accs(task, model, ckpt, verbose=False)
                task_result[model] = accs
            
            # print(f'{task:8}\t{model:24}\t{ckpt}')
            # print_scores(task, model, ckpt=ckpt, 
            #                 two_level_embeddings=False,
            #                 verbose=False)
        result[task] = task_result
    print(json.dumps(result, indent=2))


def get_all_accs():
    tasks = ['cluener', 'cmrc']
    models = ['pinyin', 'wubi', 'raw', 'bert', 'pinyin_no_index']
    suffix = 'chartokens'
    # suffix = 'twolevel'
    # suffix = None
    results = {}
    for task in tasks:
        results[task] = {}
        for model in models:
            ckpt = consts.ALL_BEST_CKPTS[model]
            if suffix:
                model += '_' + suffix
            path = Path('logs', task, model, ckpt)
            accs = get_accs_all_seed(path, verbose=False)
            results[task][model] = accs
            if len(accs) > 0: print(path, sum(accs) / len(accs))
    print(json.dumps(results, indent=2))


def get_all_res():
    tasks = ['cmrc', 'cluener']  # Two level embeddings
    models = [
        'pinyin',
        'wubi', 
        'raw', 
        'bert',
        ]

    suffix = None
    # suffix = 'no_index'
    # suffix = 'shuffled'
    # suffix = 'cws'
    # suffix = '500'
    # suffix = 'shuffled_500'
    suffix = 'chartokens'
    
    max_seq_len = 64

    # Noise settings
    noise_type = None
    list_noise_train = [0]

    # noise_type = 'phonetic'
    # list_noise_test = [10, 20, 30, 40, 50]
    
    # noise_type = 'glyph'
    # list_noise_test = [50, 100]

    # Don't change below
    if suffix is not None:
        # use_shuffled = suffix is not None and 'shuffled' in suffix
        # use_no_index = suffix is not None and 'no_index' in suffix
        use_cws = suffix is not None and 'cws' in suffix
        use_500 = suffix is not None and '500' in suffix
        two_level_embeddings = 'twolevel' in suffix
        avg_char_tokens = 'chartokens' in suffix
    ckpt = consts.ALL_BEST_CKPTS[model]
    # ckpts = get_ckpts(use_no_index, use_shuffled, use_cws, use_500)
    # ckpts['cangjie'] = 'ckpt_8804'
    
    if noise_type != None:
        noise_tasks = []
        for task in tasks:
            for noise_train in list_noise_train:
                for noise_test in list_noise_test:
                    noise_tasks.append(
                        '{}_{}_{}_{}'.format(task, noise_type, noise_train, 
                                            noise_test)
                    )
        tasks = noise_tasks 

    for task in tasks:
        print(task)
        for model in models:
            if model not in ckpts:
                continue
            print(model)
            # if ckpts is None:
            # print(ckpts)
            cs = [ckpts[model]]
            for ckpt in cs:
                print(ckpt)
                model += '_' + suffix
                print_scores(task, model, ckpt=ckpt, 
                             two_level_embeddings=two_level_embeddings,
                            #  use_shuffled=use_shuffled,
                            #  use_no_index=use_no_index,
                             use_cws=use_cws,
                             use_500=use_500,
                             max_seq_len=max_seq_len,
                             verbose=True)
    


def main():
    # get_result_all_seqlen()
    # get_all_res()
    get_all_accs()
    # get_res_long('cmrc', ['raw'])


if __name__ == '__main__':
    main()
