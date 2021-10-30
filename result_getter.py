import os
from pathlib import Path

import consts

TASK_NAMES = ['tnews', 'iflytek', 'afqmc', 'cmrc', 'c3']


def get_result_test(file):
    with open(file, 'r') as f:
        if 'c3' in str(file):
            line = f.readline().strip()
            acc = float(line.split()[-1])
        else:
            next(f)
            line = f.readline().strip()
            acc = float(line.split('\t')[-1])
    return acc


def get_res_long(task, models):
    for model in models:
        if task == 'drcd':
            if model == 'cangjie':
                ckpt = 'ckpt_8804'
            elif model == 'zhuyin':
                ckpt = 'ckpt_7992'
            model += '_simp'
        elif task == 'chid':
            model += '_shuffled_whole_def'

        path = Path(os.path.join(task, model + '_long'))
        
        
        rows = []
        print(model)
        # ckpts = [
        #     6137,
        #     7160,
        #     8184,
        #     9207,
        #     10231,
        #     11255,
        #     12278,
        #     13302,
        #     14080,
        #     15096,
        #     16120,
        #     17143,
        #     18167,
        #     18200,
        # ]
        for ckpt_dir in path.iterdir():
            scores = []
            for seed_dir in ckpt_dir.iterdir():
                file_scores = os.path.join(seed_dir, 'result_test.txt')
                acc = get_result_test(file_scores, task=task)
                print(f'{seed_dir}\t{acc}')
                scores.append(acc)
            avg = sum(scores) / len(scores)
            cnt = len(scores)
            print(f'avg = {avg} ({cnt})')


def _get_one_res(folder: Path):
    file = folder / 'result_test.txt'
    if not file.exists():
        file = folder / 'results_test.txt'
    return get_result_test(file)


def get_accs_all_seed(folder: Path, verbose=False):
    '''Return average, count'''
    seeds = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    accs = []
    for seed in seeds:
        try:
            seed_folder = folder / str(seed)
            acc = _get_one_res(seed_folder)
            accs.append(acc)
            if verbose:
                print(f"  Seed {seed}:", acc)
        except FileNotFoundError:
            continue
    return accs


def get_accs(task, model, ckpt, verbose=False):
    '''Return average, count'''
    folder = Path('logs', task, model, ckpt)
    return get_accs_all_seed(folder, verbose=verbose)


def print_scores(task, model, ckpt, two_level_embeddings=False, use_shuffled=False,
                 use_no_index=False, use_cws=False, use_500=False, max_seq_len=None,
                 suffix=None, verbose=False):
    if use_no_index:
        model += '_no_index'
    if use_shuffled:
        model += '_shuffled'
    if use_cws:
        model += '_cws'
    if use_500:
        model += '_500'
    if two_level_embeddings:
        model += '_twolevel'

    if task == 'drcd':
        if model == 'cangjie':
            ckpt = 'ckpt_8804'
        elif model == 'zhuyin':
            ckpt = 'ckpt_7992'
        model += '_trad'  # 默认使用繁体
    elif task == 'chid':
        model += '_shuffled_whole_def'

    if max_seq_len:
        model += '_seqlen' + str(max_seq_len)

    if suffix:
        model += '_' + suffix

    path = Path(os.path.join('logs', task, model, ckpt))
    rows = []
    if path.exists():
        accs = get_accs(task, model, ckpt, verbose=verbose)
        cnt = len(accs)
        avg = sum(accs) / cnt if cnt != 0 else 0
        print(f'  avg = {avg} ({cnt})')


def get_ckpts(use_no_index, use_shuffled=False, use_cws=False, use_500=False):
    # TODO: deprecated, remove
    if use_no_index:
        return consts.BEST_CKPTS_NO_INDEX
    elif use_shuffled:
        if use_500:
            return consts.BEST_CKPTS_500
        return consts.BEST_CKPTS_SHUFFLED
    elif use_cws:
        return consts.BEST_CKPTS_CWS
    elif use_500:
        return consts.BEST_CKPTS_500
    else:
        return consts.BEST_CKPTS
