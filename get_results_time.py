import json
import os

from result_getter import get_ckpts
from get_results import normalize_model_name


def get_avg_time(task, model, ckpt, seeds):
    list_time = []
    for seed in seeds:
        filename = os.path.join('logs', task, model, ckpt, str(seed), 'time_log.txt')
        if not os.path.exists(filename):
            continue
        time_data = json.load(open(filename, 'r'))
        list_time.append(float(time_data['elapsed']))
    return list_time
    
tasks = ['tnews', 'iflytek', 'ocnli', 'c3']
models = ['bert', 'pinyin', 'pinyin_no_index']
seq_lens = [8, 12, 16, 24, 32, 48, 64, 76, 80, 96, 128, 256, 384, 512]

seeds = list(range(10, 20))

result = {}
for task in tasks:
    task_result = {}
    for model in models:
        model_normalized = normalize_model_name(model)
        use_no_index = 'no_index' in model
        ckpt = get_ckpts(use_no_index=use_no_index)[model_normalized]
        for seq_len in seq_lens:
            model_name = model + '_seqlen' + str(seq_len)
            data = get_avg_time(task, model_name, ckpt, seeds)
            if data != []:
                # print(seq_len)
                # print(data)
                task_result[model_name] = data
                cnt = len(data)
                avg = round(sum(data) / len(data), 2)
                _max = round(max(data) - avg, 2)
                _min = round(min(data) - avg, 2)
                # print(f'{avg} ({cnt}), [{_min}, +{_max}]')
    result[task] = task_result
        
print(json.dumps(result, indent=2))