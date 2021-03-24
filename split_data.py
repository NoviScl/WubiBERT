import json
import random
random.seed(2021)

def _read_json(input_file):
    examples = []
    with open(input_file, "r") as f:
        data = f.readlines()
    for line in data:
        line = json.loads(line.strip())
        examples.append(line)
    return examples 

orig_data = _read_json('tnews_public/train.json')
dev_size = len(orig_data) // 10 ## sample 1/10 as dev set

dev_idx = random.sample(list(range(len(orig_data))), dev_size)

train_set = []
dev_set = []
for idx, eg in enumerate(orig_data):
    if idx in dev_idx:
        dev_set.append(eg)
    else:
        train_set.append(eg)

with open('tnews_public/tnews/dev.json', 'w+') as f:
    for eg in dev_set:
        json.dump(eg, f, ensure_ascii=False)
        f.write('\n')

with open('tnews_public/tnews/train.json', 'w+') as f:
    for eg in train_set:
        json.dump(eg, f, ensure_ascii=False)
        f.write('\n')