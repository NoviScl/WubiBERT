from pathlib import Path
import json
from collections import defaultdict


def get_bin_metrics(labels: list, preds: list) -> dict:
    '''
    Metrics for binary labels.
    Preds can have values other than 0 and 1.
    ```
                    labels
                  |  0  |  1  
            ----- | --- | --- 
    preds     0   |  A  |  B
              1   |  C  |  D 
            other |  E  |  F

    recall of 0:    A / (A + C + E)
    precision of 0: A / (A + B)
    recall of 1:    D / (B + D + F)
    precision of 1: D / (C + D)
    ```
    '''
    assert len(preds) == len(labels)
    n = len(preds)
    mat = [[0, 0] for _ in range(3)]
    for pred, label in zip(preds, labels):
        if pred not in [0, 1]:
            mat[2][label] += 1 
        else:
            mat[pred][label] += 1
    # print(mat)
    result = {}
    if mat[0][0] + mat[0][1] > 0:
        r0 = mat[0][0] / (mat[0][0] + mat[1][0] + mat[2][0])
        p0 = mat[0][0] / (mat[0][0] + mat[0][1])
        f1_0 = 2 * r0 * p0 / (r0 + p0)
        result.update({
            'recall_0': r0,
            'precision_0': p0,
            'f1_0': f1_0,
        })
    if mat[1][1] + mat[1][0] > 0:
        r1 = mat[1][1] / (mat[0][1] + mat[1][1] + mat[2][1])
        p1 = mat[1][1] / (mat[1][1] + mat[1][0])
        f1_1 = 2 * r1 * p1 / (r1 + p1)
        result.update({
            'recall_1': r1,
            'precision_1': p1,
            'f1_1': f1_1,
        })
    if 'f1_0' in result and 'f1_1' in result:
        result['macro_f1'] = (f1_0 + f1_1) / 2

    acc = (mat[0][0] + mat[1][1]) / n
    result['acc'] = acc

    return result


def get_labels(test_file: Path) -> list:
    test_data = [json.loads(line) for line in test_file.open('r')]
    labels = [int(d['label']) for d in test_data]
    return labels


def get_acc(preds, labels) -> float:
    assert len(preds) == len(labels)
    correct = 0
    for a, b in zip(preds, labels):
        if a == b:
            correct += 1
    return correct / len(preds)


def get_preds(dir) -> list:
    preds_file = dir / 'preds.json'
    if preds_file.exists():
        return json.load(preds_file.open())
    try:
        preds_text_file = dir / 'preds_text.json'
        if preds_text_file.exists():
            preds = json.load(preds_text_file.open('r'))
            pred_ids = [label_to_id.get(x, -1) for x in preds]
            return pred_ids
        else:
            preds = dir / 'preds.txt'
            return json.load(preds.open('r'))
    except Exception:
        return None


def get_result(result_dir, noise_type, labels):
    result = defaultdict()
    
    total_pos = labels.count(1)
    total_neg = labels.count(0)
    # print('total:', total_pos, total_neg)

    def get_pos_neg_acc(labels, preds):
        correct_pos = 0
        correct_neg = 0
        for label, pred in zip(labels, preds):
            if label == pred:
                if label == 0:
                    correct_neg += 1
                elif label == 1:
                    correct_pos += 1
                else:
                    raise ValueError
        acc_pos = correct_pos / total_pos
        acc_neg = correct_neg / total_neg
        # print('# correct:', correct_pos, correct_neg)
        return acc_pos, acc_neg

    def get_worst_group_acc(result):
        noisy_preds = []
        noisy_dirs = sorted(result_dir.glob(f'test_noisy_{noise_type}_*'))
        if len(noisy_dirs) != 3:
            return None
        for dir in noisy_dirs:
            preds = get_preds(dir)
            if preds is None:
                return None
            noisy_preds.append(preds)
        count = len(noisy_preds[0])
        same_preds = [None] * count
        for i in range(count):
            if all(preds[i] == noisy_preds[0][i] for preds in noisy_preds):
                same_preds[i] = noisy_preds[0][i]
        acc_pos, acc_neg = get_pos_neg_acc(labels, same_preds)
        result['pos_worst'] = acc_pos
        result['neg_worst'] = acc_neg

    def get_avg_acc(result):
        for pos_neg in ['pos', 'neg']:
            noisy_accs = [result.get(pos_neg + '_' + x, None) for x in ['acc_noisy_1', 'acc_noisy_2', 'acc_noisy_3']]
            if None in noisy_accs:
                return None
            result[f'{pos_neg}_avg'] = sum(noisy_accs) / len(noisy_accs)

    def get_metrics(result: dict, dir):
        # Clean
        preds = get_preds(dir / 'test_clean')
        if preds:
            acc_pos, acc_neg = get_pos_neg_acc(labels, preds)
            result['pos_acc_clean'] = acc_pos
            result['neg_acc_clean'] = acc_neg
        
        # Noisy
        for i in range(1, 4):
            test_name = f'test_noisy_{noise_type}_{i}'
            preds = get_preds(dir / test_name)
            if preds == None:
                continue
            acc_pos, acc_neg = get_pos_neg_acc(labels, preds)
            result[f'pos_acc_noisy_{i}'] = acc_pos
            result[f'neg_acc_noisy_{i}'] = acc_neg

    get_metrics(result, result_dir)
    get_worst_group_acc(result)
    get_avg_acc(result)
    return result


def get_table(
    labels: list,
    results_dir: Path, 
    noise_type: str, 
    model_pattern: str='*',
    ):
    # labels = get_labels(task)

    headers = {
        'model': str,
        'pos_acc_clean': float,
        # 'pos_acc_noisy_1': float,
        # 'pos_acc_noisy_2': float,
        # 'pos_acc_noisy_3': float,
        'pos_avg': float,
        'pos_worst': float,
        'neg_acc_clean': float,
        # 'neg_acc_noisy_1': float,
        # 'neg_acc_noisy_2': float,
        # 'neg_acc_noisy_3': float,
        'neg_avg': float,
        'neg_worst': float,
    }
    # types = list(headers.values())
    headers = list(headers.keys())
    score_names = [
        'pos_acc_clean',
        # 'pos_acc_noisy_1',
        # 'pos_acc_noisy_2',
        # 'pos_acc_noisy_3',
        'pos_avg',
        'pos_worst',
        'neg_acc_clean',
        # 'neg_acc_noisy_1',
        # 'neg_acc_noisy_2',
        # 'neg_acc_noisy_3',
        'neg_avg',
        'neg_worst',
    ]

    print(f'Getting results from {results_dir}')
    all_result = {}
    subdirs = sorted(d for d in results_dir.glob(model_pattern) if d.is_dir())
    for result_dir in subdirs:
        print(result_dir)
        result = get_result(result_dir, noise_type, labels)
        result = {name: result.get(name, None) for name in score_names}
        # row = [result_dir.name] + row
        all_result[result_dir.name] = result
    
    print(json.dumps(all_result, indent=2))
    # headers = [h.replace('_', ' ') for h in headers]
    # print(rows)
    # print_table(rows, headers, types)
    # dump_table(rows, headers, types, results_dir / f'table_{noise_type}.tsv')


def main():
    # results_dir = Path('results/afqmc_balanced_da_noise')
    RESULT_DIR = Path('result/realtypo/afqmc_balanced')
    TEST_FILE = Path('datasets/realtypo/afqmc_balanced/test_clean/test.json')
    labels = get_labels(TEST_FILE)
    model_pattern = '*seed*'  # get all models
    for noise_type in ['keyboard', 'asr']:
        print(f'Getting results for {noise_type}')
        get_table(labels, RESULT_DIR, noise_type, model_pattern)
    

if __name__ == '__main__':
    main()