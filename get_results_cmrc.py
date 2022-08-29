from pathlib import Path
import json




def get_labels(test_file):
    answers = [json.loads(line)['answers'] for line in open(test_file)]
    return answers

def get_worst(labels, output_dir, test_name):
    all_preds = []

    for i in range(1, 4):
        test_dir = output_dir / f'test_noisy_{test_name}_{i}'
        preds_file = test_dir / 'preds.json'
        preds = json.load(open(preds_file))
        preds = list(preds.values())
        # print(preds[:10])
        all_preds.append(preds)
        
        
    n = len(labels)
    match_cnt = 0
    for i in range(n):
        if all(preds[i] in labels[i] for preds in all_preds):
            match_cnt += 1
    return match_cnt / n
        


def get_results(result_dir: Path, test_name: str):
    results = {}
    for model_dir in sorted(result_dir.glob('*')):
        model_result = {}
        try:
            acc = get_worst(labels, model_dir, test_name)
        except:
            pass
        model_result[test_name] = round(acc * 100, 2)
        results[model_dir.name] = model_result
        
    
    # Take average of models with same seed
    keys = sorted(list(results.keys()))
    models = set([key.split('_seed')[0] for key in keys])
    model_to_keys = {model: [key for key in keys if model == key.split('_seed')[0]] for model in models}
    print(model_to_keys)
    avg_results = {}
    for model in models:
        key0 = model_to_keys[model][0]
        model_results = {}
        for score_name in results[key0]:
            scores = [results[key][score_name] for key in model_to_keys[model]]
            avg = sum(scores) / len(scores)
            model_results[score_name] = round(avg, 2)
        avg_results[model] = model_results
    
        
    print(json.dumps(avg_results, indent=2))


if __name__ == '__main__':
    TEST_FILE = Path('/home/chenyingfa/data/readin/cmrc/test_clean/test.json')
    labels = get_labels(TEST_FILE)
    labels = [[x['text'] for x in answers] for answers in labels]
    RESULT_DIR = Path(f'result/cmrc')
    noise_type = 'asr'

    get_results(RESULT_DIR, noise_type)
