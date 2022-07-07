import json
from pathlib import Path
from tokenization import ALL_TOKENIZERS
from levenshtein import correct_ratio


TOK_INFO = {
    "char": {
        'tok_type': "BertZh",
        'vocab_name': "bert_chinese_uncased_22675",
    },
    'raw': {
        'tok_type': 'RawZh',
        'vocab_name': 'raw_zh_22675',
    },
    'pinyin': {
        'tok_type': 'CommonZh',
        'vocab_name': 'pinyin_zh_22675',
    },
    'pinyin_no_index': {
        'tok_type': 'CommonZhNoIndex',
        'vocab_name': 'pinyin_no_index_22675',
    },
    'pypinyin': {
        'tok_type': 'Pypinyin',
        'vocab_name': 'pypinyin_22675_notone_noindex',
    },
    'pypinyin_nosep': {
        'tok_type': 'PypinyinNosep',
        'vocab_name': 'pypinyin_22675_notone_noindex_nosep',
    }
}


def get_tokenizer(model_name):
    vocab_name = TOK_INFO[model_name]['vocab_name']
    tok_type = TOK_INFO[model_name]['tok_type']
    return ALL_TOKENIZERS[tok_type](
        f'tokenizers/{vocab_name}.vocab',
        f'tokenizers/{vocab_name}.model',
    )

def load(file: Path, cnt: int) -> dict:
    data = {}
    for line in open(file):
        if len(data) == cnt: break
        row = json.loads(line)
        # data.append(row['sentence1'] + '；' + row['sentence2'])
        data[row['id']] =  row['question']
    return data


def tokenize(tokenizer, data: list) -> dict:
    tokens = {}
    for key, text in data.items():
        tokens[key] = tokenizer.tokenize(text)
    return tokens


def main():
    n = None

    # data_dir = Path('/home/chenyingfa/WubiBERT/datasets/realtypo/afqmc_balanced')
    data_dir = Path('/home/chenyingfa/WubiBERT/datasets/realtypo/cmrc')
    clean_data = load(data_dir / 'test_clean/test.json', n)
    noisy_data = load(data_dir / 'test_noisy_keyboard_1/test.json', n)
    model_name = 'raw'
    model_names = []
    preserved_ids = []
    for model_name in TOK_INFO:
        tokenizer = get_tokenizer(model_name)
        clean_tokens = tokenize(tokenizer, clean_data)
        noisy_tokens = tokenize(tokenizer, noisy_data)
        # print(clean_tokens)
        # print(noisy_tokens)
        ratio_sum = 0
        for key in clean_tokens:
            clean = clean_tokens[key]
            noisy = noisy_tokens[key]
            # r = correct_ratio(clean, noisy)
            r = 1 if clean == noisy else 0
            ratio_sum += r
            
            if clean == noisy:
                preserved_ids.append(key)
        # print(model_name)
        model_names.append(model_name)
        avg_ratio = ratio_sum / len(clean_tokens)
        print(f'{100*avg_ratio:.2f}')
        
        file_preserved = f'cmrc_debug/preserved_{model_name}.json'
        json.dump(preserved_ids, open(file_preserved, 'w'), ensure_ascii=False, indent=2)
    print(model_names)


if __name__ == '__main__':
    main()