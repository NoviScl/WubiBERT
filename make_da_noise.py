import json
import random
import pickle
from tokenization import CommonZhNoIndexTokenizer
import string
from pathlib import Path

vocab_prefix = 'tokenizers/wubi_no_index_22675'
tokenizer = CommonZhNoIndexTokenizer(f'{vocab_prefix}.vocab', 
                                     f'{vocab_prefix}.model')

wubi2ch = "data/wubi_to_chinese.pkl"
ch2wubi = "data/chinese_to_wubi.pkl"

pinyin2ch = "data/pinyin_to_chinese.pkl"
ch2pinyin = "data/chinese_to_pinyin.pkl"

CH2EN_PUNC = {f: t
              for f, t in zip(
                  u'，。！？【】（）％＃＠＆１２３４５６７８９０；：',
                  u',.!?[]()%#@&1234567890;:')}
puncs = list(string.punctuation) + list(CH2EN_PUNC.keys()) + list(CH2EN_PUNC.values())
puncs = list(set(puncs))

def load_dict(dict_path):
	return pickle.load(open(dict_path, "rb"))

ch2wubi = load_dict(ch2wubi)
wubi2ch = load_dict(wubi2ch)

ch2pinyin = load_dict(ch2pinyin)
pinyin2ch = load_dict(pinyin2ch)

## wubi dict has some punctuations
## we don't use puncs
for p in puncs:
    if p in ch2wubi:
        del ch2wubi[p]

## contruct same-encoding dict
same_dict_pinyin = {}
for c,enc in ch2pinyin.items():
    enc = ''.join([i for i in enc if not i.isdigit()])
    if enc not in same_dict_pinyin:
        same_dict_pinyin[enc] = [c]
    else:
        same_dict_pinyin[enc].append(c)

same_dict_wubi = {}
for c,enc in ch2wubi.items():
    enc = ''.join([i for i in enc if not i.isdigit()])
    if enc not in same_dict_wubi:
        same_dict_wubi[enc] = [c]
    else:
        same_dict_wubi[enc].append(c)

with open("sim_dict.pkl", "rb") as f:
    wubi_sim_dict = pickle.load(f)


def _read_json(input_file):
    examples = []
    with open(input_file, "r") as f:
        data = f.readlines()
    for line in data:
        line = json.loads(line.strip())
        examples.append(line)
    return examples 


def _add_noise(sent, ratio, ch_2_enc, same_dict):
    newsent = ''
    changed_chars = 0
    total_chars = 0
    for c in sent:
        try:
            enc = ch_2_enc[c]
            enc = ''.join([i for i in enc if not i.isdigit()])
        except:
            enc = None
        if random.random() < ratio:
            try:
                tmp = same_dict[enc][:]
                tmp.remove(c) ## remove itself
                newc = random.choice(tmp)
            except:
                newc = c
        else:
            newc = c
        if newc != c:
            changed_chars += 1
        total_chars += 1
        newsent += newc
        if c in ch_2_enc:
            enc_orig = ''.join([i for i in ch_2_enc[c] if not i.isdigit()])
            enc_new = ''.join([i for i in ch_2_enc[newc] if not i.isdigit()])
            if enc_orig != enc_new:
                print(enc_orig, enc_new)
    # if newsent != sent:
    #     print(sent)
    #     print(newsent)
    #     print(ratio)
    #     exit()
    return newsent, changed_chars, total_chars


TEXT_KEYS = ['sentence1', 'sentence2']


def change_pinyin(orig_data, ratio):
    # random.seed(2021)
    data = []
    ## pinyin substitute
    changed_chars = 0
    total_chars = 0
    for eg in orig_data:
        new_eg = {}
        new_eg['label'] = eg['label']
        for key in TEXT_KEYS:
            newsent, changed, total = _add_noise(eg[key], ratio, ch2pinyin, same_dict_pinyin)
            new_eg[key] = newsent
            changed_chars += changed
            total_chars += total
        data.append(new_eg)
    print(f'{changed_chars = }')
    print ("changed ratio: ", changed_chars / total_chars)
    return data


def change_wubi(orig_data, ratio=1):
    # random.seed(2021)
    data = []
    ## wubi substitute
    changed_chars = 0
    total_chars = 0
    for eg in orig_data:
        for key in TEXT_KEYS:
            newsent, changed, total = _add_noise(eg[key], ratio, ch2wubi, same_dict_wubi)
            changed_chars += changed
            total_chars += total
            if tokenizer.tokenize(eg[key]) != tokenizer.tokenize(newsent):
                print (eg[key])
                print (tokenizer.tokenize(eg[key]))
                print (newsent)
                print (tokenizer.tokenize(newsent))
                print ()
            eg[key] = newsent
        data.append(eg)
    print ("changed ratio: ", changed_chars / total_chars)
    return data


def _dump_json(data, file):
    with open(file, 'w+') as f:
        for eg in data:
            json.dump(eg, f, ensure_ascii=False)
            f.write('\n')

def gen_phonetic_data(clean_data, ratio):
    # clean_data = clean_data[:2]
    print(f'# clean data: {len(clean_data)}')
    print(f'Generating phonetic noise with nosie ratio: {ratio}')
    random.seed(0)
    
    # Make 2x amount noisy data
    noisy_data = [change_pinyin(clean_data, ratio=ratio) for _ in range(2)]
    assert all(data != noisy_data[0] for data in noisy_data[1:])
    all_data = clean_data + sum(noisy_data, [])  # Append noisy data to clean
    print(f'# examples after data augmentation: {len(all_data)}')
    return all_data
    

if __name__ == '__main__':
    filename = 'train.json'
    FILE_SRC = Path('datasets/realtypo/afqmc', filename)
    FILE_DST = Path('datasets/realtypo/afqmc/da_noise/phonetic_50', filename)
    orig_data = _read_json(FILE_SRC)

    FILE_DST.parent.mkdir(parents=True, exist_ok=True)

    # def gen_glyph_data(ratios):
    #     for ratio in ratios:
    #         print("ratio:", ratio)
    #         dir_glyph = DIR_DEST_DATA + '/glyph_' + str(int(ratio * 100))
    #         print(dir_glyph)
    #         os.makedirs(dir_glyph, exist_ok=True)
    #         data = change_wubi(orig_data, ratio=ratio)
    #         file_glyph = dir_glyph + '/test.json'
    #         _dump_json(data, file_glyph)
    #         print('')

    # gen_glyph_data([])
    data = gen_phonetic_data(orig_data, 0.5)
    print(f'Dumping to {FILE_DST}')
    _dump_json(data, FILE_DST)
