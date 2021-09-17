import json
import random
import os
import pickle
random.seed(2021)


wubi2ch = "data/wubi_to_chinese.pkl"
ch2wubi = "data/chinese_to_wubi.pkl"

pinyin2ch = "data/pinyin_to_chinese.pkl"
ch2pinyin = "data/chinese_to_pinyin.pkl"

def load_dict(dict_path):
	return pickle.load(open(dict_path, "rb"))

ch2wubi = load_dict(ch2wubi)
wubi2ch = load_dict(wubi2ch)

ch2pinyin = load_dict(ch2pinyin)
pinyin2ch = load_dict(pinyin2ch)

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

orig_data = _read_json('/home/sichenglei/CLUE/tnews/dev.json')

# print (len(orig_data))
# print (orig_data[0])
def change_pinyin(ratio=1):
    data = []
    ## pinyin substitute
    changed_chars = 0
    total_chars = 0
    for eg in orig_data:
        newsent = ""
        for c in eg["sentence"]:
            try:
                enc = ch2pinyin[c]
                enc = ''.join([i for i in enc if not i.isdigit()])
            except:
                enc = None
            if random.random() < ratio:
                try:
                    tmp = same_dict_pinyin[enc][:]
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
            if c in ch2pinyin:
                enc_orig = ''.join([i for i in ch2pinyin[c] if not i.isdigit()])
                enc_new = ''.join([i for i in ch2pinyin[newc] if not i.isdigit()])
                if enc_orig != enc_new:
                    print (enc_orig, enc_new)
        eg["sentence"] = newsent

        newsent = ""
        for c in eg["keywords"]:
            try:
                enc = ch2pinyin[c]
                enc = ''.join([i for i in enc if not i.isdigit()])
            except:
                enc = None
            if random.random() < ratio:
                try:
                    tmp = same_dict_pinyin[enc][:]
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
            if c in ch2pinyin:
                enc_orig = ''.join([i for i in ch2pinyin[c] if not i.isdigit()])
                enc_new = ''.join([i for i in ch2pinyin[newc] if not i.isdigit()])
                if enc_orig != enc_new:
                    print (enc_orig, enc_new)
        eg["keywords"] = newsent
        data.append(eg)
    print ("changed ratio: ", changed_chars / total_chars)
    return data

data = change_pinyin(ratio=1.0)
with open('noisy_data/tnews/phonetic_test.json', 'w+') as f:
    for eg in data:
        json.dump(eg, f, ensure_ascii=False)
        f.write('\n')


def change_wubi(ratio=1):
    data = []
    ## wubi substitute
    changed_chars = 0
    total_chars = 0
    for eg in orig_data:
        newsent = ""
        for c in eg["sentence"]:
            try:
                enc = ch2wubi[c]
                enc = ''.join([i for i in enc if not i.isdigit()])
            except:
                enc = None
            if random.random() < ratio:
                ## first try same encoding substitute
                if enc in same_dict_wubi and len(same_dict_wubi[enc]) > 1:
                    tmp = same_dict_wubi[enc][:]
                    tmp.remove(c) ## remove itself
                    newc = random.choice(tmp)
                # elif c in wubi_sim_dict:
                #     newc = random.choice(wubi_sim_dict[c])
                else:
                    newc = c
            else:
                newc = c
            if newc != c:
                changed_chars += 1
            total_chars += 1
            newsent += newc
            if c in ch2wubi:
                enc_orig = ''.join([i for i in ch2wubi[c] if not i.isdigit()])
                enc_new = ''.join([i for i in ch2wubi[newc] if not i.isdigit()])
                if enc_orig != enc_new:
                    print (enc_orig, enc_new)
        eg["sentence"] = newsent

        newsent = ""
        for c in eg["keywords"]:
            try:
                enc = ch2wubi[c]
                enc = ''.join([i for i in enc if not i.isdigit()])
            except:
                enc = None
            if random.random() < ratio:
                ## first try same encoding substitute
                if enc in same_dict_wubi and len(same_dict_wubi[enc]) > 1:
                    tmp = same_dict_wubi[enc][:]
                    tmp.remove(c) ## remove itself
                    newc = random.choice(tmp)
                # elif c in wubi_sim_dict:
                #     newc = random.choice(wubi_sim_dict[c])
                else:
                    newc = c
            else:
                newc = c
            if newc != c:
                changed_chars += 1
            total_chars += 1
            newsent += newc
            if c in ch2wubi:
                enc_orig = ''.join([i for i in ch2wubi[c] if not i.isdigit()])
                enc_new = ''.join([i for i in ch2wubi[newc] if not i.isdigit()])
                if enc_orig != enc_new:
                    print (enc_orig, enc_new)
        eg["keywords"] = newsent
        data.append(eg)
    print ("changed ratio: ", changed_chars / total_chars)
    return data

data = change_wubi(ratio = 1.0)
with open('noisy_data/tnews/glyph_test.json', 'w+') as f:
    for eg in data:
        json.dump(eg, f, ensure_ascii=False)
        f.write('\n')
