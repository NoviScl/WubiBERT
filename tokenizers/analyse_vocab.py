import collections
import logging
import os
import unicodedata
import six
from io import open
import pickle
# from tokenization import CommonZhTokenizer, BertZhTokenizer, RawZhTokenizer

import sentencepiece as spm

cangjie2ch = "../data/cangjie_to_chinese.pkl"
ch2cangjie = "../data/chinese_to_cangjie.pkl"

stroke2ch = "../data/stroke_to_chinese.pkl"
ch2stroke = "../data/chinese_to_stroke.pkl"

zhengma2ch = "../data/zhengma_to_chinese.pkl"
ch2zhengma = "../data/chinese_to_zhengma.pkl"

wubi2ch = "../data/wubi_to_chinese.pkl"
ch2wubi = "../data/chinese_to_wubi.pkl"

pinyin2ch = "../data/pinyin_to_chinese.pkl"
ch2pinyin = "../data/chinese_to_pinyin.pkl"

zhuyin2ch = "../data/zhuyin_to_chinese.pkl"
ch2zhuyin = "../data/chinese_to_zhuyin.pkl"

# control_char = u'0123456789abcdefghijklmnopqrstuvwxyz' 
control_char = u'0123456789abcdefghijklmnopqrstuvwxyz' 
pinyin_tones = '❁❄❂❃❅'
control_uni = [chr(ord(c)+50000) for c in control_char]
sep = chr(ord('_')+50000)


def load_dict(dict_path):
	return pickle.load(open(dict_path, "rb"))

def load_vocab_spm(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip().split()[0].strip()
            vocab[token] = index
            index += 1
    return vocab
# vocab = load_vocab_spm('shuffled_wubi_22675.vocab')
# vocab = load_vocab_spm('shuffled_pinyin_22675.vocab')
vocab = load_vocab_spm('zhuyin_zh_22675.vocab')

v = list(vocab.keys())[5:]

d_wubi = load_dict(ch2wubi)
d_pinyin = load_dict(ch2pinyin)
d_zhuyin = load_dict(ch2zhuyin)

# pinyin_d = load_dict(pinyin2ch)
# for k in pinyin_d.keys():
#     i = -1
#     while k[i].isdigit():
#         i -= 1
#     c = k[i]
#     if c not in pinyin_tones:
#         print (c)

# d_no_index = []
# for s in d_wubi.keys():
#     s = ''.join([i for i in s if not i.isdigit()])
#     d_no_index.append(s)
# d_wubi = d_no_index

# d_no_index = []
# for s in d_wubi.keys():
#     s = ''.join([i for i in s if not i.isdigit()])
#     d_no_index.append(s)
# d_wubi = d_no_index


# d_no_index = {}
# for ch in d_wubi.keys():
#     d_no_index[ch] = ''.join([i for i in d_wubi[ch] if not i.isdigit()]) 

# for ch in d_wubi.keys():
#     if ch in d_no_index:
#         d_no_index[ch] += ''.join([i for i in d_wubi[ch] if not i.isdigit()]) 
#     else:
#         d_no_index[ch] = ''.join([i for i in d_wubi[ch] if not i.isdigit()]) 

# d = d_no_index.values()
# d = list(set(d))

# print (d)
d = list(d_zhuyin.values())

counter_char = 0
counter_sub = 0
counter_compo = 0

def check_char(word):
    for c in word:
        if c.lower() not in (control_char + pinyin_tones):
            return False 
    return True

for c in v:
    ## punctuations:
    if len(c) == 1 and (c.lower() not in control_char):
        counter_char += 1
        # print (c)

    ## sub-char 
    elif (c[-1] == sep) and (sep not in c[:-1]) and (c[:-1] not in d) and check_char(c[:-1]):
        counter_sub += 1
        # print (c)
    elif check_char(c):
        counter_sub += 1
        # print (c)

    ## char
    elif (c[-1] == sep and c[:-1] in d) or (c in d):
        counter_char += 1
        # print (c)
    else:
        counter_compo += 1
        # print (c)
    
   
print (counter_sub)
print (counter_sub / (len(v)))
print (counter_char)
print (counter_char / (len(v)))
print (counter_compo)
print (counter_compo / (len(v)))