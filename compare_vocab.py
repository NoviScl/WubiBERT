import collections
import logging
import os
import unicodedata
import six
from io import open
import pickle
from tokenization import CommonZhTokenizer, BertZhTokenizer, RawZhTokenizer

import sentencepiece as spm

cangjie2ch = "/home/ubuntu/WubiBERT/data/cangjie_to_chinese.pkl"
ch2cangjie = "/home/ubuntu/WubiBERT/data/chinese_to_cangjie.pkl"

stroke2ch = "/home/ubuntu/WubiBERT/data/stroke_to_chinese.pkl"
ch2stroke = "/home/ubuntu/WubiBERT/data/chinese_to_stroke.pkl"

zhengma2ch = "/home/ubuntu/WubiBERT/data/zhengma_to_chinese.pkl"
ch2zhengma = "/home/ubuntu/WubiBERT/data/chinese_to_zhengma.pkl"

wubi2ch = "/home/ubuntu/WubiBERT/data/wubi_to_chinese.pkl"
ch2wubi = "/home/ubuntu/WubiBERT/data/chinese_to_wubi.pkl"

pinyin2ch = "/home/ubuntu/WubiBERT/data/pinyin_to_chinese.pkl"
ch2pinyin = "/home/ubuntu/WubiBERT/data/chinese_to_pinyin.pkl"

zhuyin2ch = "/home/ubuntu/WubiBERT/data/zhuyin_to_chinese.pkl"
ch2zhuyin = "/home/ubuntu/WubiBERT/data/chinese_to_zhuyin.pkl"

control_char = u'0123456789abcdefghijklmnopqrstuvwxyz' 
control_uni = [chr(ord(c)+50000) for c in control_char]

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

# raw_zh_30k_vocab = load_vocab_spm('tokenizers/sp_raw_zh_30k.vocab')
# # print (len(raw_zh_30k_vocab))

# # bert_zh_vocab = load_vocab_spm('tokenizers/bert_chinese_uncased_22675.vocab')
# vocab = load_vocab_spm('tokenizers/zhuyin_zh_22675.vocab')
# encode2ch = load_dict(zhuyin2ch)
# sep = chr(ord('_')+50000)

# counter_1 = 0
# counter_2 = 0
# for v,i in vocab.items():
#     # print (v)
#     v = v.split(sep)
#     if v[-1] == '':
#         v = v[:-1]
#     # print (v)
#     newv = ''
#     for v_ in v:
#         # print (v_)
#         v_ = v_.strip()
#         if v_ in encode2ch:
#             newv += encode2ch[v_]
#         else:
#             print (v_)

#     if len(newv) == len(v) and newv in raw_zh_30k_vocab:
#         print (v, newv)
#         counter_1 += 1

#     if len(newv) == len(v):
#         print (v, newv)
#         counter_2 += 1

# print ("overlap: {}/{} = {}".format(counter_1, 22675, counter_1 / 22675))
# print ("words: {}/{} = {}".format(counter_2, 22675, counter_2 / 22675))

# line = ' 一 个 小 测 试 ： 祝 大 家 新 年 快 乐 ！ '
# # line = '如今把事实指出，愈使魑魅魍魉无所遁形于光天化日之下了！'
# line = '紧锣密鼓'
line = '1984'
# # line = "尼采的著作对于宗教、道德、现代文化、哲学、以及科学等领域提出了广泛的批判和讨论。他的写作风格独特，经常使用格言和悖论的技巧。尼采对于后代哲学的发展影响很大，尤其是在存在主义与后现代主义上。"
# line = '缙云氏有不才子，贪于饮食，冒于货贿，侵欲崇侈，不可盈厌；聚敛积实，不知纪极；不分孤寡，不恤穷匮。天下之民以比三凶，谓之饕餮。'

raw_zh = RawZhTokenizer(vocab_file='tokenizers/sp_raw_zh_30k.vocab', model_file='tokenizers/sp_raw_zh_30k.model')
bert_zh = BertZhTokenizer(vocab_file='tokenizers/bert_chinese_uncased_22675.vocab')
cangjie_zh = CommonZhTokenizer(vocab_file='tokenizers/cangjie_zh_22675.vocab', model_file='tokenizers/cangjie_zh_22675.model')
stroke_zh = CommonZhTokenizer(vocab_file='tokenizers/stroke_zh_22675.vocab', model_file='tokenizers/stroke_zh_22675.model')
wubi_zh = CommonZhTokenizer(vocab_file='tokenizers/wubi_zh_22675.vocab', model_file='tokenizers/wubi_zh_22675.model')
zhengma_zh = CommonZhTokenizer(vocab_file='tokenizers/zhengma_zh_22675.vocab', model_file='tokenizers/zhengma_zh_22675.model')
pinyin_zh = CommonZhTokenizer(vocab_file='tokenizers/pinyin_zh_22675.vocab', model_file='tokenizers/pinyin_zh_22675.model')
zhuyin_zh = CommonZhTokenizer(vocab_file='tokenizers/zhuyin_zh_22675.vocab', model_file='tokenizers/zhuyin_zh_22675.model')

print (line)
print ("raw_zh: ", raw_zh.tokenize(line))
print ("bert_zh: ", bert_zh.tokenize(line))
print ("cangjie_zh: ", cangjie_zh.tokenize(line))
print ("stroke_zh: ", stroke_zh.tokenize(line))
print ("wubi_zh: ", wubi_zh.tokenize(line))
print ("zhengma_zh: ", zhengma_zh.tokenize(line))
print ("pinyin_zh: ", pinyin_zh.tokenize(line))
print ("zhuyin_zh: ", zhuyin_zh.tokenize(line))



# ch2encode = load_dict(ch2pinyin)
# print (ch2encode['哥'])