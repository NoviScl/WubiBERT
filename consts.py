FILENAME_TEST_RESULT = 'result_test.txt'
FILENAME_SCORES = 'scores.txt'
FILENAME_BEST_MODEL = 'best_model.bin'
FILENAME_PARAMS = 'params.json'

VOCAB_FILES = {
	'cangjie': '/home/chenyingfa/WubiBERT/tokenizers/cangjie_zh_22675.vocab', 
	'pinyin': '/home/chenyingfa/WubiBERT/tokenizers/pinyin_zh_22675.vocab', 
	'stroke': '/home/chenyingfa/WubiBERT/tokenizers/stroke_zh_22675.vocab', 
	'wubi': '/home/chenyingfa/WubiBERT/tokenizers/wubi_zh_22675.vocab', 
	'zhengma': '/home/chenyingfa/WubiBERT/tokenizers/zhengma_zh_22675.vocab', 
	'zhuyin': '/home/chenyingfa/WubiBERT/tokenizers/zhuyin_zh_22675.vocab', 
	'raw': '/home/chenyingfa/WubiBERT/tokenizers/raw_zh_22675.vocab', 
	'bert': '/home/chenyingfa/WubiBERT/tokenizers/bert_chinese_uncased_22675.vocab', 
}

TOKENIZER_TYPES = {
	'bert': 'BertZh',
	'raw': 'RawZh',
}


MODEL_NAMES = [
	'cangjie',
	'pinyin',
	'stroke',
	'wubi',
	'zhengma',
	'zhuyin',
	'raw',
	'bert',
]

BEST_CKPTS = {
    'cangjie': "ckpt_7202", # cangjie
    'pinyin': "ckpt_8804",
    'stroke': "ckpt_8804",
    # "ckpt_7992" # wubi
    'wubi': "ckpt_8804",
    'zhengma': "ckpt_8804",
    'zhuyin': "ckpt_7992",
    # "ckpt_7202" # raw
    'raw': "ckpt_8804",
    'bert': "ckpt_8601", # bert
    # cws
    # "ckpt_7202" # cws_raw
    'cws_raw': "ckpt_8804",
    # "ckpt_7993" # cws_wubi
    'cws_wubi': "ckpt_8804",
    'cws_zhuyin': "ckpt_8804",
}