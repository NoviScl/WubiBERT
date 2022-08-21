FILENAME_TEST_RESULT = 'result_test.txt'
FILENAME_SCORES = 'scores.txt'
FILENAME_BEST_MODEL = 'best_model.bin'
FILENAME_PARAMS = 'params.json'

ALL_VOCAB_FILES = {
    'cangjie': 'tokenizers/cangjie_zh_22675.vocab', 
    'pinyin': 'tokenizers/pinyin_zh_22675.vocab', 
    'stroke': 'tokenizers/stroke_zh_22675.vocab', 
    'wubi': 'tokenizers/wubi_zh_22675.vocab', 
    'zhengma': 'tokenizers/zhengma_zh_22675.vocab', 
    'zhuyin': 'tokenizers/zhuyin_zh_22675.vocab', 
    'raw': 'tokenizers/raw_zh_22675.vocab', 
    'bert': 'tokenizers/bert_chinese_uncased_22675.vocab', 
    'char': 'tokenizers/bert_chinese_uncased_22675.vocab', 
    'pinyin_concat_wubi': 'tokenizers/pinyin_concat_wubi_22675.vocab',
    'byte': 'tokenizers/byte_22675.vocab',
    
    'random_index': 'tokenizers/random_index_22675.vocab',
    'pinyin_no_index': 'tokenizers/pinyin_no_index_22675.vocab',
    'wubi_no_index': 'tokenizers/wubi_no_index_22675.vocab',
    'wubi_shuffled': 'tokenizers/shuffled_wubi_22675.vocab',
    'pinyin_shuffled': 'tokenizers/shuffled_pinyin_22675.vocab',
    'pinyin_500': 'tokenizers/small_vocab/pinyin_zh_500.vocab',
    'wubi_500': 'tokenizers/small_vocab/wubi_zh_500.vocab',
    'byte_500': 'tokenizers/small_vocab/byte_500.vocab',
    'random_index_500': 'tokenizers/small_vocab/random_index_500.vocab',
}

TOKENIZER_TYPES = {
    'cangjie': 'CommonZh',
    'pinyin': 'CommonZh',
    'stroke': 'CommonZh',
    'wubi': 'CommonZh',
    'zhengma': 'CommonZh',
    'zhuyin': 'CommonZh',
    'raw': 'RawZh',
    'bert': 'BertZh',
    'char': 'BertZh',
    'pinyin_no_index': 'CommonZhNoIndex',
    'wubi_no_index': 'CommonZhNoIndex',
    'pinyin_concat_wubi': 'PinyinConcatWubi',
    'byte': 'Byte',
    'random_index': 'RandomIndex',
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
    'char',
]


ALL_BEST_CKPTS = {
    'cangjie': 'ckpt_8804',
    'pinyin': "ckpt_8804",
    'stroke': "ckpt_8804",
    'wubi': "ckpt_8804",
    'zhengma': "ckpt_8804",
    'zhuyin': "ckpt_7992",
    'raw': "ckpt_8804",
    'bert': "ckpt_8601", # bert
    'pinyin_concat_wubi': 'ckpt_8804',
    'byte': 'ckpt_8840',
    'random_index': 'ckpt_8840',
    
    'pinyin_no_index': 'ckpt_8840',
    'wubi_no_index': 'ckpt_8840',
    'pinyin_shuffled': 'ckpt_8840',
    'wubi_shuffled': 'ckpt_8840',
    'pinyin_500': 'ckpt_7420',
    'wubi_500': 'ckpt_7420',
    'byte_500': 'ckpt_7420',
    'random_index_500': 'ckpt_7420',
    
    # cws
    'cws_raw': "ckpt_8804",
    'cws_wubi': "ckpt_8804",
    'cws_zhuyin': "ckpt_8804",
}


ALL_DIR_CKPTS = {
    "cangjie": "checkpoints/checkpoints_cangjie_22675",
    "pinyin": "checkpoints/checkpoints_pinyin_zh_22675",
    "stroke": "checkpoints/checkpoints_stroke_22675",
    "wubi": "checkpoints/checkpoints_wubi_zh_22675",
    "zhengma": "checkpoints/checkpoints_zhengma_zh_22675",
    "zhuyin": "checkpoints/checkpoints_zhuyin_zh_22675",
    "raw": "checkpoints/checkpoints_raw_zh_22675",
    "bert": "checkpoints/checkpoints_bert_zh_22675",
    "pinyin_concat_wubi": "checkpoints/checkpoints_pinyin_concat_wubi",
    'byte': 'checkpoints/ckpts_byte_22675',
    'random_index': 'checkpoints/ckpts_random_index_22675',
    
    'pinyin_no_index': 'checkpoints/checkpoints_pinyin_no_index',
    'wubi_no_index': 'checkpoints/checkpoints_wubi_no_index',
    'wubi_shuffled': 'checkpoints/checkpoints_shuffled_wubi',
    'pinyin_shuffled': 'checkpoints/checkpoints_shuffled_pinyin',
    'pinyin_shuffled_500': 'checkpoints/checkpoints_pinyin_shuffled_500',
    'wubi_shuffled_500': 'checkpoints/checkpoints_wubi_shuffled_500',
    'pinyin_500': 'checkpoints/checkpoints_pinyin_500',
    'wubi_500': 'checkpoints/checkpoints_wubi_500',
    'byte_500': 'checkpoints/checkpoints_byte_500',
    'random_index_500': 'checkpoints/checkpoints_random_index_500',
}