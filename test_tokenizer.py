#coding: utf8
import random

import consts
from tokenization import ALL_TOKENIZERS
from processors.glue import PROCESSORS, convert_examples_to_features


def get_tokenizer(tokenizer_name, suffix):
    tokenizer_type = consts.TOKENIZER_TYPES[tokenizer_name]
    if suffix == 'no_index':
        vocab_file = consts.VOCAB_FILES_NO_INDEX[tokenizer_name]
    elif suffix == '500':
        vocab_file = consts.VOCAB_FILES_500[tokenizer_name]
    elif suffix == 'shuffled_500':
        vocab_file = consts.VOCAB_FILES_SHUFFLED_500[tokenizer_name]
    model_file = vocab_file.replace('.vocab', '.model')
    return ALL_TOKENIZERS[tokenizer_type](vocab_file, model_file)


def main():
    random.seed(0)

    task = 'iflytek'
    tokenizer_name = 'pinyin'
    suffix = '500'

    # Load examples
    
    tokenizer = get_tokenizer(tokenizer_name, suffix)
    processor = PROCESSORS['tnews']()
    examples = processor.get_dev_examples('datasets/tnews/split')
    all_tokens = []
    for eg in examples[:5]:
        text = eg.text_a
        tokens = tokenizer.tokenize(text)
        
        all_tokens.append(tokens)
        
        print(text)
        print(tokens)
        print('')
    
    # text = '就读于清华大学 CS Department，的陈英发。'
    # tokens = tokenizer.tokenize(text)
    # ids = tokenizer.convert_tokens_to_ids(tokens)
    # print(tokens)
    # print(ids)


if __name__ == '__main__':
    main()