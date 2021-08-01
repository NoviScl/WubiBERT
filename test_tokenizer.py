import random

import consts
from tokenization import ALL_TOKENIZERS
from processors.glue import PROCESSORS, convert_examples_to_features


def load_tokenizers():
	tokenizers = {}
	for t in consts.VOCAB_FILES:
		if t in consts.TOKENIZER_TYPES:
			tokenizer_type = consts.TOKENIZER_TYPES[t]
		else:
			tokenizer_type = 'CommonZh'
		vocab_file = consts.VOCAB_FILES[t]
		vocab_model_file = vocab_file.replace('.vocab', '.model')
		tokenizer = ALL_TOKENIZERS[tokenizer_type](vocab_file, vocab_model_file)
		tokenizers[t] = tokenizer
	return tokenizers


def get_inverse_index(tokens, tokenizer, text):
	idx = 0
	split = [None] * len(text)
	for i, ch in enumerate(text):
		split[i] = ''.join(tokenizer.tokenize(ch))
	indices = []
	for token in tokens:
		while len(split[idx]) == 0:
			idx += 1
		print(token, len(token), idx)
		indices.append(idx)
		hi = 0
		while hi < len(token):
			hi += len(split[idx])
			idx += 1
	indices.append(len(text))
	return indices

def main():
	random.seed(123)

	# Load examples

	tokenizers = load_tokenizers()
	tokenizer_names = list(consts.VOCAB_FILES.keys())
	print('\t'.join(tokenizer_names))
	tokenizer = tokenizers['pinyin']
	text = '就椒房殿长门怨读于清华大学 CS Department，的陈英发。'
	

	line = tokenizer.convert_line(text)
	
	print(line)
	tokens = tokenizer.spm_tokenizer.encode(line, out_type=str)
	print(tokens)
	print(get_inverse_index(tokens, tokenizer, text))


if __name__ == '__main__':
	main()