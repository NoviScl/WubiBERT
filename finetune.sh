#!/bin/bash
#SBATCH -G 1
python3 finetune.py \
  --tokenizer pinyin \
  --task cmrc \
  --seed 15 \
  --char_pred chartokens