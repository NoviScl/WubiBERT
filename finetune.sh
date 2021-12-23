#!/bin/bash
#SBATCH -G 1
python3 finetune.py \
  --tokenizer raw \
  --task cmrc \
  --seed 18 \
  --char_pred chartokens