#!/usr/bin/env bash

model_name="bert-tiny"

task_name=${task_name:-"cmrc"}

init_checkpoint=${init_checkpoint:-"results/checkpoints_raw_zh/ckpt_8601.pt"}
config_file=${config_file:-"configs/bert_config_vocab30k.json"}
vocab_file=${vocab_file:-"tokenizers/sp_raw_zh_30k.vocab"}
vocab_model_file=${vocab_model_file:-"tokenizers/sp_raw_zh_30k.model"}
tokenizer_type=${tokenizer_type:-"RawZh"}

fewshot=${fewshot:-0}

data_dir=${data_dir:-"datasets/$task_name"}
out_dir=${out_dir:-"logs/$task_name"}

seed=${seed:-2}
epochs=${epochs:-2}

if [[ $fewshot == '1' ]] ; then
  fewshot = 
fi

python3 run_mrc.py \
  --train_epochs=${epochs} \
  --train_batch_size=32 \
  --lr=3e-5 \
  --gradient_accumulation_steps=2 \
  --warmup_rate=0.1 \
  --max_seq_length=512 \
  --task_name=${task_name} \
  --vocab_file=${vocab_file} \
  --vocab_model_file=${vocab_model_file} \
  --config_file=${config_file} \
  --tokenizer_type=${tokenizer_type} \
  --init_checkpoint=${init_checkpoint} \
  --out_dir=${out_dir} \
  --train_dir=${data_dir}/train_features.json \
  --train_file=${data_dir}/train.json \
  --dev_dir1=${data_dir}/dev_examples.json \
  --dev_dir2=${data_dir}/dev_features.json \
  --dev_file=${data_dir}/dev.json \
  --checkpoint_dir=${out_dir} \
  --seed=${seed} \
  --do_train \
  --fewshot=fewshot \
  # --gpu_ids="0,1" \
  # --init_restore_dir=$BERT_DIR/pytorch_model.pth \


# python test_mrc.py \
#   --gpu_ids="0" \
#   --n_batch=32 \
#   --max_seq_length=512 \
#   --task_name=${task_name} \
#   --vocab_file=${vocab_file} \
#   --bert_config_file=${config_file} \
#   --init_restore_dir=$OUTPUT_DIR/$task_name/$MODEL_NAME/
#   --output_dir=$OUTPUT_DIR/$task_name/$MODEL_NAME/ \
#   --test_dir1=${data_dir}/test_examples.json \
#   --test_dir2=${data_dir}/test_features.json \
#   --test_file=${data_dir}/cmrc2018_test_2k.json \




