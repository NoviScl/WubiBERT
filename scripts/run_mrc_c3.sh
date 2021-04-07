#!/usr/bin/env bash
model_name="bert-tiny"

task_name=${task_name:-"c3"}

# init_checkpoint=${init_checkpoint:-"results/checkpoints_raw_zh/ckpt_8601.pt"}
# config_file=${config_file:-"configs/bert_config_vocab30k.json"}
# vocab_file=${vocab_file:-"tokenizers/sp_raw_zh_30k.vocab"}
# vocab_model_file=${vocab_model_file:-"tokenizers/sp_raw_zh_30k.model"}
# tokenizer_type=${tokenizer_type:-"RawZh"}
init_checkpoint=${init_checkpoint:-"results/checkpoints_bert_zh_22675/ckpt_8601.pt"}
config_file=${config_file:-"configs/bert_config_vocab22675.json"}
vocab_file=${vocab_file:-"tokenizers/bert_chinese_uncased_22675.vocab"}
vocab_model_file=${vocab_model_file:-"tokenizers/bert_chinese_uncased_22675.model"}
tokenizer_type=${tokenizer_type:-"BertZh"}

data_dir=${data_dir:-"datasets/$task_name"}
out_dir=${out_dir:-"logs/$task_name"}

seed=${seed:-2}
epochs=${epochs:-8}
# epochs=${epochs:-8}


python3 run_c3.py \
  --num_train_epochs=${epochs} \
  --train_batch_size=24 \
  --eval_batch_size=24 \
  --gradient_accumulation_steps=6 \
  --learning_rate=2e-5 \
  --warmup_proportion=0.05 \
  --max_seq_length=512 \
  --seed=${seed} \
  --tokenizer_type=${tokenizer_type} \
  --vocab_file=${vocab_file} \
  --vocab_model_file=${vocab_model_file} \
  --config_file=${config_file} \
  --init_checkpoint=${init_checkpoint} \
  --data_dir=${data_dir} \
  --output_dir=${out_dir} \
  --do_train \
  --do_eval \
#   --output_dir=$OUTPUT_DIR/$TASK_NAME/$MODEL_NAME/ \
#   --gpu_ids="0,1,2,3" \
