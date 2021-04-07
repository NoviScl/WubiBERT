#!/usr/bin/env bash

model_name="bert-tiny"

task_name=${task_name:-"chid"}

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
epochs=${epochs:-2}
gradient_accumulation_steps=${gradient_accumulation_steps:-4}

python3 run_multichoice_mrc.py \
  --num_train_epochs=4 \
  --train_batch_size=24 \
  --predict_batch_size=24 \
  --gradient_accumulation_steps=${gradient_accumulation_steps} \
  --learning_rate=2e-5 \
  --warmup_proportion=0.06 \
  --max_seq_length=64 \
  --vocab_file=${vocab_file} \
  --vocab_model_file=${vocab_model_file} \
  --tokenizer_type=${tokenizer_type} \
  --config_file=${config_file} \
  --init_checkpoint=${init_checkpoint} \
  --data_dir=${data_dir} \
  --output_dir=${out_dir} \
  --train_file=${data_dir}/train.json \
  --train_ans_file=${data_dir}/train_answer.json \
  --predict_file=${data_dir}/dev.json \
  --predict_ans_file=${data_dir}/dev_answer.json \
  --seed=${seed}
  # --gpu_ids="0,1,2,3" \

# python test_multichoice_mrc.py \
#   --gpu_ids="0" \
#   --predict_batch_size=24 \
#   --max_seq_length=64 \
#   --vocab_file=$BERT_DIR/vocab.txt \
#   --bert_config_file=$BERT_DIR/bert_config.json \
#   --input_dir=$GLUE_DIR/$TASK_NAME/ \
#   --init_restore_dir=$OUTPUT_DIR/$TASK_NAME/$MODEL_NAME/ \
#   --output_dir=$OUTPUT_DIR/$TASK_NAME/$MODEL_NAME/ \
#   --predict_file=$GLUE_DIR/$TASK_NAME/test.json \
