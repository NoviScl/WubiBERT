#!/bin/bash

# Dataset
task_name=${task_name:-""}
train_dir=${train_dir:-""}
dev_dir=${dev_dir:-""}
test_dir=${test_dir:-""}

seed=${seed:-""}
out_dir=${out_dir:-""}
mode=${mode:-"test"}
# num_gpu=${num_gpu:-"8"}

# Hyperparameters
epochs=${epochs:-"12"}
max_steps=${13:-"-1.0"}
batch_size=${batch_size:-"32"}
gradient_accumulation_steps=${gradient_accumulation_steps:-"2"}
learning_rate=${10:-"2e-5"}
warmup_proportion=${11:-"0.1"}
max_seq_length=${max_seq_length:-512}
fewshot=${fewshot:-1}

CMD="python3"
# CMD="python"
CMD+=" run_ner.py "
CMD+="--task_name ${task_name} "
if [[ $mode == *"train"* ]] ; then
  CMD+="--do_train "
  CMD+="--train_batch_size=$batch_size "
fi
if [[ $mode == *"eval"* ]] || [[ $mode == *"test"* ]]; then
  if [[ $mode == *"eval"* ]] ; then
    CMD+="--do_eval "
  fi
  if [[ $mode == *"test"* ]] ; then
    CMD+="--do_test "
  fi
  CMD+="--eval_batch_size=$batch_size "
fi
CMD+="--gradient_accumulation_steps=$gradient_accumulation_steps "

CMD+="--tokenizer_type $tokenizer_type "
CMD+="--vocab_file=$vocab_file "
CMD+="--config_file=$config_file "
CMD+="--vocab_model_file $vocab_model_file "
CMD+="--init_checkpoint $init_checkpoint "

CMD+="--output_dir $out_dir "
# CMD+="--data_dir $data_dir "
CMD+="--train_dir $train_dir "
CMD+="--dev_dir $dev_dir "
CMD+="--test_dir $test_dir "

# CMD+="--bert_model bert-tiny "
CMD+="--seed $seed "

CMD+="--epochs $epochs "
CMD+="--warmup_proportion $warmup_proportion "
CMD+="--train_max_seq_length $max_seq_length "
CMD+="--eval_max_seq_length $max_seq_length "
CMD+="--learning_rate $learning_rate "
# CMD+="--max_steps $max_steps "

CMD+="--do_lower_case "
# CMD+="--fewshot $fewshot "

LOGFILE="${out_dir}/${seed}/logfile"

$CMD |& tee $LOGFILE

# python run_ner_crf.py \
#   --model_type=bert \
#   --model_name_or_path=$BERT_BASE_DIR \
#   --task_name=$TASK_NAME \
#   --do_train \
#   --do_eval \
#   --do_lower_case \
#   --data_dir=$GLUE_DIR/${TASK_NAME}/ \
#   --train_max_seq_length=128 \
#   --eval_max_seq_length=512 \
#   --per_gpu_train_batch_size=24 \
#   --per_gpu_eval_batch_size=24 \
#   --learning_rate=3e-5 \
#   --num_train_epochs=5.0 \
#   --logging_steps=448 \
#   --save_steps=448 \
#   --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
#   --overwrite_output_dir \
#   --seed=42
