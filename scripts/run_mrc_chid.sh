#!/usr/bin/env bash

model_name="bert-tiny"

task_name=${task_name:-"chid"}

init_checkpoint=${init_checkpoint:-""}
config_file=${config_file:-""}
vocab_file=${vocab_file:-""}
vocab_model_file=${vocab_model_file:-""}
tokenizer_type=${tokenizer_type:-""}


data_dir=${data_dir:-""}
out_dir=${out_dir:-"logs/$task_name"}

seed=${seed:-""}
epochs=${epochs:-""}
batch_size=${batch_size:-24}

CMD="python3 "
CMD+="run_multichoice_mrc.py "

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
  CMD+="--predict_batch_size=$batch_size "
fi

CMD+="--seed=${seed} "
CMD+="--vocab_file=${vocab_file} "
CMD+="--vocab_model_file=${vocab_model_file} "
CMD+="--tokenizer_type=${tokenizer_type} "
CMD+="--config_file=${config_file} "
CMD+="--init_checkpoint=${init_checkpoint} "
CMD+="--data_dir=${data_dir} "
CMD+="--output_dir=${out_dir} "

CMD+="--train_file=${data_dir}/train.json "
CMD+="--train_ans_file=${data_dir}/train_answer.json "
CMD+="--dev_file=${data_dir}/dev.json "
CMD+="--dev_ans_file=${data_dir}/dev_answer.json "

CMD+="--num_train_epochs=4 "
CMD+="--gradient_accumulation_steps=12 "

CMD+="--learning_rate=2e-5 "
CMD+="--warmup_proportion=0.06 "
CMD+="--max_seq_length=96 "


LOGFILE="${out_dir}/${seed}/logfile"

$CMD |& tee $LOGFILE
