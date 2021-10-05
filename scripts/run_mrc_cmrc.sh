#!/usr/bin/env bash
model_name="bert-tiny"
task_name=${task_name:-"nulls"}

init_checkpoint=${init_checkpoint:-""}
config_file=${config_file:-""}
vocab_file=${vocab_file:-""}
vocab_model_file=${vocab_model_file:-""}
cws_vocab_file=${cws_vocab_file:-""}
tokenizer_type=${tokenizer_type:-""}

convert_to_simplified=${convert_to_simplified:-""}
two_level_embeddings=${two_level_embeddings:-""}
debug=${debug:-0}

data_dir=${data_dir:-""}

out_dir=${out_dir:-""}

mode=${mode:-"train test"}
seed=${seed:-""}
epochs=${epochs:-"6"}

mkdir -p $out_dir

CMD="python3 "
if [ "$task_name" = "drcd" ] ; then
  CMD+="run_drcd.py "
elif [ "$task_name" = "cmrc" ] ; then
  CMD+="run_cmrc.py "
else
  echo "INVALID task_name: $task_name"
fi

if [[ $mode == *"train"* ]] ; then
  CMD+="--do_train "
fi

if [[ $mode == *"test"* ]] ; then
  CMD+="--do_test "
fi

if [ $two_level_embeddings -eq 1 ] ; then
  CMD+="--two_level_embeddings "
fi

if [ $debug -eq 1 ] ; then
  CMD+="--debug "
fi

# if [ $convert_to_simplified -eq 1 ] ; then
#   CMD+="--convert_to_simplified "
# fi

# CMD+="--task_name=${task_name} "
CMD+="--tokenizer_type=${tokenizer_type} "
CMD+="--vocab_file=${vocab_file} "
CMD+="--vocab_model_file=${vocab_model_file} "
CMD+="--config_file=${config_file} "
CMD+="--init_checkpoint=${init_checkpoint} "
CMD+="--epochs=${epochs} "
CMD+="--seed=${seed} "
CMD+="--data_dir=${data_dir} "
CMD+="--output_dir=${out_dir} "

CMD+="--batch_size=32 "
CMD+="--gradient_accumulation_steps=8 "
CMD+="--lr=3e-5 "
CMD+="--warmup_rate=0.05 "
CMD+="--max_seq_length=512 "

if [[ $cws_vocab_file != "" ]] ; then
  CMD+="--cws_vocab_file $cws_vocab_file "
fi

LOGFILE=$out_dir/$seed/logfile

$CMD |& tee $LOGFILE

