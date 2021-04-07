#!/bin/bash
set -e

echo "Container nvidia build = " $NVIDIA_BUILD_ID

init_checkpoint=${init_checkpoint:-"results/checkpoints_bert_zh_22675/ckpt_8601.pt"}
config_file=${config_file:-"configs/bert_config_vocab22675.json"}
vocab_file=${vocab_file:-"tokenizers/bert_chinese_uncased_22675.vocab"}
vocab_model_file=${vocab_model_file:-"tokenizers/bert_chinese_uncased_22675.model"}
tokenizer_type=${tokenizer_type:-"BertZh"}

# Dataset
task_name=${task_name:-"csl"}
data_dir=${data_dir:-"datasets/$task_name"}
fewshot=${fewshot:-1}

seed=${seed:-"2"}
out_dir=${out_dir:-"logs/temp"}
# mode=${mode:-"prediction"}
mode=${mode:-"train eval"}
num_gpu=${num_gpu:-"1"}

# Hyperparameters
epochs=${epochs:-"4"}
max_steps=${13:-"-1.0"}
batch_size=${batch_size:-"32"}
gradient_accumulation_steps=${gradient_accumulation_steps:-"2"}
learning_rate=${10:-"2e-5"}
warmup_proportion=${11:-"0.1"}
# precision=${14:-"fp16"}

mkdir -p $out_dir

if [ "$mode" = "eval" ] ; then
  num_gpu=1
fi

use_fp16=""
if [ "$precision" = "fp16" ] ; then
  echo "fp16 activated!"
  use_fp16="--fp16"
fi

if [ "$num_gpu" = "1" ] ; then
  export CUDA_VISIBLE_DEVICES=0
  mpi_command=""
else
  unset CUDA_VISIBLE_DEVICES
  mpi_command=" -m torch.distributed.launch --master_port=423333 --nproc_per_node=$num_gpu"
fi

CMD="python3 $mpi_command run_glue.py "
CMD+="--task_name ${task_name} "
if [[ $mode == *"train"* ]] ; then
  CMD+="--do_train "
  CMD+="--train_batch_size=$batch_size "
fi
if [[ $mode == *"eval"* ]] || [[ $mode == *"pred"* ]]; then
  if [[ $mode == *"eval"* ]] ; then
    CMD+="--do_eval "
  fi
  if [[ $mode == *"pred"* ]] ; then
    CMD+="--do_predict "
  fi
  CMD+="--eval_batch_size=$batch_size "
fi

CMD+="--gradient_accumulation_steps=$gradient_accumulation_steps "
CMD+="--do_lower_case "
CMD+="--tokenizer_type $tokenizer_type "
CMD+="--vocab_model_file $vocab_model_file "
CMD+="--data_dir $data_dir "
CMD+="--bert_model bert-tiny "
CMD+="--seed $seed "
CMD+="--init_checkpoint $init_checkpoint "
CMD+="--warmup_proportion $warmup_proportion "
CMD+="--max_seq_length 256 "
CMD+="--learning_rate $learning_rate "
CMD+="--num_train_epochs $epochs "
CMD+="--max_steps $max_steps "
CMD+="--vocab_file=$vocab_file "
CMD+="--config_file=$config_file "
CMD+="--output_dir $out_dir "
CMD+="--fewshot $fewshot "
CMD+="$use_fp16"

LOGFILE=$out_dir/$seed/logfile

$CMD |& tee $LOGFILE
