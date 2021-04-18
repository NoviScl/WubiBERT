#!/bin/bash
#SBATCH
#SBATCH -N 1
#SBATCH -n 5
#SBATCH -G 1
#SBATCH --no-requeue
#SBATCH -o slurm_output/slurm-%j.out

#SSBATCH --gres=gpu:1

# cd /data/private/zhangzhengyan/projects/PLM-Task-Agnostic-Backdoor/src
# source /data/private/zhangzhengyan/miniconda3/bin/activate backdoor
# bash $3 $1 0 $2

init_checkpoints=(
    "checkpoints_wubi_zh/ckpt_8601.pt"
    "checkpoints_raw_zh/ckpt_8601.pt"
    "checkpoints_concat_sep/ckpt_8601.pt"
    "checkpoints_bert_zh_22675/ckpt_8601.pt"
)

config_files=(
    "bert_config_vocab30k.json"
    "bert_config_vocab30k.json"
    "bert_config_vocab30k.json"
    "bert_config_vocab22675.json"
)

vocab_files=(
    "sp_wubi_zh_30k_sep.vocab"
    "sp_raw_zh_30k.vocab"
    "sp_concat_30k_sep.vocab"
    "bert_chinese_uncased_22675.vocab"
)

vocab_model_files=(
    "sp_wubi_zh_30k_sep.model"
    "sp_raw_zh_30k.model"
    "sp_concat_30k_sep.model"
    "bert_chinese_uncased_22675.model"
)

tokenizer_types=(
    "WubiZh"
    "RawZh"
    "ConcatSep"
    "BertZh"
)

output_dirs=(
    "wubi_zh"
    "raw_zh"
    "concat_sep"
    "bert_zh_22675"
)

seeds=(
    "2"
    "23"
    "234"
)

# Change these
# task_name="tnews"
# task_name="iflytek"
# task_name="wsc"
# task_name="afqmc"
# task_name="csl"
# task_name="ocnli"
# task_name="chid"
task_name="c3"

# epochs=6  # All 6 classification tasks
# epochs=3  # cmrc
# epochs=6  # chid
epochs=6  # C3
# batch_size=24

mode="train eval test"
# mode="test"

# Fewshot
fewshot=0       # 1 = true
# epochs=50
# batch_size=4

data_dir="datasets/${task_name}/split"
# data_dir="datasets/${task_name}"



if [ "$task_name" = "chid" ] || [ "$task_name" = "c3" ] || [ "$task_name" = "cmrc" ] ; then
  script="./scripts/run_mrc_${task_name}.sh"
else
  script="./scripts/run_finetune.sh"
fi

# Don't change below
for seed in ${seeds[@]}
do
    for i in {0..3}
    do
        # Model
        init_checkpoint="results/${init_checkpoints[$i]}"
        config_file="configs/${config_files[$i]}"
        vocab_file="tokenizers/${vocab_files[$i]}"
        vocab_model_file="tokenizers/${vocab_model_files[$i]}"
        tokenizer_type=${tokenizer_types[$i]}
        
        # Input (data) and output (result)
        output_dir="logs/${task_name}/${output_dirs[$i]}"

        # echo $init_checkpoint
        echo $script $task_name $tokenizer_type $seed
        # echo $seed
        # echo $script

        # Submit to slurm
        if [ fewshot = 1 ] ; then
            slurm_output_dir="slurm_output/${task_name}/${output_dirs[$i]}/fewshot"
        else
            slurm_output_dir="slurm_output/${task_name}/${output_dirs[$i]}"
        fi
        mkdir -p slurm_output/$task_name
        mkdir -p slurm_output/${task_name}/${output_dirs[$i]}
        mkdir -p $slurm_output_dir

        # echo "$slurm_output_dir/slurm-%j.out"
        
        sbatch -N 1 \
        -n 5 \
        -G 1 \
        --no-requeue \
        -o "${slurm_output_dir}/seed${seed}-%j.out" \
        --export=init_checkpoint="$init_checkpoint",\
task_name="$task_name",\
config_file="$config_file",\
vocab_file="$vocab_file",\
vocab_model_file="$vocab_model_file",\
tokenizer_type="$tokenizer_type",\
out_dir="$output_dir",\
data_dir="$data_dir",\
seed=$seed,\
epochs=$epochs,\
fewshot=$fewshot,\
batch_size=$batch_size,\
mode="$mode" \
        ${script}
    done
done
