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
    # "checkpoints_wubi_zh/ckpt_8601.pt"
    # "checkpoints_raw_zh/ckpt_8601.pt"
    # "checkpoints_concat_sep/ckpt_8601.pt"
    # "checkpoints_bert_zh_22675/ckpt_8601.pt"
    "results/checkpoints_cangjie_22675"
    "results/checkpoints_stroke_22675"
    "wubi_results/checkpoints_pinyin_zh_22675"
    "wubi_results/checkpoints_wubi_zh_22675"
    "wubi_results/checkpoints_zhuyin_zh_22675"
    "wubi_results/checkpoints_zhengma_zh_22675"
    "wubi_results/checkpoints_raw_zh_22675"
)

best_ckpts=(
    ckpt_6393
    ckpt_8795
    ckpt_5593
    ckpt_7992
    ckpt_7192
    ckpt_3994
    ckpt_3194
)

# config_files=(
#     "bert_config_vocab30k.json"
#     "bert_config_vocab30k.json"
#     "bert_config_vocab30k.json"
#     "bert_config_vocab22675.json"
# )

vocab_files=(
    "cangjie_zh_22675.vocab"
    "stroke_zh_22675.vocab"
    "pinyin_zh_22675.vocab"
    "wubi_zh_22675.vocab"
    "zhuyin_zh_22675.vocab"
    "zhengma_zh_22675.vocab"
    "/mnt/datadisk0/scl/WubiBERT/tokenizers/raw_zh_22675.vocab"
    # "sp_wubi_zh_30k_sep.vocab"
    # "sp_raw_zh_30k.vocab"
    # "sp_concat_30k_sep.vocab"
    # "bert_chinese_uncased_22675.vocab"
)

vocab_model_files=(
    "cangjie_zh_22675.model"
    "stroke_zh_22675.model"
    "pinyin_zh_22675.model"
    "wubi_zh_22675.model"
    "zhuyin_zh_22675.model"
    "zhengma_zh_22675.model"
    "/mnt/datadisk0/scl/WubiBERT/tokenizers/raw_zh_22675.model"
    # "sp_wubi_zh_30k_sep.model"
    # "sp_raw_zh_30k.model"
    # "sp_concat_30k_sep.model"
    # "bert_chinese_uncased_22675.model"
)

tokenizer_types=(
    # "CommonZh"
    # "CommonZh"
    # "CommonZh"
    # "WubiZh"
    # "RawZh"
    # "ConcatSep"
    # "BertZh"
)

config_file="configs/bert_config_vocab22675.json"

output_dirs=(
    "cangjie"
    "stroke"
    "pinyin"
    "wubi"
    "zhuyin"
    "zhengma"
    "raw"
    # "wubi_zh"
    # "raw_zh"
    # "concat_sep"
    # "bert_zh_22675"
)


classification_tasks=(
    "tnews"
    "iflytek"
    "wsc"
    "afqmc"
    "csl"
    "ocnli"
)

# Change these
# task_name="tnews"
# task_name="iflytek"
# task_name="wsc"
# task_name="afqmc"
# task_name="csl"
# task_name="ocnli"
task_name="chid"
# task_name="cmrc"
# task_name="c3"

# epochs=6  # All 6 classification tasks
# epochs=3  # cmrc
epochs=6  # chid
# epochs=6  # C3
# batch_size=24

mode="train eval test"
# mode="test"

# Fewshot
fewshot=0       # 1 = true
# epochs=50
# batch_size=4

seeds=(
    "2"
    # "23"
    # "234"
)

# Don't change below
for seed in ${seeds[@]}
do
    # loop tokenizers
    for i in {0..1}
    do
    # i=0
        # Model
        # config_file="configs/${config_files[$i]}"
        # vocab_file="tokenizers/${vocab_files[$i]}"
        vocab_file="/home/ubuntu/WubiBERT/tokenizers/22675/${vocab_files[$i]}"
        # vocab_model_file="tokenizers/${vocab_model_files[$i]}"
        vocab_model_file="/home/ubuntu/WubiBERT/tokenizers/22675/${vocab_model_files[$i]}"
        # tokenizer_type=${tokenizer_types[$i]}
        tokenizer_type="CommonZh"

        if [ "${output_dirs[$i]}" = "raw" ] ; then
            tokenizer_type="RawZh"
            vocab_file=${vocab_files[$i]}
            vocab_model_file=${vocab_model_files[$i]}
        fi

    # for task_name in ${classification_tasks[@]}
    # do
        data_dir="datasets/${task_name}/split"


        if [ "$task_name" = "chid" ] || [ "$task_name" = "c3" ] || [ "$task_name" = "cmrc" ] ; then
            script="./scripts/run_mrc_${task_name}.sh"
        else
            script="./scripts/run_finetune.sh"
        fi
    
    # init_checkpoint="results/${init_checkpoints[$i]}"
    checkpoints="/mnt/datadisk0/scl/${init_checkpoints[$i]}"
    # loop check points
    # ckpts=(`ls "$checkpoints"/*`)
    # for j in {1..11}
    # do
    #     # init_checkpoint=$checkpoints/$ckpt
    #     init_checkpoint=${ckpts[$j]}
    #     ckpt=${ckpts[$j]}
    #     ckpt=${ckpt##*/}
    #     ckpt=${ckpt%.pt}

        ckpt=${best_ckpts[$i]}
        init_checkpoint=${checkpoints}/$ckpt.pt

        # echo "ckpt = $ckpt"
        
        # break

        # echo basename $ckpt ".pt"
        # Input (data) and output (result)
        output_dir="logs/${task_name}/${output_dirs[$i]}/$ckpt"

        # echo $init_checkpoint
        echo $script 
        echo "    Task:       $task_name"
        echo "    Vocab:      $vocab_file"
        # echo "    Tokenizer:  $tokenizer_type"
        echo "    Checkpoint: $ckpt" 
        echo "    Seed:       $seed"
        # echo "    $output_dir"
        # continue

        export init_checkpoint="$init_checkpoint"
        export task_name="$task_name"
        export config_file="$config_file"
        export vocab_file="$vocab_file"
        export vocab_model_file="$vocab_model_file"
        export tokenizer_type="$tokenizer_type"
        export out_dir="$output_dir"
        export data_dir="$data_dir"
        export seed=$seed
        export epochs=$epochs
        export fewshot=$fewshot
        export batch_size=$batch_size
        export mode="$mode"

        mkdir -p "$out_dir/$seed"

        LOGFILE="$out_dir/$seed/logfile.txt"

        # $script
        $script &> $LOGFILE &
        sleep 20
    done
done


            # if [ fewshot = 1 ] ; then
            #     slurm_output_dir="slurm_output/${task_name}/${output_dirs[$i]}/fewshot"
            # else
            #     slurm_output_dir="slurm_output/${task_name}/${output_dirs[$i]}"
            # fi

            # mkdir -p slurm_output/$task_name
            # mkdir -p slurm_output/${task_name}/${output_dirs[$i]}
            # mkdir -p $slurm_output_dir

            # # echo "$slurm_output_dir/slurm-%j.out"
#         sbatch -N 1 \
#         -n 5 \
#         -G 1 \
#         --no-requeue \
#         -o "${slurm_output_dir}/seed${seed}-%j.out" \
#         --export=init_checkpoint="$init_checkpoint",\
# task_name="$task_name",\
# config_file="$config_file",\
# vocab_file="$vocab_file",\
# vocab_model_file="$vocab_model_file",\
# tokenizer_type="$tokenizer_type",\
# out_dir="$output_dir",\
# data_dir="$data_dir",\
# seed=$seed,\
# epochs=$epochs,\
# fewshot=$fewshot,\
# batch_size=$batch_size,\
# mode="$mode" \
#         ${script}