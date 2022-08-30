#!/bin/bash

data_dir="datasets/realtypo/afqmc_balanced"
# data_dir="datasets/realtypo/afqmc_balanced_da_noise/phonetic_50"

model_name="char"
model_name="raw"
model_name="pinyin"
model_name="pinyin_no_index"


for seed in {1..3}
do
    ckpt="/home/chenyingfa/models/${model_name}.pt"
    # output_dir="logs/realtypo/afqmc_balanced_da_noise/${model_name}/${seed}"
    output_dir="logs/realtypo/afqmc_balanced/${model_name}/${seed}"

    # Global arguments
    cmd="python3 run_glue.py"
    cmd+=" --task_name afqmc"
    cmd+=" --init_ckpt ${ckpt}"
    cmd+=" --output_dir ${output_dir}"
    cmd+=" --tokenizer_name ${model_name}"
    cmd+=" --config_file configs/bert_config_vocab22675.json"
    cmd+=" --seed ${seed}"

    # Training
    train_cmd="${cmd}"
    train_cmd+=" --do_train"
    train_cmd+=" --train_dir ${data_dir}"
    train_cmd+=" --dev_dir ${data_dir}"
    train_cmd+=" --epochs 4"

    logfile="${output_dir}/train.log"
    # $train_cmd | tee $logfile

    # Testing
    for test_name in \
        test_clean \
        test_noisy_keyboard_1 test_noisy_keyboard_2 test_noisy_keyboard_3 \
        test_noisy_asr_1 test_noisy_asr_2 test_noisy_asr_3
    do
        # test_dir="datasets/realtypo/afqmc_balanced/${test_name}"  # READIN noise
        # test_dir="datasets/afqmc/noisy/${test_name}"       # Synthetic noise from SCT
        test_dir="/data/private/chenyingfa/readin/text_correction/afqmc_balanced"  # Text correction

        test_cmd="${cmd}"
        test_cmd+=" --do_test"
        test_cmd+=" --test_dir ${test_dir}"
        test_cmd+=" --test_name ${test_name}_typo"

        logfile="${output_dir}/${test_name}/test_typo.log"
        mkdir -p "$output_dir/${test_name}"
        $test_cmd | tee $logfile
        # exit
    done
done
