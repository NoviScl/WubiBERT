#!/bin/bash
#SBATCH -G 1
#_ SBATCH -p rtx2080

# This scripts trains a model on CMRC with synthetic noise as DA, and tests on READIN.

# Model
model_name="char"
# model_name="raw"
# model_name="pinyin"
# model_name="pinyin_no_index"

# data_dir="datasets/realtypo/cmrc_da_noise/phonetic_50"  # Noise DA training data
train_dir="datasets/cmrc/split"  # Ordinary training data

for seed in {0..0}
do
    ckpt="/home/chenyingfa/models/${model_name}.pt"
    # output_dir="results/da_noise/cmrc/${model_name}_seed${seed}"
    output_dir="results/cmrc/${model_name}_seed${seed}"

    # Global args
    cmd="python3 run_cmrc.py"
    cmd+=" --output_dir ${output_dir}"
    cmd+=" --config_file configs/bert_config_vocab22675.json"
    cmd+=" --tokenizer_name ${model_name}"
    cmd+=" --seed $seed"

    # Training
    train_cmd="${cmd}"
    train_cmd+=" --do_train"
    train_cmd+=" --init_ckpt $ckpt"
    train_cmd+=" --train_dir ${train_dir}"
    train_cmd+=" --dev_dir ${train_dir}"
    # train_cmd+=" --two_level_embeddings"
    train_cmd+=" --batch_size 16"
    train_cmd+=" --grad_acc_steps 2"
    train_cmd+=" --log_interval 10"
    train_cmd+=" --epochs 2"
    # train_cmd+=" --num_examples 1024"

    logfile="${output_dir}/train.log"
    mkdir -p $output_dir
    $train_cmd | tee $logfile
    
    # Testing
    test_names=""
    test_names+="test_clean"
    test_names+=" test_noisy_keyboard_1 test_noisy_keyboard_2 test_noisy_keyboard_3"
    test_names+=" test_noisy_asr_1 test_noisy_asr_2 test_noisy_asr_3"
    for test_name in $test_names
    do
        # cmd+=" --test_ckpt ${output_dir}/ckpt-1567/ckpt.pt"
        data_dir="datasets/realtypo/cmrc/${test_name}"  # READIN noise

        test_cmd="${cmd}"
        test_cmd+=" --do_test"
        test_cmd+=" --test_dir ${data_dir}"
        test_cmd+=" --test_name ${test_name}"
        # test_cmd+=" --test_ckpt logs/cmrc/raw/ckpt_8804/${seed}/best_model.bin"  # Test this pinyin model

        logfile="${output_dir}/${test_name}/test.log"
        mkdir -p "${output_dir}/${test_name}"

        echo "$cmd"
        $test_cmd | tee $logfile
    done

done