#!/bin/bash
#SBATCH -G 1
#SBATCH -p rtx2080

# This scripts trains a model on CMRC with synthetic noise as DA, and tests on READIN.


# Model
model_name="char"
tok_type="BertZh"
vocab_name="bert_chinese_uncased_22675"

# model_name="raw"
# tok_type="RawZh"
# vocab_name="raw_zh_22675"

# model_name="pinyin"
# tok_type="CommonZh"
# vocab_name="pinyin_zh_22675"

# model_name="pinyin_no_index"
# tok_type="CommonZhNoIndex"
# vocab_name="pinyin_no_index_22675"

data_dir="datasets/realtypo/cmrc_da_noise/phonetic_50"

for seed in {0..2}
do
    ckpt="/home/chenyingfa/models/${model_name}.pt"
    output_dir="logs/realtypo/cmrc_da_noise/${model_name}_seed${seed}"

    # Global args
    cmd="python3 run_cmrc.py"
    # cmd+=" --data_dir $data_dir"
    cmd+=" --init_ckpt $ckpt"
    cmd+=" --output_dir ${output_dir}"
    cmd+=" --config_file configs/bert_config_vocab22675.json"
    cmd+=" --tokenizer_type $tok_type"
    cmd+=" --vocab_file tokenizers/${vocab_name}.vocab"
    cmd+=" --vocab_model_file tokenizers/${vocab_name}.model"
    cmd+=" --tokenizer_name ${model_name}"
    cmd+=" --seed $seed"


    # Training
    train_cmd="${cmd}"
    train_cmd+=" --do_train"
    train_cmd+=" --train_dir ${data_dir}"
    train_cmd+=" --dev_dir ${data_dir}"
    # train_cmd+=" --two_level_embeddings"
    train_cmd+=" --epochs 4"

    logfile="${output_dir}/train.log"
    mkdir -p $output_dir
    # $train_cmd | tee $logfile
    
    # Testing
    test_names=""
    test_names+="test_clean"
    test_names+=" test_noisy_keyboard_1 test_noisy_keyboard_2 test_noisy_keyboard_3"
    test_names+=" test_noisy_asr_1 test_noisy_asr_2 test_noisy_asr_3"
    for test_name in $test_names
    do
        # cmd+=" --test_ckpt ${output_dir}/ckpt-1567/ckpt.pt"
        test_dir="datasets/realtypo/cmrc/${test_name}"  # READIN noise

        test_cmd="${cmd}"
        test_cmd+=" --do_test"
        test_cmd+=" --test_dir ${test_dir}"
        test_cmd+=" --test_name ${test_name}"

        logfile="${output_dir}/${test_name}/test.log"
        mkdir -p "${output_dir}/${test_name}"

        echo "$cmd"
        $test_cmd | tee $logfile
    done

done