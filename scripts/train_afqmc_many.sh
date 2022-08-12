#!/bin/bash

task="afqmc"
task="afqmc_balanced"
# data_dir="datasets/${task}/split"
test_name="test_noisy_keyboard_3"
# test_name="test_clean"

data_dir="datasets/realtypo/${task}"

model_name="char"
tok_type="BertZh"
vocab_name="bert_chinese_uncased_22675"

model_name="raw"
tok_type="RawZh"
vocab_name="raw_zh_22675"

model_name="pinyin"
tok_type="CommonZh"
vocab_name="pinyin_zh_22675"

model_name="pinyin_no_index"
tok_type="CommonZhNoIndex"
vocab_name="pinyin_no_index_22675"

# model_name="pypinyin"
# tok_type="Pypinyin"
# vocab_name="pypinyin_22675_notone_noindex"



# model_name="pypinyin_12L"
# tok_type="Pypinyin"
# vocab_name="pypinyin_22675_notone_noindex"

# model_name="pypinyin_nosep_12L"
# tok_type="PypinyinNosep"
# vocab_name="pypinyin_22675_notone_noindex_nosep"


for seed in {4..5}
do
    # test_name="test_clean"
    # test_dir="datasets/realtypo/${task}/${test_name}"

    # cmd="python3 run_glue.py"
    # cmd+=" --task_name afqmc"
    # cmd+=" --train_dir=${data_dir}"
    # cmd+=" --dev_dir=${data_dir}"
    # cmd+=" --test_dir=${test_dir}"
    # cmd+=" --do_train"
    # cmd+=" --do_test"
    # cmd+=" --init_ckpt ${ckpt}"
    # cmd+=" --output_dir ${output_dir}"
    # cmd+=" --tokenizer_type ${tok_type}"
    # cmd+=" --vocab_file tokenizers/${vocab_name}.vocab"
    # cmd+=" --vocab_model_file tokenizers/${vocab_name}.model"
    # cmd+=" --config_file configs/bert_config_vocab22675.json"
    # cmd+=" --epochs 4"
    # cmd+=" --seed ${seed}"
    # cmd+=" --test_name test_clean"
    # # cmd+=" --tokenize_char_by_char"

    # logfile="${output_dir}/train.log"

    # $cmd | tee logfile


    for test_name in \
        phonetic_10
        # test_clean \
        # test_noisy_keyboard_1 test_noisy_keyboard_2 test_noisy_keyboard_3 \
        # test_noisy_asr_1 test_noisy_asr_2 test_noisy_asr_3
    do
        # test_dir="datasets/realtypo/${task}/${test_name}"  # READIN noise
        test_dir="datasets/afqmc/noisy/${test_name}"       # Synthetic noise from SCT
        ckpt="/home/chenyingfa/models/${model_name}.pt"
        output_dir="logs/realtypo/${task}/${model_name}/${seed}/"
        test_name="test_synthetic_noise_${test_name}"

        cmd="python3 run_glue.py"
        cmd+=" --task_name afqmc"
        cmd+=" --train_dir=${data_dir}"
        cmd+=" --dev_dir=${data_dir}"
        cmd+=" --test_dir=${test_dir}"
        # cmd+=" --do_train"
        cmd+=" --do_test"
        cmd+=" --init_ckpt ${ckpt}"
        cmd+=" --output_dir ${output_dir}"
        cmd+=" --tokenizer_type ${tok_type}"
        cmd+=" --vocab_file tokenizers/${vocab_name}.vocab"
        cmd+=" --vocab_model_file tokenizers/${vocab_name}.model"
        cmd+=" --config_file configs/bert_config_vocab22675.json"
        cmd+=" --epochs 4"
        cmd+=" --seed ${seed}"
        cmd+=" --test_name ${test_name}"
        # cmd+=" --tokenize_char_by_char"

        logfile="${output_dir}/test.log"

        $cmd | tee logfile

    done
done
