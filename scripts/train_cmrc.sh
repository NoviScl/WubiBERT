
data_dir="/home/chenyingfa/WubiBERT/datasets/realtypo/cmrc"
test_name="test_clean"
test_name="test_noisy_keyboard_2"


tok_name="char"
tok_type="BertZh"
vocab_name="bert_chinese_uncased_22675"
# ckpt="/home/chenyingfa/models/char.pt"
test_ckpt="/home/chenyingfa/WubiBERT/logs/cmrc/bert/ckpt_8601/11/best_model.bin"


# tok_name="raw"
# tok_type="RawZh"
# vocab_name="raw_zh_22675"
# test_ckpt="/home/chenyingfa/WubiBERT/logs/cmrc/raw/ckpt_8804/11/best_model.bin"


# tok_name="pinyin_no_index"
# tok_type="CommonZhNoIndex"
# # ckpt="/home/chenyingfa/models/${tok_name}.pt"
# test_ckpt="/home/chenyingfa/WubiBERT/logs/cmrc/pinyin_no_index/ckpt_8840/11/best_model.bin"
# vocab_name="pinyin_no_index_22675"


# tok_name="pinyin"
# tok_type="CommonZh"
# # ckpt="/home/chenyingfa/models/pinyin.pt"
# test_ckpt="/home/chenyingfa/WubiBERT/logs/cmrc/pinyin/ckpt_8804/11/best_model.bin"
# vocab_name="pinyin_zh_22675"



# output_dir="logs/realtypo/cmrc/${tok_name}_twolevel"
output_dir="logs/realtypo/cmrc/${tok_name}"
seed="0"

cmd="python3 run_cmrc.py"
cmd+=" --data_dir $data_dir"
cmd+=" --test_name $test_name"
# cmd+=" --init_checkpoint $ckpt"
cmd+=" --test_ckpt $test_ckpt"
cmd+=" --config_file configs/bert_config_vocab22675.json"
cmd+=" --tokenizer_type $tok_type"
cmd+=" --vocab_file tokenizers/${vocab_name}.vocab"
cmd+=" --vocab_model_file tokenizers/${vocab_name}.model"
cmd+=" --tokenizer_name ${tok_name}"
cmd+=" --output_dir ${output_dir}"
# cmd+=" --do_train"
cmd+=" --do_test"
cmd+=" --seed $seed"
# cmd+=" --two_level_embeddings"
cmd+=" --epochs 3"

log_file="$output_dir/$seed/test.log"
mkdir -p "$output_dir/$seed"

echo "$cmd"

$cmd | tee $log_file
