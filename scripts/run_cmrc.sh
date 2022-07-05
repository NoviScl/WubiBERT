
data_dir="/home/chenyingfa/WubiBERT/datasets/realtypo/cmrc"
test_name="test_clean"


tok_name="char"
tok_type="BertZh"
vocab_name="bert_chinese_uncased_22675"
ckpt="/home/chenyingfa/models/char.pt"

# tok_name="pinyin_no_index"
# tok_type="CommonZhNoIndex"
# ckpt="/home/chenyingfa/models/${tok_name}.pt"
# vocab_name="pinyin_no_index_22675"

# tok_name="pinyin"
# tok_type="CommonZh"
# ckpt="/home/chenyingfa/models/pinyin.pt"
# vocab_name="pinyin_zh_22675"

output_dir="logs/realtypo/cmrc/${tok_name}"
seed="0"

cmd="python3 run_cmrc.py"
cmd+=" --data_dir $data_dir"
cmd+=" --test_name $test_name"
cmd+=" --init_checkpoint $ckpt"
cmd+=" --config_file configs/bert_config_vocab22675.json"
cmd+=" --tokenizer_type $tok_type"
cmd+=" --vocab_file tokenizers/${vocab_name}.vocab"
cmd+=" --vocab_model_file tokenizers/${vocab_name}.model"
cmd+=" --tokenizer_name ${tok_name}"
cmd+=" --output_dir ${output_dir}"
cmd+=" --do_train"
cmd+=" --do_test"
cmd+=" --seed $seed"
# cmd+=" --two_level_embeddings"
cmd+=" --epochs 3"

log_file="$output_dir/$seed/train.log"
mkdir -p "$output_dir/$seed"

echo "$cmd"

$cmd | tee $log_file
