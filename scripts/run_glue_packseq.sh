task="tnews"
data_path="datasets/${task}/split"
seed="0"

# # CharTokenizer
# output_dir="results/${task}/bert_packseq"
# ckpt="checkpoints/checkpoints_bert_zh_22675/ckpt_8601.pt"
# vocab_prefix="tokenizers/bert_chinese_uncased_22675"
# tok_type="BertZh"

# Pinyin-NoIndex
output_dir="results/${task}/pinyin_no_index_packseq"
ckpt="checkpoints/checkpoints_pinyin_no_index/ckpt_8840.pt"
tok_type="CommonZhNoIndex"
vocab_prefix="tokenizers/pinyin_no_index_22675"

cmd="\
  python3 -u run_glue.py \
  --task_name ${task} \
  --data_dir $data_path \
  --init_checkpoint $ckpt \
  --output_dir $output_dir \
  --tokenizer_type $tok_type \
  --vocab_file ${vocab_prefix}.vocab \
  --vocab_model_file ${vocab_prefix}.model \
  --config_file=configs/bert_config_vocab22675.json \
  --epochs 6 \
  --do_train --do_eval --do_test \
  --max_seq_len 512 \
  --train_batch_size 16 \
  --pack_seq \
  --seed $seed \
"

mkdir -p "$output_dir"
mkdir -p "$output_dir/$seed"

logfile="$output_dir/$seed/log.txt"
cmd+=" | tee $logfile"

echo $cmd
eval $cmd

