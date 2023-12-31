llm_model_name_or_path=""
device="cuda"
dataset_dir=""
batch_size=8
gradient_accumulation_steps=1
num_epochs=2
max_seq_len=512
lr=2e-5
logging_steps=200
eval_steps=1000
save_steps=2000
save_total_limit=2
output_dir="./output"
resume_checkpoint=None
log_file="train.log"

CUDA_VISIBLE_DEVICES=0 nohup python train.py \
  --llm_model_name_or_path ${llm_model_name_or_path} \
  --device ${device} \
  --dataset_dir ${dataset_dir} \
  --batch_size ${batch_size} \
  --gradient_accumulation_steps ${gradient_accumulation_steps} \
  --num_epochs ${num_epochs} \
  --max_seq_len ${max_seq_len} \
  --lr ${lr} \
  --warmup_ratio 0.05 \
  --logging_steps ${logging_steps} \
  --save_steps ${save_steps} \
  --eval_steps ${eval_steps} \
  --save_total_limit ${save_total_limit} \
  --resume_checkpoint ${resume_checkpoint} \
  --output_dir ${output_dir} \
  >${log_file} 2>&1 &
