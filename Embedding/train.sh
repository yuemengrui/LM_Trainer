llm_model_name_or_path=""
device="cuda"
dataset_dir=""
batch_size=8
num_epochs=2
max_seq_len=512
lr=2e-5
logging_steps=100
save_steps=2000
output_dir="./output"
resume_checkpoint=None
log_file="train.log"

CUDA_VISIBLE_DEVICES=0 nohup python train.py \
  --llm_model_name_or_path ${llm_model_name_or_path} \
  --device ${device} \
  --dataset_dir ${dataset_dir} \
  --batch_size ${batch_size} \
  --num_epochs ${num_epochs} \
  --max_seq_len ${max_seq_len} \
  --lr ${lr} \
  --warmup_ratio 0.05 \
  --logging_steps ${logging_steps} \
  --save_steps ${save_steps} \
  --resume_checkpoint ${resume_checkpoint} \
  --output_dir ${output_dir} \
  >${log_file} 2>&1 &
