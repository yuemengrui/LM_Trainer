lr=1e-4
lora_rank=8
lora_alpha=32
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save=None
lora_dropout=0.05

pretrained_model=/workspace/LLM/Models/my-llama-13b
chinese_tokenizer_path=/workspace/LLM/Models/my-llama-13b
dataset_dir=/workspace/LLM/Data/Instruction_data/train
validation_file=/workspace/LLM/Data/Instruction_data/val.json
data_cache_dir=/workspace/LLM/Data/instruction_data_cache
per_device_train_batch_size=1
per_device_eval_batch_size=1
training_steps=10000
gradient_accumulation_steps=128
output_dir=/workspace/LLM/MyTrainer/SFT/output
#peft_model=None


deepspeed_config_file=ds_zero2_no_offload.json

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 nohup torchrun --nnodes 1 --nproc_per_node 6 run_sft_with_peft.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${chinese_tokenizer_path} \
    --dataset_dir ${dataset_dir} \
    --data_cache_dir ${data_cache_dir} \
    --validation_split_percentage 0.001 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --do_eval \
    --seed $RANDOM \
    --fp16 \
    --max_steps ${training_steps} \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.03 \
    --weight_decay 0 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 3 \
    --evaluation_strategy steps \
    --eval_steps 250 \
    --save_steps 50 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --max_seq_length 1024 \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --torch_dtype float16 \
    --validation_file ${validation_file} \
    --ddp_find_unused_parameters False \
    > train.log 2>&1 &
