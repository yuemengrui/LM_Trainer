# *_*coding:utf-8 *_*
# @Author : YueMengRui
import argparse


def args_parse():
    parser = argparse.ArgumentParser('LLM based Embedding')
    parser.add_argument('--llm_model_name_or_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--output_dir', default='./output', type=str)
    parser.add_argument('--log_dir', default='./logs', type=str)
    parser.add_argument("--max_seq_len", type=int, default=512, help="text max token len")
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--lr', default=2e-5, type=float)
    parser.add_argument('--warmup_ratio', default=0.05, type=float)
    parser.add_argument('--logging_steps', default=100, type=int)
    parser.add_argument('--eval_steps', default=100, type=int)
    parser.add_argument('--save_steps', default=1000, type=int)
    parser.add_argument('--save_total_limit', default=1, type=int)
    parser.add_argument('--data_parallel', action="store_true")
    parser.add_argument('--just_init_weight', action="store_true")
    parser.add_argument('--resume_checkpoint', type=str)
    args = parser.parse_args()
    return args
