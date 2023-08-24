# *_*coding:utf-8 *_*
# @Author : YueMengRui
import os
import math
import torch
from loguru import logger
import numpy as np
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F
from typing import List, Dict, Union, Optional
from transformers import AutoTokenizer, AutoModel
from .embedding_layers import build_embedding_layer


def auto_configure_device_map(num_gpus: int) -> Dict[str, int]:
    # transformer.word_embeddings 占用1层
    # transformer.final_layernorm 和 lm_head 占用1层
    # transformer.layers 占用 28 层
    # 总共30层分配到num_gpus张卡上
    num_trans_layers = 28
    per_gpu_layers = 30 / num_gpus

    # bugfix: 在linux中调用torch.embedding传入的weight,input不在同一device上,导致RuntimeError
    # windows下 model.device 会被设置成 transformer.word_embeddings.device
    # linux下 model.device 会被设置成 lm_head.device
    # 在调用chat或者stream_chat时,input_ids会被放到model.device上
    # 如果transformer.word_embeddings.device和model.device不同,则会导致RuntimeError
    # 因此这里将transformer.word_embeddings,transformer.final_layernorm,lm_head都放到第一张卡上

    # device_map = {'transformer.word_embeddings': 0,
    #               'transformer.final_layernorm': 0, 'lm_head': 0}

    device_map = {
        'transformer.embedding.word_embeddings': 0,
        'transformer.encoder.final_layernorm': 0,
        'transformer.output_layer': 0,
        'transformer.rotary_pos_emb': 0,
        'lm_head': 0
    }

    used = 2
    gpu_target = 0
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0
        assert gpu_target < num_gpus
        # device_map[f'transformer.layers.{i}'] = gpu_target
        device_map[f'transformer.encoder.layers.{i}'] = gpu_target
        used += 1

    return device_map


class ChatGLMEmbeddingModel(nn.Module):

    def __init__(self, llm_model_name_or_path: str, adapter_path=None, with_embedding_layer=True,
                 device='cuda', layer_type='MLP', **kwargs):
        super().__init__()

        logger.info(
            {'llm_model_name_or_path': llm_model_name_or_path, 'adapter_path': adapter_path, 'layer_type': layer_type,
             'with_embedding_layer': with_embedding_layer, 'device': device})
        self.device = torch.device(device)
        self.chatglm, self.tokenizer = self._load_model(llm_model_name_or_path, device)
        self.set_requires_grad_to_false()
        self.embedding_layer = None
        if with_embedding_layer:
            self.embedding_layer = build_embedding_layer(layer_type=layer_type,
                                                         hidden_size=self.chatglm.config.hidden_size,
                                                         intermediate_size=11008,
                                                         **kwargs)
            self.embedding_layer.half().to(self.device)
            if adapter_path:
                self.load_adapter_model(adapter_path)

    def get_embedding_dim(self):
        return self.chatglm.config.hidden_size

    def get_tokenizer(self):
        return self.tokenizer

    def forward(self, inputs):
        model_output = self.chatglm(**inputs, output_hidden_states=True)
        hidden_states = model_output.hidden_states[-1]  # [Seq_len, Batch, hidden_size]
        if self.embedding_layer:
            hidden_states = self.embedding_layer(hidden_states)  # # [Seq_len, Batch, hidden_size]
        output = hidden_states.transpose(0, 1)  # [Batch, Seq_len, hidden_size]

        return output

    def save_adapter_model(self, output_dir):
        torch.save(self.embedding_layer.state_dict(), os.path.join(output_dir, "chatglm_embedding_adapter.pth"))

    def load_adapter_model(self, save_dir):
        self.embedding_layer.load_state_dict(torch.load(os.path.join(save_dir, "chatglm_embedding_adapter.pth")))

    def set_requires_grad_to_false(self):
        for name, param in self.chatglm.named_parameters():
            param.requires_grad = False

    def _load_model(self,
                    model_name_or_path,
                    device='cuda',
                    device_map: Optional[Dict[str, int]] = None,
                    **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )

        if torch.cuda.is_available() and device.lower().startswith("cuda"):
            # 根据当前设备GPU数量决定是否进行多卡部署
            num_gpus = torch.cuda.device_count()
            if num_gpus < 2 and device_map is None:
                model = (
                    AutoModel.from_pretrained(
                        model_name_or_path,
                        trust_remote_code=True,
                        **kwargs)
                    .half()
                    .cuda()
                )
            else:
                from accelerate import dispatch_model

                model = (
                    AutoModel.from_pretrained(
                        model_name_or_path,
                        trust_remote_code=True,
                        **kwargs)
                    .half())
                # 可传入device_map自定义每张卡的部署情况
                if device_map is None:
                    device_map = auto_configure_device_map(num_gpus)

                model = dispatch_model(model, device_map=device_map)
        else:
            if device == 'mps':
                model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).to('mps')

            else:
                model = (
                    AutoModel.from_pretrained(
                        model_name_or_path,
                        trust_remote_code=True,
                        **kwargs)
                    .float()
                    .to(device)
                )

        return model, tokenizer
