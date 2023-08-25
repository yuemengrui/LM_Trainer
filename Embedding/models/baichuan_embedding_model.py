# *_*coding:utf-8 *_*
# @Author : YueMengRui
import os
import torch
import torch.nn as nn
from loguru import logger
import torch.utils.checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from .embedding_layers import build_embedding_layer
from .base_model import BaseModel


class BaichuanEmbeddingModel(nn.Module, BaseModel):

    def __init__(self, llm_model_name_or_path: str, adapter_path=None, with_embedding_layer=True,
                 device='cuda', layer_type='Transformer', **kwargs):
        super().__init__()

        logger.info(
            str({'llm_model_name_or_path': llm_model_name_or_path, 'adapter_path': adapter_path,
                 'layer_type': layer_type, 'with_embedding_layer': with_embedding_layer,
                 'device': device}) + str(kwargs))

        self.llm, self.tokenizer = self._load_model(llm_model_name_or_path, device)
        self.device = self.llm.device
        self.set_requires_grad_to_false()
        self.embedding_layer = None
        if with_embedding_layer:
            self.embedding_layer = build_embedding_layer(layer_type=layer_type,
                                                         d_model=self.llm.config.hidden_size,
                                                         nhead=self.llm.config.num_attention_heads,
                                                         **kwargs)
            self.embedding_layer.half().to(self.device)
            if adapter_path:
                self.load_adapter_model(adapter_path)

    def forward(self, inputs):
        model_output = self.llm(**inputs, output_hidden_states=True)
        output = model_output.hidden_states[-1]  # [Batch, Seq_len, hidden_size]
        if self.embedding_layer:
            output = self.embedding_layer(output)

        return output

    def _load_model(self, model_name_or_path, device):

        if device == 'mps':
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=True
            ).half().to('mps')
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        model.generation_config = GenerationConfig.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=False,
            trust_remote_code=True
        )

        return model, tokenizer
