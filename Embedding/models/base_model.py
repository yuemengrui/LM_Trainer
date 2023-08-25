# *_*coding:utf-8 *_*
# @Author : YueMengRui
import os
import torch


class BaseModel:

    def __init__(self, **kwargs):
        self.llm = None
        self.tokenizer = None
        self.embedding_layer = None

    def get_embedding_dim(self):
        return self.llm.config.hidden_size

    def get_tokenizer(self):
        return self.tokenizer

    def save_adapter_model(self, save_dir):
        torch.save(self.embedding_layer.state_dict(), os.path.join(save_dir, "llm_embedding_adapter.pth"))

    def load_adapter_model(self, save_dir):
        self.embedding_layer.load_state_dict(torch.load(os.path.join(save_dir, "llm_embedding_adapter.pth")))

    def set_requires_grad_to_false(self):
        for name, param in self.llm.named_parameters():
            param.requires_grad = False
