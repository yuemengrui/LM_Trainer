# *_*coding:utf-8 *_*
# @Author : YueMengRui
from .chatglm_embedding_model import ChatGLMEmbeddingModel


def build_model(**kwargs):
    return ChatGLMEmbeddingModel(**kwargs)
