# *_*coding:utf-8 *_*
# @Author : YueMengRui
from .mlp import MLP
from .transformer import TransformerLayer


def build_embedding_layer(layer_type: str = 'MLP', **kwargs):
    if layer_type == 'MLP':
        return MLP(**kwargs)
    elif layer_type == 'Transformer':
        return TransformerLayer(**kwargs)
    else:
        raise f'not support layer_type: {layer_type}'
