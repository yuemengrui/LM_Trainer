# *_*coding:utf-8 *_*
# @Author : YueMengRui
import torch.nn as nn
from torch.nn.modules.transformer import TransformerEncoderLayer, LayerNorm, TransformerEncoder


class TransformerLayer(nn.Module):

    def __init__(self, d_model: int,
                 nhead: int,
                 dim_feedforward: int,
                 num_encoder_layers: int = 1,
                 dropout: float = 0.1,
                 activation: str = 'gelu',
                 layer_norm_eps: float = 1e-5,
                 batch_first: bool = True,
                 norm_first: bool = True,
                 device=None,
                 dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                activation, layer_norm_eps, batch_first, norm_first,
                                                **factory_kwargs)
        encoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    def forward(self, x):
        return self.encoder(x)
