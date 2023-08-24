# *_*coding:utf-8 *_*
# @Author : YueMengRui
import os
import math
import torch
import numpy as np
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F
from typing import List, Dict, Union, Optional
from transformers import AutoTokenizer, AutoModel


def torch_gc(device):
    if torch.cuda.is_available():
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


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


def _config_to_kwargs(args):
    common_kwargs = {
        "dtype": args.torch_dtype,
    }
    return common_kwargs


@torch.jit.script
def apply_rotary_pos_emb(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    # x: [sq, b, np, hn]
    sq, b, np, hn = x.size(0), x.size(1), x.size(2), x.size(3)
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # truncate to support variable sizes
    rope_cache = rope_cache[:sq]
    xshaped = x.reshape(sq, -1, np, rot_dim // 2, 2)
    rope_cache = rope_cache.contiguous().view(sq, -1, 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return torch.cat((x_out2, x_pass), dim=-1)


def split_tensor_along_last_dim(
        tensor: torch.Tensor,
        num_partitions: int,
        contiguous_split_chunks: bool = False,
) -> List[torch.Tensor]:
    """Split a tensor along its last dimension.

    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.

    Returns:
        A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = tensor.size()[last_dim] // num_partitions
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, device=None, dtype=None, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(normalized_shape, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        return (self.weight * hidden_states).to(input_dtype)


class CoreAttention(nn.Module):
    def __init__(self, config, layer_number):
        super(CoreAttention, self).__init__()

        self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)

        projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_partition = projection_size
        self.hidden_size_per_attention_head = projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff
        self.coeff = coeff

        self.attention_dropout = torch.nn.Dropout(config.attention_dropout)

    def forward(self, query_layer, key_layer, value_layer, attention_mask):
        pytorch_major_version = int(torch.__version__.split('.')[0])
        if pytorch_major_version >= 2:
            query_layer, key_layer, value_layer = [k.permute(1, 2, 0, 3) for k in [query_layer, key_layer, value_layer]]
            if attention_mask is None and query_layer.shape[2] == key_layer.shape[2]:
                context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer,
                                                                                 is_causal=True)
            else:
                if attention_mask is not None:
                    attention_mask = ~attention_mask
                context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer,
                                                                                 attention_mask)
            context_layer = context_layer.permute(2, 0, 1, 3)
            new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
            context_layer = context_layer.reshape(*new_context_layer_shape)
        else:
            # Raw attention scores

            # [b, np, sq, sk]
            output_size = (query_layer.size(1), query_layer.size(2), query_layer.size(0), key_layer.size(0))

            # [sq, b, np, hn] -> [sq, b * np, hn]
            query_layer = query_layer.contiguous().view(output_size[2], output_size[0] * output_size[1], -1)
            # [sk, b, np, hn] -> [sk, b * np, hn]
            key_layer = key_layer.contiguous().view(output_size[3], output_size[0] * output_size[1], -1)

            # preallocting input tensor: [b * np, sq, sk]
            matmul_input_buffer = torch.empty(
                output_size[0] * output_size[1], output_size[2], output_size[3], dtype=query_layer.dtype,
                device=query_layer.device
            )

            # Raw attention scores. [b * np, sq, sk]
            matmul_result = torch.baddbmm(
                matmul_input_buffer,
                query_layer.transpose(0, 1),  # [b * np, sq, hn]
                key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
                beta=0.0,
                alpha=(1.0 / self.norm_factor),
            )

            # change view to [b, np, sq, sk]
            attention_scores = matmul_result.contiguous().view(*output_size)

            # ===========================
            # Attention probs and dropout
            # ===========================

            # attention scores and attention mask [b, np, sq, sk]
            if self.attention_softmax_in_fp32:
                attention_scores = attention_scores.float()
            if self.coeff is not None:
                attention_scores = attention_scores * self.coeff
            if attention_mask is None and attention_scores.shape[2] == attention_scores.shape[3]:
                attention_mask = torch.ones(output_size[0], 1, output_size[2], output_size[3],
                                            device=attention_scores.device, dtype=torch.bool)
                attention_mask.tril_()
                attention_mask = ~attention_mask
            if attention_mask is not None:
                attention_scores = attention_scores.masked_fill(attention_mask, float("-inf"))
            attention_probs = F.softmax(attention_scores, dim=-1)
            attention_probs = attention_probs.type_as(value_layer)

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            attention_probs = self.attention_dropout(attention_probs)
            # =========================
            # Context layer. [sq, b, hp]
            # =========================

            # value_layer -> context layer.
            # [sk, b, np, hn] --> [b, np, sq, hn]

            # context layer shape: [b, np, sq, hn]
            output_size = (value_layer.size(1), value_layer.size(2), query_layer.size(0), value_layer.size(3))
            # change view [sk, b * np, hn]
            value_layer = value_layer.contiguous().view(value_layer.size(0), output_size[0] * output_size[1], -1)
            # change view [b * np, sq, sk]
            attention_probs = attention_probs.contiguous().view(output_size[0] * output_size[1], output_size[2], -1)
            # matmul: [b * np, sq, hn]
            context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))
            # change view [b, np, sq, hn]
            context_layer = context_layer.contiguous().view(*output_size)
            # [b, np, sq, hn] --> [sq, b, np, hn]
            context_layer = context_layer.permute(2, 0, 1, 3).contiguous()
            # [sq, b, np, hn] --> [sq, b, hp]
            new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
            context_layer = context_layer.contiguous().view(*new_context_layer_shape)

        return context_layer


class SelfAttention(nn.Module):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(self, config, layer_number, device=None):
        super(SelfAttention, self).__init__()
        self.layer_number = max(1, layer_number)

        self.projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_attention_head = self.projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads

        self.multi_query_attention = config.multi_query_attention
        self.qkv_hidden_size = 3 * self.projection_size
        if self.multi_query_attention:
            self.num_multi_query_groups_per_partition = config.multi_query_group_num
            self.qkv_hidden_size = (
                    self.projection_size + 2 * self.hidden_size_per_attention_head * config.multi_query_group_num
            )
        self.query_key_value = nn.Linear(config.hidden_size, self.qkv_hidden_size,
                                         bias=config.add_bias_linear or config.add_qkv_bias,
                                         device=device, **_config_to_kwargs(config)
                                         )

        self.core_attention = CoreAttention(config, self.layer_number)

        # Output.
        self.dense = nn.Linear(self.projection_size, config.hidden_size, bias=config.add_bias_linear,
                               device=device, **_config_to_kwargs(config)
                               )

    def _allocate_memory(self, inference_max_sequence_len, batch_size, device=None, dtype=None):
        if self.multi_query_attention:
            num_attention_heads = self.num_multi_query_groups_per_partition
        else:
            num_attention_heads = self.num_attention_heads_per_partition
        return torch.empty(
            inference_max_sequence_len,
            batch_size,
            num_attention_heads,
            self.hidden_size_per_attention_head,
            dtype=dtype,
            device=device,
        )

    def forward(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True
    ):
        # hidden_states: [sq, b, h]

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        # =====================
        # Query, Key, and Value
        # =====================

        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        mixed_x_layer = self.query_key_value(hidden_states)

        if self.multi_query_attention:
            (query_layer, key_layer, value_layer) = mixed_x_layer.split(
                [
                    self.num_attention_heads_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                ],
                dim=-1,
            )
            query_layer = query_layer.contiguous().view(
                query_layer.size()[:-1] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
            )
            key_layer = key_layer.contiguous().view(
                key_layer.size()[:-1] + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
            )
            value_layer = value_layer.contiguous().view(
                value_layer.size()[:-1]
                + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
            )
        else:
            new_tensor_shape = mixed_x_layer.size()[:-1] + \
                               (self.num_attention_heads_per_partition,
                                3 * self.hidden_size_per_attention_head)
            mixed_x_layer = mixed_x_layer.contiguous().view(*new_tensor_shape)

            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

        # apply relative positional encoding (rotary embedding)
        if rotary_pos_emb is not None:
            query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb)
            key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb)

        # adjust key and value for inference
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            key_layer = torch.cat((cache_k, key_layer), dim=0)
            value_layer = torch.cat((cache_v, value_layer), dim=0)
        if use_cache:
            kv_cache = (key_layer, value_layer)
        else:
            kv_cache = None

        if self.multi_query_attention:
            key_layer = key_layer.unsqueeze(-2)
            key_layer = key_layer.expand(
                -1, -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1
            )
            key_layer = key_layer.contiguous().view(
                key_layer.size()[:2] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
            )
            value_layer = value_layer.unsqueeze(-2)
            value_layer = value_layer.expand(
                -1, -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1
            )
            value_layer = value_layer.contiguous().view(
                value_layer.size()[:2] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
            )

        # ==================================
        # core attention computation
        # ==================================

        context_layer = self.core_attention(query_layer, key_layer, value_layer, attention_mask)

        # =================
        # Output. [sq, b, h]
        # =================

        output = self.dense(context_layer)

        return output, kv_cache


class MLP(nn.Module):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, config, device=None):
        super(MLP, self).__init__()

        self.add_bias = config.add_bias_linear

        # Project to 4h. If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        self.dense_h_to_4h = nn.Linear(
            config.hidden_size,
            config.ffn_hidden_size * 2,
            bias=self.add_bias,
            device=device,
            **_config_to_kwargs(config)
        )

        def swiglu(x):
            x = torch.chunk(x, 2, dim=-1)
            return F.silu(x[0]) * x[1]

        self.activation_func = swiglu

        # Project back to h.
        self.dense_4h_to_h = nn.Linear(
            config.ffn_hidden_size,
            config.hidden_size,
            bias=self.add_bias,
            device=device,
            **_config_to_kwargs(config)
        )

    def forward(self, hidden_states):
        # [s, b, 4hp]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        # [s, b, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        return output


class GLMBlock(nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(self, config, layer_number, device=None):
        super(GLMBlock, self).__init__()
        self.layer_number = layer_number

        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm

        self.fp32_residual_connection = config.fp32_residual_connection

        LayerNormFunc = RMSNorm
        # Layernorm on the input data.
        self.input_layernorm = LayerNormFunc(config.hidden_size, eps=config.layernorm_epsilon, device=device,
                                             dtype=config.torch_dtype)

        # Self attention.
        self.self_attention = SelfAttention(config, layer_number, device=device)
        self.hidden_dropout = config.hidden_dropout

        # Layernorm on the attention output
        self.post_attention_layernorm = LayerNormFunc(config.hidden_size, eps=config.layernorm_epsilon, device=device,
                                                      dtype=config.torch_dtype)

        # MLP
        self.mlp = MLP(config, device=device)

    def forward(
            self, hidden_states, attention_mask=None, rotary_pos_emb=None, kv_cache=None, use_cache=True,
    ):
        # hidden_states: [s, b, h]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output, kv_cache = self.self_attention(
            layernorm_output,
            attention_mask,
            rotary_pos_emb,
            kv_cache=kv_cache,
            use_cache=use_cache
        )

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        layernorm_input = torch.nn.functional.dropout(attention_output, p=self.hidden_dropout, training=self.training)
        layernorm_input = residual + layernorm_input

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output = self.mlp(layernorm_output)

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        output = torch.nn.functional.dropout(mlp_output, p=self.hidden_dropout, training=self.training)
        output = residual + output

        return output, kv_cache


class GLMTransformer(torch.nn.Module):
    """Transformer class."""

    def __init__(self, config, num_layers=1, device=None):
        super(GLMTransformer, self).__init__()

        self.fp32_residual_connection = config.fp32_residual_connection
        self.post_layer_norm = config.post_layer_norm

        # Number of layers.
        self.num_layers = num_layers

        # Transformer layers.
        def build_layer(layer_number):
            return GLMBlock(config, layer_number, device=device)

        self.layers = torch.nn.ModuleList([build_layer(i + 1) for i in range(self.num_layers)])

        if self.post_layer_norm:
            LayerNormFunc = RMSNorm
            # Final layer norm before output.
            self.final_layernorm = LayerNormFunc(config.hidden_size, eps=config.layernorm_epsilon, device=device,
                                                 dtype=config.torch_dtype)

        self.gradient_checkpointing = False

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def forward(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_caches=None,
            use_cache: Optional[bool] = True,
            output_hidden_states: Optional[bool] = False,
    ):
        if not kv_caches:
            kv_caches = [None for _ in range(self.num_layers)]
        presents = () if use_cache else None
        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        all_self_attentions = None
        all_hidden_states = () if output_hidden_states else None
        for index in range(self.num_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer = self._get_layer(index)
            if self.gradient_checkpointing and self.training:
                layer_ret = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    kv_caches[index],
                    use_cache
                )
            else:
                layer_ret = layer(
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    kv_cache=kv_caches[index],
                    use_cache=use_cache
                )
            hidden_states, kv_cache = layer_ret
            if use_cache:
                presents = presents + (kv_cache,)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # Final layer norm.
        if self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states, presents, all_hidden_states, all_self_attentions


class ChatGLMEmbeddingModel(nn.Module):

    def __init__(self, llm_model_name_or_path: str, adapter_path=None, num_layers=1, with_embedding_layer=True,
                 device='cuda', **kwargs):
        super().__init__()

        self.device = torch.device(device)
        self.chatglm, self.tokenizer = self._load_model(llm_model_name_or_path, device)
        self.set_requires_grad_to_false()
        self.embedding_layer = None
        if with_embedding_layer:
            self.embedding_layer = GLMTransformer(self.chatglm.config, num_layers, device=self.device)
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
            hidden_states, _, _, _ = self.embedding_layer(hidden_states)
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
