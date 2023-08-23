# *_*coding:utf-8 *_*
# @Author : YueMengRui
import torch
import numpy as np
from typing import List, Dict, Union, Optional
from .chatglm_embedding_model import ChatGLMEmbeddingModel


class EmbeddingModel:
    def __init__(self, max_seq_len: int = 512, device: str = "cuda", **kwargs):
        self.max_seq_len = max_seq_len
        self.device = torch.device(device)
        self.embedding_model = ChatGLMEmbeddingModel(**kwargs)
        self.tokenizer = self.embedding_model.get_tokenizer()

    def get_embeddings(self, inputs):
        attention_mask = inputs["attention_mask"]
        model_output = self.embedding_model(inputs)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.size()).float()
        final_encoding = torch.sum(model_output * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9)

        return final_encoding

    def encode(
            self,
            sentences: Union[str, List[str]],
            batch_size: int = 64,
            convert_to_numpy: bool = True,
            convert_to_tensor: bool = False,
            **kwargs
    ):
        """
        Returns the embeddings for a batch of sentences.

        :param sentences: str/list, Input sentences
        :param batch_size: int, Batch size
        :param show_progress_bar: bool, Whether to show a progress bar for the sentences
        :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any setting from convert_to_numpy
        :param device: Which device to use for the computation
        """

        if convert_to_tensor:
            convert_to_numpy = False
        input_is_string = False
        if isinstance(sentences, str) or not hasattr(sentences, "__len__"):
            sentences = [sentences]
            input_is_string = True

        all_embeddings = []
        length_sorted_idx = np.argsort([-len(s) for s in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
        for start_index in range(0, len(sentences), batch_size):
            sentences_batch = sentences_sorted[start_index: start_index + batch_size]
            # Compute sentences embeddings
            with torch.no_grad():
                embeddings = self.get_embeddings(
                    self.tokenizer(sentences_batch, max_length=self.max_seq_len,
                                   padding=True, truncation=True, return_tensors='pt').to(self.device)
                )
            embeddings = embeddings.detach()
            if convert_to_numpy:
                embeddings = embeddings.cpu()
            all_embeddings.extend(embeddings)
        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        if input_is_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings
