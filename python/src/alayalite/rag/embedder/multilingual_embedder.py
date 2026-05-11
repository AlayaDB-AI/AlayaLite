# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
This module provides the MultilingualEmbedder class for creating embeddings
using multilingual models from the transformers library.
"""

from typing import List, Tuple

import torch
from transformers import AutoModel, AutoTokenizer

from .base import BaseEmbedding


class MultilingualEmbedder(BaseEmbedding):
    """An embedding class that uses multilingual models from Hugging Face."""

    def __init__(self, path: str = "intfloat/multilingual-e5-large") -> None:
        """
        Initializes the MultilingualEmbedder.

        Args:
            path (str): The model path or name for the multilingual transformer model.
        """
        super().__init__(path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        self.model = AutoModel.from_pretrained(self.path)

    def get_embeddings(self, texts: List[str]) -> Tuple[List[List[float]], int]:
        """
        Generates embeddings for a list of texts using a multilingual model.

        Args:
            texts (List[str]): A list of texts to embed.

        Returns:
            Tuple[List[List[float]], int]: A tuple containing the list of embeddings and the embedding dimension.
        """
        batch_dict = self.tokenizer(texts, max_length=512, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**batch_dict)

        attention_mask = batch_dict["attention_mask"]
        last_hidden = outputs.last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        embeddings = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

        if embeddings.numel() == 0:
            # Attempt to get dimension from model config if no texts are provided
            dim = self.model.config.hidden_size
            return [], dim

        dim = embeddings.shape[1]
        return embeddings.tolist(), dim
