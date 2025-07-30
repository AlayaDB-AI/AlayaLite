"""
This module provides utility functions for text processing and embedding,
including text splitting and interfacing with BAAI embedding models.
"""

from typing import List
import traceback

from FlagEmbedding import FlagAutoModel
from langchain_text_splitters import RecursiveCharacterTextSplitter


class BAAIEmbedder:
    """An embedder class that uses the BAAI FlagEmbedding models."""

    def __init__(self, path: str = "BAAI/bge-m3") -> None:
        super().__init__()
        self.path = path
        self.model = FlagAutoModel.from_finetuned(
            path,
            query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
            use_fp16=False,
        )

    def encode(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts)


def splitter(text: str, chunksize: int = 256, overlap: int = 25) -> List[str]:
    """Splits a long text into smaller chunks with overlap."""
    try:
        separators = ["\n\n", "\n", ".", "!", "！", "。"]
        text_splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=chunksize,
            chunk_overlap=overlap,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = text_splitter.split_text(text)
        for i in range(len(chunks)):
            for sep in separators:
                if chunks[i].startswith(sep):
                    chunks[i] = chunks[i][1:].strip()
                if chunks[i].endswith(sep):
                    chunks[i] = chunks[i][:-1].strip()
        return chunks
    # pylint: disable=broad-exception-caught
    except Exception as e:
        print(f"Error during splitting: {e}")
        traceback.print_exc()
        return [text]


def embedder(texts: List[str], path: str) -> List[List[float]]:
    """
    Call the embedder according to model_name.

    Parameters:
    texts (List[str]): A list of texts to enter.
    path (str): The model path

    Returns:
    List[List[float]]: A list of embedding vectors.
    """
    try:
        model_name = path.split("/")[-1].strip()
        if model_name.startswith("bge"):
            if path:
                embedder_instance = BAAIEmbedder(path=path)
            else:
                embedder_instance = BAAIEmbedder()
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        print(repr(texts))
        return embedder_instance.encode(texts)
    # pylint: disable=broad-exception-caught
    except Exception as e:
        print(f"Error during embedding: {e}")
        traceback.print_exc()
        return []
