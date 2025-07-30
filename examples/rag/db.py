# Copyright 2025 AlayaDB.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module provides functions for a Retrieval-Augmented Generation (RAG)
example, including database reset, text insertion, and querying using AlayaLite.
"""

from alayalite import Client
import traceback
from utils import splitter, embedder


# Initialize the client globally
client = Client()


def reset_db():
    """Resets the AlayaLite database."""
    client.reset()


def insert_text(
    collection_name: str, docs: str, embed_model_path: str, chunksize: int = 256, overlap: int = 25
) -> bool:
    """Splits, embeds, and inserts text into a specified collection."""
    chunks = splitter(docs, chunksize, overlap)
    print(f"Splitting text into {len(chunks)} chunks")

    embeddings = embedder(chunks, embed_model_path)
    print(f"Embedding {len(chunks)} chunks into vectors")

    if not embeddings:
        print("Fail to embed chunks. Not to insert")
        return False

    print(f"Inserting {len(chunks)} chunks")
    try:
        collection = client.get_or_create_collection(collection_name)
        items = [(str(i), chunks[i], embeddings[i], None) for i in range(len(chunks))]
        collection.insert(items)
    # pylint: disable=broad-exception-caught
    except Exception as e:
        print(f"Error during index creation: {e}")
        traceback.print_exc()
        return False
    print("Insertion done!")

    return True  # success


def query_text(collection_name: str, embed_model_path: str, query: str, top_k=5) -> str:
    """Queries the collection and retrieves the top_k most relevant documents."""
    retrieved_docs = ""
    try:
        collection = client.get_collection(collection_name)
        if collection:
            processed_query = embedder([query], embed_model_path)
            # return type: DataFrame[id, document, distance, metadata]
            query_result = collection.batch_query(processed_query, top_k)
            retrieved_docs = "\n\n".join(query_result["document"][0])
    # pylint: disable=broad-exception-caught
    except Exception as e:
        print(f"Error during retrieval: {e}")
        traceback.print_exc()

    return retrieved_docs


if __name__ == "__main__":
    from llm import ask_llm

    # Specify UTF-8 encoding for cross-platform compatibility.
    with open("test_docs.txt", "r", encoding="utf-8") as fp:
        sample_text_main = fp.read()

    # Use distinct variable names to avoid conflicts with function parameters.
    query_main = "What are higher-order chunking techniques?"
    llm_url_main = "Your LLM service base URL here"
    llm_api_key_main = "Your API key here"  # pragma: allowlist secret
    llm_model_main = "deepseek-v3"
    embed_model_path_main = "BAAI/bge-small-zh-v1.5"

    insert_text(collection_name="test", embed_model_path=embed_model_path_main, docs=sample_text_main, chunksize=128)
    retrieved_docs_main = query_text(
        collection_name="test", embed_model_path=embed_model_path_main, query=query_main, top_k=5
    )
    final_result = ask_llm(
        llm_url_main,
        llm_api_key_main,
        llm_model_main,
        query=query_main,
        retrieved_docs=retrieved_docs_main,
        is_stream=False,
    )
    print(f"=== Response ===\n{final_result}")
