"""
This module provides the Streamlit user interface for a Retrieval-Augmented Generation (RAG) Q&A application.
It handles file uploads, knowledge base management, and the chat interface.
"""

import streamlit as st
import json
from datetime import datetime
from docx import Document
from typing import Callable, Generator, Tuple
from pypdf import PdfReader

from db import reset_db, insert_text, query_text
from llm import ask_llm

# fix error print:

# but it does not exist! Ensure that it is registered via torch::class_
import torch
import os

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]
USE_STREAM = True
st.set_page_config(page_title="My RAG", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

# Init session
if "user" not in st.session_state:
    st.session_state.update({"user": None, "chat_history": [], "current_db": "default"})


def read_file(uploaded_file):
    content = ""

    file_type = uploaded_file.type

    if uploaded_file.name.endswith(".md"):
        file_type = "text/markdown"

    try:
        if file_type in ["text/plain", "text/x-markdown", "text/markdown"]:
            content = uploaded_file.read().decode("utf-8")
        elif file_type == "application/pdf":
            pdf_reader = PdfReader(uploaded_file)
            content = "\n".join([page.extract_text() for page in pdf_reader.pages])
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(uploaded_file)
            content = "\n".join([para.text for para in doc.paragraphs])
        else:
            raise ValueError(f"Unsupported file types: {uploaded_file.type} for file {uploaded_file.name}")
    except Exception as e:
        raise RuntimeError(f"Fail to read files: {uploaded_file.name}. Type: {uploaded_file.type}.") from e
    return content


def enhanced_file_processor(file):
    content = read_file(file)
    content = content.replace("\n", " ")
    content = " ".join(content.split())
    return content


def context_aware_query(
    collection_name: str,
    query: str,
    history: list,
    *,
    llm_url: str,
    llm_api_key: str,
    llm_model: str,
    embed_model_path: str,
    is_stream: bool = True,
) -> Tuple[str | Callable[[], Generator[str, None, None]], str]:
    context = "\n".join([f"Q: {q}\nA: {a}" for q, a, d in history[-3:]])
    enhanced_query = f"{context}\n\nNew question：{query}"

    retrieved_docs = query_text(
        collection_name=collection_name,
        query=enhanced_query,
        embed_model_path=embed_model_path,
    )
    return ask_llm(
        llm_url, llm_api_key, llm_model, query=query, retrieved_docs=retrieved_docs, is_stream=is_stream
    ), retrieved_docs


def main_interface():
    with st.sidebar:
        with st.expander("🛠️ Service Setting", expanded=True):
            llm_url = st.text_input("LLM Base URL", value="https://api.lkeap.cloud.tencent.com/v1/chat")
            llm_api_key = st.text_input("LLM API Key", value="Your API Key")
            llm_model = st.text_input("LLM Model", value="deepseek-v3")
            embed_model_path = st.text_input("Embedding Model", value="BAAI/bge-small-zh-v1.5")

        with st.expander("📚 Knowledge Base Management", expanded=True):
            with st.form("upload_form"):
                collection_name = "rag_collection"
                uploaded_file = st.file_uploader(
                    "Upload documents", type=["txt", "pdf", "docx", "md"], accept_multiple_files=True
                )
                if st.form_submit_button("🚀 Start processing"):
                    reset_db()
                    if uploaded_file:
                        success = True
                        for file in uploaded_file:
                            with st.spinner(f"Process {file.name}..."):
                                content = enhanced_file_processor(file)
                                success = insert_text(
                                    collection_name=collection_name,
                                    docs=content,
                                    embed_model_path=embed_model_path,
                                    chunksize=256,
                                    overlap=25,
                                )
                                if not success:
                                    break
                        if success:
                            st.success("Document processing completed")
                        else:
                            st.error("Document processing failed!")
                    else:
                        st.error("No document uploaded yet!")

        with st.expander("🗃️ Dialogue management", expanded=True):
            if st.button("🔄 Clear all records", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

            timestamp = datetime.now().strftime("%Y%m%d")
            if st.download_button(
                "💾 Export all records",
                data=json.dumps(st.session_state.chat_history),
                file_name=f"chat_{timestamp}.json",
                mime="application/json",
                use_container_width=True,
            ):
                st.toast("Exported!")

    with st.container():
        st.header("💬 RAG QA")

        # Real-time dialog display container
        chat_container = st.container(height=600, border=False)
        # Render the existing history first
        with chat_container:
            for _, (q, a, d) in enumerate(st.session_state.chat_history):
                with st.chat_message("user", avatar="👤"):
                    st.markdown(f"{q}")
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(a)
                    if d:
                        with st.popover("References"):
                            st.markdown(d)

        # Input processing
        if prompt := st.chat_input("Please enter your question..."):
            # Display user input
            with chat_container:
                with st.chat_message("user", avatar="👤"):
                    st.markdown(f"{prompt}")

                # Add a placeholder response
                with st.chat_message("assistant", avatar="🤖"):
                    answer_placeholder = st.empty()
                    answer_placeholder.markdown("▌")

            # Execute the query (keep the original logic)
            try:
                resp_or_stream, retrieved_docs = context_aware_query(
                    collection_name=collection_name,
                    query=prompt,
                    history=st.session_state.chat_history,
                    llm_url=llm_url,
                    llm_api_key=llm_api_key,
                    llm_model=llm_model,
                    embed_model_path=embed_model_path,
                    is_stream=USE_STREAM,
                )

                # Update the answer
                full_resp = ""
                if USE_STREAM:
                    for chunk in resp_or_stream():
                        full_resp += chunk
                        answer_placeholder.markdown(full_resp)
                else:
                    full_resp = resp_or_stream
                    answer_placeholder.markdown(full_resp)

                # Update session history
                st.session_state.chat_history.append((prompt, full_resp, retrieved_docs))
            # pylint: disable=broad-exception-caught
            except Exception as e:
                answer_placeholder.error(f"Handling failures: {str(e)}")
                st.session_state.chat_history.append((prompt, f"⚠️ Error: {str(e)}", ""))

            st.rerun()


def main():
    main_interface()


if __name__ == "__main__":
    main()
