FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    curl \
    binutils \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# install python3 and pip
RUN echo "tzdata tzdata/Areas select Asia" | debconf-set-selections && \
    echo "tzdata tzdata/Zones/Asia select Shanghai" | debconf-set-selections && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y python3.11 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 2 && \
    update-alternatives --config python && \
    apt install -y python3-pip

# install required packages and alayalite
RUN python3.11 -m pip install python-docx numpy streamlit langchain_text_splitters FlagEmbedding pypdf
RUN python3.11 -m pip install alayalite

# set the mirror of embedding models
RUN echo "export HF_ENDPOINT=https://hf-mirror.com" >> ~/.bashrc

WORKDIR /app
CMD ["/bin/bash"]
