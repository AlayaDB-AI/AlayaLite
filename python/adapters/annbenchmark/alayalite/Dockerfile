FROM ann-benchmarks

RUN apt-get update && apt-get install -y wget
RUN wget https://github.com/AlayaDB-AI/AlayaLite/releases/download/v1.0.0/alayalite-1.0.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl -O /home/app/alayalite-1.0.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
RUN python3 -m pip install /home/app/alayalite-1.0.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
RUN python3 -c 'import alayalite'
