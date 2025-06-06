FROM ann-benchmarks

RUN apt install -y python3.11 && \
      update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2 && \
      update-alternatives --config python3 && \
      python3 -m pip install --upgrade pip && \
      pip3 install -r requirements.txt

COPY ./ann_benchmarks/algorithms/alayalite/libstdc++.so.6 /usr/lib/x86_64-linux-gnu/
COPY ./ann_benchmarks/algorithms/alayalite/alayalite-0.1.0-cp311-cp311-linux_x86_64.whl /home/app/
RUN pip3 install /home/app/alayalite-0.1.0-cp311-cp311-linux_x86_64.whl
RUN python3 -c 'import alayalite'