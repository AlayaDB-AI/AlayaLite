FROM python:3.12-slim-bookworm

WORKDIR /
COPY . /app

RUN pip install --upgrade pip --index-url https://mirrors.sustech.edu.cn/pypi/web/simple &&\
    pip config set global.index-url https://mirrors.sustech.edu.cn/pypi/web/simple &&\
    pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

RUN python -c "import urllib.request; \
        url = 'https://github.com/AlayaDB-AI/AlayaLite/releases/download/v0.1.0a1/alayalite-0.1.0a1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl'; \
        filename = url.split('/')[-1]; \
        urllib.request.urlretrieve(url, filename)" && \
    pip install alayalite-0.1.0a1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl



EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]