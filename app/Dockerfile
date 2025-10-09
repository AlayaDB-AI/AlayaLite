FROM python:3.12-slim-bookworm

WORKDIR /
COPY . /app

RUN pip install --upgrade pip --index-url https://mirrors.sustech.edu.cn/pypi/web/simple &&\
    pip config set global.index-url https://mirrors.sustech.edu.cn/pypi/web/simple &&\
    pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

RUN python -c "import urllib.request; \
        url = 'https://github.com/AlayaDB-AI/AlayaLite/releases/download/v0.1.0a2/alayalite-0.1.0a2-cp312-cp312-linux_x86_64.whl'; \
        filename = url.split('/')[-1]; \
        urllib.request.urlretrieve(url, filename)" && \
    pip install alayalite-0.1.0a2-cp312-cp312-linux_x86_64.whl

EXPOSE 8000

# Default storage directory inside the container. Users can override by passing
# -e ALAYALITE_DATA_DIR=/path/on/container and bind-mounting a host dir.
ENV ALAYALITE_DATA_DIR=/data

# Make sure the directory exists and is writable
RUN mkdir -p /data && chown -R root:root /data

# Expose /data as a volume so it can be bind-mounted by the user
VOLUME ["/data"]

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
