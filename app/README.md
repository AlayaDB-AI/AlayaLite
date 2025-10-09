# AlayaLite - Standalone App

## Development Setup

```bash
# It is recommended to create a virtual environment first
# python -m venv .venv && source .venv/bin/activate

# Install dependencies from pyproject.toml
pip install -e '.[api]'
```

## Running Tests

```bash
pytest
```

## Running the Application

```bash
# For development with hot reload
uvicorn app.main:app --reload

# For production
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Running on Docker

### Build the Image

```bash
docker build -t alayalite-standalone .
```

### Run the Container

```bash
docker run -d --name my-alayalite-standalone -p 8000:8000 alayalite-standalone
```

### Run with Persistent Storage

To ensure your data is not lost when the container stops, mount a host directory to the container's data directory.

```bash
# Create a directory on your host machine
mkdir -p /path/to/your/data

# Run the container with a volume mount
docker run -d --name my-alayalite-standalone -p 8000:8000 \
  -v /path/to/your/data:/data \
  -e ALAYALITE_DATA_DIR=/data \
  alayalite-standalone
```

## API usage

See [API_Usage_Documentation.md](./API_Usage_Documentation.md) for full details.

### Create Collection

```bash
curl -X POST \
  http://localhost:8000/api/v1/collection/create \
  -H "Content-Type: application/json" \
  -d '{"collection_name": "test"}'

"Collection test created successfully"
```

### Insert

```bash
curl -X POST \
  http://localhost:8000/api/v1/collection/insert \
  -H "Content-Type: application/json" \
  -d '{
        "collection_name": "test",
        "items": [
          [1, "Document 1", [0.1, 0.2, 0.3], {"category": "A"}],
          [2, "Document 2", [0.4, 0.5, 0.6], {"category": "B"}]
        ]
      }'

"Successfully inserted 2 items into collection test"
```

### Query

```bash
curl -X POST \
  http://localhost:8000/api/v1/collection/query \
  -H "Content-Type: application/json" \
  -d '{
        "collection_name": "test",
        "query_vector": [[0.1, 0.2, 0.3]],
        "limit": 2,
        "ef_search": 10,
        "num_threads": 1
      }'
```
The response will be a JSON array of matching items.

### Upsert

```bash
curl -X POST \
  http://localhost:8000/api/v1/collection/upsert \
  -H "Content-Type: application/json" \
  -d '{
        "collection_name": "test",
        "items": [
          [1, "New Document 1", [0.1, 0.2, 0.3], {"category": "A"}]
        ]
      }'

"Successfully upserted 1 items into collection test"
```

### Save Collection

This will save the collection's in-memory data to the persistent storage directory.

```bash
curl -X POST \
  http://localhost:8000/api/v1/collection/save \
  -H "Content-Type: application/json" \
  -d '{"collection_name": "test"}'

"Collection test saved successfully"
```
