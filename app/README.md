# AlayaLite - Standalone App

## Development Setup

```bash
cd <path_to_your_project>/app
pip install -r requirements.txt
```

## Running Tests

```bash
cd <path_to_your_project>/app
pytest
```

## Running the Application

```bash
cd <path_to_your_project>/app
python -m uvicorn app.main:app --reload   # For development with hot reload
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000  # For production
```

## Running on Docker

```bash
cd <path_to_your_project>/app
docker build -t alayalite-standalone .
docker run -d --name my-alayalite-standalone -p 8000:8000 alayalite-standalone
```

## API usage

### create collection

```bash
curl -X POST \
  http://localhost:8000/api/v1/collection/create \
  -H "Content-Type: application/json" \
  -d '{"collection_name": "test"}'

"Collection test created successfully"
```

### insert

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

### query

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

{"id":[[1.0,2.0]],"document":[["Document 1","Document 2"]],"metadata":[[{"category":"A"},{"category":"B"}]],"distance":[[0.0,0.27000001072883606]]}
```

### upsert

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
