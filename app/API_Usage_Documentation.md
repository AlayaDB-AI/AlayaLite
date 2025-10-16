# AlayaLite 向量数据库 API 使用文档

## 基本信息
````markdown
# AlayaLite Vector Database API Usage Documentation

## Overview

- **Base path**: `/api/v1/collection/`
- **Style**: RESTful (all endpoints use POST)
- **Data format**: JSON for both requests and responses

---

## 1. Create Collection

- **Endpoint**: `/api/v1/collection/create`
- **Method**: POST
- **Request body**:

```json
{
  "collection_name": "test"
}
```

- **Example response**:

```json
"Collection test created successfully"
```

---

## 2. List Collections

- **Endpoint**: `/api/v1/collection/list`
- **Method**: POST
- **Request body**: none

- **Example response**:

```json
["test", "my_collection"]
```

---

## 3. Delete Collection

- **Endpoint**: `/api/v1/collection/delete`
- **Method**: POST
- **Request body**:

```json
{
  "collection_name": "test"
  "delete_on_disk": true
}
```

- **Example response**:

```json
"Collection test deleted successfully"
```

---

## 4. Reset Collections

- **Endpoint**: `/api/v1/collection/reset`
- **Method**: POST
- **Request body**:

```json
{
  "delete_on_disk": true
}
```

- **Example response**:

```json
"Collection reset successfully"
```

---

## 5. Insert Items

- **Endpoint**: `/api/v1/collection/insert`
- **Method**: POST
- **Request body**:

```json
{
  "collection_name": "test",
  "items": [
    [1, "Document 1", [0.1, 0.2, 0.3], {"category": "A"}],
    [2, "Document 2", [0.4, 0.5, 0.6], {"category": "B"}]
  ]
}
```

- **Example response**:

```json
"Successfully inserted 2 items into collection test"
```

---

## 6. Query Vectors

- **Endpoint**: `/api/v1/collection/query`
- **Method**: POST
- **Request body**:

```json
{
  "collection_name": "test",
  "query_vector": [[0.1, 0.2, 0.3]],
  "limit": 2,
  "ef_search": 10,
  "num_threads": 1
}
```

- **Example response**:

```json
[
  {
    "id": 1,
    "document": "Document 1",
    "vector": [0.1, 0.2, 0.3],
    "metadata": {"category": "A"},
    "score": 0.99
  }
]
```

---

## 7. Upsert (Insert or Update)

- **Endpoint**: `/api/v1/collection/upsert`
- **Method**: POST
- **Request body**:

```json
{
  "collection_name": "test",
  "items": [
    [1, "New Document 1", [0.1, 0.2, 0.3], {"category": "A"}]
  ]
}
```

- **Example response**:

```json
"Successfully upserted 1 items into collection test"
```

---

## 8. Delete by ID

- **Endpoint**: `/api/v1/collection/delete_by_id`
- **Method**: POST
- **Request body**:

```json
{
  "collection_name": "test",
  "ids": [1, 2]
}
```

- **Example response**:

```json
"Successfully deleted 2 items from collection test"
```

---

## 9. Delete by Filter

- **Endpoint**: `/api/v1/collection/delete_by_filter`
- **Method**: POST
- **Request body**:

```json
{
  "collection_name": "test",
  "filter": {"category": "A"}
}
```

- **Example response**:

```json
"Successfully deleted 1 items from collection test"
```

---

## 10. Save Collection

- **Endpoint**: `/api/v1/collection/save`
- **Method**: POST
- **Request body**:

```json
{
  "collection_name": "test"
}
```

- **Example response**:

```json
"Collection test saved successfully"
```

---

## Notes

- All endpoints use POST and accept parameters as JSON in the request body.
- Vectors should be represented as a one-dimensional array, e.g. `[0.1, 0.2, 0.3]`.
- The `items` field is a list where each element has the structure `[id, document, vector, metadata]`.
- The `score` field in query results is a similarity score — higher means more similar.

---

If you need further assistance or have special requirements, please provide more details.

````
