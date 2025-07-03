# AlayaLite 向量数据库 API 使用文档

## 基本信息

- **基础路径**：`/api/v1/collection/`
- **接口风格**：RESTful，全部为 POST 请求
- **数据格式**：请求和响应均为 JSON

---

## 1. 创建集合

- **接口**：`/api/v1/collection/create`
- **方法**：POST
- **请求参数**：

```json
{
  "collection_name": "test"
}
```

- **返回示例**：

```json
"Collection test created successfully"
```

---

## 2. 列出所有集合

- **接口**：`/api/v1/collection/list`
- **方法**：POST
- **请求参数**：无

- **返回示例**：

```json
["test", "my_collection"]
```

---

## 3. 删除集合

- **接口**：`/api/v1/collection/delete`
- **方法**：POST
- **请求参数**：

```json
{
  "collection_name": "test"
}
```

- **返回示例**：

```json
"Collection test deleted successfully"
```

---

## 4. 重置所有集合

- **接口**：`/api/v1/collection/reset`
- **方法**：POST
- **请求参数**：无

- **返回示例**：

```json
"Collection reset successfully"
```

---

## 5. 插入数据

- **接口**：`/api/v1/collection/insert`
- **方法**：POST
- **请求参数**：

```json
{
  "collection_name": "test",
  "items": [
    [1, "Document 1", [0.1, 0.2, 0.3], {"category": "A"}],
    [2, "Document 2", [0.4, 0.5, 0.6], {"category": "B"}]
  ]
}
```

- **返回示例**：

```json
"Successfully inserted 2 items into collection test"
```

---

## 6. 查询向量

- **接口**：`/api/v1/collection/query`
- **方法**：POST
- **请求参数**：

```json
{
  "collection_name": "test",
  "query_vector": [[0.1, 0.2, 0.3]],
  "limit": 2,
  "ef_search": 10,
  "num_threads": 1
}
```

- **返回示例**：

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

## 7. Upsert（插入或更新）

- **接口**：`/api/v1/collection/upsert`
- **方法**：POST
- **请求参数**：

```json
{
  "collection_name": "test",
  "items": [
    [1, "New Document 1", [0.1, 0.2, 0.3], {"category": "A"}]
  ]
}
```

- **返回示例**：

```json
"Successfully upserted 1 items into collection test"
```

---

## 8. 按ID删除

- **接口**：`/api/v1/collection/delete_by_id`
- **方法**：POST
- **请求参数**：

```json
{
  "collection_name": "test",
  "ids": [1, 2]
}
```

- **返回示例**：

```json
"Successfully deleted 2 items from collection test"
```

---

## 9. 按条件删除

- **接口**：`/api/v1/collection/delete_by_filter`
- **方法**：POST
- **请求参数**：

```json
{
  "collection_name": "test",
  "filter": {"category": "A"}
}
```

- **返回示例**：

```json
"Successfully deleted 1 items from collection test"
```

---

## 备注

- 所有接口均为 POST 方法，参数通过 JSON 传递。
- 向量（vector）请用一维数组（如 `[0.1, 0.2, 0.3]`）表示。
- `items` 字段为列表，每个元素为 `[id, document, vector, metadata]` 结构。
- 查询接口返回的 `score` 字段为相似度分数，数值越大越相似。

---

如需进一步帮助或有特殊需求，请补充说明！ 