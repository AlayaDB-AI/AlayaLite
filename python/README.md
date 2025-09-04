# AlayaLite Python Interfaces

This page introduces the Python interfaces in AlayaLite, which include two layers:

**Collection**: Manages raw data and its relationship with indexes. Users can build their knowledge base by integrating our *Collection* into other tools like [Open WebUI](https://github.com/open-webui/open-webui) and [LangChain](https://github.com/langchain-ai/langchain), or building their own RAG application following [our tutorial](https://github.com/AlayaDB-AI/AlayaLite/blob/main/examples/rag/README.md).

**Index**: Handles search, insert, and delete operations for indexes. Researchers and developers can modify it to enhance algorithms or mechanisms. We support ANN-Benchmark evaluation in [here](https://github.com/AlayaDB-AI/AlayaLite/blob/main/python/adapters/annbenchmark/alayalite).

<p align="center">
    <img src="https://github.com/AlayaDB-AI/AlayaLite/blob/main/.assets/user-portrait.svg?raw=true" width=800 alt="AlayaDB Logo"></a>
</p>

In particular, collections and indexes are managed by **Client**.
Next, we will introduce **Client**, **Collection** and **Index** in detail.

## Client
The Client provides a convenient interface for creating, retrieving, saving, and deleting collections and indices from disk.

### Initialize a client
You can initialize a client in following ways:
```python
# Initialize the client without a URL, you can process data in memory
client = Client()

# Initialize the client with a URL to load data from disk
client = Client(url='/path/to/your/directory')
```
* `list_collections()`: Returns a list of all collection names currently managed by the client.
* `list_indices()`: Returns a list of all index names currently managed by the client.

### Get a collection or index
Get a collection object or an index object
```python
# Get a collection by name
collection = client.get_collection(name='your_collection_name')

# Get an index by name
index = client.get_index(name='your_index_name')
```
* `get_collection(name: str)`: Retrieves a collection by name.
* `get_index(name: str)`: Retrieves an index by name.

### Create a collection or index
Creating a collection or an index:
```python
 Create a new collection
new_collection = client.create_collection(name='new_collection_name')

# Create a new index
new_index = client.create_index(name='new_index_name', index_type='your_index_type', data_type='your_data_type')
```

* `create_collection(name: str, **kwargs)`: Creates a new collection with the given name and parameters. Raises a RuntimeError if a collection or index with the same name already exists.
* `create_index(name: str, **kwargs)`: Creates a new index with the given name and parameters. Raises a RuntimeError if a collection or index with the same name already exists.

### Get or create a collection or index
We also provide interfaces for "getting or creating" interface:
```python
# Get or create a collection
collection = client.get_or_create_collection(name='your_collection_name')

# Get or create an index
index = client.get_or_create_index(name='your_index_name')
```

### Delete a collection
In addition, you can deleting a collection or an index by following ways:
```python
# Delete a collection
client.delete_collection(collection_name='your_collection_name', delete_on_disk=True)

# Delete an index
client.delete_index(index_name='your_index_name', delete_on_disk=True)
```

* `delete_collection(collection_name: str, delete_on_disk: bool)`: Deletes a collection by name you maintained in memory. Optionally, it can also delete the collection from disk. Raises a RuntimeError if the collection does not exist.
* `delete_index(index_name: str, delete_on_disk: bool)`: Deletes an index by name you maintained in memory. Optionally, it can also delete the index from disk. Raises a RuntimeError if the index does not exist.

### Reset the client
Or you may want to delete everything
```python
client.reset()
```
* The `reset()` method clears all collections and indices from the client, removing all loaded data from memory. Then you can use the name to create a new one.

### Save the collection or index
When you want to save all collections and indices for persistence, you can use the following interface:
```python
# Save a collection
client.save_collection(collection_name='your_collection_name')

# Save an index
client.save_index(index_name='your_index_name')
```

* `save_collection(collection_name: str)`: Saves a collection to disk. The collection schema will be stored in a JSON file. Raises a RuntimeError if the client is not initialized with a URL or the collection does not exist.
* `save_index(index_name: str)`: Saves an index to disk. The index schema will be stored in a JSON file. Raises a RuntimeError if the client is not initialized with a URL or the index does not exist.

## Collection

The Collection class is a fundamental component for managing a set of documents along with their associated metadata and embeddings.

### Initialize a collection
You can initialize a Collection instance like this:
```python
from alayalite import Collection

collection = Collection(name='your_collection_name')
```
The `__init__` method creates an empty structure to hold the collection's data, including raw document and embeddings.

### Insert to a collection
To add new documents to the collection, you can use the insert method:
```python
items = [(1, 'document1', [1.0, 2.0, 3.0], {'category': 'A'}), (2, 'document2', [4.0, 5.0, 6.0], {'category': 'B'})]

collection.insert(items)
```
* `insert(items: List[tuple])`: Inserts a list of tuples, where each tuple contains an id, a document, an embedding, and metadata.

### Upsert to a collection
If you want to insert new items or update existing ones if ids already exist, the upsert method is available:
```python
items = [(1, 'updated_document1', [1.1, 2.1, 3.1], {'category': 'A'}), (3, 'document3', [7.0, 8.0, 9.0], {'category': 'C'})]

collection.upsert(items)
```
* `upsert(items: List[tuple])`: Checks if an item with a given id already exists. If it does, the existing item is updated. If not, a new item is inserted.

### Query a batch of vectors to a collection
For retrieving data based on vector similarity, the `batch_query` method can be used:
```python
vectors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
limit = 5
ef_search = 100
num_threads = 1

results = collection.batch_query(vectors, limit, ef_search, num_threads)
```
* `batch_query(vectors: list[list[float | int]], limit: int, ef_search: int = 100, num_threads: int = 1)`: Queries the index using a batch of vectors. `ef_search` and `num_thread` will be used forwarded as the parameters of search process.

### Filter query in a collection
If you want to filter data based on metadata, you can use the filter_query method:
```python
filter = {'category': ['A', 'B'], 'status': 1}

filtered_results = collection.filter_query(filter, 10)
```
* `filter_query(filter: dict, limit: Optional[int] = None)`: Filters the collection based on the given metadata conditions.

### Delete by id
To remove documents from the collection by their id, you can use the delete_by_id method:

```python
ids = [1, 2]

collection.delete_by_id(ids)
```
* `delete_by_id(ids: List[int])`: Removes the documents with the specified ids.

### Delete by filtering metadata
If you want to delete items based on metadata filters, the `delete_by_filter` method is available:

```python
filter = {'category': 'A'}

collection.delete_by_filter(filter)
```
* `delete_by_filter(filter: dict)`: Deletes items from the collection that match the given metadata filter.

### Save a collection
To save the collection to disk for future use, you can use the save method:
```python
url = '/path/to/save/collection'

collection.save(url)
```
* `save(url: str)`: Saves the collection's data, including the DataFrame and mapping structures, to the specified directory. It also saves the index schema in a JSON file. The collection data is pickled and saved in a collection.pkl file.

### Load a collection
To load a previously saved collection from disk, you can use the load class method:
```python
url = '/path/where/collection/is/stored'
name = 'your_collection_name'

loaded_collection = Collection.load(url, name)
```

* `load(url: str, name: str)`: Loads a collection from the specified directory.

## Index

The Index class offers a python interface for handling vectors at the lowest level.

### Initialize an index
You can initialize a new Index instance in the following way:
```python
from your_module import Index, IndexParams

# Initialize with default name and default parameters
index = Index()

# Initialize with a custom name and custom parameters
params = IndexParams()
index = Index(name='your_index_name', params=params)
```
* The `__init__` method takes an optional name (default is "default") and params (an instance of IndexParams). The index is not fully initialized at this point; it's a late - initialization process. In particular, the dimension, data type, et al. is not known until it is fitted or the first vector is inserted. In addition, you can set the parameter to set the metric type or restrict the index type, data type, et al.

### Build an index
To build the index with a set of vectors, use the fit method:
```python
import numpy as np
vectors = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
ef_construction = 100
num_threads = 1

index.fit(vectors, ef_construction, num_threads)
```
* `fit(vectors: VectorLikeBatch, ef_construction: int = 100, num_threads: int = 1)`: Constructs the index using the provided 2D array of vectors. The `ef_construction` parameter controls the accuracy of index construction, and `num_threads` specifies the number of threads to use during construction. The index can only be fitted once.

### Insert a vector into the index
To insert a new vector into the index, use the insert method:
```python
vector = np.array([7.0, 8.0, 9.0])
ef = 100

inserted_id = index.insert(vector, ef)
```
* `insert(vectors: VectorLike, ef: int = 100)`: Inserts a 1D vector into the index. The `ef` parameter controls the retrieval accuracy. It returns the assigned ID of the inserted vector.


### Remove a vector from the index
To remove a vector from the index by its ID, use the remove method:
```python
id_to_remove = 0
index.remove(id_to_remove)
```
* `remove(id: int)`: Removes the vector with the specified ID from the index.

### ANN Search of a single query
For single - query searches, use the search method:
```python
query = np.array([1.0, 2.0, 3.0])
topk = 2
ef_search = 100

neighbors = index.search(query, topk, ef_search)
```
* `search(query: VectorLike, topk: int, ef_search: int = 100)`: Performs an ANN search for the given 1D query vector. It returns the top-k nearest neighbors.

### ANN Search of a batch of queries
For batch searches, use the batch_search method:
```python
queries = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
topk = 2
ef_search = 100
num_threads = 1

batch_neighbors = index.batch_search(queries, topk, ef_search, num_threads)
```
* `batch_search(queries: VectorLikeBatch, topk: int, ef_search: int = 100, num_threads: int = 1)`: Performs a batch search for multiple query vectors. It returns the top-k nearest neighbors for each query.

### Get the vector data by ID
To get the vector data associated with a given ID, use the `get_data_by_id` method:
```python
id = 0
vector_data = index.get_data_by_id(id)
```
* `get_data_by_id(id: int)`: Retrieves the vector data corresponding to the given ID.

### Get the dimensionality of the vectors
To get the dimensionality of the vectors stored in the index, use the get_dim method:
```python
dim = index.get_dim()
```
* `get_dim()`: Returns the dimension of the indexed vectors.

### Save an index
To save the index to disk, use the save method:
```python
url = '/path/to/save/index'
index_metadata = index.save(url)
```
* `save(url: str)`: Saves the index to the specified directory. It returns a dictionary containing metadata about the saved index.

### Load an index
To load an existing index from disk, use the load class method:
```python
url = '/path/where/index/is/stored'
name = 'your_index_name'
loaded_index = Index.load(url, name)
```
* `load(url: str, name: str)`: Loads an index from the specified directory. It returns an instance of the Index class with the loaded index data.

<!--

## Installation

### Local Compilation and Installation


To compile and install the Alaya Lite PY package locally, use the provided script as follows:
```bash
conda env create --file=environment.yaml
conda activate alayalite
bash build_support/pyinstall.sh
```

> The script performs the following steps:
>
> ```Bash
> # Locally compile the python package
> python -m build
>
> # Install the compiled package
> pip install python/AlayaDBLite/AlayaDBLite-*.whl --force-reinstall
> ```


### 3. Training the Dataset

```Python
max_nbrs       = 32
fit_thread_num = 96

client.fit(base_data, max_nbrs, fit_thread_num)
```

> ```Python
> def fit(
>      self,
>      vectors: VectorLikeBatch,
>      M: int,
>      num_threads: int = _defaults.NUM_THREADS,
>      R: int = _defaults.R,
>      L: int = _defaults.L,
>      index_path: str = _defaults.INDEX_PATH,
>      index_prefix: str = _defaults.INDEX_PREFIX,
>  ) -> None
>  """
>  fit the index with the given vectors
>  :param vectors: 2D array of vectors to fit the index
>  :param M: maximum number of neighbors
>  :param num_threads: number of threads to use
>  :param R: the maximum out-degree of the underlay graph, and half of R is the maximum out-degree of the overlay graph.
>  :param L: The size of the dynamic candidate list for the construction phase.
>  :param index_path: path to store the index
>  :param index_prefix: prefix of the index file
>
>  :return: None
>  """
> ``` -->
<!--
### 4. Searching

Perform batch and single searches as follows:

```Python
topk          = 10
ef            = 100
search_thread = 32

# batch search
results = client.batch_search(query_data, topk, ef, search_thread)

# single search
result  = client.search(query_data[0], topk, ef)
```

> ```Python
> def batch_search(
>      self,
>      queries: VectorLikeBatch,
>      topk: int,
>      ef_search: int = _defaults.EF_SEARCH,
>      num_threads: int = _defaults.NUM_THREADS,
>  ) -> VectorLikeBatch:
>  """
>  search for the topk nearest neighbors for a batch of queries
>  :param queries: 2D array of queries
>  :param topk: number of nearest neighbors to search for
>  :param ef_search: the number of candidate to evaluate during the search
>  :param num_threads: number of threads to use
>
>  :return: 2D array of nearest neighbors for each query
>  """
>
> def search(
>      self,
>      query: VectorLike,
>      topk: int,
>      ef_search: int = _defaults.EF_SEARCH
>  ) -> VectorLike:
>  """
>  search for the topk nearest neighbors for a single query
>  :param query: 1D array of query
>  :param topk: number of nearest neighbors to search for
>  :param ef_search: the number of candidate to evaluate during the search
>
>  :return: 1D array of nearest neighbors for the query
>  """
> ```

## Conducting ANN-Benchmark Tests

### Setup for Benchmarking

Clone the ANN-Benchmarks repository and set up the environment:

```bash
git clone git@github.com:erikbern/ann-benchmarks.git
# Ensure the `python/AlayaDBLite` is copied to the `ann_benchmarks/algorithms` directory
```
-->





<!--
Install necessary dependencies:

```Bash
pip install -r requirements.txt
```

Initialize the AlayaDBLite Docker environment:

```
python install.py --algorithm AlayaDBLite
```

### Running Benchmarks

Execute benchmarks for specified datasets:

```
python run.py --algorithm AlayaDBLite --dataset DATASET_NAME
# Replace DATASET_NAME with e.g., gist-960-euclidean
```

### Plotting the Results

Adjust permissions and generate result plots:

```
sudo chmod 777 -R results && python plot.py --dataset DATASET_NAME
```  -->
