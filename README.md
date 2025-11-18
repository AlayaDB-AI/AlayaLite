<p align="center">
  <a href="https://github.com/AlayaDB-AI"><img src="https://github.com/AlayaDB-AI/AlayaLite/blob/main/.assets/banner.jpg?raw=true" width=300 alt="AlayaDB Log"></a>
</p>


<p align="center">
    <b>AlayaLite – A Fast, Flexible Vector Database for Everyone</b>. <br />
    Seamless Knowledge, Smarter Outcomes.
</p>

<div class="column" align="middle">
  <a href="https://github.com/AlayaDB-AI/AlayaLite/releases"><img height="20" src="https://img.shields.io/badge/alayalite-blue" alt="release"></a>
  <a href="https://pypi.org/project/alayalite/"><img src="https://img.shields.io/pypi/v/alayalite" alt="PyPi"></a>
  <a href="https://github.com/AlayaDB-AI/AlayaLite/blob/main/LICENSE"><img height="20" src="https://img.shields.io/badge/license-Apache--2.0-green.svg" alt="LICENSE"></a>
  <a href="https://codecov.io/github/AlayaDB-AI/AlayaLite"><img height="20" src="https://codecov.io/github/AlayaDB-AI/AlayaLite/graph/badge.svg?token=KA6V0DHHUU" alt="codecov"></a>
  <a href="https://github.com/AlayaDB-AI/AlayaLite/actions/workflows/code-checker.yaml"><img height="20" src="https://github.com/AlayaDB-AI/AlayaLite/actions/workflows/code-checker.yaml/badge.svg?branch=main" alt="CI"></a>
</div>


## Features

- **High Performance**: Modern vector techniques integrated into a well-designed architecture.
- **Elastic Scalability**: Seamlessly scale across multiple threads, which is optimized by C++20 coroutines.
- **Adaptive Flexibility**: Easy customization for quantization methods, metrics, and data types.
- **Ease of Use**: [Intuitive APIs](https://github.com/AlayaDB-AI/AlayaLite/blob/main/python/README.md) in Python.


## Getting Started!

Get started with just one command!
```bash
pip install alayalite # install the python package.
```



Access your vectors using simple APIs.
```python
from alayalite import Client, Index
from alayalite.utils import calc_recall, calc_gt
import numpy as np

# Initialize the client and create an index. The client can manage multiple indices with distinct names.
client = Client()
index = client.create_index("default")

# Generate random vectors and queries, then calculate the ground truth top-10 nearest neighbors for each query.
vectors = np.random.rand(1000, 128).astype(np.float32)
queries = np.random.rand(10, 128).astype(np.float32)
gt = calc_gt(vectors, queries, 10)

# Insert vectors to the index
index.fit(vectors)

# Perform batch search for the queries and retrieve top-10 results
result = index.batch_search(queries, 10)

# Compute the recall based on the search results and ground truth
recall = calc_recall(result, gt)
print(recall)
```

## Benchmark

We evaluate the performance of AlayaLite against other vector database systems using [ANN-Benchmark](https://github.com/erikbern/ann-benchmarks) (compile locally and open `-march=native` in your `CMakeLists.txt` to reproduce the results). Several experimental results are presented below.

|     ![Fashion-MNIST	784 Euclidean](./.assets/fashion-mnist-784-euclidean.png)     |    ![Gist 960 Euclidean](./.assets/gist-960-euclidean.png)    |
| :---------------------------------------------------------: | :-----------------------------------------------------------: |
| <div style="text-align: center;">**Fashion-MNIST	784 Euclidean**</div> | <div style="text-align: center;">**Gist 960 Euclidean**</div> |



## Contributing

We welcome contributions to AlayaLite! If you would like to contribute, please follow these steps:

1. Start by creating an issue outlining the feature or bug you plan to work on.
2. We will collaborate on the best approach to move forward based on your issue.
3. Fork the repository, implement your changes, and commit them with a clear message.
4. Push your changes to your forked repository.
5. Submit a pull request to the main repository.

Please ensure that your code follows the coding standards of the project and includes appropriate tests.

## Acknowledgements

We would like to thank all the contributors and users of AlayaLite for their support and feedback.

## Contact

If you have any questions or suggestions, please feel free to open an issue or contact us at **dev@alayadb.ai**.


## License

[Apache 2.0](https://github.com/AlayaDB-AI/AlayaLite/blob/main/LICENSE)
