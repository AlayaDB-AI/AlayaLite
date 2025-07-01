# Download the dataset
wget -P ./data ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz

tar -xzvf ./data/gist.tar.gz -C ./data

# Do clustering first
python ./gen_cluster.py ./data/gist/gist_base.fvecs 16 ./data/gist/gist_centroids_16.fvecs ./data/gist/gist_clusterids_16.ivecs

# Indexing
./exec/hnsw_rabitq_indexing ./data/gist/gist_base.fvecs ./data/gist/gist_centroids_16.fvecs ./data/gist/gist_clusterids_16.ivecs 16 200 5 ./data/gist/hnsw_5.index

# Querying
./exec/hnsw_rabitq_querying ./data/gist/hnsw_5.index ./data/gist/gist_query.fvecs ./data/gist/gist_groundtruth.ivecs