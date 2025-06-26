import numpy as np
import struct
import faiss
import sys
from time import time

def read_ivecs(filename):
    print(f"Reading File - {filename}")
    a = np.fromfile(filename, dtype="int32")
    d = a[0]
    print(f"\t{filename} readed")
    return a.reshape(-1, d + 1)[:, 1:]
    

def read_fvecs(filename):
    return read_ivecs(filename).view("float32")


def write_ivecs(filename, m):
    print(f"Writing File - {filename}")
    n, d = m.shape
    myimt = "i" * d
    with open(filename, "wb") as f:
        for i in range(n):
            f.write(struct.pack("i", d))
            bin = struct.pack(myimt, *m[i])
            f.write(bin)
    print(f"\t{filename} wrote")


def write_fvecs(filename, m):
    m = m.astype("float32")
    write_ivecs(filename, m.view("int32"))


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print(f"Usage: {sys.argv[0]} <arg1> <arg2> <arg3> <arg4> <arg5>")
        print("arg1: path for data file, format .fvecs")
        print("arg2: number of clusters")
        print("arg3: path for centroid vectors")
        print("arg4: path for cluster ids")
        print("arg5: distance metric")
        exit(1)
    

    # path
    data_path = sys.argv[1]
    K = int(sys.argv[2])
    centroids_path = sys.argv[3]
    cluster_id_path = sys.argv[4]
    if len(sys.argv) == 6:
        distance_metric = sys.argv[5].lower()
        if distance_metric == "l2":
            metric = faiss.METRIC_L2
            print("Using L2 metric")
        elif distance_metric in ["ip", "innerproduct"]:
            metric = faiss.METRIC_INNER_PRODUCT
            print("Using InnerProduct metric")
        else:
             raise ValueError("Unsupported distance metric. Use 'l2' or 'ip'.")
    else:
        metric = faiss.METRIC_L2 # by default, L2 metric
        print("Using L2 metric by default")

    X = read_fvecs(data_path)

    dim = X.shape[1]

    t1 = time()

    # cluster data vectors
    index = faiss.index_factory(dim, f"IVF{K},Flat", metric)
    index.verbose = True
    index.train(X)

    t2 = time()
    print(f"Time for training ivf {t2-t1} secs")

    centroids = index.quantizer.reconstruct_n(0, index.nlist)
    _, cluster_id = index.quantizer.search(X, 1)

    write_ivecs(cluster_id_path, cluster_id)
    write_fvecs(centroids_path, centroids)
