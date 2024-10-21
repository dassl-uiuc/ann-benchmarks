import os
import struct
import time

import numpy as np
import diskannpy as vp
import diskannpy

from ..base.module import BaseANN


class Vamana(BaseANN):
    def __init__(self, metric, param):
        self.metric = {"angular": "cosine", "euclidean": "l2"}[metric]
        self.l_build = int(param["l_build"])
        self.max_outdegree = int(param["max_outdegree"])
        self.alpha = float(param["alpha"])
        print(f"Vamana: L_Build = {self.l_build}")
        print(f"Vamana: R = {self.max_outdegree}")
        print(f"Vamana: Alpha = {self.alpha}")
        self.num_threads = 0  # You may want to parameterize this

    def fit(self, X):
        print("Vamana: Starting Fit...")
        index_dir = "indices"
        os.makedirs(index_dir, exist_ok=True)

        def bin_to_float(binary):
            return struct.unpack("!f", struct.pack("!I", int(binary, 2)))[0]

        data_path = os.path.join(index_dir, "base.bin")
        self.name = "Vamana-{}-{}-{}".format(self.l_build, self.max_outdegree, self.alpha)
        save_path = os.path.join(index_dir, self.name)
        print("Vamana: Index Stored At: " + save_path)
        shape = [
            np.float32(bin_to_float("{:032b}".format(X.shape[0]))),
            np.float32(bin_to_float("{:032b}".format(X.shape[1]))),
        ]
        X = X.flatten()
        X = np.insert(X, 0, shape)
        X.tofile(data_path)

        if not os.path.exists(save_path):
            print("Vamana: Creating Index")
            s = time.time()
            diskannpy.build_memory_index(
                data=data_path,
                distance_metric=self.metric,
                vector_dtype=np.float32,
                index_directory=index_dir,
                complexity=self.l_build,
                graph_degree=self.max_outdegree,
                num_threads=self.num_threads,
                index_prefix=self.name,
                alpha=self.alpha,
                use_pq_build=False,
                num_pq_bytes=8,
                use_opq=False,
            )
            t = time.time()
            print(f"Vamana: Index Build Time (sec) = {t - s}")
        
        print("Vamana: Loading Index")
        s = time.time()
        self.index = diskannpy.StaticMemoryIndex(
            distance_metric=self.metric,
            vector_dtype=np.float32,
            index_directory=index_dir,
            num_threads=self.num_threads,
            initial_search_complexity=self.l_build,
            index_prefix=self.name
        )
        t = time.time()
        print(f"Vamana: Index Load Time (sec) = {t - s}")
        print("Vamana: End of Fit")

    def set_query_arguments(self, l_search):
        print(f"Vamana: L_Search = {l_search}")
        self.l_search = l_search

    def query(self, v, n):
        ids, _ = self.index.search(v, n, self.l_search)
        return ids

    def batch_query(self, X, n):
        self.num_queries = X.shape[0]
        self.result, _ = self.index.batch_search(X, n, self.l_search, self.num_threads)

    def get_batch_results(self):
        return self.result


class VamanaPQ(BaseANN):
    def __init__(self, metric, param):
        self.metric = {"angular": "cosine", "euclidean": "l2"}[metric]
        self.l_build = int(param["l_build"])
        self.max_outdegree = int(param["max_outdegree"])
        self.alpha = float(param["alpha"])
        self.chunks = int(param["chunks"])
        print(f"Vamana PQ: L_Build = {self.l_build}")
        print(f"Vamana PQ: R = {self.max_outdegree}")
        print(f"Vamana PQ: Alpha = {self.alpha}")
        print(f"Vamana PQ: Chunks = {self.chunks}")
        self.num_threads = 0  # You may want to parameterize this

    def fit(self, X):
        print("Vamana PQ: Starting Fit...")
        index_dir = "indices"
        os.makedirs(index_dir, exist_ok=True)

        if self.chunks > X.shape[1]:
            raise ValueError("Number of chunks cannot be greater than the number of dimensions")

        def bin_to_float(binary):
            return struct.unpack("!f", struct.pack("!I", int(binary, 2)))[0]

        data_path = os.path.join(index_dir, "base.bin")
        pq_path = os.path.join(index_dir, "pq_memory_index")
        self.name = "VamanaPQ-{}-{}-{}".format(self.l_build, self.max_outdegree, self.alpha)
        save_path = os.path.join(index_dir, self.name)
        print("Vamana PQ: Index Stored At: " + save_path)
        shape = [
            np.float32(bin_to_float("{:032b}".format(X.shape[0]))),
            np.float32(bin_to_float("{:032b}".format(X.shape[1]))),
        ]
        X = X.flatten()
        X = np.insert(X, 0, shape)
        X.tofile(data_path)

        if not os.path.exists(save_path):
            print("Vamana PQ: Creating Index")
            s = time.time()
            diskannpy.build_memory_index(
                data=data_path,
                distance_metric=self.metric,
                vector_dtype=np.float32,
                index_directory=index_dir,
                complexity=self.l_build,
                graph_degree=self.max_outdegree,
                num_threads=self.num_threads,
                index_prefix=self.name,
                alpha=self.alpha,
                use_pq_build=True,
                num_pq_bytes=self.chunks,
                use_opq=False,
            )
            t = time.time()
            print(f"Vamana PQ: Index Build Time (sec) = {t - s}")

        print("Vamana PQ: Loading Index")
        s = time.time()
        self.index = diskannpy.StaticMemoryIndex(
            distance_metric=self.metric,
            vector_dtype=np.float32,
            index_directory=index_dir,
            num_threads=self.num_threads,
            initial_search_complexity=self.l_build,
            index_prefix=self.name
        )
        t = time.time()
        print(f"Vamana PQ: Index Load Time (sec) = {t - s}")
        print("Vamana PQ: End of Fit")

    def set_query_arguments(self, l_search):
        print(f"Vamana PQ: L_Search = {l_search}")
        self.l_search = l_search

    def query(self, v, n):
        ids, _ = self.index.search(v, n, self.l_search)
        return ids

    def batch_query(self, X, n):
        self.num_queries = X.shape[0]
        self.result, _ = self.index.batch_search(X, n, self.l_search, self.num_threads)

    def get_batch_results(self):
        return self.result


class VamanaDiskPQ(BaseANN):
    def __init__(self, metric, param):
        self.metric = {"angular": "cosine", "euclidean": "l2"}[metric]
        self.l_build = int(param["l_build"])
        self.max_outdegree = int(param["max_outdegree"])
        self.alpha = float(param["alpha"])
        self.chunks = int(param["chunks"])
        self.cache = int(param["cache"])
        print(f"Vamana PQ: L_Build = {self.l_build}")
        print(f"Vamana PQ: R = {self.max_outdegree}")
        print(f"Vamana PQ: Alpha = {self.alpha}")
        print(f"Vamana PQ: Chunks = {self.chunks}")
        self.num_threads = 0  # You may want to parameterize this

    def fit(self, X):
        print("Vamana Disk PQ: Starting Fit...")
        index_dir = "indices"
        os.makedirs(index_dir, exist_ok=True)

        if self.chunks > X.shape[1]:
            raise ValueError("Number of chunks cannot be greater than the number of dimensions")

        def bin_to_float(binary):
            return struct.unpack("!f", struct.pack("!I", int(binary, 2)))[0]

        data_path = os.path.join(index_dir, "base.bin")
        pq_path = os.path.join(index_dir, "pq_memory_index")
        self.name = "VamanaDiskPQ-{}-{}-{}".format(self.l_build, self.max_outdegree, self.alpha)
        save_path = os.path.join(index_dir, self.name)
        print("Vamana Disk PQ: Index Stored At: " + save_path)
        shape = [
            np.float32(bin_to_float("{:032b}".format(X.shape[0]))),
            np.float32(bin_to_float("{:032b}".format(X.shape[1]))),
        ]
        X = X.flatten()
        X = np.insert(X, 0, shape)
        X.tofile(data_path)

        if not os.path.exists(save_path):
            print("Vamana Disk PQ: Creating Index")
            s = time.time()
            diskannpy.build_disk_index(
                data=data_path,
                distance_metric=self.metric,
                vector_dtype=np.float32,
                index_directory=index_dir,
                complexity=self.l_build,
                graph_degree=self.max_outdegree,
                num_threads=self.num_threads,
                search_memory_maximum=0.03,
                build_memory_maximum=120,
                pq_disk_bytes=0,
            )            
            t = time.time()
            print(f"Vamana Disk PQ: Index Build Time (sec) = {t - s}")

        print("Vamana Disk PQ: Loading Index")
        s = time.time()
        self.index = diskannpy.StaticDiskIndex(
            distance_metric=self.metric,
            vector_dtype=np.float32,
            index_directory=index_dir,
            num_threads=self.num_threads,
            num_nodes_to_cache=self.cache,
        )
        t = time.time()
        print(f"Vamana Disk PQ: Index Load Time (sec) = {t - s}")
        print("Vamana Disk PQ: End of Fit")

    def set_query_arguments(self, l_search):
        print(f"Vamana Disk PQ: L_Search = {l_search}")
        self.l_search = l_search

    def query(self, v, n):
        ids, _ = self.index.search(v, n, self.l_search)
        return ids

    def batch_query(self, X, n):
        self.num_queries = X.shape[0]
        self.result, _ = self.index.batch_search(X, n, self.l_search, self.num_threads)

    def get_batch_results(self):
        return self.result