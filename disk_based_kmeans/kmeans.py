import pandas as pd
import time
import os
import random

from utils import file_len, get_jaccard
from cluster import Cluster


class KMeans(object):

    def __init__(self, dpath, k=4, threshold=0.6, chunk_size=4000):
        self.data_path = dpath

        self.k = k
        self.threshold = threshold
        self.chunk_size = chunk_size

        self.discard = []  # list with clusters
        self.compressed = []  # list with tuples with clusteroid, [(id, content)]
        self.retained = []  # list with remained points as tuples (id, content)

    def fit(self):
        iteration = 0
        for chunk in pd.read_csv(self.data_path, chunksize=self.chunk_size):
            # If it is the first time, initialize the first k clusters
            if iteration == 0:
                random_clusters_ids = random.sample(range(self.chunk_size), self.k)
                self.discard = [Cluster(i, index, chunk['genres'][index]) for i, index in enumerate(random_clusters_ids)]

            for index, record in enumerate(chunk['genres']):
                # calculate the distance with each cluster
                dists = []
                for cluster in self.discard:
                    dists.append(utils.get_jaccard(cluster.clusteroid, record))

                 idx = index + self.chunk_size*iteration
                # if it is over than threshold add it to DC
                if max(dists) >= self.threshold:
                    self.discard[dists.index(max(dists))].add_temp_point(record, idx)
                else:
                    # add it to retained set
                    self.retained.append((idx, record))

            # end of chunk parse

            # calculate the new clusteroids in the discard set





            iteration += 1
        # end of dataset parse
        print("Iterations:", iteration)
        print(self.discard)


        # self.metadata = {i: [] for i in range(self.k)}
        # self.clusteroids = random.sample(range(self.datalen), self.k)
        # print("Clusteroid: ", self.clusteroids)

    # def read_csv(self):  # not used
    #     df = pd.read_csv(self.data_path)
    #     self.df_genres = df["genres"]
    #
    # def get_clusteroid(self, metadata):
    #     clusterroid = []
    #     for i, c in enumerate(self.clusteroids):
    #         index_points = self.metadata[i]
    #         print(i, ")len points: ", len(index_points))
    #         total_dist = []
    #         for point_i in index_points:
    #             sm = 0
    #             for point_j in index_points:
    #                 sm += get_jaccard(self.df_genres[point_i], self.df_genres[point_j], '|')
    #             total_dist.append(sm)
    #         clusterroid.append(index_points[total_dist.index(max(total_dist))])
    #     return clusterroid
    #
    # def run(self):
    #     start = time.time()
    #     iterations = 0
    #     while iterations < self.iter:
    #         for i, gern in enumerate(self.df_genres):
    #             dists = []
    #             for j, c in enumerate(self.clusteroids):
    #                 dists.append(get_jaccard(gern, self.df_genres[c], '|'))
    #             # print("Dists: ", dists)
    #             # print("min, ci", max(dists), ",", dists.index(max(dists)))
    #             self.metadata[dists.index(max(dists))].append(i)
    #
    #         print("updating clusteroids")
    #         self.metadata_stats()
    #         self.clusteroids = self.get_clusteroid(self.metadata)
    #         iterations += 1
    #         print(iterations, " iteration done in ", time.time()-start)
    #
    #     print("Finish, took: ", time.time()-start)
    #     [print(i, self.df_genres) for i in self.clusteroids]
    #
    # def metadata_stats(self):
    #     [print(i, len(self.metadata[i])) for i in self.metadata]




if __name__ == '__main__':
    data_path = os.path.join("..", "ml-25m", "movies.csv")
    x = KMeans(data_path)
    x.fit()


