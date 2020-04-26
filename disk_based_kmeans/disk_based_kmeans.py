import numpy as np
import pandas as pd
import time
import random
import os
import my_utils


class KMeans:
    df_genres = None

    def __init__(self, k=3, d="d1", iter=5):
        self.data_path = os.path.join("..", "ml-25m", "movies.csv")  # according to the distance
        self.datalen = my_utils.file_len(self.data_path)
        self.k = k
        self.distance = d
        self.iter = iter

        self.metadata = {i: [] for i in range(self.k)}
        self.clusteroids = random.sample(range(self.datalen), self.k)

    def read_csv(self):  # not used
        df = pd.read_csv(self.data_path)
        self.df_genres = df["genres"]

    def get_clusteroid(self, metadata):
        clusterroid = []
        for i, c in enumerate(self.clusteroids):
            index_points = self.metadata[i]

            total_dist = []
            for point_i in index_points:
                sum = 0
                for point_j in index_points:
                    sum += my_utils.get_jaccard(self.df_genres[point_i], self.df_genres[point_j])
                total_dist.append(sum)
            clusterroid.append(index_points[total_dist.index(min(total_dist))])
        return clusterroid

    def run(self):
        start = time.time()
        iterations = 0
        while iterations < self.iter:
            for i, gern in enumerate(self.df_genres):
                dists = []
                for j, c in enumerate(self.clusteroids):
                    dists.append(my_utils.get_jaccard(gern, self.df_genres[c]))

                self.metadata[dists.index(min(dists))].append(i)

            self.clusteroids = self.get_clusteroid(self.metadata)
            iterations += 1
            print(iterations, " iteration done in ", time.time()-start)

        print("Finish, took: ", time.time()-start)
        [print(i, self.df_genres) for i in self.clusteroids]




if __name__ == '__main__':
    kmean = KMeans(k=10, iter=5)
    kmean.read_csv()
    kmean.run()

    print("done")
