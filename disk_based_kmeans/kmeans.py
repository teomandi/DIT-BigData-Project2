import pandas as pd
import time
import os
import random

from utils import hierarchical_cluster, get_jaccard
from cluster import Cluster, RemainEntity


class KMeans(object):

    def __init__(self, dpath, k=4, threshold=0.6, chunk_size=4000):
        self.data_path = dpath

        self.k = k
        self.threshold = threshold
        self.chunk_size = chunk_size

        self.discard = []  # list with clusters
        self.remaining = []
        # self.compressed = []  # list with tuples with clusteroid, [(id, content)]
        # self.retained = []  # list with remained points as tuples (id, content)

    def fit(self):
        starting_tm = time.time()
        random_clusters_ids = []
        iteration = 0
        for chunk in pd.read_csv(self.data_path, chunksize=self.chunk_size):
            # If it is the first time, initialize the first k clusters
            if iteration == 0:
                random_clusters_ids = random.sample(range(self.chunk_size), self.k)
                self.discard = [Cluster(i, index, chunk['genres'][index]) for i, index in enumerate(random_clusters_ids)]
            for index, record in zip(chunk['movieId'], chunk['genres']):
                # idx = index + self.chunk_size*iteration
                if index in random_clusters_ids:
                    continue
                # calculate the distance with each cluster
                dists = []
                for cluster in self.discard:
                    dists.append(get_jaccard(cluster.clusteroid, record))
                # if it is over than threshold add it to DC
                if max(dists) >= self.threshold:
                    self.discard[dists.index(max(dists))].add_temp_point(index, record)
                else:
                    # add it to retained set
                    self.remaining.append(RemainEntity((index, record)))  # <--
            # calculate the new clusteroids in the discard set
            for c in self.discard:
                c.consume()
            # handle retain set
            # self.remaining = hierarchical_cluster(self.remaining, self.threshold)
            iteration += 1
        # end of dataset parse
        print("Iterations:", iteration)
        print("Took {:.3f}".format(time.time()-starting_tm))

    def export(self, export=True):
        for remain in self.remaining:
            dist = []
            for disc in self.discard:
                dist.append(get_jaccard(remain.clusteroid, disc.clusteroid))
            self.discard[dist.index(max(dist))].membership.extend([x[0] for x in remain.members])

        # checkup
        total = 0
        for disc in self.discard:
            print("# CLRD: -", disc.key, "-", disc.clusteroid, " len:: ", len(disc.membership))
            total += len(disc.membership)
        print("Total: ", total)

        if not export:
            return

        results = []
        for disc in self.discard:
            for member in disc.membership:
                results.append((member, disc.key))
        results = sorted(results, key=lambda tup: tup[0])
        with open('results.csv', 'a') as f:
            for res in results:
                f.write(str(res[0]) + "," + str(res[1]) + "\n")



if __name__ == '__main__':
    data_path = os.path.join("..", "ml-25m", "movies.csv")

    x = KMeans(data_path, k=10, threshold=0.8, chunk_size=4000)
    x.fit()
    x.export()

    # remain = []
    # for chunk in pd.read_csv(data_path, chunksize=1000):
    #     for i, record in enumerate(chunk['genres']):
    #         remain.append(RemainEntity((i, record)))
    #     break
    # compr = hierarchical_cluster(remain, 0.6)




