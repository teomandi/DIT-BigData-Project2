import time
import csv
import pandas as pd
import numpy as np
from scipy import sparse


def get_jaccard(str1, str2, delimiter="|"):
    # print("JC: ", str1, "---", str2)
    str1 = str(str1).split(delimiter)
    str2 = str(str2).split(delimiter)
    a = set(str1)
    b = set(str2)
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


"""
    Hierarchical Clusterer: Gets a dataset with remaining points, and in
    each loop trys to find the pair with the minimum distance. Then, merges
    the pair into one new point (keeping its metadata) and goes on. The process 
    stops only when there is no other pair to merge that it does not satisfies
    the minimum distance threshold. 
"""


def hierarchical_cluster(remained, threshold):
    print("hierarchical started:: ", len(remained))
    starting_tm = time.time()
    iterations = 0
    while True:
        max_dist = -1
        target_i = target_j = -1
        for i in range(len(remained)):
            for j in range(i+1, len(remained)):
                dist = get_jaccard(remained[i].clusteroid, remained[j].clusteroid)
                if dist >= threshold and dist > max_dist:
                    max_dist = dist
                    target_i, target_j = i, j
        if max_dist != -1:
            remained[target_i].merge(remained[target_j])
            del remained[target_j]
        else:
            break
        iterations += 1
        print("new iter", iterations)
    print("Hierarchical took {:.3f}".format(time.time() - starting_tm))
    print("Iterations : ", iterations)
    print("created clusters ", len(remained))
    return remained


"""
    DEPRECATED
    It takes the id from a chunk and returns the vectors with the ratings
"""


def get_rating_vectors(chunk_ids, ratings_path):
    chunk_vector = {}
    # initialize the vectors
    init_tm = time.time()
    for movie_id in chunk_ids:
        chunk_vector[movie_id] = np.zeros(162541)  # hardcoded users size
    print("init time: {:.3f}".format(time.time() - init_tm))
    # parse to get the ratings
    parse_tm = time.time()
    with open(ratings_path) as f:
        reader = csv.reader(f)
        for row in reader:
            try:  # for the first row
                user_id = int(row[0])
                movie_id = int(row[1])
                rating = float(row[2])
            except:
                continue
            if movie_id in chunk_ids:
                chunk_vector[movie_id][user_id - 1] = rating
    for movie_id in chunk_ids:
        chunk_vector[movie_id] = sparse.csr_matrix(chunk_vector[movie_id])
    print("parsing time: {:.3f}".format(time.time() - parse_tm))


class MoviesRatings(object):
    def __init__(self, ratings_path):
        print("Creating Movies structure")
        self.movies_info = {}  # { movie_id : { user_id : rating } }
        starting_tm = time.time()
        with open(ratings_path) as f:
            reader = csv.reader(f)
            for row in reader:
                try:  # for the first row
                    user_id = int(row[0])
                    movie_id = int(row[1])
                    rating = float(row[2])
                except:
                    continue
                if movie_id not in self.movies_info:
                    self.movies_info[movie_id] = {user_id: rating}
                else:
                    self.movies_info[movie_id][user_id] = rating
        print("movies_info structure created in {:.3f}".format(time.time() - starting_tm))

    def get_vector(self, movie_id):
        vector = np.zeros(162541)  # hardcoded users size
        if movie_id in self.movies_info:
            for user_id in self.movies_info[movie_id]:
                vector[user_id-1] = self.movies_info[movie_id][user_id]
        return sparse.csr_matrix(vector)