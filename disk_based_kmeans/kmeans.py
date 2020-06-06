import pandas as pd
import time
import random
import csv
import argparse
import os
from sklearn.metrics.pairwise import cosine_similarity

from utils import hierarchical_cluster, get_jaccard, recreate_file, get_rating_vectors, UsersRatings, MoviesRatings
from cluster import SimpleCluster, ComplexCluster, RemainEntity


class KMeans(object):

    def __init__(self, data_path, k=4, threshold=0.6, chunk_size=4000, distance_f="d1", ratings_path=None):
        if k > chunk_size:
            print("Error: K bigger than cluster size")
            exit(1)
        self.data_path = data_path
        self.k = k
        self.threshold = threshold
        self.chunk_size = chunk_size
        self.discard = []  # list with clusters
        self.remaining = []  # list of RemainEntity((index, record)) (mostly created for hierarchical_cluster)
        self.distance_f = distance_f
        self.absorb = self.simple_absorb
        self.details = self.simple_details
        if self.distance_f == "d1":
            self.fit = self.fit_with_new
            self.target = 'genres'
        elif self.distance_f == "d2":
            self.fit = self.fit_with_new
            self.target = 'tags'
        elif self.distance_f == "d3":
            self.fit = self.fit_with_ratings  # self.fit_with_ratings
            if ratings_path is None:
                print("Error: No ratings path was given")
                exit(1)
            self.ratings_path = ratings_path
        elif self.distance_f == "d4":
            self.fit = self.fit_with_all
            self.absorb = self.complex_absorb
            if ratings_path is None:
                print("Error: No ratings path was given")
                exit(1)
            self.ratings_path = ratings_path
            self.details = self.complex_details
        else:
            print("Error: Unknown distance function was given")
            exit(1)

    # adds the remains to the closest created cluster and exports the results
    def simple_absorb(self):
        print("Absorb starts")
        starting_tm = time.time()
        for remain in self.remaining:
            dist = []
            for cluster in self.discard:
                if self.distance_f != "d3":
                    dist.append(get_jaccard(remain.clusteroid, cluster.clusteroid))
                else:
                    dist.append(cosine_similarity(remain.clusteroid, cluster.clusteroid)[0][0])
            self.discard[dist.index(max(dist))].membership.extend([x[0] for x in remain.members])
        print("Absorb duration(s): {:.3f}".format(time.time()-starting_tm))

    def complex_absorb(self):
        print("Absorb starts")
        starting_tm = time.time()
        for remain in self.remaining:
            dist = []
            for cluster in self.discard:
                d1 = get_jaccard(cluster.clusteroid_genres, remain['genres'])
                d2 = get_jaccard(cluster.clusteroid_tags, remain['tags'])
                d3 = cosine_similarity(cluster.clusteroid_ratings, remain['ratings'])[0][0]
                _dist = 0.33 * d1 + 0.25 * d2 + 0.45 * d3
                dist.append(_dist)
            self.discard[dist.index(max(dist))].membership.append(remain['movie_id'])
        print("Absorb duration(s): {:.3f}".format(time.time()-starting_tm))

    def simple_details(self):
        total = 0
        for disc in self.discard:
            print("~> Key: ", disc.key, "|Clusteroid: ", disc.clusteroid, "|Members: ", len(disc.membership))
            total += len(disc.membership)
        print("Total Points: ", total)

    def complex_details(self):
        total = 0
        for disc in self.discard:
            print(
                "~> Key: ", disc.key,
                "|Clusteroid: Genres: ", disc.clusteroid_genres,
                "Tag: ", disc.clusteroid_tags,
                "Ratings: <sparse vector> ",
                " |Members: ", len(disc.membership))
            total += len(disc.membership)
        print("Total Points: ", total)

    def print_results(self):
        results = []
        for disc in self.discard:
            for member in disc.membership:
                results.append((member, disc.key))
        results = sorted(results, key=lambda tup: tup[0])
        print("MovieId, ClusterKey")
        for movie_id, cluster_key in results:
            print(movie_id, ",", cluster_key)

    def export(self):
        results = []
        output_name = self.distance_f + "_results.csv"
        print("Exporting in file ", output_name, "...")
        export_tm = time.time()
        for disc in self.discard:
            for member in disc.membership:
                results.append((member, disc.key))
        results = sorted(results, key=lambda tup: tup[0])
        with open(output_name, 'w') as f:
            for movie_id, cluster_key in results:
                f.write(str(movie_id) + "," + str(cluster_key) + "\n")
        print("Export duration(s): {:.3f}".format(time.time()-export_tm))

    def fit_with_new(self):
        starting_tm = time.time()
        random_clusters_ids = []
        iteration = 0
        for chunk in pd.read_csv(self.data_path, chunksize=self.chunk_size):
            loop_tm = time.time()
            # If it is the first time, initialize the first k clusters
            if iteration == 0:
                rows_id = random.sample(range(self.chunk_size), self.k)
                random_clusters_ids = [chunk['movieId'][row_id] for row_id in rows_id]
                self.discard = [
                    SimpleCluster(i, movie_id, chunk[self.target][row_id])
                    for i, (row_id, movie_id) in enumerate(zip(rows_id, random_clusters_ids))
                ]
            for movie_id, record in zip(chunk['movieId'], chunk[self.target]):
                if movie_id in random_clusters_ids:
                    continue
                # calculate the distance with each cluster
                dists = []
                for cluster in self.discard:
                    dists.append(get_jaccard(cluster.clusteroid, record))
                # if it is over than threshold add it to DC
                if max(dists) >= self.threshold:
                    self.discard[dists.index(max(dists))].add_temp_point(movie_id, record)
                else:
                    # add it to retained set
                    self.remaining.append(RemainEntity((movie_id, record)))
            # calculate the new clusteroids in the discard set
            for c in self.discard:
                c.consume()
            # handle retain set --TOO SLOW--
            # self.remaining = hierarchical_cluster(self.remaining, self.threshold)
            print("chunk ", iteration, " in: {:.3f}".format(time.time() - loop_tm))
            iteration += 1
        # end of dataset parse
        print("Total Iterations:", iteration, " Chunk Size: ", self.chunk_size)
        print("Fit duration(s): {:.3f}".format(time.time()-starting_tm))

    def fit_with_ratings(self):
        starting_tm = time.time()
        random_clusters_ids = []
        iteration = 0
        user_ratings = MoviesRatings(self.ratings_path)
        for chunk in pd.read_csv(self.data_path, chunksize=self.chunk_size):
            loop_tm = time.time()
            chunk_ids = chunk['movieId'].tolist()
            chunk_vectors = {}  # user_ratings.get_many_vectors(chunk_ids)
            for movie_id in chunk_ids:
                chunk_vectors[movie_id] = user_ratings.get_vector(movie_id)
            print("All vector created in: {:.3f}".format(time.time() - loop_tm))
            if iteration == 0:
                rows_id = random.sample(range(self.chunk_size), self.k)
                random_clusters_ids = [chunk['movieId'][row_id] for row_id in rows_id]
                self.discard = [
                    SimpleCluster(i, movie_id, chunk_vectors[movie_id])
                    for i, (row_id, movie_id) in enumerate(zip(rows_id, random_clusters_ids))
                ]
            clustering_tm = time.time()
            for movie_id in chunk_vectors:
                if movie_id in random_clusters_ids:
                    continue
                dists = []
                for cluster in self.discard:
                    cs = cosine_similarity(cluster.clusteroid, chunk_vectors[movie_id])[0][0]
                    dists.append(cs)
                if max(dists) >= self.threshold:
                    print("hit")
                    self.discard[dists.index(max(dists))].add_temp_point(movie_id, chunk_vectors[movie_id])
                else:
                    self.remaining.append(RemainEntity((movie_id, chunk_vectors[movie_id])))
            print("Clustering part took {:.3f}".format(time.time()-clustering_tm))
            for cluster in self.discard:
                cluster.consume(f="cosine")
            print("chunk ", iteration, " in: {:.3f}".format(time.time() - loop_tm))
            iteration += 1
        print("Total Iterations:", iteration, " Chunk Size: ", self.chunk_size)
        print("Fit duration(s): {:.3f}".format(time.time()-starting_tm))

    # d4 = 0.3*d1 + 0.25*d2 + 0.45*d3
    def fit_with_all(self):
        starting_tm = time.time()
        random_clusters_ids = []
        iteration = 0
        user_ratings = UsersRatings(self.ratings_path)
        for chunk in pd.read_csv(self.data_path, chunksize=self.chunk_size):
            loop_tm = time.time()
            chunk_ids = chunk['movieId'].tolist()
            chunk_vectors = {}  # user_ratings.get_many_vectors(chunk_ids)
            for movie_id in chunk_ids:
                chunk_vectors[movie_id] = user_ratings.get_vector(movie_id)
            print("All vector created in: {:.3f}".format(time.time() - loop_tm))
            if iteration == 0:
                rows_id = random.sample(range(self.chunk_size), self.k)
                random_clusters_ids = [chunk['movieId'][row_id] for row_id in rows_id]
                self.discard = [
                    ComplexCluster(
                        i,
                        movie_id,
                        chunk['genres'][row_id],
                        chunk['tags'][row_id],
                        chunk_vectors[movie_id]
                    )
                    for i, (row_id, movie_id) in enumerate(zip(rows_id, random_clusters_ids))
                ]
            clustering_tm = time.time()
            for movie_id, genres, tags in zip(chunk['movieId'], chunk['genres'], chunk['tags']):
                if movie_id in random_clusters_ids:
                    continue
                dists = []
                for cluster in self.discard:
                    d1 = get_jaccard(cluster.clusteroid_genres, genres)
                    d2 = get_jaccard(cluster.clusteroid_tags, tags)
                    d3 = cosine_similarity(cluster.clusteroid_ratings, chunk_vectors[movie_id])[0][0]
                    distance = 0.33*d1 + 0.25*d2 + 0.45*d3
                    dists.append(distance)
                point = {
                    "movie_id": movie_id,
                    "genres": genres,
                    "tags": tags,
                    "ratings": chunk_vectors[movie_id]
                }
                if max(dists) >= self.threshold:
                    self.discard[dists.index(max(dists))].add_temp_point(point)
                else:
                    self.remaining.append(point)
            print("Clustering part took {:.3f}".format(time.time() - clustering_tm))
            for cluster in self.discard:
                cluster.consume()
            print("chunk ", iteration, " in: {:.3f}".format(time.time() - loop_tm))
            iteration += 1
        print("Total Iterations:", iteration, " Chunk Size: ", self.chunk_size)
        print("Fit duration(s): {:.3f}".format(time.time() - starting_tm))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Disk Based KMeans. Project 2a Big-Data 2020',
        epilog='Enjoy the program! :)'
    )
    # required arguments
    parser.add_argument('-k', '--k',  type=int, help='Number of clusters', action='store', required=True)
    parser.add_argument('-p',
                        '--path',
                        type=str,
                        help="The path for the RECREATED data file. Relative and Absolute are accepted",
                        action='store',
                        required=True)
    parser.add_argument('-d',
                        '--distance',
                        type=str,
                        help='The selected distance function eg: d[1-4]',
                        action='store',
                        required=True)
    # optional arguments
    parser.add_argument('-c', '--chunk', type=int, help="The size of the chunk")
    parser.add_argument(
        '-t',
        '--threshold',
        type=float,
        help="The accepted threshold in order a point to belong in a cluster")
    parser.add_argument(
        '-r',
        '--ratings_path',
        type=str,
        help="The path for the ratings file. Required for d3, d4. Relative and Absolute are accepted")
    parser.add_argument(
        '-e',
        '--export',
        help="If this argument is given, the program will export the results as a csv file in the working directory ",
        default=False,
        action='store_true'
    )
    args = parser.parse_args()
    arguments = vars(args)
    print(arguments)
    # checking the arguments
    if arguments['distance'] not in ['d1', 'd2', 'd3', 'd4']:
        print("Error: not valid distance was given. Use -h for help")
        exit(1)
    if arguments['distance'] in ['d3', 'd4'] and arguments['ratings_path'] is None:
        print("Error: ", arguments['distance'], "needs the path for ratings. Use -h for help")
        exit(1)
    if arguments['threshold'] is not None:
        if arguments['threshold'] > 1:
            print("Error: Threshold should be less than 1. Use -h for help")
            exit(1)
    new_datafile_path = arguments['path']
    ratings_path = arguments['ratings_path']
    k = arguments['k']
    d = arguments['distance']
    t = arguments['threshold'] if arguments['threshold'] is not None else 0.6
    chunk = arguments['chunk'] if arguments['chunk'] is not None else 5000
    print("datafile: ", new_datafile_path)
    print("ratings_path: ", ratings_path)
    print("k: ", k)
    print("d: ", d)
    print("t: ", t)
    print("chunk: ", chunk)
    total_tm = time.time()

    kmean = KMeans(new_datafile_path, k=k, threshold=t, chunk_size=chunk, distance_f=d, ratings_path=ratings_path)

    kmean.fit()
    kmean.absorb()
    kmean.details()
    if arguments['export']:
        kmean.export()
    else:
        kmean.print_results()

    print("Total duration(s): {:.3f}".format(time.time() - total_tm))
