import time
import csv
import os
import pandas as pd
import numpy as np


def file_len(fname):
    start = time.time()
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    print("Count: ", str(i+1), " lines in ", (time.time()-start))
    return i + 1


def get_jaccard(str1, str2, is_list=False, delimiter="|"):
    # print("JC: ", str1, "---", str2)
    if not is_list:
        str1 = str(str1).split(delimiter)
        str2 = str(str2).split(delimiter)
    a = set(str1)
    b = set(str2)
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


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


def recreate_file(
        tags_path="../ml-25m/tags.csv",
        movies_path="../ml-25m/movies.csv",
        output_path="../ml-25m/new_movies_tag_file.csv"):
    with open(output_path, 'w') as fout:
        fout.write("movieId,tags,genres\n")
    print("converting tag file")
    movies_file_path = movies_path
    tags_file_path = tags_path
    iteration = 0
    starting_tm = time.time()
    for chunk in pd.read_csv(movies_file_path, chunksize=10000):
        init_tm = time.time()
        chunk_data = {}
        for movie_id, genres_val in zip(chunk['movieId'], chunk['genres']):
            chunk_data[movie_id] = ([], genres_val)
        print("init in: {:.3f}".format(time.time() - init_tm))

        chunk_tm = time.time()
        print("Parsing chunk:: ", iteration)
        with open(tags_file_path) as f:
            reader = csv.reader(f)
            for row in reader:
                try:
                    row_movie_id = int(row[1])
                except:
                    continue
                if row_movie_id in chunk_data:
                    chunk_data[row_movie_id][0].append(row[2])
        iteration += 1
        print("Chunk-data parsed in {:.3f}".format(time.time() - chunk_tm))

        writing_tm = time.time()
        with open(output_path, 'a') as fout:
            for movie_id in chunk_data:
                tags = ""
                for tag in chunk_data[movie_id][0]:
                    tags = tags + tag + "|"
                fout.write(str(movie_id) + "," + tags.replace(',', '').lower()[:-1] + "," + chunk_data[movie_id][1] + "\n")
        print("Chunk-data write in {:.3f}".format(time.time() - writing_tm))
    print("File Created in {:.3f}".format(time.time() - starting_tm))


def rating_vector():
    print("getting ratings vector")
    movies_file_path = os.path.join("..", "ml-25m", "movies.csv")
    ratings_file_path = os.path.join("..", "ml-25m", "ratings.csv")
    starting_tm = time.time()
    i = 0
    for chunk in pd.read_csv(movies_file_path, chunksize=4000):
        chunk_time = time.time()
        chunk_data = {}
        chunk_ids = chunk['movieId'].tolist()

        init_tm = time.time()
        for idd in chunk_ids:
            chunk_data[idd] = np.zeros(162541)
        print("init in: {:.3f}".format(time.time() - init_tm))

        with open(ratings_file_path) as f:
            reader = csv.reader(f)
            for row in reader:
                try:  # for the first row
                    user_id = int(row[0])
                    movie_id = int(row[1])
                    rating = float(row[2])
                except:
                    continue
            if movie_id in chunk_ids:
                chunk_data[movie_id][user_id-1] = rating
        print("chunk ", i, " in: {:.3f}".format(time.time() - chunk_time))
        # break
        i += 1
    print("Total process complete in: {:.3f}".format(time.time() - starting_tm))
