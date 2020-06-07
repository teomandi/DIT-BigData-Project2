import time
import argparse
import csv
import pandas as pd

"""
    Creates the new dataset file, which is just like the original
    movie's file but with the tags column also with the same format
    just like genres.
"""


def create_the_file(
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
                fout.write(
                    str(movie_id) + "," + tags.replace(',', '').lower()[:-1] + "," + chunk_data[movie_id][1] + "\n")
        print("Chunk-data write in {:.3f}".format(time.time() - writing_tm))
    print("File Created in {:.3f}".format(time.time() - starting_tm))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t',
                        '--tags',
                        type=str,
                        help="The path for the tags file",
                        action='store',
                        required=True)
    parser.add_argument('-m',
                        '--movies',
                        type=str,
                        help="The path for the movies file",
                        action='store',
                        required=True)
    parser.add_argument('-o',
                        '--output',
                        type=str,
                        help="The path to store the recreated file",
                        action='store',
                        required=True)
    args = parser.parse_args()
    arguments = vars(args)
    print(arguments)

    create_the_file(tags_path=arguments['tags'], movies_path=arguments['movies'], output_path=arguments['output'])
