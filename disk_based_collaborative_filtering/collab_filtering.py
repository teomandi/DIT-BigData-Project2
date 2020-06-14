import pandas as pd
import numpy as np
import os
import csv
import time
import math
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, diags, vstack, save_npz, load_npz
import argparse


class CollaborativeFiltering(object):
    def __init__(self, method="user", ratings_file_path=None, load=False):
        if ratings_file_path is None:
            print("Error: Path for Ratings is not given")
            exit(1)
        self.ratings_file_path = ratings_file_path
        self.users_ids, self.movies_ids = self.collect_movies_and_users()

        self.pivot_dir = "pivot-tables"
        self.users_per_table = 30000
        if not load:
            self.create_pivot_tables()

        if method == "user":
            self.predict = self.user_based_prediction
        elif method == "item":
            self.predict = self.item_based_prediction
        elif method == "mix":
            self.predict = self.mixed_based_prediction
        else:
            print("Error: unknown method was given: ", method)
            print("Accepted methods are only: \'user\', \'item\', \'mix\'")
            exit(1)

    def collect_movies_and_users(self):
        print("Collecting movies and users ids starts")
        iteration = 0
        movies_ids = np.array([])
        users_ids = np.array([])
        headers_tm = time.time()
        for chunk in pd.read_csv(self.ratings_file_path, chunksize=100000):
            movies_ids = np.append(movies_ids, chunk['movieId'].unique())
            users_ids = np.append(users_ids, chunk['userId'].unique())
            iteration += 1
        movies_ids = np.unique(movies_ids)
        users_ids = np.unique(users_ids)
        print("Collecting User+Movies took {:.3f}".format(time.time() - headers_tm))
        return users_ids, movies_ids

    def create_table(self, shape, rows, cols, values):
        table = csr_matrix((values, (rows, cols)), shape=shape)
        # fix the values
        tot = np.array(table.sum(axis=1).squeeze())[0]
        cts = np.diff(table.indptr)
        mu = tot / cts
        d = diags(mu, 0)
        b = table.copy()
        b.data = np.ones_like(b.data)
        ptable = (table - d*b)
        print("Table completed shape: ", shape)
        return ptable

    def create_pivot_tables(self):
        headers_tm = time.time()
        print("Creating the pivot-tables starts")
        if not os.path.isdir(self.pivot_dir):
            print("Directory for the tables created.")
            os.mkdir(self.pivot_dir)
        with open(self.ratings_file_path) as f:
            current_users_in_table = 0
            current_table_index = 0
            last_user_id = 0
            counting_users = 0
            rows = []  # for movies
            cols = []  # for users
            values = []  # for ratings
            reader = csv.reader(f)
            current_bucket_users = 0
            for row in reader:
                stored = False
                try:  # for the first row
                    user_id = int(row[0])
                    movie_id = int(row[1])
                    rating = float(row[2])
                except:
                    continue
                if last_user_id == user_id:
                    rows.append((user_id-1) - self.users_per_table * current_table_index)
                    cols.append(np.where(self.movies_ids == movie_id)[0][0])
                    values.append(rating)
                else:
                    last_user_id = user_id
                    counting_users += 1
                    # firstly we check if we have reach the amount of users in the current table
                    if current_users_in_table == self.users_per_table:
                        print("Table ", current_table_index, " is full. Table is created and stored ", current_bucket_users)
                        current_table = self.create_table(
                            (current_bucket_users, len(self.movies_ids)),
                            rows, cols, values
                        )
                        save_npz(
                            os.path.join(self.pivot_dir, str(current_table_index)+"-ptable.sparse"),
                            current_table
                        )
                        stored = True
                        current_table_index += 1
                        current_bucket_users = 1
                        current_users_in_table = 1  # because we add the first user below
                        rows = [(user_id-1) - self.users_per_table * current_table_index]
                        cols = [np.where(self.movies_ids == movie_id)[0][0]]
                        values = [rating]
                    else:
                        current_bucket_users += 1
                        current_users_in_table += 1
                        rows.append((user_id-1) - self.users_per_table * current_table_index)
                        cols.append(np.where(self.movies_ids == movie_id)[0][0])
                        values.append(rating)
        if not stored:
            print("Storing the Last table")
            current_table = self.create_table(
                (current_bucket_users, len(self.movies_ids)),
                rows, cols, values
            )
            save_npz(
                os.path.join(self.pivot_dir, str(current_table_index) + "-ptable.sparse"),
                current_table
            )
            print("Table ", current_table_index, " is full. Table is created and stored ", current_bucket_users)

        print("Creating the pivot-tables took: {:.3f}".format(time.time() - headers_tm))

    def predict_rating_for_movie(self, target_user_ratings, target_movie_id, movies_seen_by_target_user):
        first = True
        for tid in range(math.ceil(len(self.users_ids) / self.users_per_table)):
            table = load_npz(os.path.join(self.pivot_dir, str(tid) + "-ptable.sparse"))
            if first:
                target_movie_ratings = table.getcol(target_movie_id)
                first = False
            else:
                target_movie_ratings = vstack([target_movie_ratings, table.getcol(target_movie_id)])

        cfirst = True
        for m in range(math.ceil(len(self.movies_ids) / self.users_per_table)):  # reusing the 30k
            start = m * self.users_per_table
            stop = m * self.users_per_table + self.users_per_table
            if stop > len(self.movies_ids):
                stop = len(self.movies_ids)
            mfirst = True
            for tid in range(math.ceil(len(self.users_ids) / self.users_per_table)):
                table = load_npz(os.path.join(self.pivot_dir, str(tid) + "-ptable.sparse"))
                if mfirst:
                    other_movies_ratings = table[:, start:stop]
                    mfirst = False
                else:
                    other_movies_ratings = vstack([other_movies_ratings, table[:, start:stop]])
            similarities = cosine_similarity(
                other_movies_ratings.transpose(),
                target_movie_ratings.transpose()
            )
            if cfirst:
                similarity_movies = similarities
                cfirst = False
            else:
                similarity_movies = np.vstack((similarity_movies, similarities))
        similar_movies = (-similarity_movies).argsort(axis=0)
        most_similar_movies = (similar_movies[1:]).squeeze().tolist()  # put here the max selected movies
        accepted_movies = [movie for movie in most_similar_movies if movie in movies_seen_by_target_user]
        A = sum([target_user_ratings[0, movie] * similarity_movies[movie] for movie in accepted_movies])
        B = sum([similarity_movies[movie] for movie in accepted_movies])
        movie_rating = A / B
        return movie_rating[0]

    def user_based_prediction(self, target_user_id):
        prediction_tm = time.time()
        print("User-Based Prediction process for user: <", target_user_id, "> started")
        if target_user_id > len(self.users_ids)-1:
            print("Warning: User not exist on dataset")
            return []
        # get similar users
        target_table_id = int((target_user_id-1)/self.users_per_table)
        target_table = load_npz(os.path.join(self.pivot_dir, str(target_table_id)+"-ptable.sparse"))
        target_user_row_id = (target_user_id-1) - target_table_id * self.users_per_table
        target_user_ratings = target_table.getrow(target_user_row_id)
        # get the movies the target has seen
        movies_seen_by_target_user = target_user_ratings.nonzero()[1]
        if len(movies_seen_by_target_user) == 0:
            print("User has not see any movies!")
            return []
        first = True
        user_similarities = None
        for t_id in range(math.ceil(len(self.users_ids)/self.users_per_table)):
            current_table = load_npz(os.path.join(self.pivot_dir, str(t_id)+"-ptable.sparse"))
            similarities = cosine_similarity(
                current_table,
                target_user_ratings
            )
            if first:
                user_similarities = similarities
                first = False
            else:
                user_similarities = np.vstack((user_similarities, similarities))
        # get top 20 similar users
        similar_users = (-user_similarities).argsort(axis=0)
        most_similar_users = (similar_users[1:21]).squeeze().tolist()
        # get the movies that those users has seen
        movies_seen_by_similar_users = []
        for user in most_similar_users:
            table_id_with_that_user = int(user/self.users_per_table)
            users_table = load_npz(os.path.join(self.pivot_dir, str(table_id_with_that_user)+"-ptable.sparse"))
            users_row = user - table_id_with_that_user * self.users_per_table
            movies_seen_by_similar_users.extend(users_table.getrow(users_row).nonzero()[1])
        movies_seen_by_similar_users = set(movies_seen_by_similar_users)
        # movies that user has not seen but similar user has
        movies_under_consideration = list(movies_seen_by_similar_users - set(movies_seen_by_target_user))
        # for each movie get the avg and predict the best one
        movie_avg_ratings = []
        for movie in movies_under_consideration:
            movie_ratings = []
            # find which tables and which uses
            tables_id_to_check = {}
            for user in most_similar_users:
                table_id_with_that_user = int(user / self.users_per_table)
                if table_id_with_that_user not in tables_id_to_check:
                    tables_id_to_check[table_id_with_that_user] = []
                tables_id_to_check[table_id_with_that_user].append(user - table_id_with_that_user * self.users_per_table)
            for table_id in tables_id_to_check:
                table = load_npz(os.path.join(self.pivot_dir, str(table_id)+"-ptable.sparse"))
                try:
                    movie_ratings.extend(list(table[tables_id_to_check[table_id], movie].toarray().squeeze().tolist()))
                except TypeError:  # when is only one
                    movie_ratings.append(table[tables_id_to_check[table_id], movie].toarray().squeeze())
            # movie_avg_ratings.append(sum(movie_ratings) / len(movie_ratings))  # option 1
            movie_avg_ratings.append(
                sum([r*user_similarities[sid][0] for r, sid in zip(movie_ratings, most_similar_users)])
                / sum([user_similarities[sid][0] for sid in most_similar_users]))  # option 2
        best_movies_indexes = (-np.array(movie_avg_ratings)).argsort()[:20].tolist()
        predictions = [(int(self.movies_ids[movies_under_consideration[idx]]), movie_avg_ratings[idx], "u")
                       for idx in best_movies_indexes[:20]]
        return predictions

    def item_based_prediction(self, target_user_id):
        prediction_tm = time.time()
        print("Item-Based Prediction process for user: <", target_user_id, "> started")
        if target_user_id > len(self.users_ids) - 1:
            print("Warning: User not exist on dataset")
            return
        target_table_id = int((target_user_id - 1) / self.users_per_table)
        target_table = load_npz(os.path.join(self.pivot_dir, str(target_table_id) + "-ptable.sparse"))
        target_user_row_id = (target_user_id - 1) - target_table_id * self.users_per_table
        target_user_ratings = target_table.getrow(target_user_row_id)
        # get the movies the target has seen
        movies_seen_by_target_user = target_user_ratings.nonzero()[1]
        movies_seen_by_target_user = sorted(
            movies_seen_by_target_user,
            key=lambda x: target_user_ratings[0, x],
            reverse=True
        )
        movies_rate_predictions = []
        already_checked_movies = []
        should_stop = False
        print("User has seen ", len(movies_seen_by_target_user), "movies")
        if len(movies_seen_by_target_user) == 0:
            print("User has not see any movies!")
            return []
        while not should_stop:
            for movie in movies_seen_by_target_user:
                # get the movie vector from the tables
                first = True
                for tid in range(math.ceil(len(self.users_ids) / self.users_per_table)):
                    table = load_npz(os.path.join(self.pivot_dir, str(tid) + "-ptable.sparse"))
                    if first:
                        movie_ratings = table.getcol(movie)
                        first = False
                    else:
                        movie_ratings = vstack([movie_ratings, table.getcol(movie)])
                cfirst = True
                for m in range(math.ceil(len(self.movies_ids) / self.users_per_table)):  # reusing the 30k
                    start = m * self.users_per_table
                    stop = m * self.users_per_table + self.users_per_table
                    if stop > len(self.movies_ids):
                        stop = len(self.movies_ids)
                    mfirst = True
                    for tid in range(math.ceil(len(self.users_ids) / self.users_per_table)):
                        table = load_npz(os.path.join(self.pivot_dir, str(tid) + "-ptable.sparse"))
                        if mfirst:
                            other_movies_ratings = table[:, start:stop]
                            mfirst = False
                        else:
                            other_movies_ratings = vstack([other_movies_ratings, table[:, start:stop]])
                    similarities = cosine_similarity(
                        other_movies_ratings.transpose(),
                        movie_ratings.transpose()
                    )
                    if cfirst:
                        similarity_with_other_movies = similarities
                        cfirst = False
                    else:
                        similarity_with_other_movies = np.vstack((similarity_with_other_movies, similarities))
                k = 0
                move_on = False
                sorted_similar_movies = (-similarity_with_other_movies).argsort(axis=0)
                while True:
                    most_similar_movie_id = sorted_similar_movies[k][0]
                    if most_similar_movie_id in already_checked_movies or\
                            most_similar_movie_id in movies_seen_by_target_user:
                        if k >= similarity_with_other_movies.shape[0]:
                            move_on = True
                            break
                        k += 1
                        continue
                    else:
                        already_checked_movies.append(most_similar_movie_id)
                        break
                if move_on:
                    continue
                movies_rate_predictions.append(
                    (most_similar_movie_id, self.predict_rating_for_movie(
                        target_user_ratings,
                        most_similar_movie_id,
                        movies_seen_by_target_user
                    ))
                )
                if len(movies_rate_predictions) == 20:
                    should_stop = True
                    break
                if len(movies_rate_predictions) % 2 == 0:
                    print((len(movies_rate_predictions)*5), "% process done")
            print("Warning: Next Cycle")
        movies_rate_predictions = sorted(movies_rate_predictions, reverse=True, key=lambda tup: tup[1])[:20]
        predictions = [(int(self.movies_ids[movie_idx]), _rating, "i") for movie_idx, _rating in movies_rate_predictions]
        print("Item-Based Prediction took {:.3f}".format(time.time() - prediction_tm))
        return predictions

    def mixed_based_prediction(self, target_user_id):
        prediction_tm = time.time()
        print("Mixed-Based Prediction process for user: <", target_user_id, "> started")
        if target_user_id > len(self.users_ids) - 1:
            print("Warning: User not exist on dataset")
            return
        user_based_predictions = self.user_based_prediction(target_user_id)
        item_based_predictions = self.item_based_prediction(target_user_id)

        predictions = user_based_predictions + item_based_predictions
        predictions.sort(key=lambda trup: trup[1], reverse=True)
        print("Mixed-Based Prediction took {:.3f}".format(time.time() - prediction_tm))
        return predictions[:20]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Disk Based Collaborative Filtering. Project 2b Big-Data 2020',
        epilog='Enjoy the program! :)'
    )
    parser.add_argument(
        '-m',
        '--method',
        type=str,
        help='Selected method for predicting movies. Accepted Options are \'user\', \'item\', \'mix\'',
        action='store',
        required=True)
    parser.add_argument(
        '-r',
        '--ratings_path',
        type=str,
        help="The path for the ratings file. Required for d3, d4. Relative and Absolute are accepted",
        required=True)
    # optional arguments
    parser.add_argument(
        '-l',
        '--load',
        help="If this argument is given, the program will try to load the pivot table from the pivot_table_path",
        default=False,
        action='store_true'
    )

    args = parser.parse_args()
    arguments = vars(args)
    print("Given args: ", arguments)

    cf = CollaborativeFiltering(
        method=arguments['method'],
        ratings_file_path=arguments['ratings_path'],
        load=arguments['load']
    )
    try:
        while True:
            uid = input("Give user id: ")
            try:
                uid = int(uid)
            except ValueError:
                if uid == 'q':
                    break
                print("Invalid value..")
                continue
            results = cf.predict(uid)
            print("MOVIE ID, RATING, METHOD")
            for i, (movie_idx, r, m) in enumerate(results):
                print((i+1), ")", movie_idx, r, m)
    except KeyboardInterrupt:
        pass
    print("bye")
