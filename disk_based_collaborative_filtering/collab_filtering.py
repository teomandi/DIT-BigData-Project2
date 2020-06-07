import pandas as pd
import numpy as np
import os
import time
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, lil_matrix, diags
import pickle
import argparse


def pickle_store(obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)


def pickle_load(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


class CollaborativeFiltering(object):
    def __init__(self, method="user", ratings_file_path=None, pivot_table_path=None, load=False, store=False):
        if ratings_file_path is None:
            print("Error: Path for Ratings is not given")
            exit(1)
        self.ratings_file_path = ratings_file_path
        self.users_ids, self.movies_ids = self.collect_movies_and_users()
        if load:
            if pivot_table_path is not None:
                self.pivot_table = pickle_load(pivot_table_path)
            else:
                print("Error: Could not load pivot table without path")
                exit(1)
        else:
            self.pivot_table = self.create_pivot_table()
            if store:
                if pivot_table_path is not None:
                    print("Storing PivotTable")
                    pickle_store(self.pivot_table, pivot_table_path)
                else:
                    print("Warning: Could not store pivot table without path")
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

    def create_pivot_table(self):
        print("Creating pivot table starts")
        iteration = 0
        headers_tm = time.time()
        a = csr_matrix((len(self.users_ids), len(self.movies_ids)))
        for chunk in pd.read_csv(self.ratings_file_path, chunksize=100000):
            rows = [i - 1 for i in chunk['userId'].tolist()]
            cols = [np.where(self.movies_ids == j)[0][0] for j in chunk['movieId']]
            values = chunk['rating'].tolist()
            # fixed_ratings = [rat - chunk[chunk['userId'] == u].mean()['rating']
            #                  for u, rat in zip(chunk['userId'], chunk['rating'])]
            a[rows, cols] = values
            iteration += 1
        print("Creating pivot_table took: {:.3f}".format(time.time() - headers_tm))
        headers_tm = time.time()
        tot = np.array(a.sum(axis=1).squeeze())[0]
        cts = np.diff(a.indptr)
        mu = tot / cts
        d = diags(mu, 0)
        b = a.copy()
        b.data = np.ones_like(b.data)
        pivot_table = (a - d*b)
        print("Subtracting the mean from rows took: {:.3f}".format(time.time() - headers_tm))

        return pivot_table

    def predict_rating_for_movie(self, target_movie_id, target_user_id, movies_seen_by_target_user):
        similarity_movies = cosine_similarity(
            self.pivot_table.transpose(),
            self.pivot_table.getcol(target_movie_id).transpose()
        )
        similar_movies = (-similarity_movies).argsort(axis=0)
        most_similar_movies = (similar_movies[1:]).squeeze().tolist()  # put here the max selected movies
        accepted_movies = [movie for movie in most_similar_movies if movie in movies_seen_by_target_user]
        movie_rating = sum(
            [self.pivot_table[target_user_id - 1, movie] * similarity_movies[movie] for movie in accepted_movies]) /\
                       sum([similarity_movies[movie] for movie in accepted_movies])
        return movie_rating[0]

    def user_based_prediction(self, target_user_id):
        prediction_tm = time.time()
        print("User-Based Prediction process for user: <", target_user_id, "> started")
        if target_user_id > len(self.users_ids)-1:
            print("Warning: User not exist on dataset")
            return []
        # get similar users
        user_similarities = cosine_similarity(
            self.pivot_table,
            self.pivot_table.getrow(target_user_id - 1)
        )
        # get top 20 similar users
        similar_users = (-user_similarities).argsort(axis=0)
        most_similar_users = (similar_users[1:21]).squeeze().tolist()
        # get the movies that those users has seen
        movies_seen_by_similar_users = []
        for user in most_similar_users:
            movies_seen_by_similar_users.extend(self.pivot_table.getrow(user).nonzero()[1])
        movies_seen_by_similar_users = set(movies_seen_by_similar_users)
        # get the movies the target has seen
        movies_seen_by_target_user = self.pivot_table.getrow(target_user_id - 1).nonzero()[1]
        if len(movies_seen_by_target_user) == 0:
            print("User has not see any movies!")
            return []
        # movies that user has not seen but similar user has
        movies_under_consideration = list(movies_seen_by_similar_users - set(movies_seen_by_target_user))
        # for each movie get the avg and predict the best one
        movie_avg_ratings = []
        print("under cons movies are ", len(movies_under_consideration))
        for movie in movies_under_consideration:
            movie_ratings = self.pivot_table[most_similar_users, movie].toarray().squeeze().tolist()
            # movie_avg_ratings.append(sum(movie_ratings) / len(movie_ratings))  # option 1
            movie_avg_ratings.append(
                sum([r*user_similarities[sid][0] for r, sid in zip(movie_ratings, most_similar_users)])
                / sum([user_similarities[sid][0] for sid in most_similar_users]))  # option 2
        best_movies_indexes = (-np.array(movie_avg_ratings)).argsort()[:20].tolist()
        predictions = [(int(self.movies_ids[movies_under_consideration[idx]]), movie_avg_ratings[idx], "u")
                       for idx in best_movies_indexes[:20]]
        print("User-Based Prediction took {:.3f}".format(time.time() - prediction_tm))
        return predictions

    def item_based_prediction(self, target_user_id):
        prediction_tm = time.time()
        print("Item-Based Prediction process for user: <", target_user_id, "> started")
        if target_user_id > len(self.users_ids) - 1:
            print("Warning: User not exist on dataset")
            return
        movies_seen_by_target_user = self.pivot_table.getrow(target_user_id - 1).nonzero()[1].tolist()
        movies_seen_by_target_user.sort(key=lambda x: self.pivot_table[target_user_id - 1, x], reverse=True)
        movies_rate_predictions = []
        already_checked_movies = []
        should_stop = False
        print("User has seen ", len(movies_seen_by_target_user), "movies")
        if len(movies_seen_by_target_user) == 0:
            print("User has not see any movies!")
            return []
        while not should_stop:
            for movie in movies_seen_by_target_user:
                similarity_with_other_movies = cosine_similarity(
                    self.pivot_table.transpose(),
                    self.pivot_table.getcol(movie).transpose()
                )
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
                        most_similar_movie_id,
                        target_user_id,
                        movies_seen_by_target_user
                    ))
                )
                if len(movies_rate_predictions) == 20:
                    should_stop = True
                    break
                if len(movies_rate_predictions) % 2 == 0:
                    print((len(movies_rate_predictions)*5), "% process done")
            print("next cycle")
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
        '-p',
        '--pivot_table_path',
        type=str,
        help="The path for the pivot_table file. It will load or store it there according to the other arguments",
    )
    parser.add_argument(
        '-l',
        '--load',
        help="If this argument is given, the program will try to load the pivot table from the pivot_table_path",
        default=False,
        action='store_true'
    )
    parser.add_argument(
        '-s',
        '--store',
        help="If this argument is given, the program will store the generated pivot table to the pivot_table_path",
        default=False,
        action='store_true'
    )

    args = parser.parse_args()
    arguments = vars(args)
    print("Given args: ", arguments)

    cf = CollaborativeFiltering(
        method=arguments['method'],
        ratings_file_path=arguments['ratings_path'],
        pivot_table_path=arguments['pivot_table_path'],
        store=arguments['store'],
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
            for i, (movie_id, r, m) in enumerate(results):
                print((i+1), ")", movie_id, r, m)
    except KeyboardInterrupt:
        pass
    print("bye")
