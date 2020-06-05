import pandas as pd
import numpy as np
import os
import time
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import pickle


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
        pivot_table = csr_matrix((len(self.users_ids), len(self.movies_ids)))
        for chunk in pd.read_csv(ratings_path, chunksize=100000):
            rows = [i - 1 for i in chunk['userId'].tolist()]
            cols = [np.where(self.movies_ids == j)[0][0] for j in chunk['movieId']]
            user_ratings = chunk['rating'].tolist()
            avg_ratings = sum(user_ratings) / len(user_ratings)
            fixed_ratings = [rate - avg_ratings for rate in user_ratings]
            pivot_table[rows, cols] = fixed_ratings
            iteration += 1
        print("Creating pivot_table took: {:.3f}".format(time.time() - headers_tm))
        return pivot_table

    def user_based_prediction(self, target_user_id):
        print("User-Based Prediction process for user: <", target_user_id, "> started")
        if target_user_id > len(self.users_ids)-1:
            print("Warning: User not exist on dataset")
            return
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
        predictions = [(movies_under_consideration[idx], movie_avg_ratings[idx]) for idx in best_movies_indexes[:20]]
        return predictions

    def item_based_prediction(self, target_user_id):
        pass


if __name__ == '__main__':
    ratings_path = os.path.join("..", "ml-25m", "ratings.csv")
    pivot_table_path = os.path.join("..", "ml-25m", "pivot_table.sparse")
    fixed_pivot_table_path = os.path.join("..", "ml-25m", "fixed_pivot_table.sparse")

    # cf = CollaborativeFiltering(ratings_path, pivot_table_path=pivot_table_path, load=True)
    cf = CollaborativeFiltering(
        method="user",
        ratings_file_path=ratings_path,
        pivot_table_path=fixed_pivot_table_path,
        store=False,
        load=True
    )

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
        for movie_id, rating in results:
            print(int(cf.movies_ids[movie_id]), rating)
    print("bye")
