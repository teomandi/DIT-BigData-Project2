from utils import file_len, get_jaccard
import time

from sklearn.metrics.pairwise import cosine_similarity


class SimpleCluster(object):
    def __init__(self, key, movie_id, content, is_list=False):
        self.key = key
        self.clusteroid = content
        self.membership = [movie_id]
        self.is_list = is_list
        # the temporary items are kept here before they get discarded
        self.temp = []  # list with tuples (id, content)

    def add_temp_point(self, idx, content):
        self.temp.append((idx, content))

    def consume(self, f="jaccard"):
        if len(self.temp) == 0:
            return
        starting_tm = time.time()
        ex_roid_sum = 0  # clusteroid sum to each point
        sum_dists = []
        for p1 in self.temp:
            if f == "jaccard":
                _dist = get_jaccard(p1[1], self.clusteroid)
            else:
                _dist = cosine_similarity(p1[1], self.clusteroid)
            # sum of the current clusteroid
            ex_roid_sum += _dist
            # sum for each temp point
            sum = _dist
            for p2 in self.temp:
                if f =="jaccard":
                    sum += get_jaccard(p1[1], p2[1])
                else:
                    sum += cosine_similarity(p1[1], p2[1])[0][0]
            sum_dists.append(sum)
            # also update membership
            self.membership.append(p1[0])
        # find max distance
        max_dist_idx = sum_dists.index(max(sum_dists))
        # declare the new clusteroid
        if sum_dists[max_dist_idx] > ex_roid_sum:
            self.clusteroid = self.temp[max_dist_idx][1]
        self.temp = []
        print(
            "Consuming took {:.3f}".format(time.time()-starting_tm),
            ":: Key ~>", self.key,
            "membership ", len(self.membership)
        )

    def details(self):
        print("key :", self.key)
        print("clusteroid :", self.clusteroid)
        print("membership :", len(self.membership))


class ComplexCluster(object):
    def __init__(self, key, movie_id, genres, tags, ratings):
        self.key = key
        self.clusteroid_genres = genres
        self.clusteroid_tags = tags
        self.clusteroid_ratings = ratings
        self.membership = [movie_id]
        self.temp = []  # list of points where point = { mid: x, genres: x, tags: x, rating: x }

    def add_temp_point(self, point):
        self.temp.append(point)

    def consume(self):
        if len(self.temp) == 0:
            return
        starting_tm = time.time()
        ex_roid_sum = 0
        sum_dists = []
        for p1 in self.temp:
            d1 = get_jaccard(self.clusteroid_genres, p1['genres'])
            d2 = get_jaccard(self.clusteroid_tags, p1['tags'])
            d3 = cosine_similarity(self.clusteroid_ratings, p1['ratings'])[0][0]
            _dist = 0.33 * d1 + 0.25 * d2 + 0.45 * d3
            ex_roid_sum += _dist
            sum = _dist
            for p2 in self.temp:
                d1 = get_jaccard(p2['genres'], p1['genres'])
                d2 = get_jaccard(p2['tags'], p1['tags'])
                d3 = cosine_similarity(p2['ratings'], p1['ratings'])[0][0]
                sum += 0.33 * d1 + 0.25 * d2 + 0.45 * d3
            sum_dists.append(sum)
            self.membership.append(p1['movie_id'])

        max_dist_idx = sum_dists.index(max(sum_dists))
        if sum_dists[max_dist_idx] > ex_roid_sum:
            self.clusteroid_genres = self.temp[max_dist_idx]['genres']
            self.clusteroid_tags = self.temp[max_dist_idx]['tags']
            self.clusteroid_ratings = self.temp[max_dist_idx]['ratings']
        self.temp = []
        print(
            "Consuming took {:.3f}".format(time.time() - starting_tm),
            ":: Key ~>", self.key,
            "membership ", len(self.membership)
        )


# created in order to be used in hierarchical_cluster
class RemainEntity(object):
    def __init__(self, point1, is_list=False):
        self.members = [point1]  # points: (idx, content)
        self.clusteroid = point1[1]
        self.is_list = is_list

    # calculates the clusteroid
    def refresh(self):
        sum_dists = []
        for member in self.members:
            sum = 0
            for point in self.members:
                sum += get_jaccard(member[1], point[1], self.is_list)
            sum_dists.append(sum)
        self.clusteroid = self.members[sum_dists.index(min(sum_dists))][1]

    # adds a point to the entity
    def add(self, point):
        self.members.append(point)
        self.refresh()

    # merges the entity with another
    def merge(self, c):
        self.members.extend(c.members)
        self.refresh()



