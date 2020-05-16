from utils import file_len, get_jaccard
import time


class Cluster(object):

    def __init__(self, key, index, content):
        self.key = key
        self.clusteroid = content
        self.membership = [index]

        # list the temporary items are kept here
        self.temp = []  # list with tuples (id, content)

    def add_temp_point(self, content, idx):
        self.temp.append((idx, content))

    def consume(self):
        # print(self.key, " --- ", self.clusteroid, "- temp: ", len(self.temp))
        if len(self.temp) == 0:
            return
        starting_tm = time.time()
        # get the sum of the distances
        ex_roid_sum = 0
        sum_dists = []
        for p1 in self.temp:
            j_dist = get_jaccard(p1[1], self.clusteroid)
            # sum of the current clusteroid
            ex_roid_sum += j_dist
            # sum for each temp point
            sum = j_dist
            for p2 in self.temp:
                sum += get_jaccard(p1[1], p2[1])
            sum_dists.append(sum)
            # also update membership
            self.membership.append(p1[0])

        # find minimum distance
        min_dist_idx = sum_dists.index(min(sum_dists))
        # declare the new clusteroid
        if sum_dists[min_dist_idx] < ex_roid_sum:
            self.clusteroid = self.temp[min_dist_idx][1]

        # empty the temp-list
        self.temp = []
        # print("Consuming took {:.3f}".format(time.time()-starting_tm), ":: new ~> ", self.clusteroid, " membership ", len(self.membership))


    def details(self):
        print("key :", self.key)
        print("clusteroid :", self.clusteroid)
        print("membership :", len(self.membership))


class RemainEntity(object):
    def __init__(self, point1):
        self.members = [point1]  # points: (idx, content)
        self.clusteroid = point1[1]

    # calculates the clusteroid
    def refresh(self):
        sum_dists = []
        for member in self.members:
            sum = 0
            for point in self.members:
                sum += get_jaccard(member[1], point[1])
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



