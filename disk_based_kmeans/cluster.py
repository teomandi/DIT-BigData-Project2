

class Cluster(object):

    def __init__(self, key, content, index):

        self.key = key
        self.clusteroid = content
        self.membership = [index]

        # list the temporary items are kept here
        self.temp = []  # list with tuples (id, content)

    def add_temp_point(self, content, idx):
        self.temp.append((idx, content))

    def consume(self):

        # for each point in the temp get the distance with every other point and add it

        # find the point with the minimum dinstance

        # declare it as clusteroid




