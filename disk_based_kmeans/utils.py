import time


def file_len(fname):
    start = time.time()
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    print("Count: ", str(i+1), " lines in ", (time.time()-start))
    return i + 1


def get_jaccard(str1, str2, delimiter="|"):
    # print("JC: ", str1, "---", str2)
    a = set(str1.split(delimiter))
    b = set(str2.split(delimiter))
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
        # print("new iter", iterations)

    # for c in remained:
    #     print("** ", c.clusteroid, " len ", len(c.members))
    #     print(c.members)
    #     print("-----------")
    # print("Hierarchical took {:.3f}".format(time.time() - starting_tm))
    # print("Iterations : ", iterations)
    # print("created clusters ", len(remained))

    return remained






