import time


def file_len(fname):
    start = time.time()
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    print("Count: ", str(i+1), " lines in ", (time.time()-start))
    return i + 1


def get_jaccard(str1, str2, delimiter=","):
    a = set(str1.split(delimiter))
    b = set(str2.split(delimiter))
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))
