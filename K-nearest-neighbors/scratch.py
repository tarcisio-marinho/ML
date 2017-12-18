# euclidian distance
# sqrt((q[i] - p[i])**2 for i in range(n))

''' Example
q = (1, 3)
p = (2, 5)

sqrt((1 - 2)**2 + (3 - 5)**2)

'''

from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import warnings

style.use("fivethirtyeight")


def euclidian_distance(plot1, plot2):
    sum = 0.0
    for i in range(len(plot1)):
        sub = (plot1[i] - plot2[i])
        sub = sub**2
        sum +=sub

    return sqrt(sum)

def knn(data, predict, k=3):
    if(len(data) >= k):
        warnings.warn("k is set to a value less than total voting groups.")
    
    distances = []

    for group in data:
        for features in data[group]:
            ed = euclidian_distance(features, predict)
            distances.append([ed, group])

    votes = [i[1] for i in sorted(distances) [:k]]
    vote_result = Counter(votes).most_common(1)[0][0]

    return vote_result

def test():
    dataset = {'black' : [[1, 2], [2, 3], [3, 1]], 'red':[[6, 5], [7, 7], [8, 6]]}
    new_features = [2.5, 2.5]
    result = knn(dataset, new_features, k=3)
    print("the new feature is near to group: " + result)

    for i in dataset:
        for ii in dataset[i]:
            plt.scatter(ii[0], ii[1], s=100, color=i)
    
    plt.scatter(new_features[0], new_features[1])
    plt.show()

def cancer_test():
    pass


if __name__ == "__main__":
    test()

    
