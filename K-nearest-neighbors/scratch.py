# euclidian distance
# sqrt((q[i] - p[i])**2 for i in range(n))

''' Example
q = (1, 3)
p = (2, 5)

sqrt((1 - 2)**2 + (3 - 5)**2)

'''

from math import sqrt

q = [1, 3]
p = [2, 5]

def euclidian_distance(plot1, plot2):
    sum = 0.0
    for i in range(len(plot1)):
        sub = (plot1[i] - plot2[i])
        sub = sub**2
        sum +=sub

    return sqrt(sum)

print(euclidian_distance(q, p))