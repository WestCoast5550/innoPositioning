import numpy as np
import pandas as pd
from collections import namedtuple
import random
import sys
import math
from sklearn.metrics import classification_report
from scipy import sparse

import pandas as pd
import numpy as np
import matplotlib.pylab as plt


curr = []
arr = []

f = open('train.txt', 'r')
for line in f:
    if len(line) > 2:
        nums = [float(x) for x in line.strip().split(" ")]
        curr.append(nums)
    else:
        arr.append(curr)
        curr = []
data = np.array(arr)


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def mean(p1, p2):
    return Point((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

def _dtw_distance(ts_a, ts_b, d = lambda x,y: abs(x-y)):
    
    # Create cost matrix via broadcasting with large int
    #ts_a, ts_b = np.array(ts_a), np.array(ts_b)
    M, N = len(ts_a), len(ts_b)
    cost = sys.maxint * np.ones((M, N))

    k, l, minn = 0, 0 ,0
    # Initialize the first row and column
    cost[0, 0] = d(ts_a[0], ts_b[0])
    for i in xrange(1, M):
        cost[i, 0] = cost[i-1, 0] + d(ts_a[i], ts_b[0])
    
    for j in xrange(1, N):
        cost[0, j] = cost[0, j-1] + d(ts_a[0], ts_b[j])
    
    # Populate rest of cost matrix within window
    for i in xrange(1, M):
        for j in xrange(max(1, i - 10),
                        min(N, i + 10)):
            choices = cost[i - 1, j - 1], cost[i, j-1], cost[i-1, j]
            cost[i, j] = min(choices) + d(ts_a[i], ts_b[j])

    # Return DTW distance given window
    return cost[-1, -1]

def centrate(ts_a, ts_b):
    M, N = len(ts_a), len(ts_b)
    cost = sys.maxint * np.ones((M, N))
    new_trajectory = []
    k, l, minn = 0, 0, 0
    # Initialize the first row and column

    cost[0, 0] = d(ts_a[0], ts_b[0])
    for i in xrange(1, M):
        cost[i, 0] = cost[i-1, 0] + d(ts_a[i], ts_b[0])

    for j in xrange(1, N):
        cost[0, j] = cost[0, j-1] + d(ts_a[0], ts_b[j])
    
    # Populate rest of cost matrix within window
    for i in xrange(1, M):
        for j in xrange(max(1, i - 10),
                        min(N, i + 10)):
            choices = cost[i - 1, j - 1], cost[i, j-1], cost[i-1, j]
            cost[i, j] = min(choices) + d(ts_a[i], ts_b[j])

    while k < N - 1 and l < M - 1:
        minn = min(cost[k + 1][l + 1], cost[k][l + 1], cost[k + 1][l])
        if minn == cost[k + 1][l + 1]:
            k = k + 1
            l = l + 1
            new_trajectory.append(mean(ts_a[k + 1], ts_b[l + 1]))
        elif minn == cost[k][l + 1]:
            l = l + 1
            new_trajectory.append(mean(ts_a[k], ts_b[l + 1]))
        elif minn == cost[k + 1][l]:
            k = k + 1
            new_trajectory.append(mean(ts_a[k + 1], ts_b[l]))

    return new_trajectory





def k_means_clust(data, num_clust, num_iter, w = 5):
    centroids = random.sample(data,num_clust)
    counter = 0
    new_data = []
    old_data = []
    i = 0
    print len(centroids)
    for n in range(num_iter):
        counter += 1
        assignments = {}
        #assign data points to clusters
        for ind, i in enumerate(data):
            print len(i), ind
            min_dist = float('inf')
            closest_clust = None
            for c_ind, j in enumerate(centroids):
                cur_dist = _dtw_distance(i, j, distance)
                if cur_dist < min_dist:
                    min_dist = cur_dist
                    closest_clust = c_ind
            if closest_clust in assignments:
                assignments[closest_clust].append(ind)
            else:
                assignments[closest_clust] = []
    
        #recalculate centroids of clusters
        for key in assignments:
            clust_sum = 0
            old_data = assignments[key]
            while(len(new_data) > 1):
                while(i + 1 < len(old_data)):
                    new_data.append(centrate(old_data[i], old_data[j]))
                old_data = new_data
                new_data = []
            centroids[key] = new_data

    return centroids

centroids=k_means_clust(data,4,10,4)
for i in centroids:
    
    plt.plot(i)

plt.show()