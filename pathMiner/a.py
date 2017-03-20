import numpy as np
import pandas as pd
from collections import namedtuple
import random
import sys
import math
from sklearn.metrics import classification_report

def distance(p1, p2):
    return math.sqrt((p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

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
train = np.array(arr)


curr = []
arr = []

f = open('test.txt', 'r')
for line in f:
    if len(line) > 2:
        nums = [float(x) for x in line.strip().split(" ")]
        curr.append(nums)
    else:
        arr.append(curr)
        curr = []
test = np.array(arr)

def _dtw_distance(ts_a, ts_b, d = lambda x,y: abs(x-y)):
    # Create cost matrix via broadcasting with large int
    #ts_a, ts_b = np.array(ts_a), np.array(ts_b)
    M, N = len(ts_a), len(ts_b)
    cost = sys.maxint * np.ones((M, N))
    # Initialize the first row and column
    cost[0, 0] = d(ts_a[1], ts_b[1])
    for i in xrange(2, M):
        cost[i, 0] = cost[i-1, 0] + d(ts_a[i], ts_b[1])
    
    for j in xrange(2, N):
        cost[0, j] = cost[0, j-1] + d(ts_a[1], ts_b[j])
    
    # Populate rest of cost matrix within window
    for i in xrange(1, M):
        for j in xrange(max(1, i - 10),
                        min(N, i + 10)):
            choices = cost[i - 1, j - 1], cost[i, j-1], cost[i-1, j]
            cost[i, j] = min(choices) + d(ts_a[i], ts_b[j])

    # Return DTW distance given window
    return cost[-1, -1]



def knn(train,test,w):
    preds = []
    for i in enumerate(test):
        min_dist = float('inf')
        closest_seq = []
        #print i
        for j in train:
            dist = _dtw_distance(i, j, distance)
            if dist < min_dist:
                min_dist = dist
                closest_seq = j
        preds.append(closest_seq)
    return classification_report(test, preds)

print knn(train,test,4)