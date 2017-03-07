from collections import namedtuple
import math

distances = {}


def distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)



f = open('testTrajectories.txt', 'r')
Trajectory = namedtuple('Trajectory', ['points'])
Point = namedtuple('Point', ['x', 'y'])
Edge = namedtuple('Edge', ['p1', 'p2'])
trajectory_list = []
dist = 0.0
k = 0
l = 0

# reading from file
for i in range(0, 500):
    t = Trajectory([])
    for line in f:
        if not line.strip():
            if len(t.points) > 0:
                trajectory_list.append(t)
            t = Trajectory([])
        else:
            x1, x2 = line.split(' ')
            point = Point(float(x1), float(x2))
            t.points.append(point)

#creation of list of edges
for i1 in range(0, len(trajectory_list)):
    for i2 in range(min(i + 1, len(trajectory_list)-1), len(trajectory_list)):
        t1 = trajectory_list[i1]
        t2 = trajectory_list[i2]
        for p1 in t1.points:
            for p2 in t2.points:
                edge = Edge(p1, p2)
                distances[edge] = distance(p1, p2)

#estimation of distances
for ii in range(0, len(trajectory_list)):
    for jj in range(ii + 1, len(trajectory_list)):
        t1 = trajectory_list[ii]
        n = len(t1.points)
        t2 = trajectory_list[jj]
        m = len(t2.points)
        distances = [[0 for x in range(0, m)] for y in range(0, n)]
        for p1_i, p1 in enumerate(t1.points):
            for p2_i, p2 in enumerate(t2.points):
                distances[p1_i][p2_i] = distance(p1, p2)
    
#creation of distance matrix
        d_matrix = [[0 for x in range(0, m)] for y in range(0, n)]
        for p1_i, p1 in enumerate(t1.points):
            for p2_i, p2 in enumerate(t2.points):
                d_matrix[p1_i][p2_i] = distances[p1_i][p2_i] + min(d_matrix[p1_i-1][p2_i],
                                                                   d_matrix[p1_i][p2_i-1],
                                                                   d_matrix[p1_i-1][p2_i-1])

#search of minimal path
        dist = 0
        min_path_length = 1e300
        min_path = d_matrix[k][l]
        minn = 0.0
        while k < (len(t1.points) - 1) and l < (len(t2.points) - 1):
                minn = min(d_matrix[k + 1][l + 1], d_matrix[k][l + 1], d_matrix[k + 1][l])
                if minn == d_matrix[k + 1][l + 1]:
                    k = k + 1
                    l = l + 1
                if minn == d_matrix[k][l + 1]:
                    l = l + 1
                if minn == d_matrix[k + 1][l]:
                    k = k + 1
                min_path = min_path + min
        print min_path






