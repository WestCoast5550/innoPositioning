import random
from collections import namedtuple


f = open('originTrajectories.txt', 'r')
Distance = namedtuple('Distance', ['x', 'y'])
DistanceList = []
dist = 0.0

#file reading
for i in range(0, 5):
    x = []
    y = []
    for line in f:
        if not line.strip():
            break
        else:
            x1, y1 = line.split(' ')
            x.append(float(x1))
            y.append(float(y1))
    DistanceList.append(Distance(x, y))

f = open('testTrajectories.txt', 'w')
#generates test data
for j in range(len(DistanceList)):
    for k in range(0, 100):
        for i in range(len(DistanceList[j].x)):

            #adding noise
            noiseX = random.uniform(-0.1, 0.1)
            noiseY = random.uniform(-0.1, 0.1)
            f.write(repr(DistanceList[j].x[i] + noiseX) + ' ' + repr(DistanceList[j].y[i] + noiseY) + '\n')
        f.write('\n')