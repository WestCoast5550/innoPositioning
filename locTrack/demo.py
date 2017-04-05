import numpy as np
import math
import pickle
from scipy import stats
from mapget import *
import plotly
import plotly.plotly as py
import plotly.graph_objs as go

'''
building = Building503()
with open('data.pickle', 'rb') as f:
    signal_strength_matrix = pickle.load(f)

for y in range(building._3D_measures[1]-1, -1, -1):
    for x in range(0, building._3D_measures[0]):
        print("%3d" % signal_strength_matrix[x][y][1], end = ' ')
    print("\n")
'''
#for wall in building.all_walls:
#    print(wall)

#####################Map params###############################
# room size
n = 30
m = 30
#n = building._3D_measures[0]
#m = building._3D_measures[1]

# define params of signal propagation distribution


mu = np.zeros((n, m))  # mean
alpha = 3
d0 = 4
p_d0 = -53

x = np.ones((m, 1), int) * range(n)
y = np.transpose(np.ones((n, 1), int) * range(m))
d = np.sqrt((x - 0) ** 2 + (y - 0) ** 2) + 4
mu = p_d0 - 10 * alpha * np.log10(d / d0)

'''
mu = signal_strength_matrix[:,:,1]

######

mu[1] = [0, -60, -63, -66, -55, -63, -68, -50, -73, -67, 0]
mu[2] = [0, -66, -59, -70, -67, -60, -70, -55, -60 ,-78, 0]
mu[3] = [0, -62, -55, -50, -62, -65, -61, -67, -74, -81, 0]
mu[4] = [0, -71, -67, -75, -60, -71, -75, -70, -63, -77, 0]
mu[5] = [0, -70, -77, -64, -74, -66, -63, -81, -55, -90, 0]
mu[6] = [0, -75, -71, -74, -80, -70, -78, -50, -65, -85, 0]

######
'''
# variance
#sigma = np.ones((n, m))
sigma = np.full((n, m), 1)
###############################################################

# start point
pos = np.array([2, 2])

# start velocity
v = np.array([0, 0])

# time interval
dt = 1

# person's motion: acceleration distribution
mu_a = 0
sigma_a = 1

# number of samples
K = 100


def cond_prob(rssi, mu, sigma):
    cond_prob = stats.norm(mu, sigma).pdf(rssi)
    return cond_prob


def sample_generation(mu_a, sigma_a, K):
    a = np.random.normal(mu_a, sigma_a, K)  # can be changed
    p_a = stats.norm(mu_a, sigma_a).pdf(a)
    return [a, p_a]

###############Lines intersection###############################
class pt:
    x = 0
    y = 0

    def __init__(self, x_, y_):
        self.x = x_
        self.y = y_

def area(a, b, c):
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)


def intersect_1(a, b, c, d):
    if (a > b):
        a, b = b, a # swap (a, b)
    if (c > d):
        c, d = d, c # swap (c, d)
    return max(a, c) <= min(b, d)

def intersect (a, b, c, d):
	return intersect_1 (a.x, b.x, c.x, d.x) and intersect_1 (a.y, b.y, c.y, d.y) and area(a,b,c) * area(a,b,d) <= 0 and area(c,d,a) * area(c,d,b) <= 0

def intersect_with_wall(a, b, building):
    for wall in building.all_walls:
        c = pt(wall.p1.x, wall.p1.y)
        d = pt(wall.p2.x, wall.p2.y)
        if intersect(a, b, c, d):
            return True
    return False
###############################################################

###############Bayes solution##################################
def motion_map_bayes(pos, n, m, v, mu_a, sigma_a, K):
    # print("x y:" + str(x) + " " + str(y))
    samples = sample_generation(mu_a, sigma_a, K)
    motion_prob_map = np.zeros((n, m))
    acc_map = np.zeros((n, m))
    for i in range(0, len(samples[0])):
        a = samples[0][i]
        p_a = samples[1][i]
        pos_t = np.around(pos + v * dt + a*dt*dt / 2)
        if (pos_t < np.array([n , m])).all() & (pos_t >= np.array([0 , 0])).all():
        #if not intersect_with_wall(pt(pos[0], pos[1]), pt(pos_t[0], pos_t[1]), building):
            motion_prob_map[pos_t[0]][pos_t[1]] = p_a
            acc_map[pos_t[0]][pos_t[1]] = a

    return motion_prob_map, acc_map


def path_estimation_bayes(RSSI, pos, v):
    path_est = []
    path_est.append((pos[0], pos[1]))
    # print(str(0) + " " + str(0))
    for rssi in RSSI[1:]:
        # print("Velocity: " + str(v))
        rssi_prob = cond_prob(rssi, mu, sigma)
        motion_prob, acceleration = motion_map_bayes(pos, n, m, v, mu_a, sigma_a, K)
        prob = np.multiply(rssi_prob, motion_prob)
        best_samples = np.where(prob == prob.max())  # choose not just bust and first but adequate
        x = best_samples[0][0]
        y = best_samples[1][0]
        pos = np.array([x, y])
        # print(best_samples)
        # print(str(x) + " " + str(y))
        a = acceleration[x][y]
        # print("Acceleration: " + str(a))

        path_est.append((x, y))
        v = v + a * dt

    return path_est
################################################################

####################Viterbi approach############################
def sample_generation_vectors(mu_a, sigma_a, K):
    v1 = np.random.normal(mu_a, sigma_a, K) # can be changed
    v2 = np.random.normal(mu_a, sigma_a, K)
    a = np.ndarray(shape=(K,2))
    for i in range(0, K):
        a[i][0] = v1[i]
        a[i][1] = v2[i]
    p_a = stats.norm(mu_a, sigma_a).pdf(v1) * stats.norm(mu_a, sigma_a).pdf(v2)
    return [a, p_a]

def motion_map_viterbi(pos, n, m, v, mu_a, sigma_a, K):
    # print("x y:" + str(x) + " " + str(y))
    samples = sample_generation_vectors(mu_a, sigma_a, K)
    motion_prob_map = np.zeros((n, m))
    acc_map = np.zeros((n, m, 2)) #n*m*2 = (n,m) * (a_x, a_y)
    for i in range(0, len(samples[0][0])):
        a = samples[0][i]
        p_a = samples[1][i]
        pos_t = np.around(pos + v * dt + a*dt*dt / 2)
        if (pos_t < np.array([n , m])).all() & (pos_t >= np.array([0 , 0])).all():
        #if not intersect_with_wall(pt(pos[0], pos[1]), pt(pos_t[0], pos_t[1]), building):
            motion_prob_map[pos_t[0]][pos_t[1]] = p_a
            acc_map[pos_t[0]][pos_t[1]][0] = a[0]
            acc_map[pos_t[0]][pos_t[1]][1] = a[1]

    return motion_prob_map, acc_map

def motion_map_viterbi_2(pos, n, m, v, sigma_a):
    prev = pos
    pos = pos + v*dt
    distribution = stats.multivariate_normal(mean=pos, cov=[[sigma_a*dt ** 4 / 4,0],[0,sigma_a*dt ** 4 /4]])
    multi_normal = stats.multivariate_normal(mean=[0, 0], cov=[[1,0],[0,1]])
    motion_prob_map = np.zeros((n, m))
    acc_map = np.zeros((n, m, 2))
    sum = 0
    for i in range(max(int(pos[0]) - 5*sigma_a, 0), min(int(pos[0]) + 5*sigma_a, n)):
        for j in range(max(int(pos[1]) - 5*sigma_a, 0), min(int(pos[1]) + 5*sigma_a, m)):
            #if (pos_t < np.array([n , m])).all() & (pos_t >= np.array([0 , 0])).all():
            #if not intersect_with_wall(pt(prev[0], prev[1]), pt(i, j), building):
                motion_prob_map[i][j] = distribution.pdf([i, j])
                sum += motion_prob_map[i][j]
                a = 2*(np.array([i , j]) - pos - v*dt) / dt ** 2
                acc_map[i][j][0] = a[0]
                acc_map[i][j][1] = a[1]

    motion_prob_map = motion_prob_map / sum

    return motion_prob_map, acc_map


def path_estimation_viterbi(RSSI, pos, v):
    path_est = []
    step = []
    #step.append(np.ndarray(shape = (3,n,m))) #2*n*m : (0 = p, 1,2 = (v_x, v_y), (x,y)))
    step.append(np.zeros((3, n, m)))
    step[0][0][pos[0]][pos[1]] = 1
    backward = [] #n*m*2: ((x, y), backward = (_x, _y))
    #forward step
    count = 0
    for rssi in RSSI[1:]:
        prev_step = step[-1]
        curr_step = np.zeros((3, n, m)) #np.ndarray(shape = (3,n,m))
        curr_step_backward = np.zeros((n, m, 2)) #np.ndarray(shape = (n,m,2))
        rssi_prob = cond_prob(rssi, mu, sigma)
        #### connect motion matrices
        for i in range(0, n):
            for j in range(0, m):
                m_p, a = motion_map_viterbi(np.array([i, j]), n, m, np.array([prev_step[1][i][j], prev_step[2][i][j]]), mu_a, sigma_a, K)
                #m_p, a = motion_map_viterbi_2(np.array([i, j]), n, m, np.array([prev_step[1][i][j], prev_step[2][i][j]]), sigma_a)
                trans_prob = np.multiply(m_p, rssi_prob)
                temp = np.full((n,m), prev_step[0][i][j])
                trans_prob = np.multiply(trans_prob, temp)
                sum = np.sum(trans_prob)
                if (sum > 0):
                    trans_prob = trans_prob / sum
                res = np.nonzero(trans_prob)
                for k in range(0, len(res[0])):
                    x = res[0][k]
                    y = res[1][k]
                    #trans_prob = m_p[x][y] * prev_step[0][i][j] * rssi_prob[x][y]
                    if curr_step[0][x][y] < trans_prob[x][y]:
                        curr_step[0][x][y] = trans_prob[x][y]
                        curr_step_backward[x][y][0] = i
                        curr_step_backward[x][y][1] = j
                        acc = a[x][y]
                        curr_step[1][x][y] = prev_step[1][i][j] + acc[0] * dt
                        curr_step[2][x][y] = prev_step[2][i][j] + acc[1] * dt

        step.append(curr_step)
        backward.append(curr_step_backward)
        print(count)
        count += 1

    #backward step
    last_step = step[-1]
    best_samples = np.where(last_step[0] == last_step[0].max())
    x = best_samples[0][0]
    y = best_samples[1][0]
    path_est.append((x, y))
    for i in range(len(backward)-1, -1, -1):
        _x = backward[i][x][y][0]
        _y = backward[i][x][y][1]
        path_est.append((_x, _y))
        x = _x
        y = _y

    return path_est[::-1]

#################################################################

def path_generation_bayes(length):
    path = [(n/2, m/2)]
    rssi = []
    rssi.append(np.random.normal(mu[0][0], sigma[0][0], 1))
    v = 0

    for i in range(1, length):
        a = np.random.normal(0, 1)
        x = int(round(path[-1][0] + v * dt + a*dt*dt / 2, 0))
        y = int(round(path[-1][1] + v * dt + a*dt*dt / 2, 0))
        v = v + a * dt
        # print(str(dx) + " " + str(dy))
        path.append((x, y))
        rssi.append(np.random.normal(mu[x][y], sigma[x][y], 1))

    return rssi, path

def path_generation(length):
    path = [(pos[0], pos[1])]
    rssi = []
    rssi.append(np.random.normal(mu[0][0], sigma[0][0], 1))
    v_x = 0
    v_y = 0

    for i in range(1, length):
        v_x_t = v_x
        v_y_t = v_y
        x = path[-1][0]
        y = path[-1][1]
        a_x = 0
        a_y = 0
        a_x = np.random.normal(0, 1)
        a_y = np.random.normal(0, 1)
        a_x_t = a_x
        a_y_t = a_y
        x = int(round(path[-1][0] + v_x * dt + a_x*dt*dt / 2, 0))
        y = int(round(path[-1][1] + v_y * dt + a_y*dt*dt / 2, 0))
        #if not intersect_with_wall(pt(path[-1][0], path[-1][1]), pt(x, y), building) and x > 0 and x < 7 and y > 0 and y < 10:
        if x >= 0 and x < n and y >= 0 and y < m:
            v_x = v_x + a_x * dt
            v_y = v_y + a_y * dt
            print(1)
            print(str(x) + " " + str(y))
            path.append((x, y))
            rssi.append(np.random.normal(mu[x][y], sigma[x][y], 1))
        else:
            a_x = -a_x_t
            a_y = a_y_t
            v_x = -v_x_t
            v_y = v_y_t
            x = int(round(path[-1][0] + v_x * dt + a_x * dt * dt / 2, 0))
            y = int(round(path[-1][1] + v_y * dt + a_y * dt * dt / 2, 0))
            #if not intersect_with_wall(pt(path[-1][0], path[-1][1]), pt(x, y), building) and x > 0 and x < 7 and y > 0 and y < 10:
            if x >=0 and x < n and y >=0 and y < m:
                v_x = v_x + a_x * dt
                v_y = v_y + a_y * dt
                print(2)
                print(str(x) + " " + str(y))
                path.append((x, y))
                rssi.append(np.random.normal(mu[x][y], sigma[x][y], 1))
            else:
                a_x = -a_x_t
                a_y = -a_y_t
                v_x = -v_x_t
                v_y = -v_y_t
                x = int(round(path[-1][0] + v_x * dt + a_x * dt * dt / 2, 0))
                y = int(round(path[-1][1] + v_y * dt + a_y * dt * dt / 2, 0))
                #if not intersect_with_wall(pt(path[-1][0], path[-1][1]), pt(x, y), building) and x > 0 and x < 7 and y > 0 and y < 10:
                if x >= 0 and x < n and y >= 0 and y < m:
                    v_x = v_x + a_x * dt
                    v_y = v_y + a_y * dt
                    print(3)
                    print(str(x) + " " + str(y))
                    path.append((x, y))
                    rssi.append(np.random.normal(mu[x][y], sigma[x][y], 1))
                else:
                    a_x = a_x_t
                    a_y = -a_y_t
                    v_x = v_x_t
                    v_y = -v_y_t
                    x = int(round(path[-1][0] + v_x * dt + a_x * dt * dt / 2, 0))
                    y = int(round(path[-1][1] + v_y * dt + a_y * dt * dt / 2, 0))
                    #if not intersect_with_wall(pt(path[-1][0], path[-1][1]), pt(x, y), building) and x > 0 and x < 7 and y > 0 and y < 10:
                    if x >= 0 and x < n and y >= 0 and y < m:
                        v_x = v_x + a_x * dt
                        v_y = v_y + a_y * dt
                        print(4)
                        print(str(x) + " " + str(y))
                        path.append((x, y))
                        rssi.append(np.random.normal(mu[x][y], sigma[x][y], 1))
                    else:
                        a_x = 0
                        a_y = 0
                        v_x = v_x_t
                        v_y = v_y_t
                        x = path[-1][0]
                        y = path[-1][1]
                        print(5)
                        print(str(x) + " " + str(y))
                        path.append((x, y))
                        rssi.append(np.random.normal(mu[x][y], sigma[x][y], 1))
        '''
        v_x = v_x + a_x * dt
        v_y = v_y + a_y * dt
        print(str(x) + " " + str(y))
        path.append((x, y))
        rssi.append(np.random.normal(mu[x][y], sigma[x][y], 1))
        '''

    return rssi, path

def error(path, path_est):
    error = 0
    for i in range(0, len(path)):
        error = error + (path[i][0] - path_est[i][0]) ** 2 + (path[i][1] - path_est[i][1]) ** 2
    error = math.sqrt(error)

    print("Error : " + str(error))


RSSI, path = path_generation(6)
print("Test path:")
print(path)
print("RSSI:")
print(RSSI)

path_est = path_estimation_viterbi(RSSI, pos, v)
print("Path estimation:")
print(path_est)

error(path, path_est)

mu[mu == 0] = -45
trace = go.Heatmap(z=mu.transpose())
x = []
y = []
for el in path:
    x.append(el[0])
    y.append(el[1])
trace1 = go.Scatter(name = 'real path', x = x, y = y)
x = []
y = []
for el in path_est:
    x.append(el[0])
    y.append(el[1])
trace2 = go.Scatter(name = 'estimated path', x = x, y = y)
data=[trace, trace1, trace2]
plotly.offline.plot(data, filename='labelled-heatmap.html')


