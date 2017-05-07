from mapGen import *
import pickle
from scipy import stats
import numpy as np
import math
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.graph_objs import *

##############Basic parameters##################################
# time interval
dt = 1

# person's motion: acceleration distribution
mu_a = 0
sigma_a = 1

# number of samples
K = 100
################################################################

##############Statistics calculation############################
def cond_prob(rssi, mu, sigma):
    cond_prob = stats.norm(mu, sigma).pdf(rssi)
    return cond_prob


def sample_generation(mu_a, sigma_a, K):
    a = np.random.normal(mu_a, sigma_a, K)  # can be changed
    p_a = stats.norm(mu_a, sigma_a).pdf(a)
    return [a, p_a]

def sample_generation_vectors(mu_a, sigma_a, K):
    v1 = np.random.normal(mu_a, sigma_a, K) # can be changed
    v2 = np.random.normal(mu_a, sigma_a, K)
    a = np.ndarray(shape=(K,2))
    for i in range(0, K):
        a[i][0] = v1[i]
        a[i][1] = v2[i]
    p_a = stats.norm(mu_a, sigma_a).pdf(v1) * stats.norm(mu_a, sigma_a).pdf(v2)
    return [a, p_a]

def combine_prob(m_p, rssi_prob, boosting = 1e100):
    trans_prob = np.multiply(m_p, rssi_prob)
    boosting_coefficient = boosting / np.sum(trans_prob)
    return trans_prob*boosting_coefficient
################################################################

##############Environment generation############################
class VirtualEnvironment:
    def __init__(self, n, m, x0, y0):
        self.n = n
        self.m = m
        self.building = None

        self.mu =[]
        mu_t = np.zeros((self.n, self.m))  # mean
        alpha = 3
        d0 = 4
        p_d0 = -53
        x = np.ones((self.m, 1), int) * range(self.n)
        y = np.transpose(np.ones((self.n, 1), int) * range(self.m))
        d = np.sqrt((x - x0) ** 2 + (y - y0) ** 2) + d0
        mu_t = p_d0 - 10 * alpha * np.log10(d / d0)
        self.mu.append(mu_t)

        self.sigma = np.full((self.n, self.m), 0.1) #variance

class VirtualEnvironment2AP:
    def __init__(self, n, m, x0, y0, x1, y1):
        self.n = n
        self.m = m
        self.building = None
        self.sigma = np.full((self.n, self.m), 0.01) #variance
        self.mu = []

        mu1 = np.zeros((self.n, self.m))
        mu2 = np.zeros((self.n, self.m))
        alpha = 3
        d0 = 4
        p_d0 = -53
        x = np.ones((self.m, 1), int) * range(self.n)
        y = np.transpose(np.ones((self.n, 1), int) * range(self.m))
        #1
        d = np.sqrt((x - x0) ** 2 + (y - y0) ** 2) + d0
        mu1 = p_d0 - 10 * alpha * np.log10(d / d0)
        #2
        d = np.sqrt((x - x1) ** 2 + (y - y1) ** 2) + d0
        mu2 = p_d0 - 10 * alpha * np.log10(d / d0)

        self.mu.append(mu1)
        self.mu.append(mu2)

class RealEnvironment:
    def __init__(self, Building, file):
        self.building = Building()
        with open(file, 'rb') as f:
            signal_strength_matrix = pickle.load(f)

        self.n = self.building._3D_measures[0]
        self.m = self.building._3D_measures[1]
        self.mu = signal_strength_matrix[:, :, 2] #mean
        self.sigma = np.full((self.n, self.m), 0.01) #variance

################################################################

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

###################Path generation#############################
def path_generation(length, pos_0, v_0, env):
    n = env.n
    m = env.m
    mu = env.mu
    sigma = env.sigma
    building = env.building

    path = [(pos_0[0], pos_0[1])]
    rssi = []
    rssi.append(((np.random.normal(mu[pos_0[0]][pos_0[1]], sigma[pos_0[0]][pos_0[1]], 1)), 0))
    v_x = v_0[0]
    v_y = v_0[1]

    for i in range(1, length):
        print(i)
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
        #if not intersect_with_wall(pt(path[-1][0], path[-1][1]), pt(x, y), building): #and x > 0 and x < 7 and y > 0 and y < 10:
        if x >= 0 and x < n and y >= 0 and y < m:
            v_x = v_x + a_x * dt
            v_y = v_y + a_y * dt
            print(str(x) + " " + str(y))
            print(str(v_x) + " " + str(v_y))
            path.append((x, y))
            rssi.append(((np.random.normal(mu[x][y], sigma[x][y], 1)), 0))
        else:
            a_x = -a_x_t
            a_y = a_y_t
            v_x = -v_x_t
            v_y = v_y_t
            x = int(round(path[-1][0] + v_x * dt + a_x * dt * dt / 2, 0))
            y = int(round(path[-1][1] + v_y * dt + a_y * dt * dt / 2, 0))
            #if not intersect_with_wall(pt(path[-1][0], path[-1][1]), pt(x, y), building): #and x > 0 and x < 7 and y > 0 and y < 10:
            if x >=0 and x < n and y >=0 and y < m:
                v_x = v_x + a_x * dt
                v_y = v_y + a_y * dt
                print(str(x) + " " + str(y))
                print(str(v_x) + " " + str(v_y))
                path.append((x, y))
                rssi.append(((np.random.normal(mu[x][y], sigma[x][y], 1)), 0))
            else:
                a_x = -a_x_t
                a_y = -a_y_t
                v_x = -v_x_t
                v_y = -v_y_t
                x = int(round(path[-1][0] + v_x * dt + a_x * dt * dt / 2, 0))
                y = int(round(path[-1][1] + v_y * dt + a_y * dt * dt / 2, 0))
                #if not intersect_with_wall(pt(path[-1][0], path[-1][1]), pt(x, y), building): #and x > 0 and x < 7 and y > 0 and y < 10:
                if x >= 0 and x < n and y >= 0 and y < m:
                    v_x = v_x + a_x * dt
                    v_y = v_y + a_y * dt
                    print(str(x) + " " + str(y))
                    print(str(v_x) + " " + str(v_y))
                    path.append((x, y))
                    rssi.append(((np.random.normal(mu[x][y], sigma[x][y], 1)), 0))
                else:
                    a_x = a_x_t
                    a_y = -a_y_t
                    v_x = v_x_t
                    v_y = -v_y_t
                    x = int(round(path[-1][0] + v_x * dt + a_x * dt * dt / 2, 0))
                    y = int(round(path[-1][1] + v_y * dt + a_y * dt * dt / 2, 0))
                    #if not intersect_with_wall(pt(path[-1][0], path[-1][1]), pt(x, y), building): #and x > 0 and x < 7 and y > 0 and y < 10:
                    if x >= 0 and x < n and y >= 0 and y < m:
                        v_x = v_x + a_x * dt
                        v_y = v_y + a_y * dt
                        print(str(x) + " " + str(y))
                        print(str(v_x) + " " + str(v_y))
                        path.append((x, y))
                        rssi.append(((np.random.normal(mu[x][y], sigma[x][y], 1)), 0))
                    else:
                        a_x = 0
                        a_y = 0
                        v_x = v_x_t
                        v_y = v_y_t
                        x = path[-1][0]
                        y = path[-1][1]
                        print(str(x) + " " + str(y))
                        print(str(v_x) + " " + str(v_y))
                        path.append((x, y))
                        rssi.append(((np.random.normal(mu[x][y], sigma[x][y], 1)), 0))

    return rssi, path

def path_generation2AP(length, pos_0, v_0, env):
    n = env.n
    m = env.m

    mu1 = env.mu[0]
    sigma = env.sigma
    #building = env.building
    mu2 = env.mu[1]
    # building = env.building

    path = [(pos_0[0], pos_0[1])]
    rssi = []
    rssi.append(((np.random.normal(mu1[pos_0[0]][pos_0[1]], sigma[pos_0[0]][pos_0[1]], 1)), 0))
    v_x = v_0[0]
    v_y = v_0[1]

    for i in range(1, length):
        print(i)
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
        #if not intersect_with_wall(pt(path[-1][0], path[-1][1]), pt(x, y), building): #and x > 0 and x < 7 and y > 0 and y < 10:
        if x >= 0 and x < n and y >= 0 and y < m:
            v_x = v_x + a_x * dt
            v_y = v_y + a_y * dt
            print(str(x) + " " + str(y))
            print(str(v_x) + " " + str(v_y))
            path.append((x, y))
            if (x+y <= n-1):
                rssi.append(((np.random.normal(mu1[x][y], sigma[x][y], 1)), 0))
            else:
                rssi.append(((np.random.normal(mu2[x][y], sigma[x][y], 1)), 1))
        else:
            a_x = -a_x_t
            a_y = a_y_t
            v_x = -v_x_t
            v_y = v_y_t
            x = int(round(path[-1][0] + v_x * dt + a_x * dt * dt / 2, 0))
            y = int(round(path[-1][1] + v_y * dt + a_y * dt * dt / 2, 0))
            #if not intersect_with_wall(pt(path[-1][0], path[-1][1]), pt(x, y), building): #and x > 0 and x < 7 and y > 0 and y < 10:
            if x >=0 and x < n and y >=0 and y < m:
                v_x = v_x + a_x * dt
                v_y = v_y + a_y * dt
                print(str(x) + " " + str(y))
                print(str(v_x) + " " + str(v_y))
                path.append((x, y))
                if (x + y <= n - 1):
                    rssi.append(((np.random.normal(mu1[x][y], sigma[x][y], 1)), 0))
                else:
                    rssi.append(((np.random.normal(mu2[x][y], sigma[x][y], 1)), 1))
            else:
                a_x = -a_x_t
                a_y = -a_y_t
                v_x = -v_x_t
                v_y = -v_y_t
                x = int(round(path[-1][0] + v_x * dt + a_x * dt * dt / 2, 0))
                y = int(round(path[-1][1] + v_y * dt + a_y * dt * dt / 2, 0))
                #if not intersect_with_wall(pt(path[-1][0], path[-1][1]), pt(x, y), building): #and x > 0 and x < 7 and y > 0 and y < 10:
                if x >= 0 and x < n and y >= 0 and y < m:
                    v_x = v_x + a_x * dt
                    v_y = v_y + a_y * dt
                    print(str(x) + " " + str(y))
                    print(str(v_x) + " " + str(v_y))
                    path.append((x, y))
                    if (x + y <= n - 1):
                        rssi.append(((np.random.normal(mu1[x][y], sigma[x][y], 1)), 0))
                    else:
                        rssi.append(((np.random.normal(mu2[x][y], sigma[x][y], 1)), 1))
                else:
                    a_x = a_x_t
                    a_y = -a_y_t
                    v_x = v_x_t
                    v_y = -v_y_t
                    x = int(round(path[-1][0] + v_x * dt + a_x * dt * dt / 2, 0))
                    y = int(round(path[-1][1] + v_y * dt + a_y * dt * dt / 2, 0))
                    #if not intersect_with_wall(pt(path[-1][0], path[-1][1]), pt(x, y), building): #and x > 0 and x < 7 and y > 0 and y < 10:
                    if x >= 0 and x < n and y >= 0 and y < m:
                        v_x = v_x + a_x * dt
                        v_y = v_y + a_y * dt
                        print(str(x) + " " + str(y))
                        print(str(v_x) + " " + str(v_y))
                        path.append((x, y))
                        if (x + y <= n - 1):
                            rssi.append(((np.random.normal(mu1[x][y], sigma[x][y], 1)), 0))
                        else:
                            rssi.append(((np.random.normal(mu2[x][y], sigma[x][y], 1)), 1))
                    else:
                        a_x = 0
                        a_y = 0
                        v_x = v_x_t
                        v_y = v_y_t
                        x = path[-1][0]
                        y = path[-1][1]
                        print(str(x) + " " + str(y))
                        print(str(v_x) + " " + str(v_y))
                        path.append((x, y))
                        if (x + y <= n - 1):
                            rssi.append(((np.random.normal(mu1[x][y], sigma[x][y], 1)), 0))
                        else:
                            rssi.append(((np.random.normal(mu2[x][y], sigma[x][y], 1)), 1))

    return rssi, path

def error(path, path_est):
    error = 0
    for i in range(0, len(path)):
        error = error + (path[i][0] - path_est[i][0]) ** 2 + (path[i][1] - path_est[i][1]) ** 2
    error = math.sqrt(error) / 10

    print("Error : " + str(error))
###############################################################

def plot(mu, path, path_est):
    mu[mu == 0] = -35
    trace = go.Heatmap(z=mu.transpose())
    x = []
    y = []
    for el in path:
        x.append(el[0])
        y.append(el[1])
    trace1 = go.Scatter(name='real path', x=x, y=y)
    x = []
    y = []
    for el in path_est:
        x.append(el[0])
        y.append(el[1])
    trace2 = go.Scatter(name='estimated path', x=x, y=y)
    #data = [trace, trace1, trace2]

    data = Data([
        Contour(
            z = mu
        ),
        trace1, trace2
    ])
    plotly.offline.plot(data, filename='labelled-heatmap.html')
