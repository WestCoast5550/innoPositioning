
import numpy as np
import math
from scipy import stats


# ## Input data

#room size
n = 100
m = 100

#define params of signal propagation distribution

mu = np.zeros((n, m)) # mean
alpha = 3
d0 = 4
p_d0 = -53

x = np.ones((m,1),int)*range(n)
y = np.transpose(np.ones((n,1),int)*range(m))
d = np.sqrt((x-49)**2 + (y-49)**2)+4
mu = p_d0 - 10 * alpha * np.log10(d / d0)

sigma = np.ones((n, m)) #variance

#start point
x0 = 0
y0 = 0

#start velocity
v0 = 0
#time interval
dt = 1


# ## Positioning Algorithm

# Some model parameters:

#person's motion: acceleration distribution
mu_a = 0
sigma_a = 1

#number of samples
K = 100


# Supplementary functions:

def cond_prob(rssi, mu, sigma):     
    cond_prob = stats.norm(mu, sigma).pdf(rssi)
    return cond_prob

def sample_generation(mu_a, sigma_a, K):
    a = np.random.normal(mu_a, sigma_a, K) #can be changed
    p_a = stats.norm(mu_a, sigma_a).pdf(a)   
    return [a, p_a]

def motion_map(x, y, n, m, v, mu_a, sigma_a, K):
    #print("x y:" + str(x) + " " + str(y))
    samples_x  = sample_generation(mu_a, sigma_a, K)
    samples_y  = sample_generation(mu_a, sigma_a, K)
    motion_prob_map = np.zeros((n, m))
    acc_map = np.zeros((n, m))
    for i in range(0, len(samples_x[0])):
        a_x = samples_x[0][i]
        a_y = samples_y[0][i]
        p_a = samples_x[1][i] * samples_y[1][i]
        x_t = int(round(x + v*dt + a*a*dt/2, 0))
        y_t = int(round(y + v*dt + a*a*dt/2, 0))
        if (x_t < n and y_t < m and x_t >= 0 and y_t >= 0): 
            motion_prob_map[x_t][y_t] = p_a
            acc_map[x_t][y_t] = a
        
    return motion_prob_map, acc_map


# ### Path estimation

# #### First Bayes approach

def path_estimation_bayes(RSSI, x, y, v):
    path_est = []
    path_est.append((0,0))
    #print(str(0) + " " + str(0))
    for rssi in RSSI:
        print("Velocity: " + str(v))
        rssi_prob = cond_prob(rssi, mu, sigma)
        motion_prob, acceleration = motion_map(x, y, n, m, v, mu_a, sigma_a, K)
        prob = np.multiply(rssi_prob, motion_prob)
        best_samples = np.where(prob==prob.max()) #choose not just bust and first but adequate
        x = best_samples[0][0]
        y = best_samples[1][0]
        #print(best_samples)
        print(str(x) + " " + str(y))
        a = acceleration[x][y]
        #print("Acceleration: " + str(a))
        
        path_est.append((x, y))
        v = v + a*dt
        
    return path_est


# #### Viterbi approach

class Cell:
    x = 0
    y = 0
    prob = 0
    v = 0
    
    def __init__(self, x, y, prob, v):
        self.x = x
        self.y = y
        self.prob = prob
        self.v = v

def path_estimation_viterbi(RSSI, x, y, v):
    path_est = []
    steps = []
    steps.append([Cell(0,0,1,0)])
    for rssi in RSSI:
        print("rssi: "+ str(rssi))
        rssi_prob = cond_prob(rssi, mu, sigma)
        current_step = steps[-1]
        next_step = []
        for cell in current_step:
            motion_prob_map, acc_map = motion_map(cell.x, cell.y, n, m, cell.v, mu_a, sigma_a, K)
            prob = np.multiply(rssi_prob, motion_prob_map)
            non_zero_cells = np.nonzero(prob)
            for i in range(0, len(non_zero_cells[0])):
                x = non_zero_cells[0][i]
                y = non_zero_cells[1][i]
                p = prob[x][y]
                a = acc_map[x][y]
                v = cell.v + a*dt
                next_step.append(Cell(x,y,p,v))
        print(len(next_step))
        steps.append(next_step)    


# ## Test generation

def path_generation(length):
    path = [(0,0)]
    rssi = []
    rssi.append(np.random.normal(mu[0][0], sigma[0][0], 1))
    v = 0

    for i in range(1, length):
        a = np.random.normal(1, 1)
        dx = np.random.randint(0, 3) #change to generation with acceleration
        dy = np.random.randint(0, 3)
        x = int(round(path[-1][0] + v*dt + a*a*dt/2, 0))
        y = int(round(path[-1][1] + v*dt + a*a*dt/2, 0))
        v = v + a*dt
        #print(str(dx) + " " + str(dy))
        path.append((x,y))
        rssi.append(np.random.normal(mu[x][y], sigma[x][y], 1))
        
    return rssi, path

N = 5
paths = []

for i in range(0,5):
    paths.append(path_generation(5))


# ## Model evaluation

def error(path, path_est):
    error = 0
    for i in range(0, len(path)):
        error = error + (path[i][0] - path_est[i][0])**2 + (path[i][1] - path_est[i][1])**2
    error = math.sqrt(error)

    print("Error : " + str(error))


RSSI, path = path_generation(4)

path_est = path_estimation_bayes(RSSI, x0, y0, v0)

error(path, path_est)

