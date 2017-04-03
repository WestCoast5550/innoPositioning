import numpy as np
import math
from scipy import stats


#####################Map params###############################
# room size
n = 30
m = 30

# define params of signal propagation distribution

mu = np.zeros((n, m))  # mean
alpha = 3
d0 = 4
p_d0 = -53

x = np.ones((m, 1), int) * range(n)
y = np.transpose(np.ones((n, 1), int) * range(m))
d = np.sqrt((x - 49) ** 2 + (y - 49) ** 2) + 4
mu = p_d0 - 10 * alpha * np.log10(d / d0)

sigma = np.ones((n, m))  # variance
###############################################################

# start point
pos = np.array([n/2, n/2])

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
            motion_prob_map[pos_t[0]][pos_t[1]] = p_a
            acc_map[pos_t[0]][pos_t[1]] = a

    return motion_prob_map, acc_map


def path_estimation_bayes(RSSI, pos, v):
    path_est = []
    path_est.append((n/2, n/2))
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
    acc_map = np.ndarray(shape = (n, m, 2)) #n*m*2 = (n,m) * (a_x, a_y)
    for i in range(0, len(samples[0][0])):
        a = samples[0][i]
        p_a = samples[1][i]
        pos_t = np.around(pos + v * dt + a*dt*dt / 2)
        if (pos_t < np.array([n , m])).all() & (pos_t >= np.array([0 , 0])).all():
            motion_prob_map[pos_t[0]][pos_t[1]] = p_a
            acc_map[pos_t[0]][pos_t[1]][0] = a[0]
            acc_map[pos_t[0]][pos_t[1]][0] = a[1]

    return motion_prob_map, acc_map


def path_estimation_viterbi(RSSI, pos, v):
    path_est = []
    step = []
    #step.append(np.ndarray(shape = (2,n,m))) #2*n*m : (0 = p, 1,2 = (v_x, v_y), (x,y)))
    step.append(np.zeros((2, n, m)))
    step[0][0][n/2][m/2] = 1
    backward = [] #n*m*2: ((x, y), backward = (_x, _y))
    #forward step
    count = 0
    for rssi in RSSI[1:]:
        prev_step = step[-1]
        curr_step = np.zeros((2, n, m)) #np.ndarray(shape = (2,n,m))
        curr_step_backward = np.zeros((n, m, 2)) #np.ndarray(shape = (n,m,2))
        rssi_prob = cond_prob(rssi, mu, sigma)
        #### connect motion matrices
        for i in range(0, n):
            for j in range(0, m):
                m_p, a = motion_map_bayes(np.array([i, j]), n, m, prev_step[1][i][j], mu_a, sigma_a, K)
                res = np.nonzero(m_p)
                for k in range(0, len(res[0])):
                    x = res[0][k]
                    y = res[1][k]
                    trans_prob = m_p[x][y] * prev_step[0][i][j] * rssi_prob[x][y]
                    if (curr_step[0][x][y] < trans_prob):
                        curr_step[0][x][y] = trans_prob
                        curr_step_backward[x][y][0] = i
                        curr_step_backward[x][y][1] = j
                        acc = a[x][y]
                        curr_step[1][x][y] = prev_step[1][i][j] + acc * dt

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

def path_generation(length):
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


def error(path, path_est):
    error = 0
    for i in range(0, len(path)):
        error = error + (path[i][0] - path_est[i][0]) ** 2 + (path[i][1] - path_est[i][1]) ** 2
    error = math.sqrt(error)

    print("Error : " + str(error))


RSSI, path = path_generation(5)
print("Test path:")
print(path)

path_est = path_estimation_viterbi(RSSI, pos, v)
print("Path estimation:")
print(path_est)

error(path, path_est)



