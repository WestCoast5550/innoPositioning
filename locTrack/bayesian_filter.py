from basic_finctions import *
from mapGen import *

def motion_map_scalar(pos, n, m, v, mu_a, sigma_a, K, building):
    samples = sample_generation(mu_a, sigma_a, K)
    motion_prob_map = np.zeros((n, m))
    acc_map = np.zeros((n, m))
    for i in range(0, len(samples[0])):
        a = samples[0][i]
        p_a = samples[1][i]
        pos_t = np.around(pos + v * dt + a*dt*dt / 2)
        if building is None:
            if (pos_t < np.array([n , m])).all() & (pos_t >= np.array([0 , 0])).all():
                motion_prob_map[pos_t[0]][pos_t[1]] = p_a
                acc_map[pos_t[0]][pos_t[1]] = a
        else:
            if not intersect_with_wall(pt(pos[0], pos[1]), pt(pos_t[0], pos_t[1]), building):
                motion_prob_map[pos_t[0]][pos_t[1]] = p_a
                acc_map[pos_t[0]][pos_t[1]] = a

    return motion_prob_map, acc_map

def motion_map_vectors(pos, n, m, v, mu_a, sigma_a, K, building):
    a, p_a = sample_generation_vectors(mu_a, sigma_a, K)
    motion_prob_map = np.zeros((n, m))
    acc_map = np.zeros((n, m,2))
    for i in range(0, len(p_a)):
        pos_t = np.around(pos + v * dt + a[i]*dt*dt / 2)
        if building is None:
            if (pos_t < np.array([n , m])).all() & (pos_t >= np.array([0 , 0])).all():
                motion_prob_map[pos_t[0]][pos_t[1]] = p_a[i]
                acc_map[pos_t[0]][pos_t[1]][0] = a[i][0]
                acc_map[pos_t[0]][pos_t[1]][1] = a[i][1]
        else:
            if not intersect_with_wall(pt(pos[0], pos[1]), pt(pos_t[0], pos_t[1]), building):
                motion_prob_map[pos_t[0]][pos_t[1]] = p_a[i]
                acc_map[pos_t[0]][pos_t[1]][0] = a[i][0]
                acc_map[pos_t[0]][pos_t[1]][1] = a[i][1]

    return motion_prob_map, acc_map


def BayesianFilter(RSSI, env, path, pos, v):
    n = env.n
    m = env.m
    mu = env.mu
    sigma = env.sigma
    building = env.building

    path_est = []
    path_est.append((pos[0], pos[1]))

    for el in RSSI[1:]:
        rssi = el[0][0]
        AP = el[1]
        rssi_prob = cond_prob(rssi, mu[AP], sigma)
        motion_prob, acceleration = motion_map_vectors(pos, n, m, v, mu_a, sigma_a, K, building)
        prob = np.multiply(rssi_prob, motion_prob)
        best_samples = np.where(prob == prob.max())
        x = best_samples[0][0]
        y = best_samples[1][0]
        pos = np.array([x, y])
        a = acceleration[x][y]

        path_est.append((x, y))
        v = v + a * dt

    print("Path estimation:")
    print(path_est)
    error(path, path_est)

    mu = np.zeros((n, m))
    for map in env.mu:
        mu = mu + map

    plot(mu, path, path_est)
