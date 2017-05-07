from basic_finctions import *
from mapGen import *

def motion_map_samples(pos, n, m, v, mu_a, sigma_a, K, building):
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

def motion_map_distribution(pos, n, m, v, sigma_a, building, boosting = 1e100):
    prev = pos
    distribution = stats.multivariate_normal(mean=pos+v*dt, cov=[[sigma_a*dt ** 4 / 4,0],[0,sigma_a*dt ** 4 /4]])
    motion_prob_map = np.zeros((n, m))
    acc_map = np.zeros((n, m, 2))
    sum = 0
    for i in range(max(int(pos[0]) - 5*sigma_a, 0), min(int(pos[0]) + 5*sigma_a, n)):
        for j in range(max(int(pos[1]) - 5*sigma_a, 0), min(int(pos[1]) + 5*sigma_a, m)):
            #if not intersect_with_wall(pt(prev[0], prev[1]), pt(i, j), building):
                a = 2*(np.array([i , j]) - pos - v*dt) / dt ** 2
                acc_map[i][j][0] = a[0]
                acc_map[i][j][1] = a[1]
                motion_prob_map[i][j] = distribution.pdf([i, j])
                sum += motion_prob_map[i][j]

    boosting_coefficient = 0
    if sum > 0:
        boosting_coefficient = boosting / sum
    motion_prob_map = motion_prob_map * boosting_coefficient

    return motion_prob_map, acc_map


def motion_map_rotation(pos, n, m, v, sigma_a, building, boosting = 1e100):
    prev = pos
    distribution = stats.multivariate_normal(mean=pos+v*dt, cov=[[sigma_a*dt ** 4 / 4,0],[0,sigma_a*dt ** 4 /4]])

    motion_prob_map = np.zeros((n, m))
    acc_map = np.zeros((n, m, 2))
    sum = 0
    for i in range(max(int(pos[0]) - 5*sigma_a, 0), min(int(pos[0]) + 5*sigma_a, n)):
        for j in range(max(int(pos[1]) - 5*sigma_a, 0), min(int(pos[1]) + 5*sigma_a, m)):
            #if not intersect_with_wall(pt(prev[0], prev[1]), pt(i, j), building):
                a = 2*(np.array([i , j]) - pos - v*dt) / dt ** 2
                acc_map[i][j][0] = a[0]
                acc_map[i][j][1] = a[1]

                mod_a = np.sqrt(a.dot(a))
                mod_v = np.sqrt(v.dot(v))
                scalar = np.dot(a, v)
                ratio = 0
                if (mod_v * mod_a > 0):
                    ratio = scalar / (mod_a * mod_v)
                phi = math.acos(min(1,max(ratio,-1)))
                angle_prob = stats.norm(0, 0.5).pdf(phi)

                motion_prob_map[i][j] = distribution.pdf([i, j]) * angle_prob
                sum += motion_prob_map[i][j]

    boosting_coefficient = 0
    if sum > 0:
        boosting_coefficient = boosting / sum
    motion_prob_map = motion_prob_map * boosting_coefficient

    return motion_prob_map, acc_map

def Viterbi(RSSI, env, path, pos, v):
    n = env.n
    m = env.m
    mu = env.mu
    sigma = env.sigma
    building = env.building

    path_est = []
    step = []
    #step.append(np.ndarray(shape = (3,n,m)))
    step.append(np.zeros((3, n, m))) #2*n*m : (0 = p, 1,2 = (v_x, v_y), (x,y)))
    step[0][0][pos[0]][pos[1]] = 1
    step[0][1][pos[0]][pos[1]] = v[0]
    step[0][2][pos[0]][pos[1]] = v[1]
    backward = [] #n*m*2: ((x, y), backward = (_x, _y))

    #forward step
    count = 0
    for el in RSSI[1:]:
        rssi = el[0][0]
        AP = el[1]
        prev_step = step[-1]
        curr_step = np.zeros((3, n, m)) #np.ndarray(shape = (3,n,m))
        curr_step_backward = np.zeros((n, m, 2)) #np.ndarray(shape = (n,m,2))
        rssi_prob = cond_prob(rssi, mu[AP], sigma)
        #### connect motion matrices
        for i in range(0, n):
            for j in range(0, m):
                m_p, a = motion_map_rotation(np.array([i, j]), n, m, np.array([prev_step[1][i][j], prev_step[2][i][j]]), sigma_a, building)
                trans_prob = combine_prob(m_p, rssi_prob)
                temp = np.full((n,m), prev_step[0][i][j])
                trans_prob = np.multiply(trans_prob, temp)

                res = np.nonzero(trans_prob)
                for k in range(0, len(res[0])):
                    x = res[0][k]
                    y = res[1][k]
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

    path_est = path_est[::-1]

    print("Path estimation:")
    print(path_est)
    error(path, path_est)

    mu = np.zeros((n, m))
    for map in env.mu:
        mu = mu + map

    plot(mu, path, path_est)