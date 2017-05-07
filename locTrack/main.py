from basic_finctions import *
from bayesian_filter import *
from viterbi import *
from mapGen import *

def main():
    '''
    env = VirtualEnvironment(15, 15, 0, 0)
    #env = RealEnvironment(BuildingDormitory, 'signal_strength_matrix_dormitory_AP')
    '''

    # start point
    pos_0 = np.array([2, 2])
    # start velocity
    v_0 = np.array([0, 0])
    '''
    ####Single AP########
    RSSI, path = path_generation(7, pos_0, v_0, env)

    BayesianFilter(RSSI, env, path, pos_0, v_0)
    #Viterbi(RSSI, env, path, pos_0, v_0)
    #####################

    ####2AP experiments#####
    env = VirtualEnvironment2AP(15, 15, 0, 0, 14, 14)
    #path = [(2,2), (3,3), (3,4), (4,5), (6,3), (7,5), (8,8), (11,9), (11,12)]
    #path = [(2, 2), (1, 3), (2, 5), (3, 8), (4, 10), (5, 13), (7, 12), (9, 12), (11, 9)]
    path = [(2, 2), (1, 3), (2, 5), (3, 8), (4, 10), (5, 13), (6, 10), (8, 9), (9, 11), (11, 12), (13, 13), (13, 11), (13, 8)]
    RSSI = []
    for pos in path:
        x = pos[0]
        y = pos[1]
        if (x + y <= env.n - 1):
            RSSI.append(((np.random.normal(env.mu[0][x][y], env.sigma[x][y], 1)), 0))
        else:
            RSSI.append(((np.random.normal(env.mu[1][x][y], env.sigma[x][y], 1)), 1))
    Viterbi(RSSI, env, path, pos_0, v_0)
    #BayesianFilter(RSSI, env, path, pos_0, v_0)
    #########################
    '''

main()