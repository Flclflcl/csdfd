__credits__ = 'Olalekan Ogunmolu, Rodrigo Perez-Dattari (TU Delft), Rachel Thomson (MIT), Jethro Tan (PFN)'
__license__ = 'MIT'

from lyapunov_learner.lyapunov_learner import LyapunovLearner
from config import gmm_params, lyapunov_learner_params, general_params
from stabilizer.ds_stab import DynamicalSystem
from gmm.gmm import GMM
from tools import load_saved_mat_file, plot_results, u_plot
import pickle
import time
import numpy as np
import math
import data_preprocess
import matplotlib

# matplotlib.use('TkAgg')

def main(data_name):
    # Load demonstrations
    # rot = True
    rot = False
    # DS = True
    DS = True
    theta = -0.1*math.pi
    data, demo_idx, demo_length = load_saved_mat_file(data_name)
    data_, demo_idx, demo_length = load_saved_mat_file(data_name)
    x_f = 0
    # x_f = 5.00
    y_f = 0
    # x_obs = np.array([[7,10,17,17,27,22,27],
    #                  [13,6,0,13,0,6,13]])
    # x_obs_ = np.array([[7,10,17,17,27,22,27],
    #                  [13,6,0,13,0,6,13]])

    # x_obs = np.array([[1.5, 1.6, 0.3, 1.8],
    #                    [0.5, 1.7, 1.4, 1.]])
    # x_obs_ = np.array([[1.5, 1.6, 0.3, 1.8],
    #                    [0.5, 1.7, 1.4, 1.]])

    x_obs = np.array([[2, 2.7, 3.5, 4.5, 6., 5.],  # SMALL
                              [1.4, 3.5, 0, 5., 3., 2.]])
    x_obs_ = np.array([[2, 2.7, 3.5, 4.5, 6., 5.],  # SMALL
                              [1.4, 3.5, 0, 5., 3., 2.]])

    # x_obs = np.array([[27],
    #                    [13]])
    # x_obs_ = np.array([[27],
    #                  [13]])
    # x_obs = np.array([[-7, 10, 17, 17, 15, -22, -8, 22, 27],
    #                           [-13, 6, 0, -13, 0, 6, 13, 6, 13]])
    # x_obs_ = np.array([[-7, 10, 17, 17, 15, -22, -8, 22, 27],
    #                           [-13, 6, 0, -13, 0, 6, 13, 6, 13]])
    # r = [0.13, 0.13, 0.13, 0.13]
    r = [0.3, 0.3, 0.3, 0.4, 0.3, 0.3]


    # r = [1.5]

    # x_obs = np.array([[12],
    #                  [5]])
    # x_obs_ = np.array([[12],
    #                  [5]])

    x_new = np.array(([[15,22],
                     [10,15]]))
    x_new_ = np.array(([[15,22],
                     [10,15]]))
    # x_new = np.array([])
    new_obs = False
    # new_obs = True
    # DIY = True
    DIY = False

    # Initialize Lyapunov learner
    if rot is True:
        # x_obs = rotate(theta, x_obs[0, :], x_obs[1, :])
        x_obs_ = rotate(theta, x_obs_[0, :], x_obs_[1, :])
    lyapunov = LyapunovLearner()
    V_init = lyapunov.guess_init_lyap(lyapunov_learner_params)
    V = None

    # option = 'BLS'
    option = 'GMR'
    # if option != 'GMR':
    #     data = data_preprocess.state_normalzation(data)
    #     data_ = data_preprocess.state_normalzation(data_)
    #     x_f = 1
    #     y_f = 1
    # original_main = False   #original :  x
    original_main = True
    nbData = data.shape[1]
    n_obs = x_obs.shape[1]

    if original_main is False:
        data[0, :] = np.tile(x_f, [nbData, 1]).T - data[0, :]
        data[1, :] = np.tile(y_f, [nbData, 1]).T - data[1, :]
        data[2, :] = -data[2, :]
        data[3, :] = -data[3, :]
        x_obs_x = np.copy(x_obs[0, :])
        x_obs_y = np.copy(x_obs[1, :])
        x_obs[0, :] = np.tile(x_f, [n_obs, 1]).T - x_obs_x
        x_obs[1, :] = np.tile(y_f, [n_obs, 1]).T - x_obs_y
        if new_obs is True:
            n_new = x_new_.shape[1]
            x_new_x = np.copy(x_new_[0, :])
            x_new_y = np.copy(x_new_[1, :])
            x_new[0:, :] = np.tile(x_f, [n_new, 1]).T - x_new_x
            x_new[1:, :] = np.tile(y_f, [n_new, 1]).T - x_new_y
    else:
        data[0, :] = data[0, :] - np.tile(x_f, [nbData, 1]).T
        data[1, :] = data[1, :] - np.tile(y_f, [nbData, 1]).T
        data[2, :] = data[2, :]
        data[3, :] = data[3, :]
        x_obs_x = np.copy(x_obs[0, :])
        x_obs_y = np.copy(x_obs[1, :])
        x_obs[0:, :] = x_obs_x - np.tile(x_f, [n_obs, 1]).T
        x_obs[1:, :] = x_obs_y - np.tile(y_f, [n_obs, 1]).T
        if new_obs is True:
            n_new = x_new_.shape[1]
            x_new_x = np.copy(x_new_[0, :])
            x_new_y = np.copy(x_new_[1, :])
            x_new[0:, :] = x_new_x - np.tile(x_f, [n_new, 1]).T
            x_new[1:, :] = x_new_y - np.tile(y_f, [n_new, 1]).T


        # data[0, :] = data[0, :] - np.tile(x_f, [nbData, 1]).T
        # data[1, :] = data[1, :] - np.tile(y_f, [nbData, 1]).T
        # data[2, :] = data[2, :]
        # data[3, :] = data[3, :]
    # Optimize Lyapunov function
    while lyapunov.success:
        print('Optimizing the lyapunov function')
        V, J = lyapunov.learnEnergy(V_init, data, lyapunov_learner_params, x_obs, x_new, new_obs, r, original=original_main, DIY=DIY)
        if lyapunov.success:
            model_id = np.int64(time.strftime('%Y%m%d%H%M', time.localtime(time.time())))
            v = open('./V/V_small_33_' + str(model_id) + '_Ndemo=' + str(nbData) + '.pkl', 'wb')
            # v = open('./V/V_530_' + str(model_id) + '_Ndemo=' + str(nbData) + '.pkl', 'wb')
            pickle.dump(V, v)
            v.close()
            j = open('./V/J_small_33_' + str(model_id) + '_Ndemo=' + str(nbData) + '.pkl', 'wb')
            # j = open('./V/J_530_' + str(model_id) + '_Ndemo=' + str(nbData) + '.pkl', 'wb')
            pickle.dump(J, j)
            j.close()
            print('optimization succeeded without violating constraints')
            break


    # # Create dynamical system from GP and Lyapunov corrections
    # dynamical_system = DynamicalSystem(V, np.array(()), np.array(()), np.array(()), lyapunov)
    #
    # # Plot results
    # plot_results(data, data_name, demo_idx, demo_length, dynamical_system, plot_mode='velocities')
    # plot_results(data, data_name, demo_idx, demo_length, dynamical_system, plot_mode='energy_levels')

    # Optimize GMM
    gmm = GMM(num_clusters=gmm_params['num_clusters'])
    gmm.update(data.T, K=gmm_params['num_clusters'], max_iterations=gmm_params['max_iterations'])
    mu, sigma, priors = gmm.mu.T, gmm.sigma.T, gmm.logmass.T
    gmm_parameters = {'mu': mu, 'sigma': sigma, 'priors': priors}
    with open('./G/G_small_33_' + str(model_id) + '_Ndemo=' + str(nbData) + '.pkl', 'wb') as fo:              # 将数据写入pkl文件
    # with open('./G/G_530_' + str(model_id) + '_Ndemo=' + str(nbData) + '.pkl', 'wb') as fo:
        pickle.dump(gmm_parameters, fo)


    # Create dynamical system from learned GMM and Lyapunov corrections
    dynamical_system = DynamicalSystem(V, priors, mu, sigma, lyapunov)

    # Plot results

    model_path = './bls_models/bls_deep.npz'
    # if original_main is True:      # plot的数据输入用未变换的
    #     data_[0, :] = np.tile(x_f, [nbData, 1]).T - data_[0, :]
    #     data_[1, :] = np.tile(y_f, [nbData, 1]).T - data_[1, :]
    #     data_[2, :] = -data_[2, :]
    #     data_[3, :] = -data_[3, :]

    plot_results(data, data_name, demo_idx, demo_length, dynamical_system, x_obs, x_new, new_obs,  r, option, DS=DS, path=model_path, plot_mode='velocities', DIY=DIY, original=original_main)
    plot_results(data, data_name, demo_idx, demo_length, dynamical_system, x_obs, x_new, new_obs,  r, option, DS=DS, path=model_path, plot_mode='energy_levels', DIY=DIY, original=original_main)

def rotate(angle,valuex,valuey):
    rotatex = math.cos(angle)*valuex - math.sin(angle)* valuey
    rotatey = math.cos(angle)*valuey + math.sin(angle)* valuex
    x_ = np.vstack((rotatex,rotatey))
    return x_

if __name__ == '__main__':
    data_name = general_params['shape_name']
    main(data_name)
