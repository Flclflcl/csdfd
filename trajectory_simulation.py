import numpy as np
import scipy as sp
import scipy.linalg as LA
from scipy.optimize import minimize, NonlinearConstraint, BFGS
from gmm.gmm_utils import gmm_2_parameters, parameters_2_gmm, shape_DS, gmr_lyapunov
import matplotlib.pyplot as plt
from bls_pred import *
import pickle
from tools import load_saved_mat_file, plot_results, u_plot
import resample
from lyapunov_learner.lyapunov_learner import LyapunovLearner
from config import gmm_params, lyapunov_learner_params, general_params
from stabilizer.ds_stab import DynamicalSystem
from gmm.gmm import GMM
from scipy import linalg
from sympy import *
from mpl_toolkits.mplot3d import Axes3D
from math import sqrt, pow, acos
import polynomial_fitting
import scipy.io as io
import matplotlib

matplotlib.use('TkAgg')

def angle_of_vector(v1, v2):
    pi = 3.1415
    vector_prod = v1[0] * v2[0] + v1[1] * v2[1]
    length_prod = sqrt(pow(v1[0], 2) + pow(v1[1], 2)) * sqrt(pow(v2[0], 2) + pow(v2[1], 2))
    cos = vector_prod * 1.0 / (length_prod * 1.0 + 1e-6)
    return (acos(cos) / pi) * 180

def GMR(x, Vxf, sigma, mu, priors, nargout=0):
    nbData = x.shape[1]
    nbStates = sigma.shape[2]
    input = np.arange(0, Vxf['d'])
    output = np.arange(Vxf['d'], 2 * Vxf['d'])
    Pxi = []
    for i in range(nbStates):
        Pxi.append(priors[0, i] * gaussPDF(x, mu[input, i],
                                                     sigma[input[0]:(input[1] + 1),
                                                     input[0]:(input[1] + 1), i]))

    Pxi = np.reshape(Pxi, [len(Pxi), -1]).T
    beta = Pxi / np.tile(np.sum(Pxi, axis=1) + 1e-300, [nbStates, 1]).T

    y_tmp = []
    for j in range(nbStates):
        a = np.tile(mu[output, j], [nbData, 1]).T
        b = sigma[output, input[0]:(input[1] + 1), j]
        c = x - np.tile(mu[input[0]:(input[1] + 1), j], [nbData, 1]).T
        d = sigma[input[0]:(input[1] + 1), input[0]:(input[1] + 1), j]
        e = np.linalg.lstsq(d, b.T, rcond=-1)[0].T
        y_tmp.append(a + e.dot(c))

    y_tmp = np.reshape(y_tmp, [nbStates, len(output), nbData])
    # y_tmp = np.reshape(y_tmp, [nbStates, len(self.output)])
    beta_tmp = beta.T.reshape([beta.shape[1], 1, beta.shape[0]])
    y_tmp2 = np.tile(beta_tmp, [1, len(output), 1]) * y_tmp
    y = np.sum(y_tmp2, axis=0)
    
    ## Compute expected covariance matrices Sigma_y, given input x
    Sigma_y_tmp = []
    Sigma_y = []
    if nargout > 1:
        for j in range(nbStates):
            Sigma_y_tmp.append(
                sigma[output, output, j] - (
                            sigma[output, input, j] / (sigma[input, input, j]) *
                            sigma[input, output, j]))

        beta_tmp = beta.reshape(1, 1, beta.shape)
        Sigma_y_tmp2 = np.tile(beta_tmp * beta_tmp, [len(output), len(output), 1, 1]) * np.tile(Sigma_y_tmp,[1, 1, nbData,1])
        Sigma_y = np.sum(Sigma_y_tmp2, axis=3)
    return y, Sigma_y, beta

def gaussPDF(data, mu, sigma):
    nbVar, nbdata = data.shape
    data = data.T - np.tile(mu.T, [nbdata, 1])
    prob = np.sum(np.linalg.lstsq(sigma, data.T, rcond=-1)[0].T * data, axis=1)
    prob = np.exp(-0.5 * prob) / np.sqrt((2 * np.pi) ** nbVar * np.abs(np.linalg.det(sigma) + 1e-300))
    return prob.T

def ds_stabilizer(x, obs, option, Vxf, gmm_parameters, predictor):
    d = Vxf['d']
    rho0 = 1
    kappa0 = 0.1
    if x.shape[0] == 2*d:
        dx = x[d+1:2*d, :]
        x = x[:d, :]

    if option == 'GMR':
        dx, _, _ = GMR(x, Vxf, gmm_parameters['sigma'], gmm_parameters['mu'], gmm_parameters['priors'])
    elif option == 'BLS':
        dx = predictor.predict(x)
        dx = dx.T

    V, Vx = gmr_lyapunov(x, obs, Vxf['Priors'], Vxf['Mu'], Vxf['P'])

    norm_Vx = np.sum(Vx * Vx, axis=0)
    norm_x = np.sum(x * x, axis=0)
    Vdot = np.sum(Vx * dx, axis=0)
    obs = np.squeeze(obs)
    rho = rho0 * (1-np.exp(-kappa0 * norm_x)) * np.sqrt(norm_Vx)
    ind = Vdot + rho >= 0
    u = dx * 0

    if np.sum(ind) > 0:
        lambder = (Vdot[ind] + rho[ind]) / (norm_Vx[ind] + 1e-8)
        u[:, ind] = -np.tile(6*lambder, [d, 1]) * Vx[:, ind]
        dx[:, ind] = dx[:, ind] + u[:, ind]
    # ind_obs = obs > 1e-5
    # if np.sum(ind_obs) > 1e-5:
    #     lambder = (Vdot[ind] + rho[ind]) / (norm_Vx[ind] + 1e-8)
    #     u[:, ind] = -np.tile(lambder, [d, 1]) * Vx[:, ind]
    #     dx[:, ind] = dx[:, ind] + u[:, ind]
    return dx, u, Vx, Vdot

def energyContour(Vxf, D, x_obs, r_m):
    quality = 'high'
    b_plot_contour = True
    contour_levels = np.array([])

    if quality == 'high':
        nx, ny = 0.1, 0.1
    elif quality == 'medium':
        nx, ny = 1, 1
    else:
        nx, ny = 2, 2

    # x = np.arange(D[0][0], D[0][1], nx)   # linspace
    x = np.arange(D[0][0], D[0][1], nx)
    y = np.arange(D[1][0], D[1][1], ny)
    x_len = len(x)
    y_len = len(y)
    X, Y = np.meshgrid(x, y)
    x_m = np.stack([np.ravel(X), np.ravel(Y)])

    G_obs = barrier(x_m, x_obs, r_m)
    x_f = 0
    y_f = 0

    # TRAJECTORY
    index = 2

    # x_sim_re = np.array(x_sim[:, 0, index]).squeeze()  # .reshape([1,num])
    # y_sim_re = np.array(x_sim[:, 1, index]).squeeze()

    # plot_num = 80
    # rex = resample.resample(x_sim_re, plot_num)
    # x_sim_ = rex.interp()
    # rey = resample.resample(y_sim_re, plot_num)
    # y_sim_ = rey.interp()
    # x_sim_m = np.stack([x_sim_, y_sim_])
    # u_num = plot_num
    # t = np.linspace(0, simulate_length, u_num)
    # G_obs_sim = barrier(x_sim_m, x_obs)


    V, dV = gmr_lyapunov(x_m, G_obs, Vxf['Priors'], Vxf['Mu'], Vxf['P'])
    # V_sim, dV_sim = gmr_lyapunov(x_sim_m, 0, Vxf['Priors'], Vxf['Mu'], Vxf['P'])

    # V, dV = computeEnergy(x, np.array(()), G_obs, Vxf, nargout=2, original=original)
    # cmap = plt.cm.Greys
    if not contour_levels.size:
        contour_levels = np.arange(0, np.log(np.max(V)), 0.4)
        contour_levels = np.exp(contour_levels)
        if np.max(V) > 400:
            contour_levels = np.round(contour_levels)
    norm_Vx = np.sqrt(np.sum(V * V, axis=0))
    # norm_Vx_sim = np.sqrt(np.sum(V_sim * V_sim, axis=0))
    V = 10000 * V/norm_Vx     # 100000 *
    # V = V / norm_Vx
    V = V.reshape(y_len, x_len)
    # V = 100000 * V
    # V_sim = V_sim/norm_Vx_sim    # 10000 *
    # V_sim = V_sim.reshape(y_len, x_len)
    min_vel_ceil = np.ceil(np.min(V))
    max_vel_floor = np.ceil(np.max(V))
    delta_x = max_vel_floor / 100
    plt.plot(x_f, y_f, 'b*', markersize=15, linewidth=3, label='target', zorder=12)
    # plt.axis('equal')
    if b_plot_contour:
        plt.contour(X, Y, V, contour_levels, cmap='RdGy', origin='upper', linewidths=0.1)  # ,labelspacing=200
        CS = plt.contourf(X, Y, V, levels=100, cmap='RdGy')
        # cbar = plt.colorbar(CS, ticks=np.arange(0, 4000, 400))
        plt.subplots_adjust(right=0.97, left=0.13, top=0.95, bottom=0.11)
        plt.clabel(CS, inline=True, fontsize=10)
        # plt.xticks(())
        # plt.yticks(())
        x_sim_ = 0
        y_sim_ = 0
        V_sim = 0
        surface(X, Y, V, x_sim_, y_sim_, V_sim, contour_levels)

    return CS

def load_model(path):
    predictor = BLS_Pred()
    predictor.load_weight_deep(path)
    return predictor

def save_trajectory(x, dx, vu, x_obs, r,x_lim):
    num = x[0].shape[-1]
    x_sim = np.array(x)
    dx_sim = np.array(dx)
    u_sim = np.array(vu)
    state = np.array(())
    control = np.array(())
    u = np.array(())
    for i in range(x_sim.shape[-1]):   # num of demos
        if i == 0:
            state = x_sim[:, :, i]
            control = dx_sim[:, :, i]
            u = u_sim[:, :, i]
        else:
            state = np.hstack((state, x_sim[:, :, i]))
            control = np.hstack((control, dx_sim[:, :, i]))
            u = np.hstack((u, u_sim[:, :, i]))

    num = state.shape[1]
    # x_lim = [[-5, 35],
    #          [-15, 25]]
    # x_obs = np.array([[7, 10, 17, 17, 27, 22, 27],
    #                    [13, 6, 0, 13, 0, 6, 13]])
    tf = 15
    degree = 14
    testplot = polynomial_fitting.poly_fit(x_obs, x_lim, r, tf)
    for i in range(int(0.5*num)):
        x = state[:,2*i].copy()
        y = state[:,2*i+1].copy()
        data = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
        last = 1
        plot_data = testplot.polyfit(data, tf, degree, i, last, plot=True)
    testplot.plot_rectangle(plot_data, multi=True)

    # model_id = np.int64(time.strftime('%Y%m%d%H%M', time.localtime(time.time())))
    # dataNew = './trajectory/T_goal=00_demo=' + str(num) + '_' + str(model_id) + '.mat'
    # sio.savemat(dataNew, {'state': state, 'control': control, 'u': u})
    return

def plot_results(x0_all, x_obs, x_new, new_obs, r_m, r_new, goal, Vxf, gmm_params, path, plot_mode, original, extra=10, simulate_length=1200, n_points=400):
    print('plotting results ############### ')
    predict_model = load_model(path)
    x_f = goal[0]
    y_f = goal[1]

    # x_lim = [[-3, 8],
    #          [-3, 8]]
    # x_lim = [[5 - extra, 25 + extra],
    #          [0 - 1.5*extra, 15 + extra]]
    # x_lim = [[-5, 35],
    #          [-15, 25]]
    # x_lim = [[-5, 35],
    #          [-15, 25]]
    x_lim = [[-3, 8],
             [-3, 8]]
    x_obs_ = np.copy(x_obs)
    n_obs = x_obs.shape[1]
    x_obs_copy = np.copy(x_obs)
    # x_obs_[0, :] = np.tile(x_f, [n_obs, 1]).T - x_obs_copy[0, :]
    # x_obs_[1, :] = np.tile(y_f, [n_obs, 1]).T - x_obs_copy[1, :]

    x_obs_m = np.copy(x_obs)
    x_obs_m_ = np.copy(x_obs_)

    if new_obs is True:
        x_new_ = np.copy(x_new)
        r_new_ = np.copy(r_new)
        n_new = x_new.shape[1]
        x_new_x = np.copy(x_new[0, :])
        x_new_y = np.copy(x_new[1, :])
        # x_new_[0:, :] = np.tile(x_f, [n_new, 1]).T - x_new_x
        # x_new_[1:, :] = np.tile(y_f, [n_new, 1]).T - x_new_y
        x_new_[0:, :] = x_new_x
        x_new_[1:, :] = x_new_y
        x_obs_m_ = np.hstack((x_obs_m_, x_new_))
        x_obs_m = np.hstack((x_obs_m, x_new))
        r_m = np.hstack((r, r_new_))


    x_sim, x_sim_, dx_sim, u = simulate_trajectories(x0_all, x_obs_m, r_m ,goal, option, Vxf, gmm_params,
                                                    predict_model, corrected=False, trajectory_length=simulate_length, original=original)

    # Plot simulated data
    x_sim = np.array(x_sim)
    dx_sim = np.array(dx_sim)

    # plt.rcdefaults()
    plt.rcParams.update({"text.usetex": False, "font.family": "Times New Roman", "font.size": 18})


    plt.plot(x_sim[:, 0, :], x_sim[:, 1, :], color='red', linewidth=2, zorder=10)
    plt.plot(0, 0, 'b*', markersize=15, linewidth=3, label='target', zorder=12)



    # plt.axis('equal')

    save_trajectory(x_sim, dx_sim, u, x_obs_m, r_m, x_lim)
    if plot_mode == 'velocities':
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111)
        font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}
        # plt.title('Lyapunov Learner (CBF-DM)')
        plt.xlabel('Position x [m]', fontdict=font1)
        plt.ylabel('Position y [m]', fontdict=font1)
        plt.yticks(fontproperties='Times New Roman', size=18)
        plt.xticks(fontproperties='Times New Roman', size=18)
        # ax.tick_params(axis='both', labelsize=18)
        plt.style.use('seaborn-whitegrid')
        palette = plt.get_cmap('Set1')

        colorp = palette(8)

        # Get velocities
        x1_coords, x2_coords = np.meshgrid(
            np.linspace(x_lim[0][0], x_lim[0][1], n_points),
            np.linspace(x_lim[1][0], x_lim[1][1], n_points))

        # x1_coords_, x2_coords_ = np.meshgrid(          # original is False  40-0
        #     np.linspace(x_lim[0][1], x_lim[0][0], n_points),
        #     np.linspace(x_lim[1][1], x_lim[1][0], n_points))

        x_init = np.zeros([2, n_points ** 2])
        x_init_ = np.zeros([2, n_points ** 2])
        x_init[0, :] = x1_coords.reshape(-1)   # 100*100
        x_init[1, :] = x2_coords.reshape(-1)
        # x_init_[0, :] = x1_coords_.reshape(-1)  # 100*100
        # x_init_[1, :] = x2_coords_.reshape(-1)
        x_init_[0, :] = x1_coords.reshape(-1)  # 100*100
        x_init_[1, :] = x2_coords.reshape(-1)
        x_hist = []
        dx_hist = []

        for i in range(n_points ** 2):  # TODO: do this in one pass
            x, x_, dx, _ = simulate_trajectories([x_init_[:, i].reshape(-1, 1)], x_obs_m, r_m, goal, option, Vxf, gmm_parameters, predict_model, dt=0.001,
                                  corrected = False, trajectory_length=1, original=False)
            # for j in range(len(dx)):
            #     if dx[j].any() > 1.2:
            #         dx = 1.2
            x_hist.append(x[1])
            dx_hist.append(dx[1])

        x_hist = np.squeeze(x_hist)
        x_hist = np.array(x_hist)[:, :].T
        dx_hist = np.array(dx_hist)[:, :, 0]
        copy_dx = np.copy(dx_hist)

        dx_hist_ = np.array(copy_dx[:, :].T)

        obs_Swarning = barrier(x_init, x_obs_m_, r_m, k=0.6)
        obs_Nwarning = barrier(x_hist, x_obs_m_, r_m, k=0.6)   
        zeroS_condition = obs_Swarning.squeeze().T > 0.0001
        zeroN_condition = obs_Nwarning.squeeze().T > 0.0001

        # x_init_x = np.copy(x_init[0, :])
        x_init[0, :][zeroS_condition] = 0
        x_sum1 = np.sum(x_init[0, :])
        x_init[0, :][zeroN_condition] = 0  
        x_sum2 = np.sum(x_init[0, :])

        # x_init_y = np.copy(x_init[1, :])
        x_init[1, :][zeroS_condition] = 0
        x_sum3 = np.sum(x_init[1, :])
        x_init[1, :][zeroN_condition] = 0
        x_sum4 = np.sum(x_init[1, :])

        x_filter_mesh = x_init[0, :].reshape(n_points, n_points).squeeze()
        y_filter_mesh = x_init[1, :].reshape(n_points, n_points).squeeze()

        # mask = np.zeros(x_init[0, :].shape, dtype=bool)
        v_condition1 = x_init[0, :] == 0
        v_condition2 = x_init[1, :] == 0

        vel_x = np.copy(dx_hist_)[0, :]
        vel_x[v_condition1] = np.nan
        # vel_x[v_condition1] = 0
        vel_y = np.copy(dx_hist_)[1, :]
        vel_y[v_condition2] = np.nan
        # vel_y[v_condition2] = 0

        vel_x_filter_mesh = vel_x.reshape(n_points, n_points).squeeze()
        vel_y_filter_mesh = vel_y.reshape(n_points, n_points).squeeze()

        cmap = plt.cm.Greys
        color = 'black'
        plt.streamplot(
            x1_coords, x2_coords,
            vel_x_filter_mesh, vel_y_filter_mesh,
            color=color, cmap=cmap, linewidth=0.5, maxlength=0.5,
            density=2, arrowstyle='fancy', arrowsize=1, zorder=2
        )

        # Plot speed                                                       'viridis''RdGy'
        # vel = [vel_x_filter_mesh]
        # vel = np.array(vel.append(vel_y_filter_mesh))
        vel = dx_hist.reshape(n_points, n_points, -1)   
        norm_vel = np.clip(np.linalg.norm(vel, axis=2), a_min=0, a_max=2)  
        print('vel=', vel)

        # CS = plt.contourf(x1_coords, x2_coords, norm_vel, cmap='cividis', levels=50, zorder=1)
        CS = plt.contourf(x1_coords, x2_coords, norm_vel, levels=150, cmap='RdGy')

        print('norm_vel =', norm_vel)
        min_vel_ceil = np.ceil(np.min(norm_vel))
        max_vel_floor = np.ceil(np.max(norm_vel))
        delta_x = max_vel_floor / 10
        cbar = plt.colorbar(CS, ticks=np.arange(min_vel_ceil, max_vel_floor, delta_x))
        cbar.ax.set_xlabel('speed (m/s)')

        length = len(x_obs_m_[0])

        for i in range(length):
            x = np.linspace(x_obs_m_[0, i] - r_m[i], x_obs_m_[0, i] + r_m[i], 1000)
            y1 = np.sqrt(np.square(r_m[i]) - np.square((x - x_obs_m_[0, i]))) + x_obs_m_[1, i]
            y2 = -np.sqrt(np.square(r_m[i]) - np.square((x - x_obs_m_[0, i]))) + x_obs_m_[1, i]
            plt.plot(x, y1, c='k')
            plt.plot(x, y2, c='k')

            plt.fill_between(x, y1, y2, color=color, alpha=0.5, linewidth=3)

        for i in range(x_obs_m_.shape[1]):
            circle = plt.Circle((x_obs_m_[0, i], x_obs_m_[1, i]), r_m[i])
            ax.add_artist(circle)
        # plot_heatmap(vel_x_filter_mesh, x_obs, r)
        # plot_heatmap(vel_y_filter_mesh, x_obs, r)
        plt.axis('equal')

    elif plot_mode == 'energy_levels':
        fig = plt.figure(figsize=(9, 7))
        font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 25}
        # plt.title('Lyapunov Learner (CBF-DM)')
        plt.xlabel('Position x [m]', fontdict=font1)
        plt.ylabel('Position y [m]', fontdict=font1)
        plt.yticks(fontproperties='Times New Roman', size=22)
        plt.xticks(fontproperties='Times New Roman', size=22)
        # ax.tick_params(axis='both', labelsize=18)
        plt.style.use('seaborn-whitegrid')
        palette = plt.get_cmap('Set1')

        colorp = palette(8)
        # Plot trajectory
        # save_trajectory(x_sim, dx_sim, u, x_obs_m, r_m, x_lim)
        # Plot energy levels
        h = energyContour(Vxf, x_lim,  x_obs_m_, r_m)

        u_plot(x_sim, dx_sim, u, simulate_length=simulate_length)
    else:
        print('Selected print mode not valid!')
        exit()

    plt.xlim(x_lim[0])
    plt.ylim(x_lim[1])
    plt.tight_layout()
    # plt.axis('equal')
    # plt.savefig('vector_field_%s.pdf' % (plot_mode), dpi=300)
    plt.show()

def u_plot(x_sim, dx_sim, u, simulate_length=4000):

    # x_sim, x_sim_, dx_sim, u = simulate_trajectories(x_ini, x_obs, option, Vxf, gmm_params,
    #                                                  predict_model, corrected = False, trajectory_length=simulate_length,
    #                                                  original=False)
    x_sim = np.array(x_sim)
    dx_sim = np.array(dx_sim)
    u = np.array(u)
    # print('u =', u)

    index = 0
    x = np.array(x_sim[:, 0, index]).squeeze()  # .reshape([1,num])
    y = np.array(x_sim[:, 1, index]).squeeze()
    z1 = -np.array(u[:, 0, index]).squeeze()
    z2 = -np.array(u[:, 1, index]).squeeze()

    plot_num = 80
    rex = resample.resample(x, plot_num)
    x = rex.interp()
    rey = resample.resample(y, plot_num)
    y = rey.interp()
    rez1 = resample.resample(z1, plot_num)
    z1 = rez1.interp()
    rez2 = resample.resample(z2, plot_num)
    z2 = rez2.interp()

    u_num = plot_num
    t = np.linspace(0, simulate_length, u_num)
    dx1 = np.array(dx_sim[:, 0, index]).squeeze()
    dx2 = np.array(dx_sim[:, 1, index]).squeeze()
    redx1 = resample.resample(dx1, u_num)
    dxx = redx1.interp()
    redx2 = resample.resample(dx2, u_num)
    dxy = redx2.interp()

    plt.figure(figsize=(6, 4))
    plt.style.use('seaborn-whitegrid')
    palette = plt.get_cmap('Set1')
    colorp = palette(8)
    # plt.xlim([0, 40])
    # plt.ylim([0, 30])
    plt.tight_layout()
    # fig1, ax1 = plt.subplots(subplot_kw=dict(projection='3d'))
    fig1, ax1 = plt.subplots()
    plt.plot(t, dxx, color='grey', label='$dx$') 
    plt.plot(t, z1, 'o-', color='r', label='$u_x$') 
    plt.legend(loc = 0)  
    ax1.tick_params(axis='both', labelsize=16)
    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 16}
    plt.xlabel('step', fontdict=font1)
    plt.ylabel('v (m/s)', fontdict=font1)
    plt.yticks(fontproperties='Times New Roman', size=16)
    plt.xticks(fontproperties='Times New Roman', size=16)
    # plt.axis('equal')

    plt.figure(figsize=(6, 4))
    plt.style.use('seaborn-whitegrid')
    palette = plt.get_cmap('Set1')
    colorp = palette(8)
    # plt.xlim([0, 40])
    # plt.ylim([0, 30])
    plt.tight_layout()
    fig2, ax2 = plt.subplots()
    plt.title('Compare $v_y$ with $u_y$') 
    plt.plot(t, dxy, color='grey', label='$dy$')  
    plt.plot(t, z2, 'o-', color='r', label='$u_y$')  
    plt.legend(loc=0) 
    ax2.tick_params(axis='both', labelsize=16)
    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}
    plt.xlabel('step', fontdict=font1)
    plt.ylabel('v (m/s)', fontdict=font1)
    plt.yticks(fontproperties='Times New Roman', size=18)
    plt.xticks(fontproperties='Times New Roman', size=18)
    colorp = palette(8)


    z = np.vstack((z1**2, z2**2))
    norm_u = np.array(np.sqrt(np.sum(z, axis=0)))

    plt.figure(figsize=(4, 4))
    plt.style.use('seaborn-whitegrid')
    palette = plt.get_cmap('Set1')
    colorp = palette(8)
    # plt.xlim([0, 40])
    # plt.ylim([0, 30])
    plt.tight_layout()
    fig3, ax3 = plt.subplots(subplot_kw=dict(projection='3d'))   #     subplot_kw=dict(projection='3d')
    # ax3 = plt.axes(projection='3d' )
    # ax3 = Axes3D(fig3)
    ax3.stem(x, y, norm_u, linefmt='grey', markerfmt='o')
    # ax1.stem(x, y, z2)
    # ax.stem(np.squeeze(x_sim[:, 0, :]), np.squeeze(x_sim[:, 1, :]), np.squeeze(u))
    ax3.view_init(elev=25, 
                 azim=-45  
                 )
    ax3.xaxis.pane.fill = False
    ax3.yaxis.pane.fill = False
    ax3.zaxis.pane.fill = False
    ax3.tick_params(axis='both', labelsize=16)
    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}
    plt.xlabel('x [m]', fontdict=font1)
    plt.ylabel('y [m]', fontdict=font1)
    ax3.zaxis.set_rotate_label(False) 
    ax3.set_zlabel('$||u||$', fontdict=font1)
    # plt.yticks(fontproperties='Times New Roman', size=16)
    # plt.xticks(fontproperties='Times New Roman', size=16)

    # plt.title('V', fontdict=font1)
    plt.subplots_adjust(top=1, bottom=0.1, right=0.95, left=0)

    plt.show()

def surface(x, y, z, x_sim, y_sim, V_sim, contour_levels):
    fig = plt.figure(figsize=(6, 6))

    ax = fig.add_subplot(111, projection='3d')
    # ax.plot(x_sim, y_sim, V_sim, 'o-', color='r') 
    ax.plot_surface(x, y, z, cmap='RdGy')
    plt.rcParams['axes.facecolor'] = 'white'
    ax.contour(x, y, z, contour_levels, zdir='z', offset=-1, cmap='RdGy')
    plt.plot(0, 0, 'b*', markersize=15, linewidth=3, label='target', zorder=12)
    ax.view_init(elev=35,  
                 azim=-145  
                 )
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 18}
    font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}
    plt.xlabel('x [m]', fontdict=font1)
    plt.ylabel('y [m]', fontdict=font1)
    plt.yticks(fontproperties='Times New Roman', size=16)
    plt.xticks(fontproperties='Times New Roman', size=16)
    ax.zaxis.set_rotate_label(False) 
    ax.set_zlabel('$ V $', fontdict=font2)
    plt.subplots_adjust(top=1, bottom=0.02, right=0.95, left=0.1, hspace=0, wspace=0)
    plt.grid(True)
    plt.show()

def euclidean_distance(vectors, vector, Sum=True):
    diff = vectors - vector
    squared_diff = np.square(diff)
    if Sum is True:
        summed = np.sum(squared_diff, axis=0)
    else:
        summed = squared_diff
    distance = np.sqrt(summed)
    return distance  # 1xN

def barrier(x, x_so, r, k=1.1):  # x(state,length)
    g = [np.array(())]
    G = []
    G_new = []
    G_obs = []
    # r = 1.5  
    # r = r
    # R = 1.35
    R = 0.45
    gain = 1.4
    xi = np.array(0.00000000001)

    nbData = np.shape(x)[1]
    num_obs = len(x_so[1, :])
    for i in range(num_obs):
        obs_R = np.array((r[i] + R) * k)
        x_obs = x_so[:, i]
        a = euclidean_distance(x, x_obs[:, np.newaxis]) 
        # a = np.linalg.norm(x - x_obs[:, np.newaxis])
        a = np.reshape(a, [1, nbData])
        theta = a - obs_R
        c = np.sqrt(np.square(theta) + 4 * xi)
        g = 0.5 * (c - theta)
        # obs_indx = g > 1e-6
        G.append(g * gain)
    G_obs = np.sum(G, axis=0)
    # print(self.G_obs )
    # G_obs_hist.append(self.G_obs)
    return G_obs

def obs_check(x, x_so, r, k=1.1): 
    # r = 1.5 
    r = r
    # R = 1.35
    R = 0.45
    G = []
    xi = np.array(0.00000000001)

    for i in range(len(x_so[1, :])):
        obs_R = np.array((r[i] + R) * k)
        x_obs = x_so[:, i]
        a = euclidean_distance(x, x_obs[:, np.newaxis])  
        a = np.reshape(a, [1, np.shape(x)[1]])
        theta = a - obs_R
        c = np.sqrt(np.square(theta) + 4 * xi)
        g = 0.5 * (c - theta)
        G.append(g.squeeze())
    obs_index = np.array(G) > 1e-7
    return obs_index

def simulate_trajectories(x_init, x_obs, r, goal, option, Vxf, gmm_parameters, predict_model, dt=0.01, trajectory_length=3000, corrected=True, original=False):
    x_f = 0
    y_f = 0

    x_obs_ = np.copy(x_obs)
    n_obs = x_obs.shape[1]
    x_obs_copy = np.copy(x_obs)
    # x_obs_[0, :] = np.tile(x_f, [n_obs, 1]).T - x_obs_copy[0, :]
    # x_obs_[1, :] = np.tile(y_f, [n_obs, 1]).T - x_obs_copy[1, :]

    # nbData = np.array(x_init[0]).shape[1]
    # nbData = len(x_init[0][0, :])
    upgrade = 1
    if isinstance(x_init, list) is True:
        nbData = 1
        x_hist = x_init
    else:
        nbData = x_init.shape[1]
        x_hist = [x_init]
    # x_hist = [x_init]
    x_hist_x = []
    x_hist_y = []
    x_temp = []
    y_temp = []
    obs_avoid_hist = []
    u_hist = [np.zeros([2, nbData])]
    orthogonal_Vx = np.zeros([2, nbData])

    x_hist_inv_append = []
    x_hist_append = []
    h_hist = []
    for j in range(nbData):
        x_hist_x.append(x_hist[0][0, j].copy())
        x_hist_y.append(x_hist[0][1, j].copy())
    x_hist_inv = [np.vstack((x_hist_x, x_hist_y))]
    # dx_hist = [np.zeros([2, nbData])]
    # dx_hist_ = [np.zeros([2, nbData])]
    dx_hist = [np.zeros([2, nbData])]
    dx_hist_ = [np.zeros([2, nbData])]
    # dx_hist.clear()
    # dx_hist_.clear()
    x_hist[0][0, :] = np.array(x_hist[0][0, :])
    x_hist[0][1, :] = np.array(x_hist[0][1, :])
    # x_hist_inv[0][0, :] = np.tile(x_f, [nbData, 1]).T - x_hist_inv[0][0, :]   # 40，30 to 0，0
    # x_hist_inv[0][1, :] = np.tile(y_f, [nbData, 1]).T - x_hist_inv[0][1, :]
    x_hist_inv[0][0, :] = x_hist_inv[0][0, :] - np.tile(x_f, [nbData, 1]).T  
    x_hist_inv[0][1, :] = x_hist_inv[0][1, :] - np.tile(y_f, [nbData, 1]).T
    for i in range(trajectory_length):
        # if i == 631:
        #     a = 0
        x_hist_x.clear()
        x_hist_y.clear()
        x_temp.clear()
        y_temp.clear()
        # dx_hist.clear()
        x_hist_append.clear()
        obs_avoid = barrier(x_hist[i], x_obs, r)
        obs_avoid_hist.append(obs_avoid)

        # if upgrade > 1:
        #     with open('./V/V_re' + str(model_id) + '_Ndemo=' + str(nbData) + '.pkl', 'rb') as g: 
        #         gmm_parameters = pickle.load(g, encoding='bytes')
        #     with open('./V/J_re' + str(model_id) + '_Ndemo=' + str(nbData) + '.pkl', 'rb') as v:
        #         Vxf = pickle.load(v, encoding='bytes')

        dx, u, Vx, _ = ds_stabilizer(x_hist[i], obs_avoid, option, Vxf, gmm_parameters, predict_model)
        # norm_Vx = np.sum(Vx * Vx, axis=0)
        if i == 0:
            h = np.zeros([2, nbData])
        else:
            h = obs_avoid
            # h_hist.append(h)
            # dh = h - obs_avoid_hist[i - 1]
            # obs_collision = dh + 0.1*obs_avoid_hist[i - 1] >= 0
            obs_collision = h > 1e-6
            obs_collision_ind = np.squeeze(obs_collision)   # demo_index
            # ind_obs = obs > 1e-5
            if np.sum(obs_collision_ind) > 0:
                x_check = x_hist_inv[i][:, obs_collision_ind]  
                obs_index = obs_check(x_check, x_obs_, r)  
                
                # norm_Vx = np.sum(Vx * Vx, axis=0)
                # dh = h - obs_avoid_hist[i - 1]
                # if dh[obs_collision_ind]
                # norm_x = np.sum(x_hist[i] * x_hist[i], axis=0)
                # By = np.copy(Vx[:, 1])

                # for k in range(nbData):
                #     Bx = np.copy(Vx[:, k])
                #     Qx, Rx = linalg.qr(np.hstack((Bx[:, np.newaxis], By[:, np.newaxis])))
                #     orthogonal_Vx[:, k] = Qx[:, 1].squeeze()

                if np.shape(x_check)[1] == 1:
                    orthogonal_Vx_ = np.copy(Vx)
                    print('obs_index = ', obs_index)
                    print('x_obs[{a}], obs_index] = {b}'.format(a=i, b=x_obs[:, obs_index]))
                    print('obs_collision_ind = ', obs_collision_ind)
                    print('x_hist[{a}][:, obs_collision_ind] = {b}'.format(a=i, b=x_hist[i][:, obs_collision_ind]))
                    # Bx = np.copy(Vx[:, obs_collision_ind]).squeeze()
                    # Qx, Rx = linalg.qr(Bx[:, np.newaxis])
                    # v = orthogonal_Vx_[:, obs_collision_ind]
                    oV = -orthogonal_Vx_[:, obs_collision_ind]
                    # oQV = Qx[:, 1].copy()
                    # oVk = oV.copy()
                    # orthogonal_Vx_[:, obs_collision_ind] = q[:,np.newaxis]

                    # e_o = x_obs_[:, obs_index] - x_hist_inv[i][:, obs_collision_ind]
                    # e_o_tan = np.degrees(np.arctan2(e_o[1], e_o[0]))
                    # e_g = np.array(goal) - x_hist_inv[i][:, obs_collision_ind].squeeze()

                    # # e_g_tan = np.degrees(np.arctan2(e_g[1] / e_g[0]))
                    # e_g_tan = np.degrees(np.arctan2(e_g[1], e_g[0]))

                    ################  Dv  ################
                    # if e_o_tan < 45:
                    #     dx = dx
                    # else:
                    #     dx = - dx

                    # Vdot = np.sum((np.exp(orthogonal_Vx[:, obs_collision_ind]) - 1) * dx, axis=0)
                    # Vdot = np.sum(oVk.reshape(2,1) * dx[:, obs_collision_ind], axis=0)
                    dx_ = dx[:, obs_collision_ind].copy()      
                    Vdot = np.sum(oV.reshape(2, 1) * -dx_, axis=0)
                    norm_Vx = np.sum(oV * oV, axis=0)
                    lambder = Vdot / (norm_Vx + 1e-8)
                    # lambder_exp = np.exp(3*lambder)-1
                    # lambder = Vdot[obs_collision_ind]
                    # u[:, obs_collision_ind] -= np.tile(lambder, [2, 1]) * orthogonal_Vx_[:, obs_collision_ind]
                    # u_obs = Vdot * oVk.reshape(2,1)
                    u_obs = 3*lambder * oV.reshape(2, 1)
                    u[:, obs_collision_ind] += u_obs
                    # u[:, obs_collision_ind] += u_obs
                    dx[:, obs_collision_ind] = dx[:, obs_collision_ind] + u[:, obs_collision_ind]

                    ################  D_v_  ################
                    # g_x = angle_of_vector(e_g, oQV)
                    # if g_x < 90:
                    #     u_obs = oQV
                    # else:
                    #     u_obs = -oQV
                    #
                    # u[:, obs_collision_ind] -= 5*u_obs.reshape(2, 1)
                    # dx[:, obs_collision_ind] = dx[:, obs_collision_ind] + u[:, obs_collision_ind]

                else:
                    orthogonal_Vx_ = np.copy(Vx)
                    uk_list = []
                    for k in range(np.shape(x_check)[1]):
                        Bx = np.copy(Vx[:, obs_collision_ind][:, k])
                        Qx, Rx = linalg.qr(Bx[:, np.newaxis])
                        oV = orthogonal_Vx_[:, obs_collision_ind].copy()

                        oQV = Qx[:, 1]
                        oVk = -oV[:, k].copy()
                        # print('x_check = ', x_check)
                        # print('obs_index = ', obs_index)
                        # print('x_obs[{a}], obs_index] = {b}'.format(a = i,b = x_obs[:, obs_index]))
                        # print('obs_collision_ind = ', obs_collision_ind)
                        # print('x_hist[{a}][:, obs_collision_ind] = {b}'.format(a = i,b = x_hist[i][:, obs_collision_ind]))

                        print('obs_index = ', obs_index)
                        print('x_obs_length[{a}] \n obs_index[{k}]] = {b}'.format(a=i, k=k, b=x_obs[:, obs_index[:, k]]))
                        print('obs_collision_ind = ', obs_collision_ind)
                        print(
                            'x_hist_length[{a}] \n [:, obs_collision_ind[{k}]] = {b}'.format(a=i, k=k, b=x_hist[i][:, obs_collision_ind][:, k]))
                        obsk = x_obs_[:, obs_index[:, k]].squeeze()
                        xc = x_hist_inv[i][:, obs_collision_ind][:, k].copy()
                        # e_o = obsk - xc
                        # e_o_tan = np.degrees(np.arctan2(e_o[1], e_o[0]))
                        # g = np.array(goal).squeeze()
                        # e_g = g - xc
                        # e_g_tan = np.degrees(np.arctan2(e_g[1] / e_g[0]))
                        # e_g_tan = np.degrees(np.arctan2(e_g[1], e_g[0]))

                        ################  Dv  ################
                        # if e_o_tan < 45:
                        #     dx = dx
                        # else:
                        #     dx = -dx

                        dx_ = dx[:, obs_collision_ind][:,k].copy()
                        Vdot = np.sum(oVk.reshape(2, 1) * -dx_, axis=0)
                        norm_Vx = np.sum(oVk * oVk, axis=0)
                        lambder = Vdot / (norm_Vx + 1e-8)
                        # lambder_exp = np.exp(5 * lambder) - 1
                        # lambder = Vdot[obs_collision_ind][k]
                        tl = np.tile(lambder, [2, 1])
                        uk = 3*lambder * oVk
                        uk_list.append(uk)

                        ################  D_v_  ################
                        # g_x = angle_of_vector(e_g, oQV)
                        # if g_x < 90:
                        #     oQV = oQV
                        # else:
                        #     oQV = oQV
                        #
                        # uk = 5*oQV
                        # uk_list.append(uk)


                    u[:, obs_collision_ind] += np.array(uk_list).T
                    # u[:, obs_collision_ind] = np.array(uk_list).T
                    dx[:, obs_collision_ind] = dx[:, obs_collision_ind] + u[:, obs_collision_ind]

        u_hist.append(u)
        # if dx.any() < -1.5:
        #     dx = -1.5
        h_hist.append(h)
        x = x_hist[i] + dx * dt
        x_hist.append(x)
        # dx_hist_.append(-dx)
        dx_hist.append(dx)


        x_temp = [np.tile(x_f, [nbData, 1]).T - x_hist[i + 1][0, :]]  # 40，30 to 0，0
        y_temp = [np.tile(y_f, [nbData, 1]).T - x_hist[i + 1][1, :]]
        # x_temp = [x_hist[i + 1][0, :] + np.tile(x_f, [nbData, 1]).T] 
        # y_temp = [x_hist[i + 1][1, :] + np.tile(y_f, [nbData, 1]).T]
        state = np.vstack((x_temp, y_temp))
        # if lyapunov_learner.barrier(state) > 0.01:
        #     lyapunov_learner.learnEnergy(x_hist,)
        x_hist_inv_append = [state]
        x_hist_inv.append(np.squeeze(x_hist_inv_append))
        # print('max_obs_avoid=', np.max(obs_avoid_hist))

    return x_hist, x_hist_inv, dx_hist, u_hist


with open('./G/G_goal00_Ndemo=6000.pkl', 'rb') as g:  
    gmm_parameters = pickle.load(g, encoding='bytes')
with open('./V/V_goal00_Ndemo=6000.pkl', 'rb') as v:
    Vxf = pickle.load(v, encoding='bytes')
# with open('./G/G_small_33_202405221426_Ndemo=6000.pkl', 'rb') as g:  
#     gmm_parameters = pickle.load(g, encoding='bytes')
# with open('./V/V_small_33_202405221426_Ndemo=6000.pkl', 'rb') as v:
#     Vxf = pickle.load(v, encoding='bytes')

x_f = 0
y_f = 0
# x_obs = np.array([[7, 10, 17, 17, 27, 22, 27],
#                   [13, 6, 0, 13, 0, 6, 13]])
# x_obs_ = np.array([[7, 10, 17, 17, 27, 22, 27],
#                    [13, 6, 0, 13, 0, 6, 13]])
# x_obs_ = np.array([[2, 2.5, 3.5, 0.5, 4., 9., 4.],
#                    [1.4, 3.5, 0.2, 9., 9., 3., 2.]])


x_obs = np.array([[2, 2., 3.5, 0.6, 4., 6., 5.],  # RL
                          [1.2, 4., 0, 2.5, 5., 3., 2.]])
x_obs_ = np.array([[2, 2., 3.5, 0.6, 4., 6., 5.],  # RL
                          [1.2, 4., 0, 2.5, 5., 3., 2.]])

# x_obs = np.array([[-7, 10, -17, 17, 15, -22, -8, 22, 27],
#                   [-13, 6, 0, -13, 0, -6, 13, 6, 13]])
# x_obs_ = np.array([[-7, 10, -17, 17, 15, -22, -8, 22, 27],
#                    [-13, 6, 0, -13, 0, -6, 13, 6, 13]])
# r = [2.0, 3.0, 2.5, 5.0, 2.0, 3.5, 4.0, 1.5, 1.5]

# r = [2.0, 1.2, 0.8, 0.9, 1.5, 1.3, 1.6]
r = [0.35, 0.4, 0.3, 0.4, 0.4, 0.4, 0.35]

# x_new_ = np.array([[6, 4, 13, 21, 28, 36],
#                   [2, 10, -1, -6, 2, 15]])

# x_new_ = np.array([[15, 15, 15, 7],
#                   [18, 7, -5, 0]])
# new_r = [2.0, 1.2, 0.8, 0.9]

x_new_ = np.array([[6],
                  [2]])
new_r = [0.3]


# x_ini_ = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 15, 22],
#                    [4, 6, 8, 10, 12, 16, 18, 20, 22, 24, 26, 28, -8, -5, -7 ]])
# x_ini_ = np.array([[ 0, 0, 0, 0, 15],
#                    [0, 12, 18, 20, -5]])
# x_ini_ = np.array([[34, 28, 28, 21, 22, 32],
#                    [15, 10, 8, 19, 2, 3]])
# x_ini_ = np.array([[0, 0, 12, 25, 20, 35],
#                    [0, 8, -5, -8, 0, 10]])
# x_ini_ = np.array([[33, 33 ],
#                    [ 14, 0]])

# x_ini_ = np.array([[5., 4.5, 5., 3., 4., 2.5],
#                    [2., 3.5, 4.7, 2., 4., 4.5]])
x_ini_ = np.array([[6,6],
                  [5,7]])

x_ini = np.copy(x_ini_)
x_new = np.copy(x_new_)
x_obs = np.copy(x_obs_)
nbData = x_ini_.shape[1]
n_obs = x_obs.shape[1]
n_new = x_new.shape[1]
# x_ini[0, :] = np.tile(x_f, [nbData, 1]).T - x_ini_[0, :]
# x_ini[1, :] = np.tile(y_f, [nbData, 1]).T - x_ini_[1, :]
# x_obs[0, :] = np.tile(x_f, [n_obs, 1]).T - x_obs_[0, :]
# x_obs[1, :] = np.tile(y_f, [n_obs, 1]).T - x_obs_[1, :]
# x_new[0, :] = np.tile(x_f, [n_new, 1]).T - x_new_[0, :]
# x_new[1, :] = np.tile(y_f, [n_new, 1]).T - x_new_[1, :]

x0_all = [x_ini[:2,:]]
goal = [0, 0]

new_obs = False
# new_obs = True
DIY = False
option = 'GMR'
model_path = './bls_models/bls_deep.npz'  # GMR is better
predict_model = load_model(model_path)
# x_sim, x_sim_, _, _ = simulate_trajectories(x_ini, x_obs, x_new, new_obs, option, Vxf, gmm_parameters, predict_model, dt=0.01, trajectory_length=7000, corrected=True, original=False)

# plot_results(x_ini, x_obs, x_new, new_obs, r, new_r, goal, Vxf, gmm_parameters, model_path,  plot_mode='velocities', original=False)
plot_results(x_ini, x_obs, x_new, new_obs, r, new_r, goal, Vxf, gmm_parameters, model_path,  plot_mode='energy_levels', original=False)
















