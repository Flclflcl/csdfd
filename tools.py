__credits__ = 'Olalekan Ogunmolu, Rodrigo Perez-Dattari (TU Delft), Rachel Thomson (MIT), Jethro Tan (PFN)'
__license__ = 'MIT'

from bls_pred import *
import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import resample
from scipy import linalg
import polynomial_fitting
# from sympy import *
# from mpl_toolkits.mplot3d import Axes3D
# from math import sqrt, pow, acos
# from lyapunov_learner.lyapunov_learner import LyapunovLearner
# from config import gmm_params, lyapunov_learner_params, general_params, event_trigger_params
# import pickle
from sympy import *

def load_saved_mat_file(data_name):
    # """
    #     Loads a matlab file from a subdirectory.
    # """

    # dataset_path = 'data/lasa_handwriting_dataset'
    # data = sio.loadmat(os.path.join(dataset_path, data_name + '.mat'))
    # dataset = data['demos']
    # num_demos = int(dataset.shape[1])
    # demo_length = dataset[0][0]['pos'][0][0].shape[1]
    # demoIdx = []
    # demonstrations = np.empty([4, num_demos * demo_length])
    # for i in range(num_demos):
    #     pos = dataset[0][i]['pos'][0][0]
    #     vel = dataset[0][i]['vel'][0][0]
    #     demonstrations[:2, i * demo_length:(i + 1) * demo_length] = pos
    #     demonstrations[2:, i * demo_length:(i + 1) * demo_length] = vel
    #
    #     demoIdx.append(i * demo_length)

    """
        Loads a matlab file from a OPC solutions.
    """

    dataset_path = './data/lasa_handwriting_dataset'
    data = sio.loadmat(os.path.join(dataset_path, data_name + '.mat'))
    solution = data['demos']
    num_demos = int(solution.shape[1])
    demo_length = solution[0][0]['state_'][0][0].shape[1]
    demoIdx = []
    demonstrations = np.empty([4, num_demos * demo_length])
    for i in range(num_demos):
        pos = solution[0][i]['state_'][0][0]
        vel = solution[0][i]['control_'][0][0]
        demonstrations[:2, i * demo_length:(i + 1) * demo_length] = pos
        demonstrations[2:, i * demo_length:(i + 1) * demo_length] = vel
        demoIdx.append(i * demo_length)
    return demonstrations, np.array(demoIdx), demo_length

# def obs_check(x, x_so, k=1.1):    # 找是哪一个障碍物
#     r = 1.5  # 假设障碍物半径一致
#     R = 1.35
#     G = []
#     xi = np.array(0.00000000001)
#     obs_R = np.array((r + R) * k)
#     for i in range(len(x_so[1, :])):
#         x_obs = x_so[:, i]
#         a = euclidean_distance(x, x_obs[:, np.newaxis])  # 欧式距离
#         a = np.reshape(a, [1, np.shape(x)[1]])
#         theta = a - obs_R
#         c = np.sqrt(np.square(theta) + 4 * xi)
#         g = 0.5 * (c - theta)
#         G.append(g.squeeze())
#     obs_index = np.array(G) > 1e-7
#     return obs_index


def simulate_trajectories(x_init, x_obs, r, goal, dynamical_system, option, predict_model, dt=0.01, trajectory_length=7000, ds=True, original=False):
    x = x_init
    x_f = 0
    # x_f = 5.00
    y_f = 0
    nbData = np.array(x_init[0]).shape[1]
    x_hist = x_init
    x_hist_x = []
    x_hist_y = []
    x_temp = []
    y_temp = []
    u_hist = [np.zeros([2, nbData])]
    x_hist_inv_append = []
    x_hist_append = []
    h_hist = []
    obs_avoid_hist = []
    # nbData = np.shape(x_hist)[1]
    # x_obs_ = np.array([[10,10,10,20,20,20,30,30,30],
    #                  [5,18,30,5,18,30,5,18,30]])
    for j in range(nbData):
        # for j in x_init[0]:
        x_hist_x.append(x_hist[0][0, j].copy())
        x_hist_y.append(x_hist[0][1, j].copy())
    x_hist_inv = [np.vstack((x_hist_x, x_hist_y))]

    dx_hist = [np.zeros([2, nbData])]

    x_hist[0][0, :] = x_hist[0][0, :]
    x_hist[0][1, :] = x_hist[0][1, :]
    # x_hist_inv[0][0, :] = np.tile(x_f, [nbData, 1]).T - x_hist_inv[0][0, :]  # 40，30 to 0，0
    # x_hist_inv[0][1, :] = np.tile(y_f, [nbData, 1]).T - x_hist_inv[0][1, :]

    for i in range(trajectory_length):
        x_hist_x.clear()
        x_hist_y.clear()
        x_temp.clear()
        y_temp.clear()
        # dx_hist.clear()
        x_hist_inv_append.clear()
        obs_avoid = barrier(np.array(x_hist[i]), x_obs, r, k=1.05)     #输入当前步数下 各个demo的state

        dx, u, Vx, _ = dynamical_system.ds_stabilizer(x_hist[i], obs_avoid, option, predict_model, ds=ds)
        if i == 0:
            h = np.zeros([2, nbData])
        else:
            h = obs_avoid
            # h_hist.append(h)
            # dh = h - obs_avoid_hist[i - 1]
            # obs_collision = dh + 0.1*obs_avoid_hist[i - 1] >= 0
            obs_collision = h > 1e-6
            obs_collision_ind = np.squeeze(obs_collision)  # 碰撞demo序号
            # ind_obs = obs > 1e-5
            if np.sum(obs_collision_ind) > 0:
                x_check = x_hist[i][:, obs_collision_ind]  # 碰撞demo位置
                obs_index = obs_check(x_check, x_obs, r)  # 获得碰撞demo位置所对应的障碍物序号
                # norm_Vx = np.sum(Vx * Vx, axis=0)
                # dh = h - obs_avoid_hist[i - 1]
                # if dh[obs_collision_ind]
                # norm_x = np.sum(x_hist[i] * x_hist[i], axis=0)
                # By = np.copy(Vx[:, 1])

                # for k in range(nbData):
                #     Bx = np.copy(Vx[:, k])
                #     Qx, Rx = linalg.qr(np.hstack((Bx[:, np.newaxis], By[:, np.newaxis])))
                #     orthogonal_Vx[:, k] = Qx[:, 1].squeeze()
                #
                if np.shape(x_check)[1] == 1:
                    orthogonal_Vx_ = np.copy(Vx)
                    print('obs_index = ', obs_index)
                    print('x_obs[{a}], obs_index] = {b}'.format(a=i, b=x_obs[:, obs_index]))
                    print('obs_collision_ind = ', obs_collision_ind)
                    print('x_hist[{a}][:, obs_collision_ind] = {b}'.format(a=i, b=x_hist[i][:, obs_collision_ind]))
                    Bx = np.copy(Vx[:, obs_collision_ind]).squeeze()
                    Qx, Rx = linalg.qr(Bx[:, np.newaxis])
                    # v = orthogonal_Vx_[:, obs_collision_ind]
                    oV = -orthogonal_Vx_[:, obs_collision_ind]
                    oQV = Qx[:, 1].copy()
                    # oVk = oV.copy()
                    # orthogonal_Vx_[:, obs_collision_ind] = q[:,np.newaxis]

                    # e_o = x_obs[:, obs_index] - x_hist[i][:, obs_collision_ind]
                    # e_o_tan = np.degrees(np.arctan2(e_o[1], e_o[0]))
                    # e_g = np.array(goal) - x_hist[i][:, obs_collision_ind].squeeze()
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
                    u_obs = 2 * lambder * oV.reshape(2, 1)
                    u[:, obs_collision_ind] += u_obs
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
                        oV = -orthogonal_Vx_[:, obs_collision_ind].copy()

                        oQV = Qx[:, 1]
                        oVk = oV[:, k].copy()
                        # print('x_check = ', x_check)
                        # print('obs_index = ', obs_index)
                        # print('x_obs[{a}], obs_index] = {b}'.format(a = i,b = x_obs[:, obs_index]))
                        # print('obs_collision_ind = ', obs_collision_ind)
                        # print('x_hist[{a}][:, obs_collision_ind] = {b}'.format(a = i,b = x_hist[i][:, obs_collision_ind]))

                        print('obs_index = ', obs_index)
                        print('x_obs_length[{a}] \n obs_index[{k}]] = {b}'.format(a=i, k=k,
                                                                                  b=x_obs[:, obs_index[:, k]]))
                        print('obs_collision_ind = ', obs_collision_ind)
                        print(
                            'x_hist_length[{a}] \n [:, obs_collision_ind[{k}]] = {b}'.format(a=i, k=k,
                                                                                             b=x_hist[i][:,
                                                                                               obs_collision_ind][:,
                                                                                               k]))
                        # obsk = x_obs[:, obs_index[:, k]].squeeze()
                        # xc = x_hist[i][:, obs_collision_ind][:, k].copy().squeeze()
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

                        dx_ = dx[:, obs_collision_ind][:, k].copy()
                        Vdot = np.sum(oVk.reshape(2, 1) * -dx_, axis=0)
                        norm_Vx = np.sum(oVk * oVk, axis=0)
                        lambder = Vdot / (norm_Vx + 1e-8)
                        # lambder_exp = np.exp(5 * lambder) - 1
                        # lambder = Vdot[obs_collision_ind][k]
                        tl = np.tile(lambder, [2, 1])
                        uk = 2 * lambder * oVk
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
                    dx[:, obs_collision_ind] = dx[:, obs_collision_ind] + u[:, obs_collision_ind]

        u_hist.append(u)
        if dx.any() < -1.5:
            dx = -1.5
        x = x_hist[i] + dx * dt
        x_hist.append(x)
        # dx_hist.append(dx)
        dx_hist.append(dx)

        x_temp = [np.tile(x_f, [nbData, 1]).T - x_hist[i + 1][0, :]]  # 40，30 to 0，0
        y_temp = [np.tile(y_f, [nbData, 1]).T - x_hist[i + 1][1, :]]
        # x_temp = [x_hist[i + 1][0, :] + np.tile(x_f, [nbData, 1]).T] # 40，30 to 0，0
        # y_temp = [x_hist[i + 1][1, :] + np.tile(y_f, [nbData, 1]).T]
        state = np.vstack((x_temp, y_temp))
        # if lyapunov_learner.barrier(state) > 0.01:
        #     lyapunov_learner.learnEnergy(x_hist,)
        x_hist_inv_append = [state]
        x_hist_inv.append(np.squeeze(x_hist_inv_append))
    return x_hist, x_hist_inv, dx_hist, u_hist

def load_model(path):
    predictor = BLS_Pred()
    predictor.load_weight_deep(path)
    return predictor

def euclidean_distance(vectors, vector, Sum=True):
    # 计算向量的欧氏距离
    diff = vectors - vector
    squared_diff = np.square(diff)
    if Sum is True:
        summed = np.sum(squared_diff, axis=0)
    else:
        summed = squared_diff
    distance = np.sqrt(summed)
    return distance    # 1xN

# def barrier(x, x_so, r, k=1.1):  # x(state,length)
#     g = [np.array(())]
#     G = []
#     G_new = []
#     G_obs = []
#     # r = 1.5  # 假设障碍物半径一致
#     R = 1.35
#     gain = 1.4
#     xi = np.array(0.00000000001)
#     obs_R = np.array((r + R) * k)
#     nbData = np.shape(x)[1]
#     num_obs = len(x_so[1, :])
#     for i in range(num_obs):
#         x_obs = x_so[:, i]
#         a = euclidean_distance(x, x_obs[:, np.newaxis])  # 欧式距离
#         a = np.reshape(a, [1, nbData])
#         theta = a - obs_R
#         c = np.sqrt(np.square(theta) + 4 * xi)
#         g = 0.5 * (c - theta)
#         obs_indx = g > 1e-6
#         G.append(g * gain)
#     G_obs = np.sum(G, axis=0)
#     # print(self.G_obs )
#     # G_obs_hist.append(self.G_obs)
#     return G_obs


def barrier(x, x_so, r, k=1.05):  # x(state,length)
    g = [np.array(())]
    G = []
    G_new = []
    G_obs = []
    # r = 1.5  # 假设障碍物半径一致
    # r = r
    # R = 1.35
    R = 0.18

    xi = np.array(0.00000000001)

    nbData = np.shape(x)[1]
    num_obs = len(x_so[1, :])
    for i in range(num_obs):
        obs_R = np.array((r[i] + R) * k)
        x_obs = x_so[:, i]
        a = euclidean_distance(x, x_obs[:, np.newaxis])  # 当前各个demo与第i个障碍物的欧式距离

        # demo_no_crash_index = a > obs_R
        # a[demo_no_crash_index] = 0
        a = np.reshape(a, [1, nbData])
        theta = a - obs_R
        c = np.sqrt(np.square(theta) + 4 * xi)
        g = 0.5 * (c - theta)
        # obs_indx = g > 1e-6
        G.append(g)
    G_obs = np.sum(G, axis=0)
    # print(self.G_obs )
    # G_obs_hist.append(self.G_obs)
    return G_obs

def obs_check(x, x_so, r, k=1.1):    # 找是哪一个障碍物
    # r = 1.5  # 假设障碍物半径一致
    r = r
    # R = 1.35
    R = 0.45
    G = []
    xi = np.array(0.00000000001)

    for i in range(len(x_so[1, :])):
        obs_R = np.array((r[i] + R) * k)
        x_obs = x_so[:, i]
        a = euclidean_distance(x, x_obs[:, np.newaxis])  # 欧式距离
        a = np.reshape(a, [1, np.shape(x)[1]])
        theta = a - obs_R
        c = np.sqrt(np.square(theta) + 4 * xi)
        g = 0.5 * (c - theta)
        G.append(g.squeeze())
    obs_index = np.array(G) > 1e-7
    return obs_index


def obs_diy():
    x_position = np.array([np.arange(10, 19, 0.5)])
    y_position = np.array([np.arange(0, 9, 0.5)])
    obs_multi = np.concatenate((x_position, y_position), axis=0)
    # r = 1.5
    # vel_x_filter_mesh = vel_x.reshape(n_points, n_points).squeeze()
    # vel_y_filter_mesh = vel_y.reshape(n_points, n_points).squeeze()
    return obs_multi

def u_plot(x_sim, dx_sim, u, simulate_length=6000):
    # x_sim, x_sim_, dx_sim, u = simulate_trajectories(x_ini, x_obs, option, Vxf, gmm_params,
    #                                                  predict_model, corrected = False, trajectory_length=simulate_length,
    #                                                  original=False)
    x_sim = np.array(x_sim)
    dx_sim = np.array(dx_sim)
    u = np.array(u)
    # print('u =', u)

    index = -2
    x = np.array(x_sim[:, 0, index]).squeeze()  # .reshape([1,num])
    y = np.array(x_sim[:, 1, index]).squeeze()
    z1 = -np.array(u[:, 0, index]).squeeze()
    z2 = -np.array(u[:, 1, index]).squeeze()

    plot_num = 200
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
    # plt.title('Compare $v_x$ with $u_x$')  # 标题
    # plt.plot(x,y)
    # 常见线的属性有：color,label,linewidth,linestyle,marker等
    plt.plot(t, dxx, color='grey', label='$dx$')  # 'b'指：color='blue
    plt.plot(t, z1, 'o-', color='r', label='$u_x$')  # 'b'指：color='blue'
    plt.legend(loc = 0)  # 显示上面的label
    ax1.tick_params(axis='both', labelsize=16)
    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 16}
    plt.xlabel('step', fontdict=font1)
    plt.ylabel('v (m/s)', fontdict=font1)
    plt.yticks(fontproperties='Times New Roman', size=16)
    plt.xticks(fontproperties='Times New Roman', size=16)


    plt.figure(figsize=(6, 4))
    plt.style.use('seaborn-whitegrid')
    palette = plt.get_cmap('Set1')
    colorp = palette(8)
    # plt.xlim([0, 40])
    # plt.ylim([0, 30])
    plt.tight_layout()
    fig2, ax2 = plt.subplots()
    plt.title('Compare $v_y$ with $u_y$')  # 标题
    plt.plot(t, dxy, color='grey', label='$dy$')  # 'b'指：color='blue
    plt.plot(t, z2, 'o-', color='r', label='$u_y$')  # 'b'指：color='blue'
    plt.legend(loc = 0)  # 显示上面的label
    ax2.tick_params(axis='both', labelsize=16)
    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 16}
    plt.xlabel('step', fontdict=font1)
    plt.ylabel('v (m/s)', fontdict=font1)
    plt.yticks(fontproperties='Times New Roman', size=16)
    plt.xticks(fontproperties='Times New Roman', size=16)
    colorp = palette(8)

    z = np.vstack((z1**2, z2**2))
    norm_u = np.array(np.sqrt(np.sum(z, axis=0)))

    plt.figure(figsize=(4, 4))
    plt.style.use('seaborn-whitegrid')
    palette = plt.get_cmap('Set1')
    colorp = palette(8)
    plt.xlim([0, 40])
    plt.ylim([0, 30])
    plt.tight_layout()
    fig3, ax3 = plt.subplots(subplot_kw=dict(projection='3d'))
    ax3.stem(x, y, norm_u, orientation='z', linefmt='grey', markerfmt='o',)
    # ax1.stem(x, y, z2)
    # ax.stem(np.squeeze(x_sim[:, 0, :]), np.squeeze(x_sim[:, 1, :]), np.squeeze(u))
    ax3.view_init(elev=35,  # 仰角
                 azim=-45  # 方位角
                 )
    ax3.tick_params(axis='both', labelsize=16)
    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 16}
    plt.xlabel('x (m) ', fontdict=font1)
    plt.ylabel('y (m)', fontdict=font1)
    ax3.zaxis.set_rotate_label(True)  # 一定要先关掉默认的旋转设置  , rotation=90
    ax3.set_zlabel('$||u||   (m/s)$', fontdict=font1)
    plt.yticks(fontproperties='Times New Roman', size=16)
    plt.xticks(fontproperties='Times New Roman', size=16)
    # plt.title('V', fontdict=font1)
    plt.subplots_adjust(top=1, bottom=0, right=0.95, left=0, hspace=0, wspace=0)
    plt.margins(0, 0, 0)

    # plt.figure(figsize=(4, 4))
    # plt.style.use('seaborn-whitegrid')
    # palette = plt.get_cmap('Set1')
    # colorp = palette(8)
    # plt.xlim([0, 40])
    # plt.ylim([0, 30])
    # plt.tight_layout()
    # fig4, ax4 = plt.subplots(subplot_kw=dict(projection='3d'))
    # ax4.stem(y, x, z2,orientation='z', linefmt='grey', markerfmt='o',)
    # # ax1.stem(x, y, z2)
    # # ax.stem(np.squeeze(x_sim[:, 0, :]), np.squeeze(x_sim[:, 1, :]), np.squeeze(u))
    # ax4.view_init(elev=28,  # 仰角
    #              azim=12  # 方位角
    #              )
    # ax4.tick_params(axis='both', labelsize=16)
    # font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 16}
    # plt.xlabel('y (m) ', fontdict=font1)
    # plt.ylabel('x (m)', fontdict=font1)
    # ax4.zaxis.set_rotate_label(False)  # 一定要先关掉默认的旋转设置  , rotation=90
    # ax4.set_zlabel('||u_y|| (m/s)')
    # plt.yticks(fontproperties='Times New Roman', size=16)
    # plt.xticks(fontproperties='Times New Roman', size=16)
    # plt.title('Auxiliary control', fontdict=font1)
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.margins(0, 0, 0)
    # # plt.savefig('%s_vector_field_%s.pdf' % (data_name, plot_mode), dpi=300)
    plt.show()

def save_trajectory(x, dx, vu, x_obs, r, x_lim):
    num = x[0].shape[-1]
    x_sim = np.array(x)
    dx_sim = np.array(dx)
    u_sim = np.array(vu)
    state = np.array(())
    control = np.array(())
    u = np.array(())
    for i in range(x_sim.shape[-1]):
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
    tf = 10
    degree = 14
    testplot = polynomial_fitting.poly_fit(x_obs, x_lim, r)
    for i in range(int(0.5*num)):
        x = state[:,2*i].copy()
        y = state[:,2*i+1].copy()
        data = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
        last = 1
        plot_data = testplot.polyfit(data, tf, degree, i, last, plot=False)
    testplot.plot_rectangle(plot_data, multi=True)

    model_id = np.int64(time.strftime('%Y%m%d%H%M', time.localtime(time.time())))
    dataNew = './trajectory/T_goal=00_demo=' + str(num) + '_' + str(model_id) + '.mat'
    # sio.savemat(dataNew, {'state': state, 'control': control, 'u': u})
    return

def plot_heatmap(matrix, x_obs, r):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    plt.imshow(matrix, cmap='hot', interpolation='nearest')
    plt.colorbar()
    for i in range(x_obs.shape[1]):
        circle = plt.Circle((2.5*x_obs[0, i], 3.3*x_obs[1, i]), r,  linewidth=2, alpha=0.6)
        ax.add_artist(circle)
    # 设置坐标轴标签
    plt.xlabel('X')
    plt.ylabel('Y')

    # 设置标题
    plt.title('Heatmap')
    # 显示图像
    plt.show()

def plot_results(data, data_name, demoIdx, demo_length, dynamical_system, x_obs, x_new, new_obs, r, option, path, plot_mode, DIY, original,DS=True, extra=10, simulate_length=2000, n_points=200):
    # data输入为data_  0-40   original=False
    predict_model = load_model(path)

    # plt.rcdefaults()
    plt.rcParams.update({"text.usetex": False, "font.family": "Times New Roman", "font.size": 18})
    x_f = 0
    # x_f = 5.00
    y_f = 0
    goal = [x_f, y_f]

    # r = 1.5
    # x_lim = [[np.min(data[0, :]) - extra, np.max(data[0, :]) + extra],
    #          [np.min(data[1, :]) - extra, np.max(data[1, :]) + extra]]
    # x_lim = [[0 - extra, 30 + extra],
    #          [0 - extra, 15 + extra]]
    # x_lim = [[-15, 35],
    #          [-20, 30]]
    # x_lim = [[-5, 40],
    #          [-15, 30]]
    # x_lim = [[-0.5, 3.5],
    #          [-0.5, 3.5]]
    x_lim = [[-2., 8.],
             [-2., 8.]]
    # x_lim = [[np.min(data[0, :]), np.max(data[0, :])],
    #          [np.min(data[1, :]), np.max(data[1, :])]]

    nbData = data.shape[1]
    data_ = np.copy(data)
    # x0_all = [data_[:2, demoIdx]]  # finding initial points of all demonstrations
    x0_all = [data[:2, demoIdx]]  # finding initial points of all demonstrations
    nbData = x0_all[0].shape[1]
    x_hist = []
    x_hist_ = []

    if DIY is False:
        x_obs_ = np.copy(x_obs)
        n_obs = x_obs.shape[1]
        x_obs_copy = np.copy(x_obs)
        # x_obs_[0, :] = np.tile(x_f, [n_obs, 1]).T - x_obs_copy[0, :]
        # x_obs_[1, :] = np.tile(y_f, [n_obs, 1]).T - x_obs_copy[1, :]

        x_obs_m = np.copy(x_obs)
        x_obs_m_ = np.copy(x_obs_)

        if new_obs is True:
            x_new_ = np.copy(x_new)
            x_obs_m_ = np.hstack((x_obs_m_, x_new_))
            x_obs_m = np.hstack((x_obs_m, x_new))
    # Plot obstacles
    # x_obs_m = np.hstack((x_obs, x_new))
    else:
        x_obs_m, r = obs_diy()


    x_sim, x_sim_, dx_sim, u = simulate_trajectories(x0_all, x_obs_m, r, goal, dynamical_system, option, predict_model, ds=DS, trajectory_length=simulate_length, original=original)

    # save_trajectory(x_sim, dx_sim, u, x_obs_m, r, x_lim)
    # Plot simulated data
    x_sim = np.array(x_sim)
    dx_sim = np.array(dx_sim)
    plt.plot(x_sim[:, 0, :], x_sim[:, 1, :], color='white', linewidth=2, zorder=10)
    x_sim_ = np.array(x_sim_)


    # if original is False:
    #     x_f = 40.00
    #     y_f = 30.00
    #     nbData = data.shape[1]
    #     data[0, :] = np.tile(x_f, [nbData, 1]).T - data[0, :]
    #     data[1, :] = np.tile(y_f, [nbData, 1]).T - data[1, :]

    if plot_mode == 'velocities':
        # Plot demonstrations
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111)
        font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 22}
        # plt.title('Lyapunov Learner (CBF-DM)')
        plt.xlabel('Position x [m]', fontdict=font1)
        plt.ylabel('Position y [m]', fontdict=font1)
        plt.yticks(fontproperties='Times New Roman', size=18)
        plt.xticks(fontproperties='Times New Roman', size=18)

        # ax.tick_params(axis='both', labelsize=18)
        plt.style.use('seaborn-whitegrid')
        palette = plt.get_cmap('Set1')

        colorp = palette(8)

        cmap = plt.cm.Greys
        color = 'black'
        for i in range(int(data.shape[1] / demo_length)):
            plt.scatter(data_[0, i * demo_length:(i + 1) * demo_length], data_[1, i * demo_length:(i + 1) * demo_length],
                        color='white', zorder=5, alpha=0.5)

        # Plot goal
        plt.scatter(goal[0], goal[1], linewidth=4, color='blue', zorder=10)

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
        dx_hist = []

        # for ox in range(n_points):
        #     for oy in range(n_points):

        for i in range(n_points ** 2):  # TODO: do this in one pass
            x, x_, dx, _ = simulate_trajectories([x_init[:, i].reshape(-1, 1)], x_obs_m, r, goal, dynamical_system, option, predict_model, ds=DS, trajectory_length=1, original=original)

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

        obs_Swarning = barrier(x_init, x_obs_m, r, k=0.6)
        obs_Nwarning = barrier(x_hist, x_obs_m, r, k=0.6)   # x_ini的下一步会发生碰撞
        zeroS_condition = obs_Swarning.squeeze().T > 0.0001
        zeroN_condition = obs_Nwarning.squeeze().T > 0.0001
        # mask = np.logical_not(zero_condition)

        # x_init_x = np.copy(x_init[0, :])
        # x_filter = x_init_x[mask]
        # x_init_y = np.copy(x_init[1, :])
        # y_filter = x_init_y[mask]
        # n_new_point = np.round(np.sqrt(np.shape(x_filter)))
        # # x_init_ = np.vstack([x_filter, y_filter])
        # x_filter_mesh = x_filter.reshape(n_new_point, n_new_point, -1)
        # y_filter_mesh = y_filter.reshape(n_new_point, n_new_point, -1)

        # x_init_dx = np.copy(dx_hist[0, :])
        # dx_filter = x_init_dx[mask]
        # x_init_dy = np.copy(dx_hist[1, :])
        # dy_filter = x_init_dy[mask]

        # # x_init_ = np.vstack([x_filter, y_filter])
        # dx_filter_mesh = dx_filter.reshape(n_new_point, n_new_point, -1)
        # dy_filter_mesh = dy_filter.reshape(n_new_point, n_new_point, -1)

        # x_init_x = np.copy(x_init[0, :])
        x_init[0, :][zeroS_condition] = 0
        x_sum1 = np.sum(x_init[0, :])
        x_init[0, :][zeroN_condition] = 0   # 把置零条件归到一处
        x_sum2 = np.sum(x_init[0, :])

        # x_init_y = np.copy(x_init[1, :])
        x_init[1, :][zeroS_condition] = 0
        x_sum3 = np.sum(x_init[1, :])
        x_init[1, :][zeroN_condition] = 0
        x_sum4 = np.sum(x_init[1, :])

        x_filter_mesh = x_init[0, :].reshape(n_points, n_points).squeeze()
        y_filter_mesh = x_init[1, :].reshape(n_points, n_points).squeeze()

        mask = np.zeros(x_init[0, :].shape, dtype=bool)
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

        # Plot vector field
        # vel[v_condition1 | v_condition2] = 0


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
        norm_vel = np.clip(np.linalg.norm(vel, axis=2), a_min=0, a_max=7)  # norm相当于求了v_xy
        print('vel=', vel)

        # CS = plt.contourf(x1_coords, x2_coords, norm_vel, cmap='cividis', levels=50, zorder=1)
        CS = plt.contourf(x1_coords, x2_coords, norm_vel, levels=50, cmap='RdGy')

        print('norm_vel =', norm_vel)
        min_vel_ceil = np.ceil(np.min(norm_vel))
        max_vel_floor = np.ceil(np.max(norm_vel))
        delta_x = max_vel_floor / 10
        cbar = plt.colorbar(CS, ticks=np.arange(0, max_vel_floor, delta_x))
        cbar.ax.set_xlabel('speed (m/s)')
        # , location = 'bottom'

        # plt.plot(x_sim[:, 0, :], x_sim[:, 1, :], color='white', linewidth=2, zorder=10)

        length = len(x_obs_m[0])


        for i in range(length):
            x = np.linspace(x_obs_m[0, i] - r[i], x_obs_m[0, i] + r[i], 1000)
            y1 = np.sqrt(r[i] ** 2 - (x - x_obs_m[0, i]) ** 2) + x_obs_m[1, i]
            y2 = -np.sqrt(r[i] ** 2 - (x - x_obs_m[0, i]) ** 2) + x_obs_m[1, i]
            plt.plot(x, y1, c='k')
            plt.plot(x, y2, c='k')

            # 填充色块
            plt.fill_between(x, y1, y2, color=color, alpha=0.5,  linewidth=3)

            # # 填充白色搞阴影
            # for i in np.linspace(0, 1, 10):  # 第三个参数调整间距
            #     a = x = np.linspace(-8, 4, 9)  # 可以调位置
            #     b = a + i
            #     c = a + i - 0.1
            #     plt.fill_between(a, b, c, facecolor='white')


        # for i in range(x_obs_m_.shape[1]):
        #     circle = plt.Circle((x_obs_m_[0, i], x_obs_m_[1, i]), r, color='grey', linewidth=3)
        #     ax.add_artist(circle)
        # plot_heatmap(vel_x_filter_mesh, x_obs, r)
        # plot_heatmap(vel_y_filter_mesh, x_obs, r)

    elif plot_mode == 'energy_levels':
        # Plot demonstrations

        for i in range(int(data.shape[1] / demo_length)):
            plt.scatter(data_[0, i * demo_length:(i + 1) * demo_length], data_[1, i * demo_length:(i + 1) * demo_length],
                        color='blue', zorder=5, alpha=0.5)

        # Plot energy levels
        dynamical_system.lyapunov.energyContour(dynamical_system.Vxf, x_lim, x_obs_m, r, original=True, DIY=DIY)
        # u_plot(x_sim, dx_sim, u, simulate_length=simulate_length)

    else:
        print('Selected print mode not valid!')
        exit()

    plt.xlim(x_lim[0])
    plt.ylim(x_lim[1])
    plt.tight_layout()
    plt.savefig('%s_vector_field_%s.pdf' % (data_name, plot_mode), dpi=300)
    plt.show()




