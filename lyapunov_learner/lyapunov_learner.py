__credits__ = 'Olalekan Ogunmolu, Rodrigo Perez-Dattari (TU Delft), Rachel Thomson (MIT), Jethro Tan (PFN)'
__license__ = 'MIT'

import numpy as np
import scipy as sp
import scipy.linalg as LA
from scipy.optimize import minimize, NonlinearConstraint, BFGS
from gmm.gmm_utils import gmm_2_parameters, parameters_2_gmm, shape_DS, gmr_lyapunov
import matplotlib.pyplot as plt
from bls import *
import pickle
import data_preprocess
import matplotlib

# matplotlib.use('Qt5Agg')

class LyapunovLearner():
    def __init__(self):
        """
            Class that estimates lyapunov energy function
        """
        self.G_obs = 0
        self.Nfeval = 0
        self.success = True   # boolean indicating if constraints were violated
        self.G_obs_hist = []

    def guess_init_lyap(self, Vxf0):
        """
        This function guesses the initial lyapunov function
        """
        b_initRandom = Vxf0['int_lyap_random']
        retrain = Vxf0['int_lyap_re']
        Vxf0['Mu'] = np.zeros((Vxf0['d'], Vxf0['L'] + 1))
        Vxf0['P'] = np.zeros((Vxf0['d'], Vxf0['d'], Vxf0['L'] + 1))

        if b_initRandom:
            """
             If `rowvar` is True (default), then each row represents a
            variable, with observations in the columns. Otherwise, the relationship
            is transposed: each column represents a variable, while the rows
            contain observations.
            """
            Vxf0['Priors'] = np.random.rand(Vxf0['L'] + 1, 1)

            for l in range(Vxf0['L'] + 1):
                tempMat = np.random.randn(Vxf0['d'], Vxf0['d'])
                Vxf0['Mu'][:, l] = np.random.randn(Vxf0['d'])
                Vxf0['P'][:, :, l] = tempMat
        else:
            Vxf0['Priors'] = np.ones((Vxf0['L'] + 1, 1))
            Vxf0['Priors'] = Vxf0['Priors'] / np.sum(Vxf0['Priors'])
            Vxf0['P'] = []
            for l in range(Vxf0['L'] + 1):
                Vxf0['P'].append(np.eye(Vxf0['d'], Vxf0['d']))

            Vxf0['P'] = np.reshape(Vxf0['P'], [Vxf0['L'] + 1, Vxf0['d'], Vxf0['d']])

        Vxf0.update(Vxf0)

        if retrain:
            Vxf0['Priors'] = np.ones((Vxf0['L'] + 1, 1))
            Vxf0['Priors'] = Vxf0['Priors'] / np.sum(Vxf0['Priors'])
            Vxf0['P'] = []
            for l in range(Vxf0['L'] + 1):
                Vxf0['P'].append(np.eye(Vxf0['d'], Vxf0['d']))

            Vxf0['P'] = np.reshape(Vxf0['P'], [Vxf0['L'] + 1, Vxf0['d'], Vxf0['d']])
            Vxf0.update(Vxf0)
        return Vxf0


    def matVecNorm(self, x):
        return np.sqrt(np.sum(x**2, axis=0))


    def euclidean_distance(self, vectors, vector, Sum=True):
        # 计算向量的欧氏距离
        diff = vectors - vector
        squared_diff = np.square(diff)
        if Sum is True:
            summed = np.sum(squared_diff, axis=0)
        else:
            summed = squared_diff
        distance = np.sqrt(summed)
        return distance    # 1xN

    def barrier(self, x, x_so, r, k=1.1):  # x(state,length)
        g = [np.array(())]
        G = []
        G_new = []
        G_obs = []
        # r = 1.5  # 假设障碍物半径一致
        r = r
        # R = 1.35
        R = 0.45
        gain = 1.4
        xi = np.array(0.00000000001)

        nbData = np.shape(x)[1]
        num_obs = len(x_so[1, :])
        for i in range(num_obs):
            obs_R = np.array((r[i] + R) * k)
            x_obs = x_so[:, i]
            a = self.euclidean_distance(x, x_obs[:, np.newaxis])  # 欧式距离
            # a = np.linalg.norm(x - x_obs[:, np.newaxis])
            a = np.reshape(a, [1, nbData])
            theta = a - obs_R
            c = np.sqrt(np.square(theta) + 4 * xi)
            g = 0.5 * (c - theta)
            # obs_indx = g > 1e-6
            G.append(g * gain)
        self.G_obs = np.sum(G, axis=0)
        # print(self.G_obs )
        # self.G_obs_hist.append(self.G_obs)
        return G_obs

    def obs_check(self, x, x_so, r, k=1.1):  # 找是哪一个障碍物
        # r = 1.5  # 假设障碍物半径一致
        r = r
        R = 1.35
        G = []
        xi = np.array(0.00000000001)

        for i in range(len(x_so[1, :])):
            obs_R = np.array((r[i] + R) * k)
            x_obs = x_so[:, i]
            a = self.euclidean_distance(x, x_obs[:, np.newaxis])  # 欧式距离
            a = np.reshape(a, [1, np.shape(x)[1]])
            theta = a - obs_R
            c = np.sqrt(np.square(theta) + 4 * xi)
            g = 0.5 * (c - theta)
            G.append(g.squeeze())
        obs_index = np.array(G) > 1e-7
        return obs_index

    def barrier(self, x, x_so, r, k=1.1):   # x(state,length)
        g = [np.array(())]            # next
        G = []
        G_new = []
        r = 1.5   # 假设障碍物半径一致
        R = 1.35
        gain = 1
        xi = np.array(0.00000000001)
        obs_R = np.array((r+R)*k)
        nbData = np.shape(x)[1]
        num_obs = len(x_so[1, :])
        for i in range(num_obs):
            x_obs = x_so[:, i]
            a = self.euclidean_distance(x, x_obs[:, np.newaxis])   # 欧式距离
            a = np.reshape(a, [1, nbData])
            theta = a - obs_R
            c = np.sqrt(np.square(theta) + 4*xi)
            g = c - theta
            G.append(0.5*g*gain)
            # if new_obs is True:
            #     num_obs_new = len(x_ne[1, :])
            #     for j in range(num_obs_new):
            #         x_new = x_ne[:, j]
            #         a_ = self.euclidean_distance(x, x_new[:, np.newaxis])  # 欧式距离
            #         theta_ = a_ - obs_R
            #         c_ = np.sqrt(np.square(theta_) + 4 * xi)
            #         g = c_ - theta_
            #         g += g
            #         G_new.append(g * 0.5*gain)
            #     G += G_new
        self.G_obs = np.sum(G, axis=0)
        # print(self.G_obs )
        self.G_obs_hist.append(self.G_obs)

    def obs_diy(self):     # 多个密集的obs叠加
        x_position = np.array([np.arange(10, 19, 0.5)])
        y_position = np.array([np.arange(0, 9, 0.5)])
        obs_multi = np.concatenate((x_position, y_position), axis=0)
        # r = 1.5
        # vel_x_filter_mesh = vel_x.reshape(n_points, n_points).squeeze()
        # vel_y_filter_mesh = vel_y.reshape(n_points, n_points).squeeze()
        return obs_multi

    def obj(self, p, x, xd, d, L, w, options, x_obs, r, original, DIY):
        L_p = 0
        Vxf = shape_DS(p, d, L, L_p, options)
        Vxf.update(Vxf)    # 字典对应键值的更新
        # x_f = 40
        # y_f = 30
        # nbData = x.shape[1]
        # x[0, :] = x[0, :] - np.tile(x_f, [nbData, 1]).T
        # x[1, :] = x[1, :] - np.tile(y_f, [nbData, 1]).T
        if DIY is True:
            obs = self.obs_diy()
        else:
            obs = x_obs
        # self.barrier(x, x_obs)
        self.barrier(x, obs, r, k=1.1)
        _, Vx = self.computeEnergy(x, np.array(()), self.G_obs, Vxf, nargout=2, original=original)

        Vdot = np.sum(Vx * xd, axis=0)  # derivative of J w.r.t. xd
        norm_Vx = np.sqrt(np.sum(Vx * Vx, axis=0))
        norm_xd = np.sqrt(np.sum(xd * xd, axis=0))
        Vdot = np.expand_dims(Vdot, axis=0)
        norm_Vx = np.expand_dims(norm_Vx, axis=0)
        norm_xd = np.expand_dims(norm_xd, axis=0)
        butt = norm_Vx * norm_xd

        J = Vdot / (butt + w)
        J[np.where(norm_xd == 0)] = 0
        J[np.where(norm_Vx == 0)] = 0
        J[np.where(Vdot > 0)] = J[np.where(Vdot > 0)] ** 2
        # if new_obs is True:
        #     w = 1e-4
        # else:
        #     w = w
        J[np.where(Vdot < 0)] = -w * J[np.where(Vdot < 0)] ** 2
        # J += self.G_obs
        J = np.sum(J, axis=1)
        return J

    def callback_opt(self, Xi, y):
        print('Iteration: {}   Cost: {}'.format([self.Nfeval], y.fun))
        self.Nfeval += 1

    def optimize(self, obj_handle, ctr_handle_ineq, ctr_handle_eq, p0):
        nonl_cons_ineq = NonlinearConstraint(ctr_handle_ineq, -np.inf, 0, jac='3-point', hess=BFGS())
        nonl_cons_eq = NonlinearConstraint(ctr_handle_eq, 0, 0, jac='3-point', hess=BFGS())

        solution = minimize(obj_handle,
                            np.reshape(p0, [len(p0)]),
                            hess=BFGS(),
                            constraints=[nonl_cons_eq, nonl_cons_ineq],
                            method='trust-constr', tol=1e-6, options={'disp': True, 'initial_constr_penalty': 1.5, 'maxiter': 300}
                            , callback=self.callback_opt)     #
        print('solution_x =', solution.x)
        print('solution_func =', solution.fun)
        return solution.x, solution.fun

    def ctr_eigenvalue_ineq(self, p, d, L, options):
        # This function computes the derivative of the constrains w.r.t.
        # optimization parameters.
        L_p = 0
        Vxf = shape_DS(p, d, L, L_p, options)
        if L > 0:
            c = np.zeros(((L + 1) * d + (L + 1) * options['optimizePriors'], 1))  # +options.variableSwitch
        else:
            c = np.zeros((d, 1))

        for k in range(L + 1):
            lambder = sp.linalg.eigvals(Vxf['P'][k, :, :] + (Vxf['P'][k, :, :]).T)
            lambder = np.divide(lambder.real, 2.0)
            lambder = np.expand_dims(lambder, axis=1)
            c[k * d:(k + 1) * d] = -lambder.real + options['tol_mat_bias']

        if L > 0 and options['optimizePriors']:
            c[(L + 1) * d:(L + 1) * d + L + 1] = np.reshape(-Vxf['Priors'], [L + 1, 1])

        return np.reshape(c, [len(c)])

    def ctr_eigenvalue_eq(self, p, d, L, options):
        # This function computes the derivative of the constrains w.r.t.
        # optimization parameters.
        L_p = 0
        Vxf = shape_DS(p, d, L, L_p, options)
        if L > 0:
            if options['upperBoundEigenValue']:
                ceq = np.zeros((L + 1, 1))
            else:
                ceq = np.array(())  # zeros(L+1,1);
        else:
            ceq = (np.ravel(Vxf['P']).T).dot(np.ravel(Vxf['P'])) - 2

        for k in range(L + 1):
            lambder = sp.linalg.eigvals(Vxf['P'][k, :, :] + (Vxf['P'][k, :, :]).T)
            lambder = np.divide(lambder.real, 2.0)
            lambder = np.expand_dims(lambder, axis=1)
            if options['upperBoundEigenValue']:
                ceq[k] = 1.0 - np.sum(lambder.real)  # + Vxf.P(:,:,k+1)'

        return np.reshape(ceq, [len(ceq)])

    def check_constraints(self, p, ctr_handle, d, L, options):
        c = -ctr_handle(p)

        if L > 0:
            c_P = c[:L*d].reshape(d, L).T
        else:
            c_P = c

        i = np.where(c_P <= 0)
        # self.success = True

        if i:
            self.success = False
        else:
            self.success = True

        if L > 1:
            if options['optimizePriors']:
                c_Priors = c[L*d+1:L*d+L]
                i = np.nonzero(c_Priors < 0)

                if i:
                    self.success = False
                else:
                    self.success = True

            if len(c) > L*d+L:
                c_x_sw = c[L*d+L+1]
                if c_x_sw <= 0:
                    self.success = False
                else:
                    self.success = True

    def computeEnergy(self, X, Xd, obs, Vxf, nargout=2, original=True, retrain=False):
        d = X.shape[0]
        nDemo = 1
        if nDemo > 1:
            X = X.reshape(d, -1)
            Xd = Xd.reshape(d, -1)
        if original is True:
            x_f = 0
            y_f = 0
            nbData = X .shape[1]
            X[0, :] = X[0, :] - np.tile(x_f, [nbData, 1]).T
            X[1, :] = X[1, :] - np.tile(y_f, [nbData, 1]).T
            # Xd[0, :] = - Xd[0, :]
            # Xd[1, :] = - Xd[1, :]

        if retrain:
            v = open('./V/V_re.pkl', 'wb')
            pickle.dump(Vxf, v)
            v.close()

        V, dV = gmr_lyapunov(X, obs, Vxf['Priors'], Vxf['Mu'], Vxf['P'])

        if nargout > 1:
            if not Xd:
                Vdot = dV
            else:
                Vdot = np.sum((Xd) * dV, axis=0)       # dx = np.array(())
        if nDemo > 1:
            V = V.reshape(-1, nDemo).T
            if nargout > 1:
                Vdot = Vdot.reshape(-1, nDemo).T

        return V, Vdot

    def bls_learnDS(self, x, xd):
        d = x.shape[0]
        x = x.reshape(-1, d)
        xd = xd.reshape(-1, d)
        length = len(x)
        print('length = ', length)
        normalize = data_preprocess
        X, Xd = normalize.state_normalzation(x, xd)
        rate = 0.9
        state_train = X[0:int(length * rate), :]
        control_train = Xd[0:int(length * rate), :]
        state_test = X[int(length * rate):2 * int(length * rate), :]
        control_test = Xd[int(length * rate):2 * int(length * rate):, :]
        # traindata,testdata,trainlabel,testlabel = train_test_split(X,Y,test_size=0.2,random_state = 2018)
        print(state_train.shape, control_train.shape, state_test.shape, control_test.shape)

        NumFea = 6
        NumWin = 30
        NumEnhan = 136
        NumWinE = 1

        # best_loss = 0.1
        bls = broadnet_mapping(maptimes=NumWin,
                               enhencetimes=NumWinE,
                               traintimes=100,
                               map_function='linear',
                               enhence_function='tanh',
                               # batchsize = 'auto',
                               enhancebs=NumEnhan,
                               batchsize=NumFea,
                               acc=0.001,
                               step=1,
                               reg=0.001,
                               deep_num=2)
        # print(bls)
        bls.fit(state_train, control_train)
        predictlabel = bls.predict(state_test)
        test_mae = show_accuracy(predictlabel, control_test)
        bls.save_model()
        print('test_mae =', test_mae)
        # if np.sum(test_mae) < best_loss:
        #     bls.save_model()
        #     best_loss = np.sum(test_mae)



    def learnEnergy(self, Vxf0, Data, options, x_obs, x_new, new_obs, r, original, DIY):
        d = Vxf0['d']
        x = Data[:d, :]   # x
        xd = Data[d:, :]   # dx

        # Transform the Lyapunov model to a vector of optimization parameters
        for l in range(Vxf0['L']):
            try:
                Vxf0['P'][l + 1, :, :] = sp.linalg.solve(Vxf0['P'][l + 1, :, :], sp.eye(d))   # 验证是否正定
            except sp.linalg.LinAlgError as e:
                print('Error lyapunov solver.')

        # in order to set the first component to be the closest Gaussian to origin
        to_sort = self.matVecNorm(Vxf0['Mu'])
        idx = np.argsort(to_sort, kind='mergesort')   # 排序 返回索引号序列
        Vxf0['Mu'] = Vxf0['Mu'][:, idx]
        Vxf0['P'] = Vxf0['P'][idx, :, :]
        p0 = gmm_2_parameters(Vxf0, options)

        # account for targets in x and xd
        obj_handle = lambda p: self.obj(p, x, xd, d, Vxf0['L'], Vxf0['w'], options, x_obs, r, original, DIY)    # p的函数呀 优化p
        ctr_handle_ineq = lambda p: self.ctr_eigenvalue_ineq(p, d, Vxf0['L'], options)
        ctr_handle_eq = lambda p: self.ctr_eigenvalue_eq(p, d, Vxf0['L'], options)

        popt, J = self.optimize(obj_handle, ctr_handle_ineq, ctr_handle_eq, p0)   # 返回优化后的p和学习到的目标函数值

        # transforming back the optimization parameters into the GMM model
        Vxf = parameters_2_gmm(popt, d, Vxf0['L'], options)
        Vxf['Mu'][:, 0] = 0
        Vxf['L'] = Vxf0['L']
        Vxf['d'] = Vxf0['d']
        Vxf['w'] = Vxf0['w']
        self.success = True

        sumDet = 0
        for l in range(Vxf['L'] + 1):
            sumDet += np.linalg.det(Vxf['P'][l, :, :])

        Vxf['P'][0, :, :] = Vxf['P'][0, :, :] / sumDet
        Vxf['P'][1:, :, :] = Vxf['P'][1:, :, :] / np.sqrt(sumDet)

        self.bls_learnDS(x, xd)

        return Vxf, J


    def surface(self, x, y, z, contour_levels):
        # fig = plt.figure(figsize=(6, 6))
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot_surface(x, y, z, cmap='RdGy')
        # ax.contour(x, y, z, contour_levels, zdir='z', offset=-1, cmap='RdGy', origin='upper')
        #
        # ax.view_init(elev=35,  # 仰角
        #              azim=135  # 方位角
        #              )
        # ax.tick_params(axis='both', labelsize=16)
        # font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 16}
        # plt.xlabel('x (m) ', fontdict=font1)
        # plt.ylabel('y (m)', fontdict=font1)
        # ax.zaxis.set_rotate_label(True)  # 一定要先关掉默认的旋转设置  , rotation=90
        # ax.set_zlabel('$ V $', fontdict=font1)
        # plt.subplots_adjust(top=0.95, bottom=0.1, right=0.95, left=0.1)
        # plt.show()

        fig = plt.figure(figsize=(6, 6))
        # cmap = 'viridis'
        cmap = 'RdGy'
        ax = fig.add_subplot(111, projection='3d')
        ax_t = plt.gca()

        # for i in range(x_sim.shape[2]):
        #     ax.plot(x_sim[1:, 0, i], x_sim[1:, 1, i], v[:, i] , 'o-', color='b')

        ax.plot_surface(x, y, z, cmap=cmap, alpha=0.96)

        plt.rcParams['axes.facecolor'] = 'white'
        ax.contour(x, y, z, contour_levels, zdir='z', offset=-1, cmap=cmap)
        plt.plot(0, 0, 'b*', markersize=15, linewidth=3, label='target', zorder=12)
        ax.view_init(elev=35,  # 仰角
                     azim=-145  # 方位角
                     )
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        # ax.tick_params(axis='both', labelsize=16)
        # font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}
        # font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 16}
        # plt.xlabel('x [m]', fontdict=font1)
        # plt.ylabel('y [m]', fontdict=font1)
        # plt.yticks(fontproperties='Times New Roman', size=16)
        # plt.xticks(fontproperties='Times New Roman', size=16)
        ax.zaxis.set_rotate_label(False)  # 一定要先关掉默认的旋转设置  , rotation=90
        # ax.set_zlabel('$ V $', fontdict=font2)
        # ax.set_zlabel('V', fontsize=16, fontname='Times New Roman')
        plt.ticklabel_format(style='sci', scilimits=(0, 0), axis='both')
        # ax.zaxis.set_tick_params(labelsize=20, labelrotation=45)

        ax.set_xlabel('x [m]', fontsize=20, labelpad=3, fontname='Times New Roman')
        ax.set_ylabel('y [m]', fontsize=20, labelpad=3, fontname='Times New Roman')
        ax.set_zlabel('V', fontsize=20, labelpad=3, fontname='Times New Roman')

        ax.xaxis.set_tick_params(labelsize=16, pad=1)
        ax.yaxis.set_tick_params(labelsize=16, pad=1)
        ax.zaxis.set_tick_params(labelsize=16, pad=1)  # 设置x轴、y轴和z轴刻度字体和字号
        for t in ax.xaxis.get_major_ticks():
            t.label.set_fontsize(16)
            t.label.set_fontname('Times New Roman')

        for t in ax.yaxis.get_major_ticks():
            t.label.set_fontsize(16)
            t.label.set_fontname('Times New Roman')

        for t in ax.zaxis.get_major_ticks():
            t.label.set_fontsize(16)
            t.label.set_fontname('Times New Roman')
            # formatter = ticker.ScalarFormatter(useMathText=True)
        # formatter.set_scientific(True)
        # formatter.set_powerlimits((0, 0))
        # ax_t.z

        # ax.axis('off')
        # ax.set_zticks([])  # 不显示z坐标轴
        plt.subplots_adjust(top=1, bottom=0.02, right=0.95, left=0.1, hspace=0, wspace=0)
        plt.grid(True)
        plt.show()

    def energyContour(self, Vxf, D, x_obs_, r, original, DIY):
        quality ='high'
        b_plot_contour = True
        contour_levels = np.array([])

        if quality == 'high':
            nx, ny = 0.1, 0.1
        elif quality == 'medium':
            nx, ny = 1, 1
        else:
            nx, ny = 2, 2

        # # x = np.arange(D[0][0], D[0][1], nx)   # linspace
        x = np.arange(D[0][0], D[0][1], nx)
        y = np.arange(D[1][0], D[1][1], ny)
        # x_ = np.arange(D[0][1], D[0][0], nx)
        # y_ = np.arange(D[1][1], D[1][0], ny)
        x_len = len(x)
        y_len = len(y)
        X, Y = np.meshgrid(x, y)
        x = np.stack([np.ravel(X), np.ravel(Y)])
        # X_, Y_ = np.meshgrid(x_, y_)
        # x_ = np.stack([np.ravel(X_), np.ravel(Y_)])

        if DIY is True:
            obs = self.obs_diy()
        else:
            obs = x_obs_
        # self.barrier(x, x_obs)
        self.barrier(x, obs, r, k=1.1)

        # self.barrier(x, x_obs_)    # 坐标x是没有做转换的，所以obs也应该一致

        # print('x_len=', x_len)
        # print('x.shape[0]', x.shape[0])
        # print('x.shape[1]', x.shape[1])

        # if original is True:     # learning的时候 data_inv， plot的时候 data一致
        #     x_f = 40.00
        #     # x_f = 5
        #     y_f = 30.00
        #     nbData = x.shape[1]
        #     x[0, :] = x[0, :] - np.tile(x_f, [nbData, 1]).T
        #     x[1, :] = x[1, :] - np.tile(y_f, [nbData, 1]).T
        #     # x[0, :] = x[0, :] - np.tile(x_f, [nbData, 1]).T
        #     # x[1, :] = x[1, :] - np.tile(y_f, [nbData, 1]).T

        V, dV = self.computeEnergy(x, np.array(()), self.G_obs, Vxf, nargout=2, original=True)   # 这里要算V，因此需要临时对x做转换

        if not contour_levels.size:
            contour_levels = np.arange(0, np.log(np.max(V)), 0.1)
            contour_levels = np.exp(contour_levels)
            if np.max(V) > 40:
                contour_levels = np.round(contour_levels)
        norm_Vx = np.sqrt(np.sum(V * V, axis=0))
        V = V / norm_Vx
        V = V.reshape(y_len, x_len)
        print('V = ', V)
        min_vel_ceil = np.ceil(np.min(V))
        max_vel_floor = np.ceil(np.max(V))
        delta_x = max_vel_floor / 100

        plt.rcParams.update({"text.usetex": False, "font.family": "Times New Roman", "font.size": 18})
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

        plt.plot(0, 0, 'g*', markersize=15, linewidth=3, label='target', zorder=12)
        if b_plot_contour:
            # h = plt.contour(X, Y, V, contour_levels, cmap='RdGy', origin='upper', linewidths=0.1)  # ,labelspacing=200
            CS = plt.contourf(X, Y, V, levels=160, cmap='RdGy')
            # cbar = plt.colorbar(CS, ticks=np.arange(0, 2500, 500))
            # cbar.ax.set_xlabel('V')
            # plt.clabel(CS, inline=True, fontsize=10)
            # plt.xticks(())
            # plt.yticks(())

            self.surface(X, Y, V, contour_levels)
        return CS

    #, location='bottom'
