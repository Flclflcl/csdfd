__credits__ = 'Olalekan Ogunmolu, Rodrigo Perez-Dattari (TU Delft), Rachel Thomson (MIT), Jethro Tan (PFN)'
__license__ = 'MIT'

import numpy as np
import scipy.linalg
from scipy.cluster.vq import kmeans2
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt

class resample:
    def __init__(self,x, num):
        self.x = x
        # self.y = y
        self.num = num
        self.kind = [3]
        self.len = len(self.x)-1

    # def __len__(self):
    #     return len(self.x)

    def interp(self):

        index = np.linspace(0, self.len, self.num)
        # print(index)
        # print(index.shape)

        # 增加
        # f = interpolate.interp1d(self.x, self.y, kind='cubic')

        # 减少
        index = np.trunc(index).astype('int64')
        output = self.x[index]
        return output


def logsum(vec, axis=0, keepdims=True):
    #TODO: Add a docstring.
    maxv = np.max(vec, axis=axis, keepdims=keepdims)
    maxv[maxv == -float('inf')] = 0
    return np.log(np.sum(np.exp(vec-maxv), axis=axis, keepdims=keepdims)) + maxv


def check_sigma(A):
    """
        checks if the sigma matrix is symmetric
        positive definite before inverting via cholesky decomposition
    """
    eigval = np.linalg.eigh(A)[0]
    if np.array_equal(A, A.T) and np.all(eigval>0):
        # logger.debug("sigma is pos. def. Computing cholesky factorization")
        return A
    else:
        # find lowest eigen value
        eta = 1e-6  # regularizer for matrix multiplier
        low = np.amin(np.sort(eigval))
        Anew = low * A + eta * np.eye(A.shape[0])
        return Anew


class GMM(object):
    """ Gaussian Mixture Model. """
    def __init__(self, num_clusters=6, init_sequential=False, eigreg=False, warmstart=True):
        self.init_sequential = init_sequential
        self.eigreg = eigreg
        self.warmstart = warmstart
        self.sigma = None

        # mine June 26
        self.K = num_clusters
        self.fail = None

        # regularization parameters
        self.eta = 1e-6
        self.delta = 1e-4
        self.eta_min = 1e-6
        self.delta_nut = 2

    def inference(self, pts):
        """
            Evaluate dynamics prior.
            Args:
                pts: A N x D array of points.
        """
        # Compute posterior cluster weights.
        logwts = self.clusterwts(pts)

        # Compute posterior mean and covariance.
        mu0, Phi = self.moments(logwts)

        # Set hyperparameters.
        m = self.N
        n0 = m - 2 - mu0.shape[0]

        # Normalize.
        m = float(m) / self.N
        n0 = float(n0) / self.N
        return mu0, Phi, m, n0

    def clusterwts(self, data):
        """
        Compute cluster weights for specified points under GMM.
        Args:
            data: An N x D array of points
        Returns:
            A K x 1 array of average cluster log probabilities.
        """
        # Compute probability of each point under each cluster.
        logobs = self.estep(data)

        # Renormalize to get cluster weights.
        logwts = logobs - logsum(logobs, axis=1)

        # Average the cluster probabilities.
        logwts = logsum(logwts, axis=0) - np.log(data.shape[0])
        return logwts.T

    def reg_sched(self, increase=False):
        # increase mu
        if increase:
            self.delta = max(self.delta_nut, self.delta * self.delta_nut)
            eta = self.eta * 1.1
        else: # decrease eta
            eta = self.eta
            eta *= 0.09
        self.eta = eta

    def estep(self, data):
        """
        Compute log observation probabilities under GMM.
        Args:
            data: A N x D array of points.
        Returns:
            logobs: A N x K array of log probabilities (for each point
                on each cluster).
        """
        # Constants.
        N, D = data.shape
        K = self.sigma.shape[0]

        logobs = -0.5*np.ones((N, K))*D*np.log(2*np.pi)

        self.fail = True
        while(self.fail):

            self.fail = False

            for i in range(K):
                # print('sigma i ', self.sigma[i].shape, np.eye(self.sigma[i].shape[-1]).shape)
                # print('eta: ', self.eta)
                self.sigma[i] += self.eta * np.eye(self.sigma[i].shape[-1])
                mu, sigma = self.mu[i], self.sigma[i]
                # logger.debug('sigma: {}\n'.format(sigma))
                try:
                    L = scipy.linalg.cholesky(sigma, lower=True)
                except LinAlgError as e:
                    self.fail = True
                    break
                logobs[:, i] -= np.sum(np.log(np.diag(L)))
                diff = (data - mu).T
                soln = scipy.linalg.solve_triangular(L, diff, lower=True)
                logobs[:, i] -= 0.5*np.sum(soln**2, axis=0)

            if self.fail:
                old_eta = self.eta
                self.reg_sched(increase=True)
            else:
                # if successful, decrese mu
                old_eta = self.eta
                self.reg_sched(increase=False)

        logobs += self.logmass.T
        return logobs

    def moments(self, logwts):
        """
            Compute the moments of the cluster mixture with logwts.
            Args:
                logwts: A K x 1 array of log cluster probabilities.
            Returns:
                mu: A (D,) mean vector.
                sigma: A D x D covariance matrix.
        """
        # Exponentiate.
        wts = np.exp(logwts)

        # Compute overall mean.
        mu = np.sum(self.mu * wts, axis=0)

        # Compute overall covariance.
        diff = self.mu - np.expand_dims(mu, axis=0)
        diff_expand = np.expand_dims(self.mu, axis=1) * \
                np.expand_dims(diff, axis=2)
        wts_expand = np.expand_dims(wts, axis=2)
        sigma = np.sum((self.sigma + diff_expand) * wts_expand, axis=0)
        return mu, sigma

    def kmeans_plot(self, data, label, centroid, x_f, y_f):
        z = np.array(data.T).copy()
        centroid_ = np.array(centroid.T).copy()
        nbData = z.shape[1]
        nbCentroid = centroid_.shape[1]
        # z[0, :] = np.tile(x_f, [nbData, 1]).T - z[0, :]
        # z[0, :] = z[0, :] + np.tile(x_f, [nbData, 1]).T
        # z[1, :] = np.tile(y_f, [nbData, 1]).T - z[1, :]
        # centroid_[0, :] = np.tile(x_f, [nbCentroid, 1]).T - centroid_[0, :]
        # centroid_[0, :] = centroid_[0, :] + np.tile(x_f, [nbCentroid, 1]).T
        # centroid_[1, :] = np.tile(y_f, [nbCentroid, 1]).T - centroid_[1, :]

        plot_num = 300
        z_x = resample(z[0, :], plot_num)
        z_xre = z_x.interp()
        z_y = resample(z[1, :], plot_num)
        z_yre = z_y.interp()
        z_re = np.vstack((z_xre, z_yre)).T
        label_ = resample(label[::-1], plot_num)
        label_re = label_.interp()
        w0 = z_re[label_re == 0]
        w1 = z_re[label_re == 1]
        w2 = z_re[label_re == 2]
        w3 = z_re[label_re == 3]
        w4 = z_re[label_re == 4]
        w5 = z_re[label_re == 5]

        plt.tick_params(axis='both', labelsize=16)
        font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}
        plt.xlabel('x (m) ', fontdict=font1)
        plt.ylabel('y (m)', fontdict=font1)

        plt.plot(w0[:, 0], w0[:, 1], 'o', alpha=0.3, label='cluster 0')
        plt.plot(w1[:, 0], w1[:, 1], 'd', alpha=0.3, label='cluster 1')
        plt.plot(w2[:, 0], w2[:, 1], 's', alpha=0.3, label='cluster 2')
        plt.plot(w3[:, 0], w3[:, 1], 'p', alpha=0.3, label='cluster 3')
        plt.plot(w4[:, 0], w4[:, 1], 'h', alpha=0.3, label='cluster 4')
        plt.plot(w5[:, 0], w5[:, 1], 'v', alpha=0.3, label='cluster 5')
        plt.plot(centroid_[0, :], centroid_[1, :], 'k*', label='centroids')
        plt.plot(0, 0, 'g*', markersize=15, linewidth=3, label='target', zorder=12)
        plt.axis('equal')
        plt.subplots_adjust(top=0.95, bottom=0.15)
        plt.legend(loc=0)
        plt.legend(shadow=True)
        plt.show()

    def update(self, data, K=None, max_iterations=100):
        """
        Run EM to update clusters.
        Args:
            data: An N x D data matrix, where N = number of data points.
            K: Number of clusters to use.
        """
        # Constants.
        N  = data.shape[0]
        Do = data.shape[1]

        if K is None:
            K = self.K

        if (not self.warmstart or self.sigma is None or K != self.sigma.shape[0]):
            # Initialization.
            self.sigma = np.zeros((K, Do, Do))
            self.mu = np.zeros((K, Do))
            self.logmass = np.log(1.0 / K) * np.ones((K, 1))
            self.mass = (1.0 / K) * np.ones((K, 1))
            self.N = data.shape[0]
            N = self.N

            # Set initial cluster indices.
            use_kmeans = True
            # use_kmeans = False
            if not self.init_sequential and not use_kmeans:
                cidx = np.random.randint(0, K, size=(1, N))
                for i in range(K):
                    cluster_idx = (cidx == i)[0]
                    mu = np.mean(data[cluster_idx, :], axis=0)
                    diff = (data[cluster_idx, :] - mu).T
                    sigma = (1.0 / K) * (diff.dot(diff.T))
                    self.mu[i, :] = mu
                    self.sigma[i, :, :] = sigma + np.eye(Do) * 2e-6
            else:
                # Initialize clusters with kmeans
                self.mu, cidx = kmeans2(data, K)
                iter = 1000
                for j in range(iter):
                    self.mu, cidx = kmeans2(data, K)
                    for i in range(K):
                        cluster_idx = (np.reshape(cidx, [1, len(cidx)]) == i)[0]
                        sigma = np.cov(data[cluster_idx, :].T, data[cluster_idx, :].T)[:Do, :Do]
                        self.sigma[i, :, :] = sigma + np.eye(Do) * 2e-6

                    if not np.isnan(self.sigma).any():
                        break

                    if j == (iter - 1):
                        print('Initialization of gaussians in GMM failed.')
                        exit()


        prevll = -float('inf')
        for itr in range(max_iterations):
            # E-step: compute cluster probabilities.
            logobs = self.estep(data)

            # Compute log-likelihood.
            ll = np.sum(logsum(logobs, axis=1))
            if ll < prevll:
                break
            if np.abs(ll-prevll) < 1e-5*prevll:
                break
            prevll = ll

            # Renormalize to get cluster weights.
            logw = logobs - logsum(logobs, axis=1)
            assert logw.shape == (N, K)

            # Renormalize again to get weights for refitting clusters.
            logwn = logw - logsum(logw, axis=0)
            assert logwn.shape == (N, K)
            w = np.exp(logwn)

            # M-step: update clusters.
            # Fit cluster mass.
            self.logmass = logsum(logw, axis=0).T
            self.logmass = self.logmass - logsum(self.logmass, axis=0)
            assert self.logmass.shape == (K, 1)
            self.mass = np.exp(self.logmass)

            # Reboot small clusters.
            w[:, (self.mass < (1.0 / K) * 1e-4)[:, 0]] = 1.0 / N
            # Fit cluster means.
            w_expand = np.expand_dims(w, axis=2)
            data_expand = np.expand_dims(data, axis=1)
            self.mu = np.sum(w_expand * data_expand, axis=0)
            # Fit covariances.
            wdata = data_expand * np.sqrt(w_expand)
            assert wdata.shape == (N, K, Do)
            for i in range(K):
                # Compute weighted outer product.
                XX = wdata[:, i, :].T.dot(wdata[:, i, :])
                mu = self.mu[i, :]
                self.sigma[i, :, :] = XX - np.outer(mu, mu)

                if self.eigreg:  # Use eigenvalue regularization.
                    raise NotImplementedError()
                else:  # Use quick and dirty regularization.
                    sigma = self.sigma[i, :, :]
                    self.sigma[i, :, :] = 0.5 * (sigma + sigma.T) + 1e-6 * np.eye(Do)

        self.kmeans_plot(data, cidx, self.mu, x_f=5, y_f=30)