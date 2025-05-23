import numpy as np
from numba import njit

class UKF(object):
    def __init__(self, dim_x, dim_z, Q, R, kappa=0.0):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_v = np.shape(Q)[0]
        self.dim_n = np.shape(R)[0]
        self.dim_a = self.dim_x + self.dim_v + self.dim_n

        self.n_sigma = (2 * self.dim_a) + 1

        self.kappa = 3 - self.dim_a
        self.alpha = 0.01
        self.beta = 2.0

        self.Q = np.array(Q)
        self.R = np.array(R)

        alpha_2 = self.alpha**2
        self.lambda_ = alpha_2 * (self.dim_a + self.kappa) - self.dim_a
        self.sigma_scale = np.sqrt(self.dim_a + self.lambda_)

        self.W0 = self.lambda_ / (self.dim_a + self.lambda_)
        self.Wm = np.full(self.n_sigma, 0.5 / (self.dim_a + self.lambda_))
        self.Wm[0] = self.W0
        self.Wc = self.Wm.copy()
        self.Wc[0] += (1 - alpha_2 + self.beta)

        self.x_a = np.zeros(self.dim_a)
        self.P_a = np.zeros((self.dim_a, self.dim_a))

        self.idx1, self.idx2 = self.dim_x, self.dim_x + self.dim_v

    def predict(self, f, x, P):
        self._initialize_augmented_state(x, P)
        xa_sigmas = self.sigma_points(self.x_a, self.P_a)

        xx_sigmas = xa_sigmas[:self.dim_x, :]
        v_sigmas = xa_sigmas[self.idx1:self.idx2, :]

        assert v_sigmas.shape == (self.dim_v, self.n_sigma), \
            f"v_sigmas shape {v_sigmas.shape} != ({self.dim_v},{self.n_sigma})"

        y_sigmas = self._transform_sigma_points(f, xx_sigmas, v_sigmas)
        y, Pyy = self.calculate_mean_and_covariance(y_sigmas)

        self.x_a[:self.dim_x] = y
        self.P_a[:self.dim_x, :self.dim_x] = Pyy

        return y, Pyy, xx_sigmas

    def correct(self, h, x, P, z):
        self._initialize_augmented_state(x, P)
        xa_sigmas = self.sigma_points(self.x_a, self.P_a)

        xx_sigmas = xa_sigmas[:self.dim_x, :]
        n_sigmas = xa_sigmas[self.idx2:, :]

        y_sigmas = self._transform_sigma_points(h, xx_sigmas, n_sigmas)
        assert y_sigmas.shape[0] == self.dim_z, "Measurement function output mismatch"

        y, Pyy = self.calculate_mean_and_covariance(y_sigmas)
        Pxy = self.calculate_cross_correlation(x, xx_sigmas, y, y_sigmas)

        K = Pxy @ np.linalg.pinv(Pyy)
        x = x + (K @ (z - y))
        P = P - (K @ Pyy @ K.T)

        return x, P, xx_sigmas

    def sigma_points(self, x, P):
        # Regularize P outside of Numba
        P_reg = P + np.eye(P.shape[0]) * 1e-6
        return compute_sigma_points(x, P_reg, self.sigma_scale)


    def calculate_mean_and_covariance(self, y_sigmas):
        diffs = y_sigmas - y_sigmas[:, [0]]
        y_mean = np.sum(y_sigmas * self.Wm, axis=1)
        Pyy = self.Wc[0] * np.outer(diffs[:, 0], diffs[:, 0])
        Pyy += (diffs[:, 1:] * self.Wc[1:]) @ diffs[:, 1:].T
        return y_mean, Pyy

    def calculate_cross_correlation(self, x, x_sigmas, y, y_sigmas):
        dx = x_sigmas - x[:, None]
        dy = y_sigmas - y[:, None]
        Pxy = self.Wc[0] * (dx[:, [0]] @ dy[:, [0]].T)
        Pxy += dx[:, 1:] @ np.diag(self.Wc[1:]) @ dy[:, 1:].T
        return Pxy

    def _initialize_augmented_state(self, x, P):
        self.x_a = np.zeros(self.dim_a)
        self.P_a = np.zeros((self.dim_a, self.dim_a))
        self.x_a[:self.dim_x] = x
        self.P_a[:self.dim_x, :self.dim_x] = P.copy()
        self.P_a[self.idx1:self.idx2, self.idx1:self.idx2] = self.Q
        self.P_a[self.idx2:, self.idx2:] = self.R

    def _transform_sigma_points(self, func, xx_sigmas, noise_sigmas):
        y_sigmas = np.zeros((self.dim_x if func.__name__ == 'f' else self.dim_z, self.n_sigma))
        for i in range(self.n_sigma):
            y_sigmas[:, i] = func(xx_sigmas[:, i], noise_sigmas[:, i])
        return y_sigmas


@njit
def compute_sigma_points(x, P, sigma_scale):
    nx = x.shape[0]
    x_sigma = np.zeros((nx, 2 * nx + 1))
    x_sigma[:, 0] = x
    S = np.linalg.cholesky(P)
    for i in range(nx):
        x_sigma[:, i + 1] = x + sigma_scale * S[:, i]
        x_sigma[:, i + nx + 1] = x - sigma_scale * S[:, i]
    return x_sigma