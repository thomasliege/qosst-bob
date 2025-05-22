import numpy as np
from numba import njit

class UKF(object):
    def __init__(self, dim_x, dim_z, Q, R, kappa=0.0):
        """
        UKF class constructor
        Args:
            dim_x : state vector x dimension
            dim_z : measurement vector z dimension
        """
        # Setting dimensions
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_v = np.shape(Q)[0]
        self.dim_n = np.shape(R)[0]
        self.dim_a = self.dim_x + self.dim_v + self.dim_n

        # Number of sigma points
        self.n_sigma = (2 * self.dim_a) + 1

        # Scaling parameters
        self.kappa = 3 - self.dim_a
        self.alpha = 0.01
        self.beta = 2.0

        self.Q = np.array(Q)
        self.R = np.array(R)

        alpha_2 = self.alpha**2
        self.lambda_ = alpha_2 * (self.dim_a + self.kappa) - self.dim_a

        # Scaling coefficient for sigma points
        self.sigma_scale = np.sqrt(self.dim_a + self.lambda_)

        # Unscented weights
        self.W0 = self.lambda_ / (self.dim_a + self.lambda_)
        # self.Wi = 0.5 / (self.dim_a + self.lambda_)
        self.Wi = np.full(self.n_sigma, 0.5 / (self.dim_a + self.lambda_))  # Create an array of weights
        self.Wi[0] = self.W0  # Set the first weight to W0
        self.Wc0 = self.W0 + (1 - alpha_2 + self.beta)

        # Preallocate augmented state and covariance
        self.x_a = np.zeros(self.dim_a)
        self.P_a = np.zeros((self.dim_a, self.dim_a))

        self.idx1, self.idx2 = self.dim_x, self.dim_x + self.dim_v

        # Initialize noise covariance in augmented matrix
        self.P_a[self.idx1:self.idx2, self.idx1:self.idx2] = self.Q
        self.P_a[self.idx2:, self.idx2:] = self.R

    def predict(self, f, x, P):
        """Prediction step of the UKF."""
        self._initialize_augmented_state(x, P)

        # Generate sigma points
        xa_sigmas = self.sigma_points(self.x_a, self.P_a)

        # Split sigma points
        xx_sigmas = xa_sigmas[:self.dim_x, :]
        xn_sigmas = xa_sigmas[self.idx1:self.idx2, :]
        # xx_sigmas: (dim_x, n_sigma), xn_sigmas: (dim_n, n_sigma)
        assert xx_sigmas.shape == (self.dim_x, self.n_sigma), f"xx_sigmas shape {xx_sigmas.shape} != ({self.dim_x},{self.n_sigma})"
        assert xn_sigmas.shape == (self.dim_n, self.n_sigma), f"xn_sigmas shape {xn_sigmas.shape} != ({self.dim_n},{self.n_sigma})"

        # Transform sigma points through process function
        y_sigmas = self._transform_sigma_points(f, xx_sigmas, xn_sigmas)

        # Calculate mean and covariance
        y, Pyy = self.calculate_mean_and_covariance(y_sigmas)

        # Update augmented state and covariance
        self.x_a[:self.dim_x] = y
        self.P_a[:self.dim_x, :self.dim_x] = Pyy

        return y, Pyy, xx_sigmas

    def correct(self, h, x, P, z):
        """Correction step of the UKF."""
        self._initialize_augmented_state(x, P)

        # Generate sigma points
        xa_sigmas = self.sigma_points(self.x_a, self.P_a)

        # Split sigma points
        xx_sigmas = xa_sigmas[:self.dim_x, :]
        xn_sigmas = xa_sigmas[self.idx2:, :]

        # Transform sigma points through measurement function
        y_sigmas = self._transform_sigma_points(h, xx_sigmas, xn_sigmas)

        # Calculate mean and covariance
        y, Pyy = self.calculate_mean_and_covariance(y_sigmas)

        # Calculate cross-correlation
        Pxy = self.calculate_cross_correlation(x, xx_sigmas, y, y_sigmas)

        # Kalman gain
        K = Pxy @ np.linalg.pinv(Pyy)

        # Update state and covariance
        x = x + (K @ (z - y))
        P = P - (K @ Pyy @ K.T)

        return x, P, xx_sigmas

    def sigma_points(self, x, P):
        """Generate sigma points."""
        return compute_sigma_points(x, P, self.sigma_scale)

    def calculate_mean_and_covariance(self, y_sigmas):
        """Calculate mean and covariance of sigma points."""
        diffs = y_sigmas - y_sigmas[:, [0]]  # Vectorized difference
        y_mean = np.sum(y_sigmas * self.Wi, axis=1)  # Weighted mean
        Pyy = self.Wc0 * np.outer(diffs[:, 0], diffs[:, 0])
        Pyy += (diffs[:, 1:] * self.Wi[1:]) @ diffs[:, 1:].T  # Corrected covariance calculation
        return y_mean, Pyy

    def calculate_cross_correlation(self, x, x_sigmas, y, y_sigmas):
        """Calculate cross-correlation between state and measurement."""
        dx = x_sigmas - x[:, None]
        dy = y_sigmas - y[:, None]
        Pxy = self.Wc0 * (dx[:, [0]] @ dy[:, [0]].T)
        Pxy += dx[:, 1:] @ np.diag(self.Wi) @ dy[:, 1:].T
        return Pxy

    def _initialize_augmented_state(self, x, P):
        """Initialize the augmented state and covariance matrix."""
        self.x_a[:self.dim_x] = x
        self.P_a[:self.dim_x, :self.dim_x] = P
        self.P_a[self.idx1:self.idx2, self.idx1:self.idx2] = self.Q
        self.P_a[self.idx2:, self.idx2:] = self.R

    def _transform_sigma_points(self, func, xx_sigmas, noise_sigmas):
        """Transform sigma points through a given function."""
        y_sigmas = np.zeros((self.dim_x, self.n_sigma))
        for i in range(self.n_sigma):
            # Each xx_sigmas[:, i] is (dim_x,), noise_sigmas[:, i] is (dim_n,)
            assert xx_sigmas[:, i].shape == (self.dim_x,), f"xx_sigmas[:, {i}].shape = {xx_sigmas[:, i].shape}"
            assert noise_sigmas[:, i].shape == (self.dim_n,), f"noise_sigmas[:, {i}].shape = {noise_sigmas[:, i].shape}"
            y_sigmas[:, i] = func(xx_sigmas[:, i], noise_sigmas[:, i])
        return y_sigmas


@njit
def compute_sigma_points(x, P, sigma_scale):
    """Compute sigma points."""
    nx = x.shape[0]
    x_sigma = np.zeros((nx, 2 * nx + 1))
    x_sigma[:, 0] = x
    S = np.linalg.cholesky(P + np.eye(nx) * 1e-6)
    for i in range(nx):
        x_sigma[:, i + 1] = x + sigma_scale * S[:, i]
        x_sigma[:, i + nx + 1] = x - sigma_scale * S[:, i]
    return x_sigma