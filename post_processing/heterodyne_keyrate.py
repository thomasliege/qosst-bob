import numpy as np
from utils import heterodyne_gaussian_to_densitymatrix
from math import erfc
from qosst_skr.utils import g
from tqdm import tqdm
import qutip as qt

####### Mutual Information #######

def I_AB_heterodyne_ppA(Va, Vb, C, P_A_x, P_A_p):
    """
    Computes I_AB^x = 0.5 * log2( VAx / VAx|Bx ), where VAx = (Vx + 1)/2,
    and VAx|Bx = VAx - Cx^2/(2 Vbx). All variances in SNU.
    """
    Valice_x = (Va[0] + 1) / 2
    Valice_p = (Va[1] + 1) / 2

    Vbob_x = (Vb[0] + 1) / 2
    Vbob_p = (Vb[1] + 1) / 2

    VAx_cond = Valice_x - C[0]**2 / (4 * Vbob_x)
    VAp_cond = Valice_p - C[1]**2 / (4 * Vbob_p)

    IAB_x = 0.5 * P_A_x * np.log2(Valice_x / VAx_cond)
    IAB_p = 0.5 * P_A_p * np.log2(Valice_p / VAp_cond)

    I_AB = IAB_x + IAB_p
    return I_AB

def I_AB_heterodyne_true(Va, T, xi, eta, vel):
    """
    Computes I_AB for heterodyne detection without post-selection.

    Parameters
    ----------
    Va : float
        Alice's variance of modulation (in SNU).
    T : float
        Transmittance of the channel.
    xi : float
        Excess noise of the channel (in SNU).
    eta : float
        Efficiency of the detector.
    vel : float
        Electronic noise of the detector (in SNU).
    Returns
    -------
    I_AB : float
        Mutual information in bits.
    """
    chi_line = 1 / T - 1 + xi
    chi_het = (1 + (1 - eta) + 2 * vel) / eta
    chi_tot = chi_line + chi_het / T
    return np.log2((1 + Va + chi_tot) / (1 + chi_tot))

def I_AB_heterodyne_ppB(Va, Vb, C, cutoff, delta, x_range, p_range, na, P_B):
    """
    Computes I_AB after Bob's post-selection.

    Parameters
    ----------
    Va : np.ndarray
        Alice's total variance in SNU (e.g. Va = V_mod + 1).
    Vb : np.ndarray
        Bob’s variance at his detector in SNU.
    C : np.ndarray
        Alice-Bob covariance (same as paper's C_x or C_p).
    cutoff : np.ndarray
        Bob’s post-selection threshold c_x (or c_p).
    delta : float
        Discretization step Δ used in your Bob post-processing.
    x_range : tuple
        (xmin, xmax) range for the grid.
    p_range : tuple
        (pmin, pmax) range for the grid.
    na : float
        Alice's post-selection bound (if any); set to np.inf for no bound.
    P_B : float
        Probability of Bob's post-selection.

    Returns
    -------
    I_AB : float
        Mutual information in bits after Bob's post-selection.
    """
    print(f"Pb_theoretical: {P_B}   cutoff: {cutoff}   na: {na}")
    # Construct axes and 4D grid for joint (xa, pa, xb, pb) distribution
    xa_vals = np.arange(x_range[0], x_range[1] + delta, delta)
    pa_vals = np.arange(p_range[0], p_range[1] + delta, delta)
    xb_vals = np.arange(x_range[0], x_range[1] + delta, delta)
    pb_vals = np.arange(p_range[0], p_range[1] + delta, delta)
    
    # Create 4D meshgrid: shape will be (len(xa), len(pa), len(xb), len(pb))
    xa, pa, xb, pb = np.meshgrid(xa_vals, pa_vals, xb_vals, pb_vals, indexing='ij')

    # Bivariate Gaussian, sigma_a and sigma_b from Eq. (28)
    k = 4
    # Flatten 4D grids to 1D arrays for vectorized computation
    xa_flat = xa.flatten()
    pa_flat = pa.flatten()
    xb_flat = xb.flatten()
    pb_flat = pb.flatten()
    v = np.array([xa_flat, pa_flat, xb_flat, pb_flat])
    
    ######################## Tres bizarre de ne considérer que Vx et pas Vp ########################
    sigma_AB_C = np.array([[(Va[0] + 1) / 2, 0, C[0] / 2, 0],
                           [0, (Va[1] + 1) / 2, 0, -C[1] / 2],
                            [C[0] / 2, 0, (Vb[0] + 1) / 2, 0],
                            [0, -C[1] / 2, 0, (Vb[1] + 1) / 2]])
    sigma_AB_C_inv = np.linalg.inv(sigma_AB_C)

    denominator = np.sqrt((2 * np.pi)**k * np.linalg.det(sigma_AB_C))
    # Compute quadratic form for each point: v.T @ Sigma_inv @ v
    numerator = np.exp(-0.5 * np.sum(v * (sigma_AB_C_inv @ v), axis=0))
    # = equivalent to: for each column i: v[:,i]^T @ sigma_inv @ v[:,i])

    fAB_flat = numerator / denominator
    fAB = fAB_flat.reshape(xa.shape)   # Reshape back to 4D grid

    # Build mask: Bob's circular post-selection + Alice's bounds
    mask_bob = (xb**2 + pb**2 >= cutoff**2)
    mask_alice = (np.abs(xa) <= na) & (np.abs(pa) <= na)
    mask = mask_bob & mask_alice

    fAB_post = fAB * mask / P_B

    # Sum over xb and pb (axes 2 and 3) to get p_A(xa, pa)
    fA = np.sum(fAB_post, axis=(2, 3)) * delta**2
    # Sum over xa and pa (axes 0 and 1) to get p_B(xb, pb)
    fB = np.sum(fAB_post, axis=(0, 1)) * delta**2

    # Entropies and mutual information
    H_A = -np.sum(fA[mask_alice.any(axis=(0,1))] * np.log2(fA[mask_alice.any(axis=(0,1))])) * delta**2  # 2D integral over xa, pa
    H_B = -np.sum(fB[mask_bob.any(axis=(0,1))] * np.log2(fB[mask_bob.any(axis=(0,1))])) * delta**2  # 2D integral over xb, pb
    H_AB = -np.sum(fAB_post[mask] * np.log2(fAB_post[mask])) * delta**4  # 4D integral
    I_AB = H_A + H_B - H_AB

    return I_AB


####### Holevo bound computation #######

def compute_covariances_heterodyne(
    Va: np.ndarray,
    Vb: np.ndarray,
    transmittance: float,
    excess_noise: float,
    W_noise: np.ndarray,
    eta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    V_E_x =  transmittance * W_noise[0] + (1 - transmittance) * (Va[0] + excess_noise)
    V_E_p =  transmittance * W_noise[1] + (1 - transmittance) * (Va[1] + excess_noise)

    r1 = 0.5*np.log(W_noise[0] + np.sqrt(W_noise[0]**2 - W_noise[0] / W_noise[1]))
    r2 = 0.5* np.log(W_noise[0] / W_noise [1]) - 0.5*np.log(W_noise[0] + np.sqrt(W_noise[0]**2 - W_noise[0] / W_noise[1]))
    C_E_x = 0.5 * (-np.exp(2 * r1) + np.exp(2 * r2))
    C_E_p = 0.5 * (-np.exp(-2 * r1) + np.exp(-2 * r2))

    C_E1_Bx = np.sqrt(eta * transmittance * (1 - transmittance)) * (W_noise[0] - (Va[0] + excess_noise))
    C_E1_Bp = np.sqrt(eta * transmittance * (1 - transmittance)) * (W_noise[1] - (Va[1] + excess_noise))
    C_E2_Bx = np.sqrt(eta * (1 - transmittance)) * C_E_x
    C_E2_Bp = np.sqrt(eta * (1 - transmittance)) * C_E_p

    sigma_c = np.array([[C_E1_Bx, 0],
                        [0, C_E1_Bp],
                        [C_E2_Bx, 0],
                        [0, C_E2_Bp]]).reshape(4,2)
    
    sigma_E = np.array([[V_E_x, 0, C_E_x, 0],
                        [0, V_E_p, 0, C_E_p],
                        [C_E_x, 0, W_noise[0], 0],
                        [0, C_E_p, 0, W_noise[1]]])
    
    ########## Erreur dans le papier sur Vb_x et Vb_p ? A discuter ##########
    sigma_E_cond_x = sigma_E - 1 / (Vb[0] + 1) * sigma_c @ sigma_c.T
    sigma_E_cond_p = sigma_E - 1 / (Vb[1] + 1) * sigma_c @ sigma_c.T
    
    return sigma_c, sigma_E_cond_x, sigma_E_cond_p, sigma_E

def mu_E_cond_heterodyne(xb, pb, channel_params):
    """
    Compute Eve's conditional mean vector mu_E(xb) for heterodyne measurement.
    The user must provide the correlation vector between Bob and Eve modes (sigma_c)
    and the relevant variances; channel_params is a dict containing sigma_c and Vb.
    """
    sigma_c = channel_params['sigma_c']
    Vb_x = channel_params['Vb_x']
    Vb_p = channel_params['Vb_p']
    vec = np.array([xb, pb])
    # invV = np.array([[1.0 / ((Vb_x + 1) / 2), 0.0], [0.0, 1.0 / ((Vb_p + 1) / 2)]])
    # mu = sigma_c @ (invV @ vec) / np.sqrt(2)
    invV = np.array([[1.0 / (Vb_x + 1), 0.0], [0.0, 1.0 / (Vb_p + 1)]])
    mu = sigma_c @ (invV @ vec)
    return mu

def holevo_bound_heterodyne_ppB(
    sigma_E_cond_cov,
    sigma_c,
    V_bob,
    P_B,
    cutoff,
    delta,
    x_range,
    p_range,
    ncut=20
):
    """
    Compute Holevo bound after Bob's post-selection on x quadrature using the density-matrix averaging method.
    """
    def pb_tilde(x, p):
        if x**2 + p**2 < cutoff**2:
            return 0.0
        return (
            np.exp(
                -x**2 / (V_bob[0] + 1)
                -p**2 / (V_bob[1] + 1)
            )
            / (np.pi * np.sqrt((V_bob[0] + 1)*(V_bob[1] + 1)) * P_B)
        )
    # discretize
    x_bins = np.arange(x_range[0] + delta/2.0, x_range[1], delta)
    p_bins = np.arange(p_range[0] + delta/2.0, p_range[1], delta)
    # Create 2D grid for heterodyne (xb, pb)
    pb_tilde_list = np.array([[pb_tilde(x, p) for p in p_bins] for x in x_bins])

    print(f"Sum pb_tilde over grid: {np.sum(pb_tilde_list) * delta**2} (should be 1.0)")

    # prepare channel_params for mu function
    channel_params = {'sigma_c': sigma_c, 'Vb_x': V_bob[0], 'Vb_p': V_bob[1]}

    # 2) Build conditional density matrices rho_{E|x,p}
    rhos = []
    total_points = len(x_bins) * len(p_bins)
    with tqdm(total=total_points, desc="[Holevo] Computing density matrices") as pbar:
        for i, x in enumerate(x_bins):
            for j, p in enumerate(p_bins):
                pb = pb_tilde_list[i, j]
                if pb < 1e-8:
                    rhos.append(None)
                    pbar.update(1)
                    continue
                mu = mu_E_cond_heterodyne(x, p, channel_params)
                rho = heterodyne_gaussian_to_densitymatrix(sigma_E_cond_cov, mu, ncut=ncut)
                rhos.append(rho)
                pbar.update(1)

    # 3) Average state
    rho_avg = 0.0
    pb_flat = pb_tilde_list.flatten()
    for p, rho in zip(pb_flat, rhos):
        if rho is None:
            continue
        else:
            rho_avg += p * rho * delta**2

    # 4) Entropy
    S_avg = qt.entropy_vn(rho_avg)

    # conditional entropy: use x=0, p=0
    mu0 = mu_E_cond_heterodyne(0.0, 0.0, channel_params)
    rho0 = heterodyne_gaussian_to_densitymatrix(sigma_E_cond_cov, mu0, ncut=ncut)
    S_cond = qt.entropy_vn(rho0)

    holevo_bits = S_avg - S_cond

    return holevo_bits, rho_avg

def holevo_bound_heterodyne_ppA(Va: float, T: float, xi: float, eta: float, Vel: float) -> float:
    """
    Compute the Holevo bound on the information between Bob and Eve in the case of the trusted heterodyne detector, with Gaussian modulation,
    in the asymptotic scenario.

    Args:
        Va (float): Alice's variance of modulation (in SNU).
        T (float): transmittance of the channel.
        xi (float): excess noise of the channel (in SNU).
        eta (float): efficiency of the detector.
        Vel (float): electronic noise of the detector (in SNU).

    Returns:
        float: Holevo's bound on the information between Eve and Bob in bits per symbol.
    """
    def I_E(Va: float, T: float, xi: float, eta: float, Vel: float):
        V = Va + 1
        chi_line = 1 / T - 1 + xi
        chi_het = (1 + (1 - eta) + 2 * Vel) / eta
        chi_tot = chi_line + chi_het / T
        A = V**2 * (1 - 2 * T) + 2 * T + T**2 * (V + chi_line) ** 2
        B = T**2 * (V * chi_line + 1) ** 2
        C = (
            1
            / (T * (V + chi_tot)) ** 2
            * (
                A * chi_het**2
                + B
                + 1
                + 2 * chi_het * (V * B**0.5 + T * (V + chi_line))
                + 2 * T * (V**2 - 1)
            )
        )
        D = (V + B**0.5 * chi_het) ** 2 / (T * (V + chi_tot)) ** 2

        lambda_1 = (0.5 * (A + (A**2 - 4 * B) ** 0.5)) ** 0.5
        lambda_2 = (0.5 * (A - (A**2 - 4 * B) ** 0.5)) ** 0.5
        lambda_3 = (0.5 * (C + (C**2 - 4 * D) ** 0.5)) ** 0.5
        lambda_4 = (0.5 * (C - (C**2 - 4 * D) ** 0.5)) ** 0.5
        lambda_5 = 1

        return (
            g((lambda_1 - 1) / 2)
            + g((lambda_2 - 1) / 2)
            - g((lambda_3 - 1) / 2)
            - g((lambda_4 - 1) / 2)
            - g((lambda_5 - 1) / 2)
        )
    
    I_E_x = I_E(Va[0], T, xi, eta, Vel)
    I_E_p = I_E(Va[1], T, xi, eta, Vel)

    return I_E_x, I_E_p

####### Heterodyne Keyrate #######

def keyrate_heterodyne_ppB(
        P_A_x: float,
        P_A_p: float,
        P_B: float,
        I_AB: float,
        IE: float,
        beta: float = 0.95,
        ):
    return P_A_x * P_A_p * P_B * (beta * I_AB - IE)


def keyrate_heterodyne_ppA(
        P_A_x: float,
        P_A_p: float,
        I_AB: float,
        IE: float,
        beta: float = 0.95,
        ):
    return P_A_x * P_A_p * (beta * I_AB - IE)