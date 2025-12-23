import numpy as np
from utils import heterodyne_gaussian_to_densitymatrix, von_neumann_entropy
from math import erfc
import time

####### Mutual Information #######

# TODO
def I_AB_gaussian_heterodyne(V_A, V_bob, C):
    """
    """

def I_AB_discrete_heterodyne(Va, Vb, C, cutoff, delta, x_range, p_range):
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

    Returns
    -------
    I_AB : float
        Mutual information in bits after Bob's post-selection.
    """

    # Construct axes and 2D grid
    xa_vals = np.arange(x_range[0], x_range[1] + delta, delta)
    xb_vals = np.arange(x_range[0], x_range[1] + delta, delta)
    xa, xb = np.meshgrid(xa_vals, xb_vals, indexing='ij')
    pa_vals = np.arange(p_range[0], p_range[1] + delta, delta)
    pb_vals = np.arange(p_range[0], p_range[1] + delta, delta)
    pa, pb = np.meshgrid(pa_vals, pb_vals, indexing='ij')

    # Bivariate Gaussian, sigma_a and sigma_b from Eq. (28)
    k = 4
    v = np.array([xa, pa, xb, pb]).reshape(k, -1)
    sigma_AB_C = np.array([[(Va + 1) / 2, 0, C[0] / 2, 0],
                           [0, (Va + 1) / 2, 0, -C[1] / 2],
                            [C[0] / 2, 0, (Vb[0] + 1) / 2, 0],
                            [0, -C[1] / 2, 0, (Vb[1] + 1) / 2]])
    sigma_AB_C_inv = np.linalg.inv(sigma_AB_C)

    denominator = np.sqrt((2 * np.pi)**k * np.abs(sigma_AB_C))
    numerator = np.exp(-0.5 * v.T @ sigma_AB_C_inv @ v)

    fAB = numerator / denominator

    assert Vb[0] == Vb[1], "Vbob_x and Vbob_p should be equal for theoretical acceptance probability."
    Vb = Vb[0]
    P_B_theoretical = np.exp(-cutoff**2 / (2 * Vb)) 
    eps = 1e-300
    PB = max(P_B_theoretical, eps)
    print(f"Post-selection acceptance probabilities: PB={PB:.3e}")
    mask = -cutoff**2 < xb**2 + pb**2 < cutoff**2
    fAB_post = fAB * mask / PB

    # Normalize (small numerical drift)
    Z = np.sum(fAB_post) * delta**2
    fAB_post /= Z

    # Compute marginals p_A and p_B
    fA = np.sum(fAB_post, axis=1) * delta   # sum over xb
    fB = np.sum(fAB_post, axis=0) * delta   # sum over xa

    # Avoid log(0)
    eps = 1e-20
    fAB_safe = np.maximum(fAB_post, eps)
    fA_safe  = np.maximum(fA, eps)
    fB_safe  = np.maximum(fB, eps)

    # Entropies and mutual information
    H_A = -np.sum(fA_safe * np.log2(fA_safe)) * delta
    H_B = -np.sum(fB_safe * np.log2(fB_safe)) * delta
    H_AB = -np.sum(fAB_safe * np.log2(fAB_safe)) * delta**2
    I_AB = H_A + H_B - H_AB

    return I_AB


####### Holevo bound computation #######

def compute_covariances_heterodyne(
    V_A_x: float,
    V_A_p: float,
    Vb_x: float,
    Vb_p: float,
    transmittance: float,
    excess_noise: float,
    W_noise: float,
    eta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    V_E_x =  transmittance * W_noise[0] + (1 - transmittance) * (V_A_x + excess_noise)
    V_E_p =  transmittance * W_noise[1] + (1 - transmittance) * (V_A_p + excess_noise)

    
    r1 = 0.5*np.log(W_noise[0] + np.sqrt(W_noise[0]**2 - W_noise[0] / W_noise[1]))
    r2 = 0.5* np.log(W_noise[0] / W_noise [1]) - 0.5*np.log(W_noise[0] + np.sqrt(W_noise[0]**2 - W_noise[0] / W_noise[1]))
    C_E_x = 0.5 * (-np.exp(2 * r1) + np.exp(2 * r2))
    C_E_p = 0.5 * (-np.exp(-2 * r1) + np.exp(-2 * r2))

    C_E1_Bx = np.sqrt(eta * transmittance * (1 - transmittance)) * (W_noise[0] - (V_A_x + excess_noise))
    C_E1_Bp = np.sqrt(eta * transmittance * (1 - transmittance)) * (W_noise[1] - (V_A_p + excess_noise))
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
    sigma_E_cond_x = sigma_E - 1 / Vb_x * sigma_c @ sigma_c.T
    sigma_E_cond_p = sigma_E - 1 / Vb_p * sigma_c @ sigma_c.T
    
    return sigma_c, sigma_E_cond_x, sigma_E_cond_p

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
    invV = np.array([[1.0 / (Vb_x + 1) / 2, 0.0], [0.0, 1.0 / (Vb_p + 1) / 2]])
    mu = sigma_c / np.sqrt(2) @ (invV @ vec)
    return mu

def holevo_bound_heterodyne(
    sigma_E_cond_cov,
    sigma_c,
    V_bob,
    cutoff,
    delta,
    x_range,
    p_range,
    ncut=20
):
    """
    Compute Holevo bound after Bob's post-selection on x quadrature using the density-matrix averaging method.
    """
    print(f"[Holevo] Starting with ncut={ncut}, delta={delta}")
    t_start = time.time()

    # 1) Build post-selected probability distribution tilde{p}_b(x)
    assert V_bob[0] == V_bob[1], "Vbob_x and Vbob_p should be equal for theoretical acceptance probability."
    Vb = V_bob[0]
    P_B_theoretical = np.exp(-cutoff**2 / (2 * Vb)) 
    eps = 1e-300
    P_B = max(P_B_theoretical, eps)
    print(f"[Holevo] Post-selection probabilities: PB={P_B:.3e}")

    def pb_tilde(x, p):
        if abs(x) <= cutoff[0]:
            return 0.0
        return np.exp(-x**2 / (V_bob[0] + 1) - p**2 / (V_bob[1] + 1)) / (np.pi * (np.sqrt((V_bob[0] + 1) * (V_bob[1] + 1)) * P_B))

    # discretize
    x_bins = np.arange(x_range[0] + delta/2.0, x_range[1], delta)
    p_bins = np.arange(p_range[0] + delta/2.0, p_range[1], delta)
    pb_tilde_list = np.array([pb_tilde(x, p) for x in x_bins and p in p_bins])
    print(f"[Holevo] Check the size of pb_tilde_list: {pb_tilde_list.shape}, expected ({len(x_bins)}, {len(p_bins)})")
    print(f"[Holevo] Grid sizes: x={len(x_bins)}, p={len(p_bins)} (total {len(x_bins) + len(p_bins)} points)")
    print(f"[Holevo] x_range={x_range}, p_range={p_range}")

    # normalize (numerical)
    if pb_tilde_list.sum() <= 0:
        raise ValueError("Post-selected pb had zero mass on chosen grid; enlarge x_range or p_range or shrink cutoff.")
    pb_tilde_list = pb_tilde_list / np.sum(pb_tilde_list)
    
    # Count non-negligible points
    n_significant = np.sum(pb_tilde_list > 1e-8)
    print(f"[Holevo] Significant points: total={n_significant}/{len(x_bins) * len(p_bins)}")

    # prepare channel_params for mu function
    channel_params = {'sigma_c': sigma_c, 'Vb_x': V_bob[0], 'Vb_p': V_bob[1]}

    # 2) Build conditional density matrices rho_{E|x,p}
    t_start = time.time()
    rhos = []
    for i, (x, p, pb) in enumerate(zip(x_bins, p_bins, pb_tilde_list)):
        if pb < 1e-8:
            rhos.append(None)
            continue
        mu = mu_E_cond_heterodyne(x, p, channel_params)
        rho = heterodyne_gaussian_to_densitymatrix(sigma_E_cond_cov, mu, ncut=ncut)
        rhos.append(rho)
    t_elapsed = time.time() - t_start
    print(f"[Holevo] x-quadrature done in {t_elapsed:.2f}s ({t_elapsed/n_significant:.3f}s per point)")

    # 3) Average state
    rho_avg = None
    for p, rho in zip(pb_tilde_list, rhos):
        if rho is None:
            continue
        if rho_avg is None:
            rho_avg = p * rho
        else:
            rho_avg = rho_avg + p * rho

    # 4) Entropy
    S_avg = von_neumann_entropy(rho_avg)
    print(f"[Holevo] S(rho_E): {S_avg:.4f} bits")

    # conditional entropy: use x=0, p=0
    mu0 = mu_E_cond_heterodyne(0.0, 0.0, channel_params)
    rho0 = heterodyne_gaussian_to_densitymatrix(sigma_E_cond_cov, mu0, ncut=ncut)
    S_cond = von_neumann_entropy(rho0)
    print(f"[Holevo] S(rho_E|x=0, p=0): {S_cond:.4f} bits")

    holevo_bits = S_avg - S_cond

    t_total = time.time() - t_start
    print(f"[Holevo] Total time: {t_total:.2f}s")
    print(f"[Holevo] Result: χ_EB={holevo_bits:.4f} bits")

    return holevo_bits



####### Heterodyne Keyrate #######

def keyrate_heterodyne(
        P_A_x: float,
        P_B_x: float,
        I_AB_x: float,
        IE_x: float,
        P_A_p: float,
        P_B_p: float,
        I_AB_p: float,
        IE_p: float,
        beta: float = 0.95,
        ):
    return 0.5 * P_B_x * P_A_x * (beta * I_AB_x - IE_x) + 0.5 * P_B_p * P_A_p * (beta * I_AB_p - IE_p)
