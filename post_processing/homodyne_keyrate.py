import numpy as np
from utils import homodyne_gaussian_to_densitymatrix, von_neumann_entropy
from math import erfc
from qosst_skr.utils import g
import tqdm

####### Mutual Information #######

def I_AB_homodyne_ppA(Va, Vb, C, P_A_x, P_A_p):
    """
    Computes I_AB^x = 0.5 * log2( VAx / VAx|Bx ), where VAx = (Vx + 1)/2,
    and VAx|Bx = VAx - Cx^2/(2 Vbx). All variances in SNU.
    """
    VAx = (Va[0] + 1.0) / 2.0
    VAp = (Va[1] + 1.0) / 2.0

    VAx_cond = VAx - (C[0]**2) / (2.0 * Vb[0])
    VAp_cond = VAp - (C[1]**2) / (2.0 * Vb[1])
    
    I_AB = 0.5 * P_A_x * 0.5 * np.log2(VAx / VAx_cond) + 0.5 * P_A_p * 0.5 * np.log2(VAp / VAp_cond)
    return I_AB

def I_AB_homodyne_ppB(Va, Vb, C, cutoff, delta, x_range, p_range, na, nb):
    """
    Computes I_AB after Bob's post-selection exactly as in the paper (Eqs. 28–34).

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
    sigma_a_x = np.sqrt((Va[0] + 1) / 2)
    sigma_b_x = np.sqrt(Vb[0])
    rho_x = C[0] / np.sqrt(2 * sigma_a_x * sigma_b_x)
    sigma_a_p = np.sqrt((Va[1] + 1) / 2)
    sigma_b_p = np.sqrt(Vb[1])
    rho_p = C[1] / np.sqrt(2 * sigma_a_p * sigma_b_p)

    denominator_x = 2 * np.pi * sigma_a_x * sigma_b_x * np.sqrt(1 - rho_x**2)
    denominator_p = 2 * np.pi * sigma_a_p * sigma_b_p * np.sqrt(1 - rho_p**2)

    Q_x = ((xa / sigma_a_x)**2 
         - (2 * rho_x * xa * xb) / (sigma_a_x * sigma_b_x)
         + (xb / sigma_b_x)**2)
    
    Q_p = ((pa / sigma_a_p)**2 
         - (2 * rho_p * pa * pb) / (sigma_a_p * sigma_b_p)
         + (pb / sigma_b_p)**2)

    fAB_x = np.exp(-Q_x / (2 * (1 - rho_x**2))) / denominator_x
    fAB_p = np.exp(-Q_p / (2 * (1 - rho_p**2))) / denominator_p

    # Apply Bob's post-selection (Eq. 30,31) 
    PB_x = erfc(cutoff[0] / np.sqrt(2 * Vb[0]))      # stable for large cutoff
    PB_p = erfc(cutoff[1] / np.sqrt(2 * Vb[1]))
    eps = 1e-300
    PB_x = max(PB_x, eps)
    PB_p = max(PB_p, eps)
    print(f"Post-selection acceptance probabilities: PB_x = {PB_x}, PB_p = {PB_p}")
    mask_x = (np.abs(xb) > cutoff[0]) & (np.abs(xa) <= na)                # Bob only keeps |xb| > cutoff
    mask_p = (np.abs(pb) > cutoff[1]) & (np.abs(pa) <= nb)                # Bob only keeps |pb| > cutoff
    fAB_post_x = fAB_x * mask_x / PB_x                  # post-selected joint distribution (Eq. 31)
    fAB_post_p = fAB_p * mask_p / PB_p

    # Normalize (small numerical drift)
    Z_x = np.sum(fAB_post_x) * delta**2
    fAB_post_x /= Z_x
    Z_p = np.sum(fAB_post_p) * delta**2
    fAB_post_p /= Z_p

    # Compute marginals p_A and p_B
    fA_x = np.sum(fAB_post_x, axis=1) * delta   # sum over xb
    fB_x = np.sum(fAB_post_x, axis=0) * delta   # sum over xa
    fA_p = np.sum(fAB_post_p, axis=1) * delta   # sum over pb
    fB_p = np.sum(fAB_post_p, axis=0) * delta   # sum over pa

    # Avoid log(0)
    eps = 1e-20
    fAB_safe_x = np.maximum(fAB_post_x, eps)
    fA_safe_x  = np.maximum(fA_x, eps)
    fB_safe_x  = np.maximum(fB_x, eps)
    fAB_safe_p = np.maximum(fAB_post_p, eps)
    fA_safe_p  = np.maximum(fA_p, eps)
    fB_safe_p  = np.maximum(fB_p, eps)

    # Entropies and mutual information
    H_A_x = -np.sum(fA_safe_x * np.log2(fA_safe_x)) * delta
    H_B_x = -np.sum(fB_safe_x * np.log2(fB_safe_x)) * delta
    H_AB_x = -np.sum(fAB_safe_x * np.log2(fAB_safe_x)) * delta**2

    H_A_p = -np.sum(fA_safe_p * np.log2(fA_safe_p)) * delta
    H_B_p = -np.sum(fB_safe_p * np.log2(fB_safe_p)) * delta
    H_AB_p = -np.sum(fAB_safe_p * np.log2(fAB_safe_p)) * delta**2

    I_AB_x = H_A_x + H_B_x - H_AB_x
    I_AB_p = H_A_p + H_B_p - H_AB_p

    return I_AB_x, I_AB_p

####### Holevo bound computation #######

def compute_covariances_homodyne(
    V_A_x: float,
    V_A_p: float,
    Vb_x: float,
    Vb_p: float,
    transmittance: float,
    excess_noise: float,
    W_noise: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    V_E_x =  transmittance * W_noise[0] + (1 - transmittance) * (V_A_x + excess_noise)
    V_E_p =  transmittance * W_noise[1] + (1 - transmittance) * (V_A_p + excess_noise)

    
    r1 = 0.5*np.log(W_noise[0] + np.sqrt(W_noise[0]**2 - W_noise[0] / W_noise[1]))
    r2 = 0.5* np.log(W_noise[0] / W_noise [1]) - 0.5*np.log(W_noise[0] + np.sqrt(W_noise[0]**2 - W_noise[0] / W_noise[1]))
    C_E_x = 0.5 * (-np.exp(2 * r1) + np.exp(2 * r2))
    C_E_p = 0.5 * (-np.exp(-2 * r1) + np.exp(-2 * r2))

    C_E1_Bx = np.sqrt(transmittance * (1 - transmittance)) * (W_noise[0] - (V_A_x + excess_noise))
    C_E1_Bp = np.sqrt(transmittance * (1 - transmittance)) * (W_noise[1] - (V_A_p + excess_noise))
    C_E2_Bx = np.sqrt(1 - transmittance) * C_E_x
    C_E2_Bp = np.sqrt(1 - transmittance) * C_E_p

    sigma_c = np.array([[C_E1_Bx, 0],
                        [0, C_E1_Bp],
                        [C_E2_Bx, 0],
                        [0, C_E2_Bp]]).reshape(4,2)
    
    sigma_E = np.array([[V_E_x, 0, C_E_x, 0],
                            [0, V_E_p, 0, C_E_p],
                            [C_E_x, 0, W_noise[0], 0],
                            [0, C_E_p, 0, W_noise[1]]])
    
    sigma_E_cond_x = sigma_E - 1 / Vb_x * sigma_c @ np.array([[1.0, 0.0],
                                                                  [0.0, 0.0]]) @ sigma_c.T
    sigma_E_cond_p = sigma_E - 1 / Vb_p * sigma_c @ np.array([[0.0, 0.0],
                                                                  [0.0, 1.0]]) @ sigma_c.T
    
    return sigma_c, sigma_E_cond_x, sigma_E_cond_p

def mu_E_cond_homodyne(xb, pb, channel_params):
    """
    Compute Eve's conditional mean vector mu_E(xb) for homodyne measurement.
    The user must provide the correlation vector between Bob and Eve modes (sigma_c)
    and the relevant variances; channel_params is a dict containing sigma_c and Vb.
    """
    sigma_c = channel_params['sigma_c']
    Vb_x = channel_params['Vb_x']
    Vb_p = channel_params['Vb_p']
    vec = np.array([xb, pb])
    invV = np.array([[1.0 / Vb_x, 0.0], [0.0, 1.0 / Vb_p]])
    mu = sigma_c @ (invV @ vec)
    return mu

def holevo_bound_homodyne_ppB(
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
    # 1) Build post-selected probability distribution tilde{p}_b(x)
    PB_x = erfc(cutoff[0] / np.sqrt(2.0 * V_bob[0]))
    PB_p = erfc(cutoff[1] / np.sqrt(2.0 * V_bob[1]))
    eps = 1e-300
    PB_x = max(PB_x, eps)
    PB_p = max(PB_p, eps)

    def pb_tilde_x(x):
        if abs(x) < cutoff[0]:
            return 0.0
        return np.exp(-x**2 / (2.0 * V_bob[0])) / (np.sqrt(2.0 * np.pi * V_bob[0]) * PB_x)
    def pb_tilde_p(p):
        if abs(p) < cutoff[1]:
            return 0.0
        return np.exp(-p**2 / (2.0 * V_bob[1])) / (np.sqrt(2.0 * np.pi * V_bob[1]) * PB_p)

    # discretize
    x_bins = np.arange(x_range[0] + delta/2.0, x_range[1], delta)
    pb_tilde_x_list = np.array([pb_tilde_x(x) for x in x_bins])
    p_bins = np.arange(p_range[0] + delta/2.0, p_range[1], delta)
    pb_tilde_p_list = np.array([pb_tilde_p(p) for p in p_bins])

    # normalize (numerical)
    if pb_tilde_x_list.sum() <= 0 or pb_tilde_p_list.sum() <= 0:
        raise ValueError("Post-selected pb had zero mass on chosen grid; enlarge x_range or shrink cutoff.")
    pb_tilde_x_list = pb_tilde_x_list / np.sum(pb_tilde_x_list)
    pb_tilde_p_list = pb_tilde_p_list / np.sum(pb_tilde_p_list)
    
    # prepare channel_params for mu function
    channel_params = {'sigma_c': sigma_c, 'Vb_x': V_bob[0], 'Vb_p': V_bob[1]}

    # 2) Build conditional density matrices rho_{E|x}
    rhos_x = []
    rhos_p = []
    total_points = len(x_bins) * len(p_bins)
    with tqdm(total=total_points, desc="[Holevo] Computing density matrices") as pbar:
        for i, (x, pb_x) in enumerate(zip(x_bins, pb_tilde_x_list)):
            if pb_x < 1e-8:
                rhos_x.append(None)
                pbar.update(1)
                continue
            mu_x = mu_E_cond_homodyne(x, 0.0, channel_params)
            rho_x = homodyne_gaussian_to_densitymatrix(sigma_E_cond_cov[0], mu_x, ncut=ncut)
            rhos_x.append(rho_x)
            pbar.update(1)

        for i, (p, pb_p) in enumerate(zip(p_bins, pb_tilde_p_list)):
            if pb_p < 1e-8:
                rhos_p.append(None)
                pbar.update(1)
                continue
            mu_p = mu_E_cond_homodyne(0.0, p, channel_params)
            rho_p = homodyne_gaussian_to_densitymatrix(sigma_E_cond_cov[1], mu_p, ncut=ncut)
            rhos_p.append(rho_p)
            pbar.update(1)
   
    # 3) Average state
    rho_avg_x = None
    rho_avg_p = None
    for px, rho in zip(pb_tilde_x_list, rhos_x):
        if rho is None:
            continue
        if rho_avg_x is None:
            rho_avg_x = px * rho
        else:
            rho_avg_x = rho_avg_x + px * rho
    for pp, rho in zip(pb_tilde_p_list, rhos_p):
        if rho is None:
            continue
        if rho_avg_p is None:
            rho_avg_p = pp * rho
        else:
            rho_avg_p = rho_avg_p + pp * rho

    # 4) Entropies
    S_avg_x = von_neumann_entropy(rho_avg_x)
    S_avg_p = von_neumann_entropy(rho_avg_p)

    # conditional entropy: use x=0
    mu0 = mu_E_cond_homodyne(0.0, 0.0, channel_params)
    rho0_x = homodyne_gaussian_to_densitymatrix(sigma_E_cond_cov[0], mu0, ncut=ncut)
    S_cond_x = von_neumann_entropy(rho0_x)
    rho0_p = homodyne_gaussian_to_densitymatrix(sigma_E_cond_cov[1], mu0, ncut=ncut)
    S_cond_p = von_neumann_entropy(rho0_p)

    holevo_bits_x = S_avg_x - S_cond_x
    holevo_bits_p = S_avg_p - S_cond_p

    return holevo_bits_x, holevo_bits_p

def holevo_bound_homodyne_ppA(Va: float, T: float, xi: float, eta: float, Vel: float) -> float:
        """
        Compute the Holevo bound on the information between Bob and Eve in the case of the trusted homodyne detector, with Gaussian modulation,
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
        def I_E(Va, T, xi, eta, Vel):
            V = Va + 1
            chi_line = 1 / T - 1 + xi
            chi_hom = (1 + Vel) / eta - 1
            chi_tot = chi_line + chi_hom / T
            A = V**2 * (1 - 2 * T) + 2 * T + T**2 * (V + chi_line) ** 2
            B = T**2 * (V * chi_line + 1) ** 2
            C = (V * B**0.5 + T * (V + chi_line) + A * chi_hom) / (T * (V + chi_tot))
            D = B**0.5 * (V + B**0.5 * chi_hom) / (T * (V + chi_tot))
            nu_1 = (0.5 * (A + (A**2 - 4 * B) ** 0.5)) ** 0.5
            nu_2 = (0.5 * (A - (A**2 - 4 * B) ** 0.5)) ** 0.5
            nu_3 = (0.5 * (C + (C**2 - 4 * D) ** 0.5)) ** 0.5
            nu_4 = (0.5 * (C - (C**2 - 4 * D) ** 0.5)) ** 0.5
            return (
                g((nu_1 - 1) / 2)
                + g((nu_2 - 1) / 2)
                - g((nu_3 - 1) / 2)
                - g((nu_4 - 1) / 2)
            )
        I_E_x = I_E(Va[0], T, xi, eta, Vel)
        I_E_p = I_E(Va[1], T, xi, eta, Vel)
        return I_E_x, I_E_p
    


####### Homodyne Keyrate #######

def keyrate_homodyne_ppB(
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

def keyrate_homodyne_ppA(
        I_AB_x: float,
        IE_x: float,
        beta: float = 0.95,
        ):
    return beta * I_AB_x - IE_x