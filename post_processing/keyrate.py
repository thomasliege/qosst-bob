import numpy as np
from post_processing.utils import gaussian_to_densitymatrix
import qutip as qt
from math import erf


####### Mutual Information #######

def I_AB_gaussian_homodyne(V_A, V_bob, C):
    """
    Computes I_AB^x = 0.5 * log2( VAx / VAx|Bx ), where VAx = (Vx + 1)/2,
    and VAx|Bx = VAx - Cx^2/(2 Vbx). All variances in SNU.
    """
    VAx = (V_A + 1.0) / 2.0
    VAx_cond = VAx - (C**2) / (2.0 * V_bob)
    return 0.5 * np.log2(VAx / VAx_cond)

def I_AB_discrete_homodyne(Va, Vb, C, cutoff, delta, x_range):
    """
    Computes I_AB after Bob's post-selection exactly as in the paper (Eqs. 28–34).

    Parameters
    ----------
    Va : float
        Alice's total variance in SNU (e.g. Va = V_mod + 1).
    Vb : float
        Bob’s variance at his detector in SNU.
    C : float
        Alice-Bob covariance (same as paper's C_x or C_p).
    cutoff : float
        Bob’s post-selection threshold c_x (or c_p).
    delta : float
        Discretization step Δ used in your Bob post-processing.
    x_range : tuple
        (xmin, xmax) range for the grid (e.g. (-15,15)).

    Returns
    -------
    I_AB : float
        Mutual information in bits after Bob's post-selection.
    """

    # Construct axes and 2D grid
    xa_vals = np.arange(x_range[0], x_range[1] + delta, delta)
    xb_vals = np.arange(x_range[0], x_range[1] + delta, delta)
    xa, xb = np.meshgrid(xa_vals, xb_vals, indexing='ij')

    # Bivariate Gaussian, sigma_a and sigma_b from Eq. (28)
    sigma_a = np.sqrt((Va + 1) / 2)
    sigma_b = np.sqrt(Vb)
    rho = C / np.sqrt(2 * sigma_a * sigma_b)

    denominator = 2 * np.pi * sigma_a * sigma_b * np.sqrt(1 - rho**2)

    Q = ((xa / sigma_a)**2 
         - (2 * rho * xa * xb) / (sigma_a * sigma_b)
         + (xb / sigma_b)**2)

    fAB = np.exp(-Q / (2 * (1 - rho**2))) / denominator

    # Apply Bob's post-selection (Eq. 30,31) 
    PB = 1 - erf(cutoff / np.sqrt(2 * Vb))      # probability Bob keeps the sample
    mask = (np.abs(xb) > cutoff)                # Bob only keeps |xb| > cutoff
    fAB_post = fAB * mask / PB                  # post-selected joint distribution (Eq. 31)

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

def compute_covariances_homodyne(
    V_A_x: float,
    V_A_p: float,
    Vb_x: float,
    Vb_p: float,
    transmittance: float,
    excess_noise: float,
    W_noise: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    V_E_x =  transmittance * W_noise[0] + (1 - transmittance) * (V_A_x + excess_noise[0])
    V_E_p =  transmittance * W_noise[1] + (1 - transmittance) * (V_A_p + excess_noise[1])

    
    r1 = 0.5*np.log(W_noise[0] + np.sqrt(W_noise[0]**2 - W_noise[0] / W_noise[1]))
    r2 = 0.5* np.log(W_noise[0] / W_noise [1]) - 0.5*np.log(W_noise[0] + np.sqrt(W_noise[0]**2 - W_noise[0] / W_noise[1]))
    C_E_x = 0.5 * (-np.exp(2 * r1) + np.exp(2 * r2))
    C_E_p = 0.5 * (-np.exp(-2 * r1) + np.exp(-2 * r2))

    C_E1_Bx = np.sqrt(transmittance * (1 - transmittance)) * (W_noise[0] - (V_A_x + excess_noise[0]))
    C_E1_Bp = np.sqrt(transmittance * (1 - transmittance)) * (W_noise[1] - (V_A_p + excess_noise[1]))
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

def von_neumann_entropy(rho: qt.Qobj):
    evals = rho.eigenenergies()
    evals = np.real_if_close(evals)
    evals = np.maximum(evals, 0.0)
    tot = np.sum(evals)
    if tot <= 0:
        return 0.0
    probs = evals / tot
    probs = probs[probs > 0]
    S = -np.sum(probs * np.log2(probs))
    return S

# -------------------------
# mu_E conditional (example for homodyne x outcome)
# -------------------------
def mu_E_cond(xb, pb, channel_params):
    """
    Compute Eve's conditional mean vector mu_E(xb) for homodyne measurement result xb.
    The user must provide the correlation vector between Bob and Eve modes (sigma_c)
    and the relevant variances; channel_params is a dict containing sigma_c and Vb.
    For the homodyne of x, only q quadratures shift.
    """
    sigma_c = channel_params['sigma_c']  # shape (4,2) mapping [q1,p1,q2,p2] vs [x_b, p_b]
    Vb_x = channel_params['Vb_x']
    Vb_p = channel_params['Vb_p']
    # conditional mean: mu_E = sigma_c @ (Vb^{-1} * [xb, 0]) (see paper Eq. 25)
    vec = np.array([xb, pb])
    invV = np.array([[1.0 / Vb_x, 0.0], [0.0, 1.0 / Vb_p]])
    mu = sigma_c @ (invV @ vec)
    return mu

def holevo_bound(
    sigma_E_cond_cov,
    sigma_c,
    Vb_x,
    cutoff_x,
    delta,
    x_range,
    ncut=20
):
    """
    Compute Holevo bound after Bob's post-selection on x quadrature using the density-matrix averaging method.
    This code apply the following steps:
     1) Build the covariance matrices sigma_E and sigma_E_cond following paper Eqs. (14–17)
     2) For each discretized Bob outcome xb, then compute the conditional mean μE(xb) (paper Eq. (25)).
     3) For each xb, convert the conditional Gaussian state (covariance sigma_E_cond and mean mu_E(x_b)) 
        to a Fock-space density matrix rho_{E|x_b}. To do that we:
            - perform a Williamson decomposition of sigma_E_cond →  sigma = S_w^T diag(nu1,nu1,nu2,nu2) S_w.
            - create a tensor product thermal state in Fock basis with occupations nbar_i = (nu_i - 1)/2.
            - implement the Gaussian unitary corresponding to S_w by decomposing S_w with a Bloch–Messiah-type construction.
            - apply displacement (two-mode displacement) to that state to produce rho_{E|x_b}.
     4) Form ρE=∑p~b(xi) ρE∣xi using the post-selected Bob distribution p~b (paper Eq. (26)).
     5) Compute von Neumann entropies S(ρE) and S(ρE∣x=0) and return Holevo bound χEB = S(ρE)−S(ρE∣x=0).
    Parameters
    ----------
    sigma_E_cov : np.ndarray (4x4)
        Eve's unconditional covariance matrix.
    sigma_E_cond_cov : np.ndarray (4x4)
        Eve's conditional covariance after Bob homodyne (x) -- typically same for all xb.
    sigma_c : np.ndarray (4x2)
        correlation matrix between Eve modes (q1,p1,q2,p2) and Bob modes (x_b,p_b).
    Vb_x : float
        Bob variance for x quadrature.
    cutoff_x : float
        Bob's box cutoff for x quadrature.
    delta : float
        discretization step.
    x_range : tuple (xmin, xmax)
        range of xb to discretize.
    ncut : int
        Fock cutoff per mode.

    Returns
    -------
    holevo_bits : float
        Holevo bound in bits after post-selection.
    """

    # 1) Build post-selected probability distribution tilde{p}_b(x)
    PB = 1.0 - erf(cutoff_x / np.sqrt(2.0 * Vb_x))

    def pb_tilde(x):
        if abs(x) <= cutoff_x:
            return 0.0
        return np.exp(-x**2 / (2.0 * Vb_x)) / (np.sqrt(2.0 * np.pi * Vb_x) * PB)

    # discretize
    x_bins = np.arange(x_range[0] + delta/2.0, x_range[1], delta)
    p_list = np.array([pb_tilde(x) for x in x_bins])

    # normalize (numerical)
    if p_list.sum() <= 0:
        raise ValueError("Post-selected pb had zero mass on chosen grid; enlarge x_range or shrink cutoff.")
    p_list = p_list / np.sum(p_list)

    # prepare channel_params for mu function
    channel_params = {'sigma_c': sigma_c, 'Vb_x': Vb_x}

    # 2) Build conditional density matrices rho_{E|x}
    rhos = []
    for x, px in zip(x_bins, p_list):
        if px < 1e-16:
            rhos.append(None)
            continue
        mu = mu_E_cond(x, 0.0, channel_params)  # length 4
        rho = gaussian_to_densitymatrix(sigma_E_cond_cov, mu, ncut=ncut)
        rhos.append(rho)

    # 3) Average state
    rho_avg = None
    for px, rho in zip(p_list, rhos):
        if rho_avg is None:
            rho_avg = px * rho
        else:
            rho_avg = rho_avg + px * rho

    # 4) Entropies
    S_avg = von_neumann_entropy(rho_avg)

    # conditional entropy: use x=0
    mu0 = mu_E_cond(0.0, 0.0, channel_params)
    rho0 = gaussian_to_densitymatrix(sigma_E_cond_cov, mu0, ncut=ncut)
    S_cond = von_neumann_entropy(rho0)

    holevo_bits = S_avg - S_cond
    return holevo_bits



####### Homodyne Keyrate #######

def keyrate(
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
