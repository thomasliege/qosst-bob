import numpy as np
from math import erfc

def bob_homodyne_post_selection(bob_symbols, Vbob, cutoff):
    """
    Clean Bob post-processing function keeping only the information needed
    for the theoretical discrete IAB computation.

    Parameters
    ----------
    bob_symbols : ndarray
        Bob’s raw homodyne results.
    Vbob : float
        Bob’s variance in SNU.
    cutoff : float
        Bob's acceptance cutoff in SNU.

    Returns
    -------
    post_selection_bob : dict
        {
            "bob_symbols_kept": array of kept symbols,
            "mask": boolean mask for accepted symbols,
            "P_B_empirical": empirical acceptance probability,
            "P_B_theoretical": theoretical acceptance probability (paper),
        }
    """

    # Apply Bob's box filter independently on x and p
    x_vals = bob_symbols.real
    p_vals = bob_symbols.imag

    mask_x = np.abs(x_vals) > cutoff[0]
    mask_p = np.abs(p_vals) > cutoff[1]

    # Bob keeps the sample only if both quadratures are accepted
    mask = mask_x & mask_p

    bob_symbols_kept = bob_symbols[mask]

    # Empirical acceptance probabilities
    P_B_empirical_x = np.mean(mask_x)
    P_B_empirical_p = np.mean(mask_p)

    # Theoretical acceptance probabilities (paper Eq. 21)
    P_B_theoretical_x = erfc(cutoff[0] / np.sqrt(2 * Vbob[0]))
    P_B_theoretical_p = erfc(cutoff[1] / np.sqrt(2 * Vbob[1]))

    return {
        "bob_symbols_kept": bob_symbols_kept,
        "mask": mask,
        "P_B_empirical_x": float(P_B_empirical_x),
        "P_B_empirical_p": float(P_B_empirical_p),
        "P_B_theoretical_x": float(P_B_theoretical_x),
        "P_B_theoretical_p": float(P_B_theoretical_p),
    }

def bob_heterodyne_post_selection(bob_symbols, Vbob, cutoff):
    """
    Clean Bob post-processing function keeping only the information needed
    for the theoretical discrete IAB computation.

    Parameters
    ----------
    bob_symbols : complex ndarray
        Bob’s raw homodyne results (real = x, imag = p).
    Vbob : tuple of floats
        Bob’s (Vb_x, Vb_p) variances in SNU.
    cutoff : float

    Returns
    -------
    post_selection_bob : dict
        {
            "bob_symbols_kept": array of kept symbols,
            "mask": boolean mask for accepted symbols,
            "P_B_empirical_x": empirical acceptance on x,
            "P_B_empirical_p": empirical acceptance on p,
            "P_B_theoretical_x": theoretical acceptance probability (paper),
            "P_B_theoretical_p": theoretical acceptance probability (paper),
        }
    """
    x_vals = bob_symbols.real
    p_vals = bob_symbols.imag
    print(f"[Bob Post-selection] Provided Vbob: x={Vbob[0]:.4f}, p={Vbob[1]:.4f}")
    print(f"[Bob Post-selection] Cutoff={cutoff:.4f}")

    mask = (x_vals**2 + p_vals**2) >= cutoff**2

    bob_symbols_kept = bob_symbols[mask]

    return {
        "bob_symbols": bob_symbols_kept,
        "mask": mask,
    }

def compute_PB(cutoff, Vb_x, Vb_p, delta, x_range, p_range):
    """
    Compute P_B(c) = ∫∫ F_B(xb, pb) * p_b(xb, pb) dxb dpb
    
    Parameters:
    -----------
    cutoff : float
        The post-selection threshold c
    Vb_x, Vb_p : float
        Bob's variances in SNU
    delta : float
        Grid spacing
    x_range, p_range : tuple
        Integration ranges (xmin, xmax), (pmin, pmax)
    """
    # Create 2D grid
    xb_vals = np.arange(x_range[0], x_range[1] + delta, delta)
    pb_vals = np.arange(p_range[0], p_range[1] + delta, delta)
    xb, pb = np.meshgrid(xb_vals, pb_vals, indexing='ij')
    
    # Bob's probability distribution (Gaussian in SNU)
    p_b = np.exp(-xb**2 / (Vb_x + 1) - pb**2 / (Vb_p + 1)) / (np.pi * np.sqrt((Vb_x + 1) * (Vb_p + 1)))
    
    # Post-selection function F_B (indicator function for post-selection region)
    # For circular post-selection: F_B = 1 if xb² + pb² ≥ c²
    F_B = (xb**2 + pb**2 >= cutoff**2).astype(float)
    
    # Compute integral as Riemann sum
    P_B = np.sum(F_B * p_b) * delta**2
    
    return P_B