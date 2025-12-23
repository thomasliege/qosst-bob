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

    # Apply Bob's box filter independently on x and p
    x_vals = bob_symbols.real
    p_vals = bob_symbols.imag

    # Check actual variances
    actual_var_x = np.var(x_vals)
    actual_var_p = np.var(p_vals)
    print(f"[Bob Post-selection] Provided Vbob: x={Vbob[0]:.4f}, p={Vbob[1]:.4f}")
    print(f"[Bob Post-selection] Actual variance: x={actual_var_x:.4f}, p={actual_var_p:.4f}")
    print(f"[Bob Post-selection] Cutoff={cutoff:.4f}")

    mask = (x_vals**2 + p_vals**2) >= cutoff**2

    bob_symbols_kept = bob_symbols[mask]

    # Empirical acceptance probabilities
    P_B_empirical = np.mean(mask)

    # Theoretical acceptance probabilities
    p_b = 1 / (np.pi * np.sqrt((Vbob[0] + 1)*(Vbob[1] + 1))) * np.exp(- (x_vals**2)/(Vbob[0] + 1) - (p_vals**2)/(Vbob[1] + 1))
    # TODO: Vbob_x != V_bob_p, maybe need to compute the actual acceptance proba
    ######################### Mistake in the paper, P_B = 1 - erf(-cutoff/sqrt(2 Vb)) #########################
    # P_B_theoretical = erfc(-cutoff**2 / (2 * Vbob[0]))  # Approximation assuming Vbob_x = Vbob_p
    P_B_theoretical = np.exp(-cutoff**2 / (2 * Vbob[0])) 

    return {
        "bob_symbols": bob_symbols_kept,
        "mask": mask,
        "P_B_empirical": float(P_B_empirical),
        "P_B_theoretical": float(P_B_theoretical),
    }