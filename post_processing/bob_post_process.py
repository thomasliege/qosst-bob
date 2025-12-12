import numpy as np
from scipy.special import erf


def bob_homodyne_post_selection(bob_symbols, Vbob, cutoff):
    """
    Clean Bob post-processing function keeping only the information needed
    for the theoretical discrete IAB computation.

    Parameters
    ----------
    bob_symbols : complex ndarray
        Bob’s raw homodyne results (real = x, imag = p).
    Vbob : tuple of floats
        Bob’s (Vb_x, Vb_p) variances in SNU.

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

    mask_x = np.abs(x_vals) > cutoff[0]
    mask_p = np.abs(p_vals) > cutoff[1]

    # Bob keeps the sample only if both quadratures are accepted
    mask = mask_x & mask_p

    bob_symbols_kept = bob_symbols[mask]

    # Empirical acceptance probabilities
    P_B_empirical_x = np.mean(mask_x)
    P_B_empirical_p = np.mean(mask_p)

    # Theoretical acceptance probabilities (paper Eq. 21)
    P_B_theoretical_x = 1 - erf(cutoff[0] / np.sqrt(2 * Vbob[0]))
    P_B_theoretical_p = 1 - erf(cutoff[1] / np.sqrt(2 * Vbob[1]))

    return {
        "bob_symbols_kept": bob_symbols_kept,
        "mask": mask,
        "P_B_empirical_x": float(P_B_empirical_x),
        "P_B_empirical_p": float(P_B_empirical_p),
        "P_B_theoretical_x": float(P_B_theoretical_x),
        "P_B_theoretical_p": float(P_B_theoretical_p),
    }