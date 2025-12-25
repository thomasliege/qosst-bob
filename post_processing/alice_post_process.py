import numpy as np

def alice_post_selection(alice_symbols: np.ndarray, gain: np.ndarray, Va: np.ndarray):
    # Compute the filter for each quadrature
    filter_x = np.exp(-1 * gain[0]**2 * alice_symbols.real**2)
    filter_p = np.exp(-1 * gain[1]**2 * alice_symbols.imag**2)

    # Select only the symbols that pass both filters
    uniform_x = np.random.uniform(0, 1, size=alice_symbols.shape[0])
    uniform_p = np.random.uniform(0, 1, size=alice_symbols.shape[0])
    
    mask_x = uniform_x < filter_x
    mask_p = uniform_p < filter_p
    mask = mask_x & mask_p

    new_alice_symbols = alice_symbols[mask]

    # Compute the acceptance probability
    P_A_empirical = np.mean((uniform_x < filter_x) & (uniform_p < filter_p)) # Empirical
    P_A_theoretical_x = 1 / np.sqrt(1 + 2 * gain[0]**2 * Va[0])  # Theoretical
    P_A_theoretical_p = 1 / np.sqrt(1 + 2 * gain[1]**2 * Va[1])

    # Get the new effective variance modulation
    Vmod_x_tilde = Va[0] / (1 + 2 * gain[0]**2 * Va[0])
    Vmod_p_tilde = Va[1] / (1 + 2 * gain[1]**2 * Va[1])
    Vmod_tilde = np.array([Vmod_x_tilde, Vmod_p_tilde])

    post_selection_alice = {
        'alice_symbols': new_alice_symbols,
        'mask': mask,
        'P_A_empirical': P_A_empirical,
        'P_A_theoretical_x': P_A_theoretical_x,
        'P_A_theoretical_p': P_A_theoretical_p,
        'Vmod_tilde': Vmod_tilde,
        }

    return post_selection_alice
