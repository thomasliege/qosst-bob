import numpy as np

def alice_homodyne_post_selection(alice_symbols: np.ndarray, gain: np.ndarray, Va: float):
    # Compute the filter for each quadrature
    filter_x = np.exp(-1 * gain[0]**2 * alice_symbols.real**2)
    filter_p = np.exp(-1 * gain[1]**2 * alice_symbols.imag**2)

    # Select only the symbols that pass both filters
    uniform_pick = np.random.uniform(0, 1, size=alice_symbols.real.shape)
    mask = (uniform_pick < filter_x) & (uniform_pick < filter_p)
    new_alice_symbols = alice_symbols[mask]

    # Compute the acceptance probability
    P_A_empirical = np.mean((uniform_pick < filter_x) & (uniform_pick < filter_p)) # Empirical
    P_A_x_theoretical = 1 / np.sqrt(1 + 2 * gain[0]**2 * Va)  # Theoretical
    P_A_p_theoretical = 1 / np.sqrt(1 + 2 * gain[1]**2 * Va)

    # Get the new effective variance modulation
    Vmod_new_x = Va / (1 + 2 * gain[0]**2 * Va)
    Vmod_new_p = Va / (1 + 2 * gain[1]**2 * Va)
    Vmod_new = np.array([Vmod_new_x, Vmod_new_p])

    post_selection_alice = {
        'alice_symbols': new_alice_symbols,
        'mask': mask,
        'P_A_empirical': P_A_empirical,
        'P_A_x_theoretical': P_A_x_theoretical,
        'P_A_p_theoretical': P_A_p_theoretical,
        'Vmod_new': Vmod_new,
        }

    return post_selection_alice