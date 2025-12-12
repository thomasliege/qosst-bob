import numpy as np
from post_processing.alice_post_process import alice_homodyne_post_selection
from post_processing.bob_post_process import bob_homodyne_post_selection
from post_processing.keyrate import I_AB_discrete_homodyne, holevo_bound, keyrate, compute_covariances_homodyne

def post_processing(
        alice_symbols: np.ndarray,
        bob_symbols: np.ndarray,
        transmittance: float,
        excess_noise: float,
        alice_photon_number: float,
        W_noise: float,
        beta: float = 0.95,
        gain: np.ndarray = np.array([0.3, 0.3]),
        cutoff: np.ndarray = np.array([0.5, 0.5]),
        delta: float = 0.1,
        x_range: tuple = (-5.0, 5.0),
        p_range: tuple = (-5.0, 5.0),
        ):
    # Alice post-selection
    post_selection_alice = alice_homodyne_post_selection(
        alice_symbols, gain, alice_photon_number * 2
    )
    Va_eff_x = post_selection_alice['Vmod_new'][0]
    Va_eff_p = post_selection_alice['Vmod_new'][1]
    alice_symbols = post_selection_alice['alice_symbols']
    bob_symbols = bob_symbols[post_selection_alice['mask']]

    V_A_x = Va_eff_x + 1
    Vbob_x = transmittance * V_A_x + transmittance * excess_noise + 1
    
    V_A_p = Va_eff_p + 1
    Vbob_p = transmittance * V_A_p + transmittance * excess_noise + 1

    Vbob = np.array([Vbob_x, Vbob_p])

    # Bob post-selection
    post_selection_bob = bob_homodyne_post_selection(
        bob_symbols, Vbob, cutoff
    )
    alice_symbols = alice_symbols[post_selection_bob['mask']]
    bob_symbols = post_selection_bob['bob_symbols']

    # Mutual information
    Cx = np.sqrt(transmittance * (V_A_x**2 - 1))
    Cp = np.sqrt(transmittance * (V_A_p**2 - 1))
    I_AB_x = I_AB_discrete_homodyne(
        Va      = V_A_x,
        Vb      = Vbob[0],
        C       = Cx,
        cutoff  = cutoff[0],
        delta   = delta,
        x_range = x_range
    )

    I_AB_p = I_AB_discrete_homodyne(
        Va      = V_A_p,
        Vb      = Vbob[1],
        C       = Cp,
        cutoff  = cutoff[1],
        delta   = delta,
        x_range = p_range
    )

    # Holevlo bound
    # Compute covariance matrices for Eve (homodyne case)
    sigma_c, sigma_E_cond_x, sigma_E_cond_p = compute_covariances_homodyne(
        V_A_x,
        V_A_p,
        Vbob_x,
        Vbob_p,
        transmittance,
        excess_noise,
        W_noise,
    )
    
    IE_x = holevo_bound(
            sigma_E_cond_x,
            sigma_c,
            Vbob_x,
            cutoff[0],
            delta,
            x_range,
            ncut=20
        )
    IE_p = holevo_bound(
            sigma_E_cond_p,
            sigma_c,
            Vbob_p,
            cutoff[1],
            delta,
            p_range,
            ncut=20
        )
    # Compute key rate
    P_A_x = post_selection_alice['P_A_x_theoretical']
    P_B_x = post_selection_bob['P_B_x_theoretical']
    P_A_p = post_selection_alice['P_A_p_theoretical']
    P_B_p = post_selection_bob['P_B_p_theoretical']

    skr = keyrate(
        P_A_x,
        P_B_x,
        I_AB_x,
        IE_x,
        P_A_p,
        P_B_p,
        I_AB_p,
        IE_p,
        beta
    )

    return skr
