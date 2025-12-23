import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from alice_post_process import alice_post_selection
from bob_post_process import bob_homodyne_post_selection, bob_heterodyne_post_selection
from homodyne_keyrate import I_AB_discrete_homodyne, holevo_bound_homodyne, keyrate_homodyne, compute_covariances_homodyne
from heterodyne_keyrate import I_AB_discrete_heterodyne, holevo_bound_heterodyne, keyrate_heterodyne, compute_covariances_heterodyne
from plot import histogram_comparison
from utils import normalize
from qosst_skr.gaussian_trusted_heterodyne_asymptotic import GaussianTrustedHeterodyneAsymptotic
from qosst_bob.data import ExcessNoiseResults
from qosst_core.configuration import Configuration
import matplotlib.pyplot as plt

def homodyne_post_processing(
        alice_symbols: np.ndarray,                   # SNU
        bob_symbols: np.ndarray,                     # SNU
        transmittance: float,
        excess_noise: float,                         # SNU
        vel: float,                                  # SNU
        eta: float,
        alice_photon_number: float,
        W_noise: np.ndarray = np.array([1.0, 1.0]),  # SNU
        beta: float = 0.95,
        gain: np.ndarray = np.array([0.3, 0.3]),     # SNU
        cutoff: np.ndarray = np.array([0, 1.0]),     # SNU
        delta: float = 0.1,
        plot: bool = False,
        homodyne: bool = False,
        ):
    excess_noise = excess_noise / transmittance
    transmittance = transmittance / eta

    # Alice post-selection
    print("Alice post-selection...")
    post_selection_alice = alice_post_selection(
        alice_symbols, gain, alice_photon_number * 2
    )
    Va_eff_x = post_selection_alice['Vmod_new'][0]
    Va_eff_p = post_selection_alice['Vmod_new'][1]
    print("Alice Vmod after post-selection:", post_selection_alice['Vmod_new'])
    alice_symbols_ppA = post_selection_alice['alice_symbols']
    bob_symbols_ppA = bob_symbols[post_selection_alice['mask']]

    V_A_x = Va_eff_x + 1
    Vbob_x = transmittance * V_A_x + transmittance * excess_noise + 1
    
    V_A_p = Va_eff_p + 1
    Vbob_p = transmittance * V_A_p + transmittance * excess_noise + 1

    Vbob = np.array([Vbob_x, Vbob_p])
    print("Bob Vbob after Alice post-selection:", Vbob)

    # Bob post-selection
    print("Bob post-selection...")
    post_selection_bob = bob_homodyne_post_selection(
        bob_symbols_ppA, Vbob, cutoff
    )
    alice_symbols_ppB = alice_symbols_ppA[post_selection_bob['mask']]
    bob_symbols_ppB = post_selection_bob['bob_symbols']

    # Determine x_range and p_range based on percentiles
    q_vals = np.real(bob_symbols_ppB)
    p_vals = np.imag(bob_symbols_ppB)
    q_lo, q_hi = np.quantile(q_vals, [0.001, 0.999])
    p_lo, p_hi = np.quantile(p_vals, [0.001, 0.999])
    pad_q = 0.5
    pad_p = 0.5
    x_range = (q_lo - pad_q, q_hi + pad_q)
    p_range = (p_lo - pad_p, p_hi + pad_p)
    print("Bob x_range after post-selection:", x_range)
    print("Bob p_range after post-selection:", p_range)

    if plot:
        fig_alice, (ax_alice_real, ax_alice_imag) = histogram_comparison(alice_symbols, alice_symbols_ppA, alice_symbols_ppB, title='Alice Symbols')
        fig_alice.suptitle('Alice Symbols', fontsize=14, y=1.02)
        
        fig_bob, (ax_bob_real, ax_bob_imag) = histogram_comparison(bob_symbols, bob_symbols_ppA, bob_symbols_ppB, cutoff=cutoff, title='Bob Symbols')
        fig_bob.suptitle('Bob Symbols', fontsize=14, y=1.02)
        plt.show()

    # Mutual information
    print("Mutual Information computation...")
    
    Cx = np.sqrt(transmittance * (V_A_x**2 - 1))
    Cp = np.sqrt(transmittance * (V_A_p**2 - 1))
    I_AB_x, I_AB_p = I_AB_discrete_homodyne(
        Va      = np.array([V_A_x, V_A_p]),
        Vb      = Vbob,
        C       = np.array([Cx, Cp]),
        cutoff  = cutoff,
        delta   = delta,
        x_range = x_range,
        p_range = p_range
    )
    
    print(f"I_AB_x: {I_AB_x}, I_AB_p: {I_AB_p}")

    # Holevlo bound
    print("Holevo Bound computation...")
    sigma_c, sigma_E_cond_x, sigma_E_cond_p = compute_covariances_homodyne(
        V_A_x,
        V_A_p,
        Vbob_x,
        Vbob_p,
        transmittance,
        excess_noise,
        W_noise,
    )
    
    IE_x, IE_p = holevo_bound_homodyne(
            np.array([sigma_E_cond_x, sigma_E_cond_p]),
            sigma_c,
            np.array([Vbob_x, Vbob_p]),
            cutoff,
            delta,
            x_range,
            p_range,
            ncut=10
        )
    
    print(f"IE_x: {IE_x}, IE_p: {IE_p}")

    # Compute key rate
    print("Secret Key Rate computation...")
    P_A_x = post_selection_alice['P_A_theoretical_x']
    P_B_x = post_selection_bob['P_B_theoretical_x']
    P_A_p = post_selection_alice['P_A_theoretical_p']
    P_B_p = post_selection_bob['P_B_theoretical_p']
    
    P_B_exp_x = post_selection_bob['P_B_empirical_x']
    P_B_exp_p = post_selection_bob['P_B_empirical_p']

    print(f"Theoretical Bob acceptance probability x: {P_B_x}, Empirical: {P_B_exp_x}")
    print(f"Theoretical Bob acceptance probability p: {P_B_p}, Empirical: {P_B_exp_p}")

    skr = 100e6 * keyrate_homodyne(
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
    skr_raw = 100e6 * GaussianTrustedHeterodyneAsymptotic.skr(
            Va=2 * alice_photon_number,
            T=transmittance,
            xi=excess_noise,
            eta=eta,
            Vel=vel,
            beta=0.95
        )
    
    print("Secret Key rate before post-process:", skr_raw/1e3, "kbits/s")
    print("Secret Key rate after post-process:", skr/1e3, "kbits/s")

    return skr

def heterodyne_post_processing(
        alice_symbols: np.ndarray,                   # SNU
        bob_symbols: np.ndarray,                     # SNU
        transmittance: float,
        excess_noise: float,                         # SNU
        vel: float,                                  # SNU
        eta: float,
        alice_photon_number: float,                  # SNU
        W_noise: np.ndarray = np.array([1.0, 1.0]),  # SNU
        beta: float = 0.95,
        gain: np.ndarray = np.array([0.3, 0.3]),     # SNU
        cutoff: np.ndarray = np.array([0, 1.0]),     # SNU
        delta: float = 0.1,
        plot: bool = False,
        ):
    excess_noise = excess_noise / transmittance
    transmittance = transmittance / eta

    # Alice post-selection
    print("Alice post-selection...")
    post_selection_alice = alice_post_selection(
        alice_symbols, gain, alice_photon_number * 2
    )
    Va_eff_x = post_selection_alice['Vmod_new'][0]
    Va_eff_p = post_selection_alice['Vmod_new'][1]
    print("Alice Vmod after post-selection:", post_selection_alice['Vmod_new'])
    alice_symbols_ppA = post_selection_alice['alice_symbols']
    bob_symbols_ppA = bob_symbols[post_selection_alice['mask']]

    
    V_A_x = Va_eff_x + 1
    Vbob_x = eta * transmittance * (V_A_x + excess_noise) + eta * (1 - transmittance) * W_noise[0] + (1 - eta) + 2 * vel 
    
    V_A_p = Va_eff_p + 1
    Vbob_p = eta * transmittance * (V_A_p + excess_noise) + eta * (1 - transmittance) * W_noise[1] + (1 - eta) + 2 * vel 

    Vbob = np.array([Vbob_x, Vbob_p])
    print("Bob Vbob after Alice post-selection:", Vbob)

    # Bob post-selection
    print("Bob post-selection...")
    post_selection_bob = bob_heterodyne_post_selection(
        bob_symbols_ppA, Vbob, cutoff
    )
    alice_symbols_ppB = alice_symbols_ppA[post_selection_bob['mask']]
    bob_symbols_ppB = post_selection_bob['bob_symbols']

    # Determine x_range and p_range based on percentiles
    q_vals = np.real(bob_symbols_ppB)
    p_vals = np.imag(bob_symbols_ppB)
    q_lo, q_hi = np.quantile(q_vals, [0.001, 0.999])
    p_lo, p_hi = np.quantile(p_vals, [0.001, 0.999])
    pad_q = 0.5
    pad_p = 0.5
    x_range = (q_lo - pad_q, q_hi + pad_q)
    p_range = (p_lo - pad_p, p_hi + pad_p)
    print("Bob x_range after post-selection:", x_range)
    print("Bob p_range after post-selection:", p_range)

    if plot:
        fig_alice, (ax_alice_real, ax_alice_imag) = histogram_comparison(alice_symbols, alice_symbols_ppA, alice_symbols_ppB, title='Alice Symbols')
        fig_alice.suptitle('Alice Symbols', fontsize=14, y=1.02)
        
        fig_bob, (ax_bob_real, ax_bob_imag) = histogram_comparison(bob_symbols, bob_symbols_ppA, bob_symbols_ppB, cutoff=cutoff, title='Bob Symbols')
        fig_bob.suptitle('Bob Symbols', fontsize=14, y=1.02)
        plt.show()

    # Mutual information
    print("Mutual Information computation...")
    Cx = np.sqrt(transmittance * (V_A_x**2 - 1))
    Cp = np.sqrt(transmittance * (V_A_p**2 - 1))
    I_AB_x, I_AB_p = I_AB_discrete_homodyne(
        Va      = np.array([V_A_x, V_A_p]),
        Vb      = Vbob,
        C       = np.array([Cx, Cp]),
        cutoff  = cutoff,
        delta   = delta,
        x_range = x_range,
        p_range = p_range
    )
    print(f"I_AB_x: {I_AB_x}, I_AB_p: {I_AB_p}")

    # Holevlo bound
    print("Holevo Bound computation...")
    # Compute covariance matrices for Eve (homodyne case)
    sigma_c, sigma_E_cond, _ = compute_covariances_heterodyne(
        V_A_x,
        V_A_p,
        Vbob_x,
        Vbob_p,
        transmittance,
        excess_noise,
        W_noise,
        eta,
    )
    
    IE = holevo_bound_heterodyne(
            sigma_E_cond,
            sigma_c,
            np.array([Vbob_x, Vbob_p]),
            cutoff,
            delta,
            x_range,
            p_range,
            ncut=10
        )
    
    print(f"IE: {IE}")

    # Compute key rate
    print("Secret Key Rate computation...")
    P_A_x = post_selection_alice['P_A_theoretical_x']
    P_A_p = post_selection_alice['P_A_theoretical_p']
    
    P_B = post_selection_bob['P_B_theoretical']
    P_B_exp = post_selection_bob['P_B_empirical']

    print(f"Theoretical Bob acceptance probability: {P_B}, Empirical: {P_B_exp}")

    skr = 100e6 * keyrate_heterodyne(
        P_A_x,
        P_B,
        I_AB,
        IE,
        beta
    )
    skr_raw = 100e6 * GaussianTrustedHeterodyneAsymptotic.skr(
            Va=2 * alice_photon_number,
            T=transmittance,
            xi=excess_noise,
            eta=eta,
            Vel=vel,
            beta=0.95
        )
    
    print("Secret Key rate before post-process:", skr_raw/1e3, "kbits/s")
    print("Secret Key rate after post-process:", skr/1e3, "kbits/s")

    return skr


def main():
    folder = 'C:\\Users\\tliege\\LIP6\\qosst-bob\\post_processing\\data\\'
    config_bob = Configuration(folder + 'config_bob.toml')
    alice_symbols = np.load(folder + 'alice_symbols.npy')
    bob_symbols = np.load(folder + 'bob_symbols.npy')
    alice_photon_number = np.load(folder + 'n.npy')
    results = ExcessNoiseResults.load(folder + 'results.qosst')
    transmittance = results.transmittance[0]
    excess_noise = results.excess_noise_bob[0]
    vel = results.electronic_noise[0]
    eta = config_bob.bob.eta
    shot = results.shot_noise[0]
    print(shot)

    # Normalize the data
    data_alice_normalized, data_bob_normalized = normalize(
        alice_symbols,
        bob_symbols,
        shot,
        alice_photon_number
    )

    skr = post_processing(
        data_alice_normalized,
        data_bob_normalized,
        transmittance,
        excess_noise,
        vel,
        eta,
        alice_photon_number,
        W_noise = np.array([1.02, 1.01]),
        beta = 0.95,
        gain= np.array([0.6, 0.6]),
        cutoff= np.array([0, 1.0]),
        delta = 0.5,
        plot = True,
    )

if __name__ == "__main__":
    main()