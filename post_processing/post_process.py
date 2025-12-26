import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from alice_post_process import alice_post_selection
from bob_post_process import bob_homodyne_post_selection, bob_heterodyne_post_selection, compute_PB
from homodyne_keyrate import I_AB_homodyne_ppB, I_AB_homodyne_ppA, holevo_bound_homodyne_ppB, holevo_bound_homodyne_ppA, keyrate_homodyne_ppB, keyrate_homodyne_ppA, compute_covariances_homodyne
from heterodyne_keyrate import I_AB_heterodyne_ppB, I_AB_heterodyne_ppA, holevo_bound_heterodyne_ppB, holevo_bound_heterodyne_ppA, keyrate_heterodyne_ppB, keyrate_heterodyne_ppA, compute_covariances_heterodyne, I_AB_heterodyne_true
from plot import histogram_comparison, keyrate_comparison_plot
from utils import normalize
from qosst_skr.gaussian_trusted_heterodyne_asymptotic import GaussianTrustedHeterodyneAsymptotic
from qosst_bob.data import ExcessNoiseResults
from qosst_core.configuration import Configuration
import matplotlib.pyplot as plt
from test.test_utils import compare_densities

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
        ):
    excess_noise = excess_noise / transmittance
    transmittance = transmittance / eta

    # Alice post-selection
    print("[Alice post-selection]...")
    Va = np.array([alice_photon_number * 2, alice_photon_number * 2])
    post_selection_alice = alice_post_selection(
        alice_symbols, gain, Va
    )
    Vmod_tilde_x = post_selection_alice['Vmod_tilde'][0]
    Vmod_tilde_p = post_selection_alice['Vmod_tilde'][1]
    print("[Alice post-selection] Vmod after post-selection:", post_selection_alice['Vmod_tilde'])
    alice_symbols_ppA = post_selection_alice['alice_symbols']
    bob_symbols_ppA = bob_symbols[post_selection_alice['mask']]
    P_A_x = post_selection_alice['P_A_theoretical_x']
    P_A_p = post_selection_alice['P_A_theoretical_p']

    Valice_x = (Vmod_tilde_x + 1)
    Vbob_x = transmittance * (Valice_x + excess_noise) + 1 - transmittance
    
    Valice_p = (Vmod_tilde_p + 1)
    Vbob_p = transmittance * (Valice_p + excess_noise) + 1 - transmittance
    Vbob = np.array([Vbob_x, Vbob_p])
    print("[Alice post-selection] Bob Vbob after Alice post-selection:", Vbob)

    Cx = np.sqrt(transmittance * (Valice_x**2 - 1))
    Cp = np.sqrt(transmittance * (Valice_p**2 - 1))

    print("[Alice post-selection] Mutual Information computation")
    I_AB_ppA = I_AB_homodyne_ppA(
        V_A     = np.array([Valice_x, Valice_p]),
        V_bob   = Vbob,
        C       = np.array([Cx, Cp]),
        P_A_x    = P_A_x,
        P_A_p    = P_A_p,
    )
    
    print("[Alice post-selection] Holevo bound computation")
    I_E_ppA = holevo_bound_homodyne_ppA(
        Va      = np.array([Vmod_tilde_x, Vmod_tilde_p]),
        T       = transmittance,
        xi      = excess_noise,
        eta     = eta,
        Vel     = vel
    )
    
    skr_ppA = keyrate_homodyne_ppA(I_AB_ppA, I_E_ppA, beta)

    # Bob post-selection
    print("[Bob post-selection]...")
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
    print("[Bob post-selection] Bob x_range after post-selection:", x_range)
    print("[Bob post-selection] Bob p_range after post-selection:", p_range)

    # Mutual information
    print("[Bob post-selection] Mutual Information computation")
    
    na = max(np.abs(alice_symbols.real).max(), np.abs(alice_symbols.imag).max())
    nb = max(np.abs(bob_symbols.real).max(), np.abs(bob_symbols.imag).max())
    I_AB_x, I_AB_p = I_AB_homodyne_ppB(
        Va      = np.array([Valice_x, Valice_p]),
        Vb      = Vbob,
        C       = np.array([Cx, Cp]),
        cutoff  = cutoff,
        delta   = delta,
        x_range = x_range,
        p_range = p_range,
        na      = na,
        nb      = nb,
    )

    # Holevlo bound
    print("[Bob post-selection] Holevo Bound computation...")
    sigma_c, sigma_E_cond_x, sigma_E_cond_p = compute_covariances_homodyne(
        Valice_x,
        Valice_p,
        Vbob_x,
        Vbob_p,
        transmittance,
        excess_noise,
        W_noise,
    )
    
    IE_x, IE_p = holevo_bound_homodyne_ppB(
            np.array([sigma_E_cond_x, sigma_E_cond_p]),
            sigma_c,
            np.array([Vbob_x, Vbob_p]),
            cutoff,
            delta,
            x_range,
            p_range,
            ncut=10
        )

    # Compute key rate
    P_A_x = post_selection_alice['P_A_theoretical_x']
    P_B_x = post_selection_bob['P_B_theoretical_x']
    P_A_p = post_selection_alice['P_A_theoretical_p']
    P_B_p = post_selection_bob['P_B_theoretical_p']
    
    P_B_exp_x = post_selection_bob['P_B_empirical_x']
    P_B_exp_p = post_selection_bob['P_B_empirical_p']

    print(f"[Bob post-selection] Theoretical Bob acceptance probability x: {P_B_x}, Empirical: {P_B_exp_x}")
    print(f"[Bob post-selection] Theoretical Bob acceptance probability p: {P_B_p}, Empirical: {P_B_exp_p}")

    print("[Bob post-selection] Secret Key Rate computation...")
    skr_ppB = 100e6 * keyrate_homodyne_ppB(
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
    
    print("[FINAL] Raw secret Key rate :", skr_raw/1e3, "kbits/s")
    print("[FINAL] Alice post-selection secret Key rate :", skr_ppA/1e3, "kbits/s")
    print("[FINAL] Bob post-selection secret Key rate :", skr_ppB/1e3, "kbits/s")

    if plot:
        fig_alice, (ax_alice_real, ax_alice_imag) = histogram_comparison(alice_symbols, alice_symbols_ppA, alice_symbols_ppB, title='Alice Symbols')
        fig_alice.suptitle('Alice Symbols', fontsize=14, y=1.02)
        
        fig_bob, (ax_bob_real, ax_bob_imag) = histogram_comparison(bob_symbols, bob_symbols_ppA, bob_symbols_ppB, cutoff=cutoff, title='Bob Symbols')
        fig_bob.suptitle('Bob Symbols', fontsize=14, y=1.02)
        plt.show()

    return skr_raw, skr_ppA, skr_ppB

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
        cutoff: float = 1.0,     # SNU
        delta: float = 0.1,
        plot: bool = False,
        ):
    excess_noise = excess_noise / transmittance
    transmittance = transmittance / eta

    # Alice post-selection
    print("[Alice post-selection]...")
    Va = np.array([alice_photon_number * 2, alice_photon_number * 2])
    post_selection_alice = alice_post_selection(
        alice_symbols, gain, Va
    )
    Vmod_tilde_x = post_selection_alice['Vmod_tilde'][0]
    Vmod_tilde_p = post_selection_alice['Vmod_tilde'][1]
    P_A_x = post_selection_alice['P_A_theoretical_x']
    P_A_p = post_selection_alice['P_A_theoretical_p']
    print(f"[Alice post-selection] Theoretical Alice acceptance probability x: {P_A_x}, p: {P_A_p}")

    print("[Alice post-selection] Alice Vmod after post-selection:", post_selection_alice['Vmod_tilde'])
    alice_symbols_ppA = post_selection_alice['alice_symbols']
    bob_symbols_ppA = bob_symbols[post_selection_alice['mask']]

    Vx = (Vmod_tilde_x + 1)
    Vbx_m = (eta * transmittance * (Vx + excess_noise) + eta * (1 - transmittance) * W_noise[0] + (1 - eta) + 1) / 2 + vel
    Vbx = 2 * Vbx_m - 1
    
    Vp = (Vmod_tilde_p + 1)
    Vbp_m = (eta * transmittance * (Vp + excess_noise) + eta * (1 - transmittance) * W_noise[1] + (1 - eta) + 1) / 2 + vel
    Vbp = 2 * Vbp_m - 1

    print("[Alice post-selection] Bob Vbob after Alice post-selection:", np.array([Vbx, Vbp]))

    Cx = np.sqrt(eta * transmittance * (Vx**2 - 1))
    Cp = np.sqrt(eta * transmittance * (Vp**2 - 1))

    I_AB_ppA = I_AB_heterodyne_ppA(
        Va       = np.array([Vx, Vp]),
        Vb       = np.array([Vbx, Vbp]),
        C        = np.array([Cx, Cp]),
        P_A_x    = P_A_x,
        P_A_p    = P_A_p,
    )

    print(f"[Alice post-selection] Mutual Information computation : {I_AB_ppA:.4f} bits/symbols")

    I_E_ppA_x, I_E_ppA_p = holevo_bound_heterodyne_ppA(
        Va      = np.array([Vmod_tilde_x, Vmod_tilde_p]),
        T       = transmittance,
        xi      = excess_noise,
        eta     = eta,
        Vel     = vel
    )
    I_E_ppA = 0.5 * P_A_x * I_E_ppA_x + 0.5 * P_A_p * I_E_ppA_p
    print(f"[Alice post-selection] Holevo bound computation : {I_E_ppA:.4f} bits/symbols")
    
    skr_ppA = 100e6 * keyrate_heterodyne_ppA(
        P_A_x,
        P_A_p,
        I_AB_ppA,
        I_E_ppA,
        beta
    )
    print(f"[Alice post-selection] Secret Key Rate computation : {skr_ppA/1e3:.4f} kbits/s")

    # Bob post-selection
    print("[Bob post-selection]...")
    post_selection_bob = bob_heterodyne_post_selection(
        bob_symbols_ppA, np.array([Vbx, Vbp]), cutoff
    )
    alice_symbols_ppB = alice_symbols_ppA[post_selection_bob['mask']]
    bob_symbols_ppB = post_selection_bob['bob_symbols']

    # Determine x_range and p_range based on percentiles
    q_vals = np.real(bob_symbols_ppB)
    p_vals = np.imag(bob_symbols_ppB)
    q_lo, q_hi = np.quantile(q_vals, [0.001, 0.999])
    p_lo, p_hi = np.quantile(p_vals, [0.001, 0.999])
    pad_q = 10.0
    pad_p = 10.0
    x_range = (q_lo - pad_q, q_hi + pad_q)
    p_range = (p_lo - pad_p, p_hi + pad_p)
    print("[Bob post-selection] Bob x_range after post-selection:", x_range)
    print("[Bob post-selection]Bob p_range after post-selection:", p_range)

    P_B = compute_PB(cutoff, Vbx, Vbp, delta, x_range, p_range)
    print(f"[Bob post-selection] Computed Bob acceptance probability: {P_B}")


    # Mutual information
    # na = max(np.abs(alice_symbols.real).max(), np.abs(alice_symbols.imag).max())
    na = 100
    
    I_AB_ppB = I_AB_heterodyne_ppB(
        Va      = np.array([Vx, Vp]),
        Vb      = np.array([Vbx, Vbp]),
        C       = np.array([Cx, Cp]),
        cutoff  = cutoff,
        delta   = delta,
        x_range = x_range,
        p_range = p_range,
        na     = na,
        P_B     = P_B,
    )
    print(f"[Bob post-selection] Mutual Information computation : {I_AB_ppB:.4f} bits/symbols")

    # Holevlo bound
    # Compute covariance matrices for Eve 
    sigma_c, sigma_E_cond, _, sigma_E = compute_covariances_heterodyne(
        np.array([Vx, Vp]),
        np.array([Vbx, Vbp]),
        transmittance,
        excess_noise,
        W_noise,
        eta,
    )
    
    I_E_ppB, rho_avg = holevo_bound_heterodyne_ppB(
            sigma_E_cond,
            sigma_c,
            np.array([Vbx, Vbp]),
            P_B,
            cutoff,
            delta,
            x_range,
            p_range,
            ncut=10
        )
    print(f"[Bob post-selection] Holevo Bound computation : {I_E_ppB:.4f} bits/symbols")

    compare_densities(rho_avg, sigma_E, ncut=10)
    # Compute key rate
    
    P_B = post_selection_bob['P_B_theoretical']

    print(f"[Bob post-selection] Theoretical Bob acceptance probability: {P_B}")

    print("[Bob post-selection] Secret Key Rate computation...")
    skr_ppB = 100e6 * keyrate_heterodyne_ppB(
        P_A_x,
        P_A_p,
        P_B,
        I_AB_ppB,
        I_E_ppB,
        beta
    )
    print(f"[Bob post-selection] Secret Key Rate computation : {skr_ppB/1e3:.4f} kbits/s")
    skr_raw, iab_raw, ie_raw = GaussianTrustedHeterodyneAsymptotic.skr(
            Va=2 * alice_photon_number,
            T=transmittance,
            xi=excess_noise,
            eta=eta,
            Vel=vel,
            beta=0.95
        )
    skr_raw = skr_raw * 100e6
    
    print("[FINAL] Raw secret Key rate :", skr_raw/1e3, "kbits/s")
    print("[FINAL] Alice post-selection secret Key rate :", skr_ppA/1e3, "kbits/s")
    print("[FINAL] Bob post-selection secret Key rate :", skr_ppB/1e3, "kbits/s")

    if plot:
        fig_alice, (ax_alice_real, ax_alice_imag) = histogram_comparison(alice_symbols, alice_symbols_ppA, alice_symbols_ppB, title='Alice Symbols')
        fig_alice.suptitle('Alice Symbols', fontsize=14, y=1.02)
        
        fig_bob, (ax_bob_real, ax_bob_imag) = histogram_comparison(bob_symbols, bob_symbols_ppA, bob_symbols_ppB, cutoff=cutoff, title='Bob Symbols')
        fig_bob.suptitle('Bob Symbols', fontsize=14, y=1.02)

        fig, ax = keyrate_comparison_plot(
            results = {
                'raw': {'beta_I_AB': iab_raw, 'I_E': ie_raw, 'KR': skr_raw/100e6},
                'alice_ps': {'beta_I_AB': beta * I_AB_ppA, 'I_E': I_E_ppA, 'KR': skr_ppA/100e6},
                'bob_ps': {'beta_I_AB': beta * I_AB_ppB, 'I_E': I_E_ppB, 'KR': skr_ppB/100e6}
            }
            )
        plt.show()

    return skr_raw, skr_ppA, skr_ppB


def main():
    folder = 'C:\\Users\\tliege\\LIP6\\qosst-bob\\post_processing\\data\\good_data\\'
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

    # Normalize the data
    data_alice_normalized, data_bob_normalized = normalize(
        alice_symbols,
        bob_symbols,
        shot,
        alice_photon_number
    )

    skr_raw, skr_ppA, skr_ppB = heterodyne_post_processing(
        data_alice_normalized,
        data_bob_normalized,
        transmittance,
        excess_noise,
        vel,
        eta,
        alice_photon_number,
        W_noise = np.array([1.0, 1.0]),
        beta = 0.95,
        gain= np.array([0, 0]),
        cutoff= 0.0,
        delta = 0.2,
        plot = True,
    )

if __name__ == "__main__":
    main()