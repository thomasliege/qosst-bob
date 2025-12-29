import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from heterodyne_keyrate import compute_covariances_heterodyne, holevo_bound_heterodyne_ppB, mu_E_cond_heterodyne
from utils import normalize
from qosst_bob.data import ExcessNoiseResults
from qosst_core.configuration import Configuration
from qosst_skr.gaussian_trusted_heterodyne_asymptotic import GaussianTrustedHeterodyneAsymptotic
from alice_post_process import alice_post_selection
from bob_post_process import bob_heterodyne_post_selection
from utils import williamson

def test_I_E_ppB(alice_symbols, bob_symbols, alice_photon_number, transmittance, excess_noise, eta, delta, vel):
    Va = np.array([alice_photon_number * 2, alice_photon_number * 2])

    print(f"X var : {np.var(alice_symbols.real)}, P var : {np.var(alice_symbols.imag)}")
    post_selection_alice = alice_post_selection(
        alice_symbols, np.array([0.0, 0.0]), Va
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
    Vbx_m = (eta * transmittance * (Vx + excess_noise) + eta * (1 - transmittance) + (1 - eta) + 1) / 2 + vel
    Vbx = 2 * Vbx_m - 1
    
    Vp = (Vmod_tilde_p + 1)
    Vbp_m = (eta * transmittance * (Vp + excess_noise) + eta * (1 - transmittance) + (1 - eta) + 1) / 2 + vel
    Vbp = 2 * Vbp_m - 1

    print("[Bob post-selection]...")
    post_selection_bob = bob_heterodyne_post_selection(
        bob_symbols_ppA, np.array([Vbx, Vbp]), 0.0)
    
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
    
    sigma_c, sigma_E_cond, _, sigma_E = compute_covariances_heterodyne(
        np.array([Vx, Vp]),
        np.array([Vbx, Vbp]),
        transmittance,
        excess_noise,
        np.array([1.0, 1.0]),
        eta,
    )
    
    I_E_ppB, rho_avg = holevo_bound_heterodyne_ppB(
            sigma_E_cond,
            sigma_c,
            np.array([Vbx, Vbp]),
            1.0,
            0.0,
            delta,
            x_range,
            p_range,
            ncut=10
        )
    skr_raw, iab_raw, ie_raw = GaussianTrustedHeterodyneAsymptotic.skr(
            Va=2 * alice_photon_number,
            T=transmittance,
            xi=excess_noise,
            eta=eta,
            Vel=vel,
            beta=0.95
        )
    print(f"Raw I_E: {ie_raw:.6f} | PP-B I_E: {I_E_ppB:.6f}")
    return I_E_ppB, ie_raw

def test_mu():
    xb = 10.0
    pb = 10.0
    channel_params = {
        'Vb_x': 5.0,
        'Vb_p': 5.0,
        'sigma_c': np.array([[1.0, 0.0], [0.0, 1.0]])
    }
    mu = mu_E_cond_heterodyne(
        xb,
        pb,
        channel_params
    )
    print("Mu:", mu)
    print("Expected Mu: [1.6667, 1.6667]")
    return mu

def test_eigenvalues(T, xi, vel, eta, alice_photon_number):
    skr_raw, iab_raw, ie_raw = GaussianTrustedHeterodyneAsymptotic.skr(
            Va=2 * alice_photon_number,
            T=T,
            xi=xi,
            eta=eta,
            Vel=vel,
            beta=0.95
        )
    Vx = (2 * alice_photon_number + 1)
    Vbx_m = (eta * T * (Vx + xi) + eta * (1 - T) + (1 - eta) + 1) / 2 + vel
    Vbx = 2 * Vbx_m - 1
    
    Vp = (2 * alice_photon_number + 1)
    Vbp_m = (eta * T * (Vp + xi) + eta * (1 - T) + (1 - eta) + 1) / 2 + vel
    Vbp = 2 * Vbp_m - 1

    sigma_c, sigma_E_cond_x, sigma_E_cond_p, sigma_E = compute_covariances_heterodyne(
        np.array([Vx, Vp]),
        np.array([Vbx, Vbp]),
        T,
        xi,
        np.array([1.0, 1.0]),
        eta,
    )
    _, nu_E = williamson(sigma_E)
    _, nu_E_cond_x = williamson(sigma_E_cond_x)

    print("Symplectic eigenvalues of sigma_E:", nu_E)
    print("Symplectic eigenvalues of sigma_E_cond_x:", nu_E_cond_x)

    # Sigma AB
    sigma_AB = np.array([
        [Vx, 0.0, np.sqrt(eta * T) * (Vx**2 - 1), 0.0],
        [0.0, Vp, 0.0, -np.sqrt(eta * T) * (Vp**2 - 1)],
        [np.sqrt(eta * T) * (Vx**2 - 1), 0.0, Vbx, 0.0],
        [0.0, -np.sqrt(eta * T) * (Vp**2 - 1), 0.0, Vbp]
    ])



def main():
    folder = 'C:\\Users\\tliege\\LIP6\\qosst-bob\\post_processing\\data\\good_data\\'
    config_bob = Configuration(folder + 'config_bob.toml')
    alice_symbols = np.load(folder + 'alice_symbols.npy')
    bob_symbols = np.load(folder + 'bob_symbols.npy')
    alice_photon_number = np.load(folder + 'n.npy')
    results = ExcessNoiseResults.load(folder + 'results.qosst')
    T = results.transmittance[0]
    xi = results.excess_noise_bob[0]
    vel = results.electronic_noise[0]
    eta = config_bob.bob.eta
    shot = results.shot_noise[0]
    xi = xi / T
    T = T / eta

    # Normalize the data
    data_alice_normalized, data_bob_normalized = normalize(
        alice_symbols,
        bob_symbols,
        shot,
        alice_photon_number
    )

    test_mu()
    test_eigenvalues(T, xi, vel, eta, alice_photon_number)

    I_E_ppB, ie_raw = test_I_E_ppB(
        data_alice_normalized,
        data_bob_normalized,
        alice_photon_number,
        T,
        xi,
        eta,
        delta=0.5,
        vel=vel
    )

if __name__ == "__main__":
    main()