import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from heterodyne_keyrate import I_AB_heterodyne_ppA, I_AB_heterodyne_true, holevo_bound_heterodyne_ppA, keyrate_heterodyne_ppA
from utils import normalize
from qosst_bob.data import ExcessNoiseResults
from qosst_core.configuration import Configuration
from qosst_skr.gaussian_trusted_heterodyne_asymptotic import GaussianTrustedHeterodyneAsymptotic
from alice_post_process import alice_post_selection

def test_I_AB_ppA(Va, T, xi, vel, eta, P_A_x=1.0, P_A_p=1.0):
    Vx = (Va[0] + 1)
    Vbx_m = (eta * T * (Vx + xi) + eta * (1 - T) + (1 - eta) + 1 + 2 * vel) / 2
    Vbx = 2 * Vbx_m - 1

    Vp = (Va[1] + 1)
    Vbp_m = (eta * T * (Vp + xi) + eta * (1 - T) + (1 - eta) + 1 + 2 * vel) / 2
    Vbp = 2 * Vbp_m - 1

    Cx = np.sqrt(eta * T * (Vx**2 - 1))
    Cp = np.sqrt(eta * T * (Vp**2 - 1))

    I_AB_ppA = I_AB_heterodyne_ppA(
        Va       = np.array([Vx, Vp]),
        Vb       = np.array([Vbx, Vbp]),
        C        = np.array([Cx, Cp]),
        P_A_x    = P_A_x,
        P_A_p    = P_A_p,
    )
    I_AB_true = I_AB_heterodyne_true(Va, T, xi, eta, vel)
    return I_AB_ppA, I_AB_true

def test_IE_ppA(Va, T, xi, eta, vel, P_A_x=1.0, P_A_p=1.0):
    I_E_ppA_x, I_E_ppA_p = holevo_bound_heterodyne_ppA(
        Va      = Va,
        T       = T,
        xi      = xi,
        eta     = eta,
        Vel     = vel
    )
    I_E_ppA = 0.5 * P_A_x * I_E_ppA_x + 0.5 * P_A_p * I_E_ppA_p
    return I_E_ppA

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
    beta = 0.95

    Va = 2 * alice_photon_number
    xi = xi / T
    T = T / eta

    # Normalize the data
    data_alice_normalized, data_bob_normalized = normalize(
        alice_symbols,
        bob_symbols,
        shot,
        alice_photon_number
    )

    test_I_AB_ppA_result = test_I_AB_ppA(
        Va=np.array([2 * alice_photon_number, 2 * alice_photon_number]),
        T=T,
        xi=xi,
        vel=vel,
        eta=eta,
        P_A_x=1.0,
        P_A_p=1.0
    )
    print(f"Test I_AB_ppA: {test_I_AB_ppA_result[0]:.6f} bits/symbols, I_AB_true: {test_I_AB_ppA_result[1]:.6f} bits/symbols")

if __name__ == "__main__":
    main()