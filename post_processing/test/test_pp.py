import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qosst_bob.data import ExcessNoiseResults
from alice_post_process import alice_homodyne_post_selection
from bob_post_process import bob_homodyne_post_selection
import numpy as np
from plot import histogram_comparison
import matplotlib.pyplot as plt

def test_alice_pp(alice_symbols, bob_symbols, gain, alice_photon_number, plot=False):
    post_selection_alice = alice_homodyne_post_selection(
        alice_symbols, gain, alice_photon_number * 2
    )
    alice_symbol_pp = post_selection_alice['alice_symbols']
    bob_symbols_pp = bob_symbols[post_selection_alice['mask']]

    if plot:
        fig_alice, (ax_alice_real, ax_alice_imag) = histogram_comparison(alice_symbols, alice_symbol_pp)
        fig_alice.suptitle('Alice Symbols', fontsize=14, y=1.02)
        
        fig_bob, (ax_bob_real, ax_bob_imag) = histogram_comparison(bob_symbols, bob_symbols_pp)
        fig_bob.suptitle('Bob Symbols', fontsize=14, y=1.02)

def test_bob_pp(alice_symbols, bob_symbols, cutoff, transmittance, excess_noise, alice_photon_number, gain, plot=False):
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

    post_selection_bob = bob_homodyne_post_selection(
        bob_symbols, Vbob, cutoff
    )
    alice_symbol_pp = alice_symbols[post_selection_bob['mask']]
    bob_symbols_pp = post_selection_bob['bob_symbols']

    if plot:
        fig_alice, (ax_alice_real, ax_alice_imag) = histogram_comparison(alice_symbols, alice_symbol_pp)
        fig_alice.suptitle('Alice Symbols', fontsize=14, y=1.02)
        
        fig_bob, (ax_bob_real, ax_bob_imag) = histogram_comparison(bob_symbols, bob_symbols_pp)
        fig_bob.suptitle('Bob Symbols', fontsize=14, y=1.02)

def main():
    folder = 'C:\\Users\\tliege\\LIP6\\qosst-bob\\post_processing\\data\\'
    alice_symbols = np.load(folder + 'alice_symbols.npy')
    bob_symbols = np.load(folder + 'bob_symbols.npy')
    alice_photon_number = np.load(folder + 'n.npy')
    results = ExcessNoiseResults.load(folder + 'results.qosst')

    gain = np.array([0.1, 0.1])

    test_alice_pp(alice_symbols, bob_symbols, gain, alice_photon_number, plot=True)

    cutoff = np.array([15, 8])
    # test_bob_pp(alice_symbols, bob_symbols, cutoff, results.transmittance[0], results.excess_noise_bob[0], alice_photon_number, gain, plot=True)

    plt.show()

if __name__ == "__main__":
    main()