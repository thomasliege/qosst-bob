import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
from alice_post_process import alice_post_selection
from utils import normalize
from qosst_bob.data import ExcessNoiseResults
from qosst_core.configuration import Configuration
from qosst_skr.gaussian_trusted_heterodyne_asymptotic import GaussianTrustedHeterodyneAsymptotic
from heterodyne_keyrate import I_AB_heterodyne_ppA, I_AB_heterodyne_true, holevo_bound_heterodyne_ppA, keyrate_heterodyne_ppA

from test.test_alice_pp import test_I_AB_ppA, test_IE_ppA

def optimize_cutoff(cutoff, gain, data_alice_normalized, Va, T, xi, vel, eta, beta):
    # Initialize 2D array for key rates
    keyrate_ppA_map = np.zeros((len(cutoff), len(cutoff)))
    
    # 2D scan over c1 and c2
    for i, c1 in enumerate(cutoff):
        for j, c2 in enumerate(cutoff):
            post_selection_alice = alice_post_selection(
                data_alice_normalized, np.array([gain,gain]), np.array([Va, Va])
            )

            Vmod_tilde_x = post_selection_alice['Vmod_tilde'][0]
            Vmod_tilde_p = post_selection_alice['Vmod_tilde'][1]
            P_A_x = post_selection_alice['P_A_theoretical_x']
            P_A_p = post_selection_alice['P_A_theoretical_p']
            
            keyrate_ppA_map[i, j] = skr_ppB
        
    # Find best key rate and corresponding gains
    max_idx = np.unravel_index(np.argmax(keyrate_ppA_map), keyrate_ppA_map.shape)
    best_c1 = cutoff[max_idx[0]]
    best_c2 = cutoff[max_idx[1]]
    best_keyrate = keyrate_ppA_map[max_idx]
    return keyrate_ppA_map, best_c1, best_c2, best_keyrate

def plot_gain_optimization(gain, keyrate_ppA_map, best_g1, best_g2, best_keyrate, skr_raw):
    # Create colormap plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Colormap of key rate vs (g1, g2)
    im = ax.imshow(keyrate_ppA_map.T, origin='lower', aspect='auto',
                   extent=[gain[0], gain[-1], gain[0], gain[-1]],
                   cmap='viridis')
    ax.plot(best_g1, best_g2, 'r*', markersize=25, markeredgecolor='white', 
            markeredgewidth=2, label=f'Optimum: g1={best_g1:.3f}, g2={best_g2:.3f}')
    
    # Add text box with key information
    textstr = f'Max Key Rate: {best_keyrate / 1000:.2f} kbit/s\nRaw Key Rate: {100e6 * skr_raw / 1000:.2f} kbit/s\nImprovement: {(best_keyrate/(100e6 * skr_raw) - 1)*100:.1f}%'
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black', linewidth=2)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props, family='monospace')
    
    ax.set_xlabel('g1 (X quadrature gain)', fontsize=14, fontweight='bold')
    ax.set_ylabel('g2 (P quadrature gain)', fontsize=14, fontweight='bold')
    ax.set_title('Key Rate vs Alice Gains (g1, g2)', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Key Rate (bit/s)', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('C:\\Users\\tliege\\LIP6\\qosst-bob\\post_processing\\plot\\opt_gain_colormap.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as 'opt_gain_colormap.png'")
    plt.show()


def main():
    folder = 'C:\\Users\\tliege\\LIP6\\qosst-bob\\post_processing\\data\\'
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
    
    gain = np.linspace(0, 1.0, 30)
    
    keyrate_ppA_map, best_g1, best_g2, best_keyrate = optimize_gain(
        gain, data_alice_normalized, Va, T, xi, vel, eta, beta
    )
    
    # Calculate raw key rate for reference
    skr_raw, iab_raw, ie_raw = GaussianTrustedHeterodyneAsymptotic.skr(
            Va=2 * alice_photon_number,
            T=T,
            xi=xi,
            eta=eta,
            Vel=vel,
            beta=0.95
        )
    
    plot_gain_optimization(gain, keyrate_ppA_map, best_g1, best_g2, best_keyrate, skr_raw)

if __name__ == "__main__":
    main()