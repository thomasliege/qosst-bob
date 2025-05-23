import numpy as np
import matplotlib.pyplot as plt
from qosst_bob.phase_recovery.utils import phase_noise_correction_ukf, evaluate_estimation
from qosst_bob.phase_recovery.laser_simulation import laser_simulated
import matplotlib.cm as cm

# Parameters
LINEWIDTH = 2e3           # linewidth (used for noise scaling)
DT = 1e-7                 # time step
T = 1e-3                  # total simulation time
MU = 1.2                  # desired mean phase (rad)
THETA = 1e5               # reversion rate (adjust to get desired behavior)
SNR = -40 # dB

def ukf_simulation(phi, linewidth_estimated):
    # Simulate noisy measurements (complex signal)
    measurements = np.cos(phi) + 1j * np.sin(phi)

    # Use UKF to estimate the phase
    # Initial covariance
    P = np.array([[1e3]])
    # Process noise
    Q = np.array([[2 * np.pi * linewidth_estimated * DT]])
    # Measurement noise
    shotnoise = 1 / 10**(SNR / 10)
    R = np.array([[shotnoise, 0], [0, shotnoise]])
    estimated_phase = phase_noise_correction_ukf(measurements, P, Q, R)
    estimated_phase_unwrapped = np.unwrap(estimated_phase)

    # mae, mse, whitin_range, max_error = evaluate_estimation(phi, estimated_phase_unwrapped, Q, R, DT)
    return estimated_phase_unwrapped

def get_colors(n):
    # Choose a colormap (viridis or cividis are good options)
    cmap = cm.get_cmap('viridis', n)  # 'viridis' or 'cividis'
    colors = [cmap(i) for i in range(n)]
    return colors

def plot_ukf(t: np.ndarray, theoretical: np.ndarray, ukf_estimation: np.ndarray, linewidth_array: np.ndarray):
    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(t, theoretical, color='gold', label='Simulated θ (2kHz)', linewidth=1)
    colors = get_colors(len(ukf_estimation))
    for idx, estimations in enumerate(ukf_estimation):
        plt.plot(t, estimations, '--', color=colors[idx], label=f'{linewidth_array[idx]*1e-3} kHz', linewidth=1)
    plt.xscale('log')
    plt.ylim(0, 1.8)
    plt.xlabel("Sample")
    plt.ylabel(r"$\theta$ [rad]")
    plt.title("Simulated Phase Noise with UKF Estimation")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to run the simulation.
    """
    linewidth_list = [1e2, 2e3, 1e4, 5e4, 1e5]
    t, theoretical_phase = laser_simulated(LINEWIDTH, DT, T, MU, THETA)
    estimations = []
    for l in linewidth_list:
        estimations.append(ukf_simulation(theoretical_phase, l))
    plot_ukf(t, theoretical_phase, estimations, linewidth_list)

if __name__ == "__main__":
    main()