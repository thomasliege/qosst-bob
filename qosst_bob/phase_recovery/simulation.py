import numpy as np
import matplotlib.pyplot as plt
from qosst_bob.phase_recovery.utils import phase_noise_correction_ukf, evaluate_estimation
from qosst_bob.phase_recovery.laser_simulation import laser_simulated

# Parameters
LINEWIDTH = 2e3           # linewidth (used for noise scaling)
DT = 1e-7                 # time step
T = 1e-3                  # total simulation time
MU = 1.2                  # desired mean phase (rad)
THETA = 1e5               # reversion rate (adjust to get desired behavior)
SNR = 40 # dB

def ukf_simulation(t, phi):
    # Simulate noisy measurements (complex signal)
    measurements = np.cos(phi) + 1j * np.sin(phi)

    # Use UKF to estimate the phase
    # Initial covariance
    P = np.array([[100]])
    # Process noise
    Q = np.array([[2 * np.pi * LINEWIDTH * DT]])
    # Measurement noise
    shotnoise = 1 / 10**(SNR / 10)
    R = np.array([[shotnoise, 0], [0, shotnoise]])
    estimated_phase = phase_noise_correction_ukf(measurements, P, Q, R)
    estimated_phase_unwrapped = np.unwrap(estimated_phase)

    mae, mse, whitin_range, max_error = evaluate_estimation(phi, estimated_phase_unwrapped, Q, R, DT)

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(t, phi, color='gold', label='Simulated θ (2kHz)', linewidth=1)
    plt.plot(t, estimated_phase_unwrapped, '--', color='blue', label='UKF Estimated Phase', linewidth=1)
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
    t, phi = laser_simulated(LINEWIDTH, DT, T, MU, THETA)
    ukf_simulation(t, phi)

if __name__ == "__main__":
    main()