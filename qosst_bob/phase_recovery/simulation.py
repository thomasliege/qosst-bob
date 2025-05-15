import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from qosst_bob.phase_recovery.utils import phase_noise_correction_ukf

# Parameters
FS = 1e5                  # Sampling rate (Hz)
N = int(1e4)              # Number of samples
NOISE_STD = 0.2           # Standard deviation of fluctuations
MEAN_PHASE = 1.2          # Centered around 1.2 rad
CUTOFF = 2e3              # 2 kHz cutoff (band-limiting phase noise)

def simulate_laser():
    """
    Simulate laser phase noise using a low-pass Butterworth filter.
    The phase noise is generated as white Gaussian noise, filtered to a specified cutoff frequency,
    and then added to a mean phase offset. The UKF is used to estimate the phase.
    """
    # Generate white Gaussian noise
    np.random.seed(0)
    white_noise = np.random.randn(N) * NOISE_STD

    # Design low-pass Butterworth filter
    b, a = butter(N=4, Wn=CUTOFF / (0.5 * FS), btype='low')

    # Apply zero-phase filter to get band-limited noise
    phase_noise = filtfilt(b, a, white_noise)

    # Add mean offset
    theta = MEAN_PHASE + phase_noise

    # Simulate noisy measurements (complex signal)
    measurements = np.cos(theta) + 1j * np.sin(theta)

    # Use UKF to estimate the phase
    estimated_phase = phase_noise_correction_ukf(measurements, elec_noise=0.1, shot_noise=0.25)
    estimated_phase_unwrapped = np.unwrap(estimated_phase)

    # Plot
    samples = np.arange(1, N + 1)
    plt.figure(figsize=(10, 4))
    plt.plot(samples, theta, color='gold', label='Simulated θ (2kHz)', linewidth=1)
    plt.plot(samples, estimated_phase_unwrapped, '--', color='blue', label='UKF Estimated Phase', linewidth=1)
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
    # Simulate laser phase noise
    simulate_laser()

if __name__ == "__main__":
    main()