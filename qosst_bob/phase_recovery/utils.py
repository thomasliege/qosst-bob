# Updated utils.py with improvements
import numpy as np
from ukf import UKF
import sys
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from numba import njit
import seaborn as sns
import pandas as pd

F_PILOTS_OFFSET = 2

def create_ukf(nx, nz, Q, N):
    assert np.array(Q).shape == (nx, nx), f"Q shape {np.array(Q).shape} != ({nx},{nx})"
    assert np.array(N).shape == (nz, nz), f"N shape {np.array(N).shape} != ({nz},{nz})"
    return UKF(dim_x=nx, dim_z=nz, Q=Q, R=N, kappa=(3 - nx))

def run_ukf(ukf: UKF, measurement, theta_0, P_0, f, h, verbose=False):
    estimates = np.zeros(len(measurement))
    x, P = theta_0, P_0

    for iteration, z in enumerate(measurement):
        if verbose and (iteration % 10 == 0 or iteration == len(measurement) - 1):
            percent = int((iteration + 1) / len(measurement) * 100)
            bar = '=' * (percent // 2) + '-' * (50 - percent // 2)
            sys.stdout.write(f'\r|{bar}| {percent}%')
            sys.stdout.flush()

        z_real = np.array([z.real, z.imag])
        x, P, _ = ukf.predict(f, x, P)
        x, P, _ = ukf.correct(h, x, P, z_real)

        assert x.shape[0] == ukf.dim_x, f"x shape {x.shape} != ({ukf.dim_x},)"
        estimates[iteration] = x[0]

    if verbose:
        sys.stdout.write('\n')
    return estimates

@njit
def f(x, q):
    return x + q

@njit
def h(x, n, A=1.0):
    assert n.shape == (2,), f"n shape {n.shape} != (2,)"
    z = np.empty(2, dtype=np.float64)
    z[0] = A * np.cos(x[0]) + n[0]
    z[1] = A * np.sin(x[0]) + n[1]
    return z

def phase_noise_correction_ukf(measurement: np.ndarray, elec_noise=3.2, shot_noise=1.5) -> np.ndarray:
    theta_0 = np.array([0.0])
    P_0 = np.array([[3.0]])
    Q = np.array([[elec_noise]])
    N = np.array([[shot_noise, 0], [0, shot_noise]])
    ukf = create_ukf(1, 2, Q, N)
    print("UKF created")
    expected_angle = run_ukf(ukf, measurement, theta_0, P_0, f, h, verbose=True)
    return expected_angle

def simulate_data(T, Q_true_val, R_true_val):
    np.random.seed(42)
    true_phases = np.zeros(T)
    measurements = np.zeros(T, dtype=complex)
    for t in range(1, T):
        true_phases[t] = true_phases[t-1] + np.random.normal(0, np.sqrt(Q_true_val))
        noisy_phase = true_phases[t] + np.random.normal(0, np.sqrt(R_true_val))
        measurements[t] = np.cos(noisy_phase) + 1j * np.sin(noisy_phase)
    return true_phases, measurements

def compute_error(estimated_phase, true_phase):
    return np.mean(np.abs(np.unwrap(estimated_phase) - np.unwrap(true_phase)))

def optimize_ukf_parameters(true_phases, measurements, Q_values, R_values):
    best_error = float('inf')
    best_params = None
    all_results = []

    for Q_val in Q_values:
        for R_val in R_values:
            estimated = phase_noise_correction_ukf(measurements, elec_noise=Q_val, shot_noise=R_val)
            error = compute_error(estimated, true_phases)
            all_results.append({"Q": Q_val, "R": R_val, "Error": error})
            if error < best_error:
                best_error = error
                best_params = (Q_val, R_val)
            print(f"Q={Q_val:.3f}, R={R_val:.3f} --> Error={error:.5f}")

    return best_params, all_results

def scan_heatmap(true_phases, measurements):
    Q_scan = np.linspace(0.01, 1.0, 10)
    R_scan = np.linspace(0.01, 1.0, 10)
    best_params, results = optimize_ukf_parameters(true_phases, measurements, Q_scan, R_scan)

    print("\n✅ Best UKF Params:")
    print(f"   Q = {best_params[0]}, R = {best_params[1]}")

    df = pd.DataFrame(results)
    df["Q"] = df["Q"].astype(float).round(2)
    df["R"] = df["R"].astype(float).round(2)
    df["Error"] = df["Error"].astype(float)
    pivot = df.pivot(index="Q", columns="R", values="Error")

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(pivot, annot=False, fmt=".3f", cmap="viridis")
    ax.set_xticklabels([f"{float(label.get_text()):.2f}" for label in ax.get_xticklabels()], rotation=45)
    ax.set_yticklabels([f"{float(label.get_text()):.2f}" for label in ax.get_yticklabels()], rotation=0)

    plt.title("UKF Error Heatmap")
    plt.xlabel("R values (Measurement noise)")
    plt.ylabel("Q values (Process noise)")
    plt.savefig('./qosst_bob/phase_recovery/plot/UKF_error_heatmap.png')
    plt.show()

def main():
    T = 500
    Q_true_val = 0.1
    R_true_val = 0.25
    true_phases, measurements = simulate_data(T, Q_true_val, R_true_val)
    scan_heatmap(true_phases, measurements)

    expected_angle = phase_noise_correction_ukf(measurements, elec_noise=0.45, shot_noise=1)
    expected_angle_unwrapped = np.unwrap(expected_angle)

    diff_ukf = np.abs(expected_angle_unwrapped - np.unwrap(true_phases))
    print(f"Mean absolute error with best parameters: {np.mean(diff_ukf)}")

    basic_angle = np.unwrap(np.angle(measurements))
    diff_basic = np.abs(basic_angle - true_phases)

    plt.figure(figsize=(10, 4))
    plt.plot(true_phases, label='True Phase')
    plt.plot(expected_angle_unwrapped, '--', label='UKF Estimated Phase')
    plt.plot(basic_angle, '--', label='Basic')
    plt.title("Unscented Kalman Filter Phase Tracking")
    plt.xlabel("Time Step")
    plt.ylabel("Phase (radians)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.figure(figsize=(10, 4))
    plt.plot(diff_ukf, label=f'UKF diff, mean = {np.mean(diff_ukf):.4f}')
    plt.plot(diff_basic, '--', label=f'Basic diff, mean = {np.mean(diff_basic):.4f}')
    plt.xlabel("Time Step")
    plt.ylabel("Phase difference with theory (radians)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

if __name__ == "__main__":
    main()