import numpy as np
from ukf import UKF
import sys
import matplotlib.pyplot as plt
from numba import njit
import seaborn as sns
import pandas as pd
from tqdm import tqdm

def create_ukf(Q, R):
    nx = Q.shape[0]
    nz = R.shape[0]
    assert Q.shape == (nx, nx), f"Q shape {Q.shape} != ({nx},{nx})"
    assert R.shape == (nz, nz), f"R shape {R.shape} != ({nz},{nz})"
    return UKF(dim_x=nx, dim_z=nz, Q=Q, R=R, kappa=(3 - nx))

def run_ukf(ukf: UKF, measurement, theta_0, P_0, f, h, verbose=False):
    estimates = np.zeros(len(measurement))
    x, P = theta_0, P_0

    iterator = tqdm(enumerate(measurement), total=len(measurement), disable=not verbose, desc="Running UKF")
    for iteration, z in iterator:
        z_real = np.array([z.real, z.imag])
        x, P, _ = ukf.predict(f, x, P)
        x, P, _ = ukf.correct(h, x, P, z_real)

        assert x.shape[0] == ukf.dim_x, f"x shape {x.shape} != ({ukf.dim_x},)"
        estimates[iteration] = x[0]

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

def phase_noise_correction_ukf(measurement: np.ndarray, P, Q, R, theta_0=None) -> np.ndarray:
    if theta_0 is None:
        theta_0 = np.zeros((Q.shape[0],))
    ukf = create_ukf(Q, R)
    print("UKF created")
    expected_angle = run_ukf(ukf, measurement, theta_0, P, f, h, verbose=True)
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
            Q = np.array([[Q_val]])
            R = np.array([[R_val, 0], [0, R_val]])
            P = np.array([[0.1]])
            estimated = phase_noise_correction_ukf(measurements, P, Q, R)
            error = compute_error(estimated, true_phases)
            all_results.append({"Q": Q_val, "R": R_val, "Error": error})
            if error < best_error:
                best_error = error
                best_params = (Q_val, R_val)
            print(f"Q={Q_val:.3f}, R={R_val:.3f} --> Error={error:.5f}")

    return best_params, all_results

def evaluate_estimation(true_phase, estimated_phase, Q, R, dt):
    error = np.unwrap(estimated_phase) - np.unwrap(true_phase)
    
    mae = np.mean(np.abs(error))
    mse = np.mean(error**2)
    
    # Theoretical standard deviation of phase due to Q
    phase_std = np.sqrt(Q[0, 0] / dt)  # Approximate
    
    # 95% confidence band
    in_bounds = np.abs(error) < 2 * phase_std
    within_band = np.mean(in_bounds)

    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Square Error: {mse}")
    print(f"Within_95%_confidence: {within_band}")
    print(f"Max Error: {np.max(np.abs(error))}")
    
    return mae, mse, within_band, np.max(np.abs(error))


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

    P = np.array([[0.1]])
    Q = np.array([[0.45]])
    R = np.array([[1e-4, 0], [0, 1e-4]])

    expected_angle = phase_noise_correction_ukf(measurements, P, Q, R)
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