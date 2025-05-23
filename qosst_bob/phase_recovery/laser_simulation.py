import numpy as np
import matplotlib.pyplot as plt

# Parameters
linewidth = 2e3           # linewidth (used for noise scaling)
dt = 1e-7                 # time step
T = 1e-3                  # total simulation time
mu = 1.2                  # desired mean phase (rad)
theta = 1e5               # reversion rate (adjust to get desired behavior)

def laser_simulated(linewidth, dt, T, mu, theta):
    n_steps = int(T / dt)
    t = np.linspace(0, T, n_steps)
    sigma = np.sqrt(2 * np.pi * linewidth)  # noise strength

    # Initialize phase array
    phi = np.zeros(n_steps)
    phi[0] = mu  # start at mean

    # Simulate OU process
    for i in range(1, n_steps):
        dW = np.random.normal(0, np.sqrt(dt))
        phi[i] = phi[i-1] + theta * (mu - phi[i-1]) * dt + sigma * dW

    # Simulated laser field
    E = np.exp(1j * phi)
    return t, phi

def plot_laser_simulation(t, phi):
    # Plotting
    plt.figure(figsize=(10, 4))
    plt.plot(t, phi)
    plt.axhline(mu, color='red', linestyle='--', label='Mean Phase = 1.2 rad')
    plt.title("Mean-Reverting Laser Phase (Ornstein–Uhlenbeck Process)")
    plt.xscale('log')
    plt.xlabel("Time (s)")
    plt.ylabel("Phase (rad)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def main():
    t, phi = laser_simulated(linewidth, dt, T, mu, theta)
    plot_laser_simulation(t, phi)

if __name__ == "__main__":
    main()