import numpy as np
from qosst_bob.data import ExcessNoiseResults
import os
import logging

def comapre_indices(logger: logging, a: np.ndarray, b: np.ndarray) -> bool:
    """
    Compare two numpy arrays and return True if they are equal, False otherwise.

    Parameters
    ----------
    a : np.ndarray
        First array to compare.
    b : np.ndarray
        Second array to compare.

    Returns
    -------
    bool
        True if the arrays are equal, False otherwise.
    """
    logger.debug(f"Comparing arrays:\n{a}\n{b}")
    if a.shape != b.shape:
        logger.debug(f"Shapes are different: {a.shape} vs {b.shape}")
    logger.info(f"Array a: {a}")
    logger.info(f"Array b: {b}")
    return np.array_equal(a, b)

def plot_excess_noise_results(folder: str, threshold: float = 0.01):
    """
    Plot the excess noise results.

    Parameters
    ----------
    results : ExcessNoiseResults
        The excess noise results to plot.
    save_path : str
        The path to save the plot.
    """
    import matplotlib.pyplot as plt

    results_excess_noise = os.path.join(folder, "results_excessnoise_2025-05-07_05-02-45.npy")
    data = ExcessNoiseResults.load(results_excess_noise)
    num_rep = data.num_rep
    excess_noise_bob = data.excess_noise_bob
    transmittance = data.transmittance
    photon_number = data.photon_number
    electronic_noise = data.electronic_noise
    shot_noise = data.shot_noise

    # Print the indices where transmittance is less than the threshold
    indices = np.where(transmittance < threshold)[0]
    print("Indices where transmittance is less than the threshold:", indices)

    if not os.path.exists(folder + 'plot/'):
        os.makedirs(folder + 'plot/')

    plt.figure()
    plt.plot(transmittance, "o")
    plt.xlabel("Round")
    plt.ylabel("Transmittance")
    plt.grid()
    plt.savefig(folder + 'plot/' + '200_acq_transmittance.png')

    plt.figure()
    plt.plot(excess_noise_bob, "o")
    plt.xlabel("Round")
    plt.ylabel("$\\xi_B$")
    plt.grid()
    plt.savefig(folder + 'plot/' + '200_acq_excess_noise.png')
    plt.show()

def main():
    folder = "./qosst_bob/synchronization/data/"
    plot_excess_noise_results(folder)

if __name__ == "__main__":
    main()