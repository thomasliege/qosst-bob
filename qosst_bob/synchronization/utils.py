import numpy as np
from qosst_bob.data import ExcessNoiseResults
import os
import logging
from qosst_core.configuration.config import Configuration
import matplotlib.pyplot as plt
import glob

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

def plot_fft(folder: str):
    """
    Plot the FFT of all acquisition signals in the folder/acq/ directory,
    along with the electronic noise and shot noise results.

    Parameters
    ----------
    folder : str
        The folder containing the results and config.toml.
    """

    # Find all acquisition files in folder/acq/
    acq_folder = os.path.join(folder, "acq")
    acq_files = sorted(glob.glob(os.path.join(acq_folder, "*signal*.npy")))

    # Load electronic noise 
    electronic_noise_path = os.path.join(folder, "electronic_noise.qosst")
    electronic_noise = np.load(electronic_noise_path, allow_pickle=True)
    if electronic_noise_path.endswith('.qosst'):
        electronic_noise = electronic_noise.data[0]

    # Load config.toml
    config_path = os.path.join(folder, "config.toml")
    configuration = Configuration(config_path)
    adc_rate = configuration.bob.adc.rate
    exclusion_zones = configuration.bob.dsp.exclusion_zone_pilots

    for acq_file in acq_files:
        print(f"Processing file: {acq_file}")
        plt.figure()
        data = np.load(acq_file)
        if len(data.shape) == 2:
            data = data[0]

        # Load electronic + shot noise region if available
        if configuration.bob.switch.switching_time:
            end_electronic_shot_noise = int(
                configuration.bob.switch.switching_time * configuration.bob.adc.rate
            )
            electronic_shot_noise_data = data[: end_electronic_shot_noise]

        plt.psd(
            data,
            NFFT=2048,
            Fs=adc_rate,
            label=f"Signal: {os.path.basename(acq_file)}",
        )

        if electronic_shot_noise_data is not None:
            plt.psd(
                electronic_shot_noise_data,
                NFFT=2048,
                Fs=adc_rate,
                label=f"Shot noise: {os.path.basename(acq_file)}",
            )

        plt.psd(
            electronic_noise,
            NFFT=2048,
            Fs=adc_rate,
            label="Electronic noise",
        )

        for begin_zone, end_zone in exclusion_zones:
            plt.axvspan(begin_zone, end_zone, alpha=0.3, color="black")
        plt.legend(fancybox=True, shadow=True)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Power Spectral Density [dBm/Hz]")
        plt.title("PSD vs. frequency")
        plt.grid(True)
    plt.show()

def main():
    # folder = "./qosst_bob/synchronization/data/"
    # plot_excess_noise_results(folder)

    folder = "./qosst_bob/synchronization/data/semi_bad_acq/"
    plot_fft(folder)

if __name__ == "__main__":
    main()