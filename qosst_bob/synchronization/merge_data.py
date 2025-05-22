"""
Script to merge the acquisitions and then analyse the stability of the performances
"""

import os
import glob
import numpy as np
from pathlib import Path
import argparse
from qosst_core.configuration.config import Configuration
from qosst_bob.parameters_estimation.base import complex_to_real

def parameter_estimation_concatenated(args):
    """
    Estimate the transmittance, excess noise and electronic noise by
    using the covariance method.
    """
    data_folder = args.data_folder + 'merged/'
    transmittance_result = []
    excess_noise_bob_result = []
    vel_result = []
    
    # Load configuration
    configuration = Configuration(args.file)

    # Number of concatenated files
    num_files = len([name for name in os.listdir(data_folder) if name.endswith('.npy')])
    num_packets = num_files // 4
    print(f"Number of merged files: {num_files}, packets : {num_packets}")

    for idx in range(num_packets):
        alice_photon_number = np.load(data_folder + f'merged_photon_number_packet{idx}.npy')[0]
        alice_symbols = np.load(data_folder + f'merged_alice_symbols_packet{idx}.npy')[0]
        alice_symbols = complex_to_real(alice_symbols)
        bob_symbols = np.load(data_folder + f'merged_signal_packet{idx}.npy')[0][0]
        vel = np.load(data_folder + f'merged_vel_packet{idx}.npy')[0]

        conversion_factor = np.sqrt(
            alice_photon_number / np.mean(np.abs(alice_symbols) ** 2)
        )
        factor = np.cov([alice_symbols, bob_symbols])[0][1].real / np.var(alice_symbols)

        excess_noise_bob = np.var(factor * alice_symbols - bob_symbols) - 1 - vel

        transmittance = factor**2 / conversion_factor**2

        transmittance_result.append(transmittance)
        excess_noise_bob_result.append(excess_noise_bob)
        vel_result.append(vel)

        skr = (
            configuration.frame.quantum.symbol_rate
            * configuration.bob.parameters_estimation.skr_calculator.skr(
                Va=2 * alice_photon_number,
                T=transmittance / configuration.bob.eta,
                xi=excess_noise_bob / transmittance,
                eta=configuration.bob.eta,
                Vel=vel,
                beta=0.95,
            )
        )

        print(f"Acquisition n°{idx} - Transmittance: {transmittance}")
        print(f"Acquisition n°{idx} - Excess noise (Bob): {excess_noise_bob}")
        print(f"Acquisition n°{idx} - Electronic noise: {vel}")
        print(f"Acquisition n°{idx} - Secret key rate: {skr * 1e-6} MBit/s")
        print("-" * 40)

    return transmittance_result, excess_noise_bob_result, vel_result

def merge(folder: str, N_merge: int = 10):
    """
    Merge the acquisitions and then analyse the stability of the performances.
    """
    # Find all Alice symbols files and sort them
    alice_symbols_files = sorted(glob.glob(os.path.join(folder, "acq*_alice_symbols-*")))
    bob_symbols = np.load(folder + 'bob_symbols.npy')
    vel = np.load(folder + 'vel.npy')
    indices = np.load(folder + 'indices.npy')
    alice_photon_number_files = sorted(glob.glob(os.path.join(folder, "acq*_n-*")))

    print(f"len bob_symbols : {len(bob_symbols)}")
    print(f"len alice_symbols : {len(alice_symbols_files)}")
    assert len(bob_symbols) == len(alice_symbols_files)

    num_acquisitions = len(alice_symbols_files)
    if num_acquisitions == 0:
        print("No acquisitions found.")
        return

    # Merge by packets
    for i in range(0, num_acquisitions, N_merge):
        alice_symbols_merged = []
        n_merged = []
        packet_files = alice_symbols_files[i:i+N_merge]
        signal_merged = bob_symbols[i:i+N_merge]
        indices_merged = indices[i:i+N_merge]
        vel_merged = vel[i:i+N_merge]
        packet_n = alice_photon_number_files[i:i+N_merge]

        for idx, (alice_file, n_file) in enumerate(zip(packet_files, packet_n)):
            alice_symbols = np.load(alice_file)
            n = np.load(n_file)
            alice_symbols_merged.append(alice_symbols[indices_merged[idx]])
            n_merged.append(n)

        # Concatenate arrays
        alice_symbols_merged = np.concatenate(alice_symbols_merged)

        # Save merged arrays
        if not os.path.exists(folder + 'merged/'):
            os.mkdir(os.path.join(folder, 'merged'))
        packet_idx = i // N_merge
        np.save(os.path.join(folder + 'merged/', f"merged_alice_symbols_packet{packet_idx}.npy"), alice_symbols_merged)
        np.save(os.path.join(folder + 'merged/', f"merged_signal_packet{packet_idx}.npy"), signal_merged)
        np.save(os.path.join(folder + 'merged/', f"merged_photon_number_packet{packet_idx}.npy"), n_merged)
        np.save(os.path.join(folder + 'merged/', f"merged_vel_packet{packet_idx}.npy"), vel_merged)
        print(f"Saved merged packet {packet_idx} with {len(packet_files)} acquisitions.")


def _create_parser() -> argparse.ArgumentParser:
    """
    Create the parser for qosst-bob-synchronization-analysis.

    Returns:
        argparse.ArgumentParser: parser for the qosst-bob-synchronization-analysis.
    """
    default_config_location = Path(os.getcwd()) / "./qosst_bob/synchronization/data/acq_test/config.toml"
    parser = argparse.ArgumentParser(prog="qosst-bob-synchronization-analysis")

    # Batch parser
    parser.add_argument("data_folder", type=str, help="Folder containing all acquisitions")
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Verbosity level."
    )
    parser.add_argument(
        "-f", "--file", default=default_config_location, help=f"Path of the configuration file. Default : {default_config_location}."
    )
    parser.add_argument(
        "--n", type=int, default=10, help="Number of files to merge."
    )
    parser.add_argument(
        "--estimate", type=str, default=False, help="Estimation of the concatenated data."
    )
    return parser


def main():
    parser = _create_parser()
    args = parser.parse_args()
    
    if args.estimate:
        transmittance, exces_noise_bob, vel = parameter_estimation_concatenated(args)

    else:
        merge(args.data_folder, N_merge=args.n)

if __name__ == "__main__":
    main()