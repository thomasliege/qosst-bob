"""
Script to analyze the results of the synchronization process.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import logging
import argparse
from pathlib import Path
from typing import Tuple
import glob

from qosst_bob import __version__
from qosst_bob.dsp.dsp import dsp_bob, special_dsp, find_global_angle
from qosst_core.infos import get_script_infos
from qosst_core.logging import create_loggers
from qosst_core.configuration.config import Configuration

from qosst_bob.synchronization.utils import comapre_indices

logger = logging.getLogger(__name__)


def synchronization_analysis(
    config: Configuration,
    data: np.ndarray,
    electronic_noise_data: np.ndarray,
    electronic_shot_noise_data: np.ndarray,
    all_alice_symbols: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    logger.info("Starting offline DSP")

    if config.bob.switch.switching_time:
        end_electronic_shot_noise = int(
            config.bob.switch.switching_time * config.bob.adc.rate
        )
        data = data[end_electronic_shot_noise :]

    quantum_symbols, params, dsp_debug = dsp_bob(data, config)

    # Correct global phase of each frame of quantum symbols
    all_indices = []
    last_indice = -1
    for i, frame in enumerate(quantum_symbols):
        logger.info(
            "Finding global angle at frame %i/%i",
            i + 1,
            len(quantum_symbols),
        )

        # Generate indices
        indices = np.arange(last_indice + 1, last_indice + 1 + len(frame))
        np.random.shuffle(indices)
        indices = indices[: int(len(frame) * config.bob.parameters_estimation.ratio)]

        alice_symbols = all_alice_symbols[indices]

        # Find global angle
        angle, cov = find_global_angle(
            frame[indices - (last_indice + 1)], alice_symbols
        )

        logger.info("Global angle found : %.2f with cov %.2f", angle, cov)

        quantum_symbols[i] = np.exp(1j * angle) * frame

        # Add indices and symbols, update last_indice
        all_indices.append(indices)
        last_indice = last_indice + len(frame)

    quantum_symbols = np.concatenate(quantum_symbols)
    all_indices = np.concatenate(all_indices)

    begin_data = dsp_debug.begin_data

    logger.info(
        "Time between end of shot noise and signal : %f ms",
        begin_data / config.bob.adc.rate * 1e3,
    )

    logger.info("Applying DSP on elec and elec+shot noise data")

    params.elec_noise_estimation_ratio = config.bob.dsp.elec_noise_estimation_ratio
    params.elec_shot_noise_estimation_ratio = config.bob.dsp.elec_shot_noise_estimation_ratio

    electronic_symbols, electronic_shot_symbols = special_dsp(
        electronic_noise_data, electronic_shot_noise_data, params
    )

    return quantum_symbols, all_indices, electronic_symbols, electronic_shot_symbols


def run_analysis(args):
    """
    Run the analysis of the synchronization process.
    """

    # Load configuration
    configuration = Configuration(args.file)

    # Load data and symbols
    electronic_noise_data = np.load(args.elec_data, allow_pickle=True)
    if args.elec_data.endswith('.qosst'):
        electronic_noise_data = electronic_noise_data.data[0]
    
    data = np.load(args.data)
    if len(data.shape) == 2:
        data = data[0]
        
    if configuration.bob.switch.switching_time:
        end_electronic_shot_noise = int(
            configuration.bob.switch.switching_time * configuration.bob.adc.rate
        )
        electronic_shot_noise_data = data[: end_electronic_shot_noise]
    else:
        electronic_shot_noise_data = np.load(args.elec_shot_data, allow_pickle=True)
        if args.elec_shot_data.endswith('.qosst'):
            electronic_shot_noise_data = electronic_shot_noise_data.data[0]
    
    alice_symbols = np.load(args.alice_symbols)
    if args.indices is not None:
        indices_from_file = np.load(args.indices)
        alice_symbols = alice_symbols[np.argsort(indices_from_file)]
        logger.info("Undo alice_symbols shuffling")

    # Offline DSP
    quantum_symbols, indices, electronic_symbols, electronic_shot_symbols = synchronization_analysis(
        configuration,
        data,
        electronic_noise_data,
        electronic_shot_noise_data,
        alice_symbols,
    )

    # # Comparing indices
    # if args.indices is not None:
    #     logger.info("Comparing indices")
    #     indices_from_file = np.load(args.indices)
    #     if not comapre_indices(logger, indices, indices_from_file):
    #         logger.error("Indices are not the same.")
    
    alice_photon_number = np.load(args.alice_photon_number)[0]
    if args.alice_photon_number.endswith('.qosst'):
        alice_photon_number = alice_photon_number.data[0]

    # Parameters estimation
    transmittance, excess_noise, electronic_noise = (
        configuration.bob.parameters_estimation.estimator.estimate(
            alice_symbols[indices],
            quantum_symbols[indices],
            alice_photon_number,
            electronic_symbols,
            electronic_shot_symbols,
        )
    )

    skr = (
        configuration.frame.quantum.symbol_rate
        * configuration.bob.parameters_estimation.skr_calculator.skr(
            Va=2 * alice_photon_number,
            T=transmittance / configuration.bob.eta,
            xi=excess_noise / transmittance,
            eta=configuration.bob.eta,
            Vel=electronic_noise,
            beta=0.95,
        )
    )

    logger.info("Transmittance: %f", transmittance)
    logger.info("Excess noise (Bob): %f", excess_noise)
    logger.info("Electronic noise: %f", electronic_noise)
    logger.info("Secret key rate: %f MBit/s", skr * 1e-6)
    return {
        "transmittance": transmittance,
        "excess_noise": excess_noise,
        "electronic_noise": electronic_noise,
        "skr": skr,
    }


def batch_analysis_acquisitions(args):
    """
    Run analysis for all acquisitions in the given folder.
    """
    logger.info("Beginning batch analysis")
    data_folder = args.data_folder
    
    # Find all Alice symbols files
    logger.info(f"Searching for acquisitions in {data_folder}")
    alice_symbols_files = sorted(glob.glob(os.path.join(data_folder, "acq*_alice_symbols-*")))
    results = []
    
    for alice_symbols_path in alice_symbols_files:
        # Extract acquisition prefix and timestamp
        base = os.path.basename(alice_symbols_path)
        acq_prefix = base.split("_alice_symbols")[0]
        # Reconstruct other file paths
        # Find the corresponding signal and photon number files, ignoring timestamp
        signal_files = glob.glob(os.path.join(data_folder, f"{acq_prefix}_signal-*"))
        photon_number_files = glob.glob(os.path.join(data_folder, f"{acq_prefix}_n-*"))
        index_files = glob.glob(os.path.join(data_folder, f"{acq_prefix}_indices-*"))
        if not signal_files or not photon_number_files or not index_files:
            logger.warning(f"Missing files for acquisition {acq_prefix}, skipping.")
            continue
        data_path = signal_files[0]
        alice_photon_number_path = photon_number_files[0]
        index_path = index_files[0]
        print(data_path)
        # Skip if files are missing
        if not (os.path.exists(data_path) and os.path.exists(alice_photon_number_path) and os.path.exists(index_path)):
            logger.warning(f"Missing files for acquisition {acq_prefix}, skipping.")
            continue

        # Build args Namespace with normalized paths
        data_path = os.path.normpath(data_path)
        alice_symbols_path = os.path.normpath(alice_symbols_path)
        alice_photon_number_path = os.path.normpath(alice_photon_number_path)
        index_path = os.path.normpath(index_path)

        args = argparse.Namespace(
            data=data_path,
            alice_symbols=alice_symbols_path,
            alice_photon_number=alice_photon_number_path,
            indices=index_path,
            file=args.file,
            elec_data=args.elec_data,
            elec_shot_data=None,
            save=False,
            verbose=2,
        )
        print(args)

        logger.info("=" * 60)
        logger.info(f"Processing acquisition {acq_prefix}")
        logger.info("=" * 60)
        result = run_analysis(args)
        results.append(result)
    return results



def _create_parser() -> argparse.ArgumentParser:
    """
    Create the parser for qosst-bob-synchronization-analysis.

    Returns:
        argparse.ArgumentParser: parser for the qosst-bob-synchronization-analysis.
    """
    default_config_location = Path(os.getcwd()) / "./qosst_bob/synchronization/data/config.toml"
    default_electronic_noise_location = "./qosst_bob/synchronization/data/electronic_noise.qosst"
    parser = argparse.ArgumentParser(prog="qosst-bob-synchronization-analysis")

    subparsers = parser.add_subparsers(dest="mode", required=True, help="Mode: single or batch")

    # Single acquisition parser
    single_parser = subparsers.add_parser("single", help="Analyze a single acquisition")
    single_parser.add_argument("data", type=str, help="Path to the data.")
    single_parser.add_argument("alice_symbols", type=str, help="Path to Alice's symbols.")
    single_parser.add_argument("alice_photon_number", type=str, help="Mean number of photons per symbol at Alice side.")
    single_parser.add_argument(
        "-f", "--file", default=default_config_location, help=f"Path of the configuration file. Default : {default_config_location}."
    )
    single_parser.add_argument(
        "--elec_data", type=str, default=default_electronic_noise_location, help="Path to the electronic noise data."
    )
    single_parser.add_argument(
        "--elec_shot_data", type=str, default=None, help="(Optional) Path to the electronic and shot noise data."
    )
    single_parser.add_argument(
        "--indices", type=str, default=None, help="(Optional) Path to the indices found."
    )
    single_parser.add_argument(
        "--save", type=bool, default=False, help="Save the data. Default: False."
    )
    single_parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Verbosity level."
    )

    # Batch parser
    batch_parser = subparsers.add_parser("batch", help="Analyze all acquisitions in a folder")
    batch_parser.add_argument("data_folder", type=str, help="Folder containing all acquisitions")
    batch_parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Verbosity level."
    )
    batch_parser.add_argument(
        "-f", "--file", default=default_config_location, help=f"Path of the configuration file. Default : {default_config_location}."
    )
    batch_parser.add_argument(
        "--elec_data", type=str, default=default_electronic_noise_location, help="Path to the electronic noise data."
    )
    return parser


def main():
    print(get_script_infos())
    parser = _create_parser()
    args = parser.parse_args()

    create_loggers(args.verbose, args.file)

    if args.mode == "single":
        run_analysis(args)
    elif args.mode == "batch":
        batch_analysis_acquisitions(args)

if __name__ == "__main__":
    main()