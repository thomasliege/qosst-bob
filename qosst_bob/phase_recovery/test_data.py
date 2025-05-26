import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging
from pathlib import Path
import os

from qosst_core.logging import create_loggers
from qosst_core.infos import get_script_infos
from qosst_core.configuration.config import Configuration

from scipy.ndimage import uniform_filter1d
from scipy.signal import oaconvolve
from qosst_bob.phase_recovery.utils import phase_noise_correction_ukf
from qosst_bob.dsp.resample import (
    _best_sampling_point_float,
    upsample
)
from qosst_bob.dsp.dsp import dsp_bob, special_dsp, find_global_angle


logger = logging.getLogger(__name__)


def dsp_subframe(data, config):
    symbol_rate = config.frame.quantum.symbol_rate
    dac_rate = config.bob.dsp.alice_dac_rate
    adc_rate = config.bob.adc.rate
    num_symbols = config.frame.quantum.num_symbols
    roll_off = config.frame.quantum.roll_off
    frequency_shift = config.frame.quantum.frequency_shift
    num_pilots = config.frame.pilots.num_pilots
    pilots_frequencies = config.frame.pilots.frequencies
    synchro_cls = config.frame.synchronization.synchronization_cls
    zc_length = config.frame.synchronization.zc_length
    zc_root = config.frame.synchronization.zc_root
    mls_nbits = config.frame.synchronization.mls_nbits
    synchro_rate = config.frame.synchronization.rate
    process_subframes = config.bob.dsp.process_subframes
    subframe_length = config.bob.dsp.subframes_size
    subframe_subdivision = config.bob.dsp.subframes_subdivisions
    fir_size = config.bob.dsp.fir_size,
    tone_filtering_cutoff = config.bob.dsp.tone_filtering_cutoff
    abort_clock_recovery = config.bob.dsp.abort_clock_recovery
    excl = config.bob.dsp.exclusion_zone_pilots
    pilot_phase_filtering_size = config.bob.dsp.pilot_phase_filtering_size
    pilot_frequency_filtering_size = config.bob.dsp.pilot_frequency_filtering_size
    num_samples_fbeat_estimation = config.bob.dsp.num_samples_fbeat_estimation
    num_samples_pilot_search = config.bob.dsp.num_samples_pilot_search
    symbol_timing_oversampling = config.bob.dsp.symbol_timing_oversampling
    schema = config.bob.schema

    logger.info("Starting General DSP with direct pilot tracking")

    # Find pilot frequency
    if num_pilots < 1:
        logger.error("At least one pilot is required... Aborting")
        return None, None, None

    if num_pilots > 2:
        logger.warning(
            "More than 2 pilots were given but only two are necessary for recovery with unshared clock and unshared LO. Taking the two first pilots (%.2f MHz, %.2f MHz)",
            pilots_frequencies[0] * 1e-6,
            pilots_frequencies[1] * 1e-6,
        )

    f_pilot_1 = pilots_frequencies[0]

    # Convert the data to float32
    data = data.astype('f')

    # Create the synchronization sequence object
    synchro_obj = synchro_cls(
        root=zc_root,
        length=zc_length,
        nbits=mls_nbits,
    )

    # Use the base DAC rate if the sample rate of the synchronization sequence has not been
    # provided.
    if synchro_rate == 0:
        synchro_rate = dac_rate
    synchro_oversampling = int(adc_rate / synchro_rate)

    logger.info("Computing envelope for approximate synchronization sequence search")
    envelope = np.abs(data[::synchro_oversampling])
    envelope = uniform_filter1d(envelope, synchro_obj.length)
    preamble_synchro_start = (np.argmax(envelope) - synchro_obj.length // 2) * synchro_oversampling

    # The pilot frequencies are estimated on a large sample
    # (typically 10M points) taken after the synchronization sequence.
    pilot_start_point = preamble_synchro_start + 2 * zc_length * synchro_oversampling
    data_pilots = data[pilot_start_point:pilot_start_point+num_samples_pilot_search]

    if num_pilots == 2:
        f_pilot_2 = pilots_frequencies[1]
        logger.info("Searching for pilots")
        f_pilot_real_1, f_pilot_real_2 = find_two_pilots(data_pilots, adc_rate, excl=excl)
        logger.info(
            "Pilots found at %f MHz and %f MHz",
            f_pilot_real_1 * 1e-6,
            f_pilot_real_2 * 1e-6,
            )

        # Measure the clock difference
        delta_f = (f_pilot_real_2 - f_pilot_real_1) / (f_pilot_2 - f_pilot_1)
        logger.info(
            "Tone difference : %.6f (expected value : %.2f)",
            (f_pilot_real_2 - f_pilot_real_1) * 1e-6,
            (f_pilot_2 - f_pilot_1) * 1e-6,
        )
        logger.info("Difference of clock is estimated at %.6f", delta_f)

        if abort_clock_recovery != 0 and np.abs(1 - delta_f) > abort_clock_recovery:
            logger.warning(
                "Clock recovery algorithm aborted due to too high mismatch (%f > %f). Taking adc_rate as real adc_value.",
                np.abs(1 - delta_f),
                abort_clock_recovery,
            )
            equi_adc_rate = adc_rate
        else:
            logger.debug("Clock mismatch was accepted.")
            equi_adc_rate = adc_rate / delta_f

        if dsp_debug:
            dsp_debug.equi_adc_rate = equi_adc_rate
            dsp_debug.delta_frequency_pilots = delta_f
            dsp_debug.real_pilot_frequencies = [f_pilot_real_1, f_pilot_real_2]
    else:
        # Only one pilot found, so we assume that the ADC rate perfectly
        # matches the specified value.
        equi_adc_rate = adc_rate

        logger.info("Searching for the pilot")
        f_pilot_real_1 = find_one_pilot(data_pilots, equi_adc_rate, excl=excl)
        dsp_debug.real_pilot_frequencies = [f_pilot_real_1]

    # Correct estimates with true ADC rate (if estimated).
    f_pilot_1 *= equi_adc_rate / adc_rate
    sps = equi_adc_rate / symbol_rate
    f_beat = f_pilot_real_1 - f_pilot_1

    logger.info("Equivalent ADC rate is %.6f MHz", equi_adc_rate * 1e-6)
    logger.info("Equivalent SPS is %.6f", sps)

    logger.info('Searching for start of the synchronization sequence')
    synchro_search_start = max(preamble_synchro_start - 4 * synchro_obj.length * synchro_oversampling, 0)
    synchro_search_end = synchro_search_start + 8 * synchro_obj.length * synchro_oversampling
    data_synchro = data[synchro_search_start:synchro_search_end]
    shift = np.exp(-1j * 2 * np.pi * np.arange(len(data_synchro)) * f_beat / equi_adc_rate)
    begin_synchro, end_synchro = synchronize(
        data_synchro * shift, synchro_obj,
        resample=equi_adc_rate / synchro_rate)
    begin_synchro += synchro_search_start
    end_synchro += synchro_search_start

    begin_data = end_synchro
    end_data = int(
        begin_data + num_symbols * np.ceil(sps + 1)
    )  # We take a bit more of what is needed to be sure to have all symbols
    useful_data = data[begin_data:end_data]

    begin_subframe = 0
    end_subframe = int(np.ceil(subframe_length * (sps + 1) - 0.5))
    result = []
    num_symbols_recovered = 0

    # Number of samples to include from the previous subframe to account for
    # the boundary conditions of the filters.
    num_samples_previous_subframe = max(pilot_phase_filtering_size, fir_size)

    # Pre-compute the RRC filter.
    _, rrc_filter = root_raised_cosine_filter(
        int(10 * sps + 2),
        roll_off,
        1 / symbol_rate,
        equi_adc_rate,
    )
    rrc_filter = (rrc_filter[1:] / np.sqrt(sps)).astype(np.complex64)

    # Pre-compute the complex exponential for shifting.
    shift_size = subframe_length * (sps + 1) + num_samples_previous_subframe
    shift_up = np.exp(
        1j
        * 2
        * np.pi
        * np.arange(shift_size)
        * (f_pilot_1 - frequency_shift)
        / equi_adc_rate
    ).astype(np.complex64)

    # Pre-compute the filter extracting the pilot tone.
    pilot_bp_filter = (firwin(fir_size, tone_filtering_cutoff / equi_adc_rate) * np.exp(
        1j * 2 * np.pi * np.arange(fir_size) * f_pilot_real_1 / equi_adc_rate
    )).astype(np.complex64)

    pilot_phase_list = [0]

        

def ukf_analysis(args):
    folder = args.data_folder
    config = Configuration(folder + 'config.toml')
    for f in os.listdir(folder + 'acq/'):
        if 'signal' in f:
            data = np.load(folder + f)
    if len(data.shape) == 2:
        data = data[0]
    if config.bob.switch.switching_time:
        end_electronic_shot_noise = int(
            config.bob.switch.switching_time * config.bob.adc.rate
        )
        data = data[end_electronic_shot_noise :]

    quantum_symbols, params, dsp_debug, initial_phase_list = dsp_subframe(data, config)



def _create_parser() -> argparse.ArgumentParser:
    """
    Create the parser for qosst-bob-ukf.

    Returns:
        argparse.ArgumentParser: parser for the qosst-bob-ukf.
    """
    parser = argparse.ArgumentParser(prog="qosst-bob-synchronization-analysis")

    parser.add_argument("data_folder", type=str, help="Path to the data.")
    parser.add_argument("--n_frames", type=int, default=1, help="Number of subframes to analyse, -1 means analysing the whole dataset.")
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Verbosity level."
    )
    return parser


def main():
    print(get_script_infos())
    parser = _create_parser()
    args = parser.parse_args()

    create_loggers(args.verbose, args.file)


    
    
if __name__ == "__main__":
    main()