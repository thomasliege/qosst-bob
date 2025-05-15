import sys
import logging
import numpy as np

from qosst_core.infos import get_script_infos

from qosst_bob import __version__
from qosst_bob.bob import Bob
from qosst_bob.data import TransmittanceResults
import os

from qosst_bob.atm_channel.utils import save_alice_symbols, save_bob_symbols, save_shot_noise_symbols, save_electronic_noise_symbols

logger = logging.getLogger(__name__)

def save_dsp_data(bob: Bob, savefolder: str, logger: logging):
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)
    dsp_data = savefolder + f'DSP_data.npz'
    alice_photon_number = bob.get_alice_photon_number()
    np.savez(dsp_data, alice_photon_number = alice_photon_number, alice_symbols = bob.alice_symbols, indices = bob.indices)
    logger.info("DSP data saved !")
    save_shot_noise_symbols(bob, 0, '', savefolder, logger)
    save_electronic_noise_symbols(bob, 0, '', savefolder, logger)

def main():
    """
    Main entrypoint of the script.
    """
    print(get_script_infos())

    config = './config-turb-channel.toml'
    savefolder = './qosst_bob/phase_recovery/data/'

    bob = Bob(config)
    bob.open_hardware()
    logger.info("Connecting Bob.")
    bob.connect()

    logger.info("Loading electronic noise data.")
    bob.load_electronic_noise_data()

    logger.info("Identification.")
    bob.identification()

    logger.info("Initialization")
    bob.initialization()

    if bob.notifier:
        bob.notifier.send_notification("Experiment started.")

    if not bob.config.bob.switch.switching_time:
        logger.info("Manual shot noise calibration")
        bob.get_electronic_shot_noise_data()

    logger.info("Starting QIE")
    bob.quantum_information_exchange()

    save_dsp_data(bob, savefolder, logger)

    bob.close()

if __name__ == "__main__":
    main()
