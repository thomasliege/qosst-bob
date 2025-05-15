import numpy as np
import matplotlib.pyplot as plt
from qosst_bob.dsp.dsp import dsp_bob
from qosst_core.configuration.config import Configuration
from qosst_bob.atm_channel.base_atm_channel import estimate_atm

def do_dsp(alice_symbols, signal_data, electronic_symbols, electronic_shot_symbols, alice_photon_number, config_path):
    
    data = signal_data[0]
    config = Configuration(config_path)
    quantum_symbols, params, dsp_debug = dsp_bob(data, config)
    (
        transmittance,
        excess_noise_bob,
        electronic_noise,
    ) = estimate_atm(
        alice_symbols,
        quantum_symbols,
        alice_photon_number,
        electronic_symbols,
        electronic_shot_symbols,
    )
