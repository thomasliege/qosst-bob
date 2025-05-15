import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.signal import oaconvolve
from qosst_bob.phase_recovery.utils import phase_noise_correction_ukf
from qosst_bob.dsp.resample import (
    _best_sampling_point_float,
    upsample
)
from qosst_bob.dsp.dsp import dsp_bob, special_dsp, find_global_angle

def dsp_subframe(file):
        # Parameters
        data = np.load(file)
        pilot_data = data['pilot_data']
        subframe_data = data['subframe']
        shift_up = data['shift_up']
        rrc_filter = data['rrc_filter']
        sps = data['sps']
        symbol_timing_oversampling = data['symbol_timing_oversampling']
        begin_subframe = data['begin_subframe']
        begin_extended_subframe = data['begin_extended_subframe']
        subframe_length = data['subframe_length']
        subframe_subdivision = data['subframe_subdivision']

        print(f'Subframe data : {subframe_data}')
        # Phase correction
        # pilot_angle = (phase_noise_correction_ukf(pilot_data))
        # pilot_angle = (pilot_angle + np.pi) % (2 * np.pi) - np.pi
        pilot_angle = np.angle(pilot_data)
        
        
        pilot_phase_filtering_size = 4000
        if pilot_phase_filtering_size > 1:
            pilot_angle = np.unwrap(pilot_angle.astype('d'))
            if pilot_phase_filtering_size > 1:
                pilot_angle = uniform_filter1d(
                    pilot_angle,
                    pilot_phase_filtering_size
                )
        clean_pilot = np.exp(-1j * pilot_angle).astype(np.complex64)

        subframe_data *= clean_pilot
        subframe_data *= shift_up[:len(subframe_data)]

        subframe_data = oaconvolve(subframe_data, rrc_filter, "same")

        # Ignore the extra samples at the beginning of the frame
        subframe_data = subframe_data[begin_subframe - begin_extended_subframe:]

        if symbol_timing_oversampling != 1:
            subframe_data = upsample(subframe_data, symbol_timing_oversampling, 2)

        best_t = _best_sampling_point_float(
            subframe_data,
            sps * symbol_timing_oversampling
        )
        print(f'Best sampling point : {best_t}')
        best_grid = np.round(
            best_t + sps * symbol_timing_oversampling * np.arange(
                subframe_length)
        ).astype(int)

        subframe_data = subframe_data[best_grid]

        chunk_length = len(subframe_data) // subframe_subdivision
        for i in range(subframe_subdivision):
            start = i * chunk_length
            print(f'Start : {start}')
            print(f'Chunk length : {chunk_length}')
            if i != subframe_subdivision - 1:
                result = subframe_data[start:start + chunk_length]
            else:
                result = subframe_data[start:]
        return result

def find_globa_angle(result, all_alice_symbols):
    indices = np.arange(0, len(result))
    np.random.shuffle(indices)
    indices = indices[: int(len(result)) // 2]
    print(indices)
    alice_symbols = all_alice_symbols[indices]

    # Find global angle
    angle, cov = find_global_angle(
        result[indices], alice_symbols
    )
    print(f'Angle found : {angle}, with covriance : {cov}')

def plot_angle(file):
    data = np.load(file)
    angle_ukf = np.unwrap(data['ukf'][0:10000])
    basic_angle = np.unwrap(data['basic'][0:10000])
    # angle_ukf = data['ukf'][9900:10000]
    # basic_angle = data['basic'][9900:10000]

    plt.figure(figsize=(10, 4))
    plt.plot(angle_ukf, '--', label='UKF Estimated Phase')
    plt.plot(basic_angle, '--', label='Basic')
    plt.title("Unscented Kalman Filter Phase Tracking)")
    plt.xlabel("Time Step")
    plt.ylabel("Phase (radians)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    # file = './qosst_bob/phase_recovery/data/phase_estimation.npz'
    # plot_angle(file)

    result = dsp_subframe('./qosst_bob/phase_recovery/data/subframe.npz')
    np.savez('./qosst_bob/phase_recovery/test/result_basic.npz', result = result)

    data = np.load('./qosst_bob/phase_recovery/test/result_basic.npz')
    result = data['result']
    all_alice_symbols = np.load('./data_test_UKF/symbols_alice.npy' , allow_pickle=True)
    for k in range(10):
        find_globa_angle(result, all_alice_symbols)
    
    
if __name__ == "__main__":
    main()