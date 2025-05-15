from qosst_bob.data import ExcessNoiseResults
import os
import shutil
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def split_syncho(folder : str, threshold: float = 0.01):
    """
    Split the synchronisation results into two parts and save them.

    Parameters
    ----------
   
    """
    results_excess_noise = os.path.join(folder, "results_excessnoise_2025-05-07_05-02-45.npy")
    data = ExcessNoiseResults.load(results_excess_noise)
    num_rep = data.num_rep
    excess_noise_bob = data.excess_noise_bob
    transmittance = data.transmittance
    photon_number = data.photon_number
    electronic_noise = data.electronic_noise
    shot_noise = data.shot_noise

    indices = np.zeros(num_rep)
    for i in range(num_rep):
        if transmittance[i] < threshold:
            indices[i] = 1
    
    # Split data into two parts
    transmittance_good = transmittance[indices == 0]
    excess_noise_bob_good = excess_noise_bob[indices == 0]
    photon_number_good = photon_number[indices == 0]
    electronic_noise_good = electronic_noise[indices == 0]
    shot_noise_good = shot_noise[indices == 0]

    transmittance_bad = transmittance[indices == 1]
    excess_noise_bob_bad = excess_noise_bob[indices == 1]
    photon_number_bad = photon_number[indices == 1]
    electronic_noise_bad = electronic_noise[indices == 1]
    shot_noise_bad = shot_noise[indices == 1]

    # Save the split data
    if not os.path.exists(folder + 'split_data/'):
        os.makedirs(folder + 'split_data/')
    np.savez(os.path.join(folder, 'split_data', 'good_data.npz'),
             transmittance=transmittance_good,
             excess_noise_bob=excess_noise_bob_good,
             photon_number=photon_number_good,
             electronic_noise=electronic_noise_good,
             shot_noise=shot_noise_good)
    np.savez(os.path.join(folder, 'split_data', 'bad_data.npz'),
             transmittance=transmittance_bad,
             excess_noise_bob=excess_noise_bob_bad,
             photon_number=photon_number_bad,
             electronic_noise=electronic_noise_bad,
             shot_noise=shot_noise_bad)
    
    # Save the corresponding acquisitions
    good_indices = np.where(indices == 0)[0]
    bad_indices = np.where(indices == 1)[0]

    acq_folder = os.path.join(folder, 'acq')
    acq_files = sorted(os.listdir(acq_folder))

    # Group files by acquisition number
    acq_dict = defaultdict(list)
    for fname in acq_files:
        # Extract acquisition number (e.g., 'acq0' from 'acq0_alice_symbols-...')
        prefix = fname.split('_')[0]
        if prefix.startswith('acq'):
            try:
                acq_num = int(prefix[3:])
                acq_dict[acq_num].append(fname)
            except ValueError:
                continue

    # Copy files to split_data/acquisitions/good/ and bad/
    good_dir = os.path.join(folder, 'split_data', 'acquisitions', 'good')
    bad_dir = os.path.join(folder, 'split_data', 'acquisitions', 'bad')
    os.makedirs(good_dir, exist_ok=True)
    os.makedirs(bad_dir, exist_ok=True)

    for idx in good_indices:
        for fname in acq_dict.get(idx, []):
            shutil.copy(os.path.join(acq_folder, fname), os.path.join(good_dir, fname))
    for idx in bad_indices:
        for fname in acq_dict.get(idx, []):
            shutil.copy(os.path.join(acq_folder, fname), os.path.join(bad_dir, fname))
    if not os.path.exists(folder + 'split_data/acquisitions/'):
        os.makedirs(folder + 'split_data/acquisitions/')

    # Plot the split data
    plt.figure()
    plt.plot(transmittance_good, "o", color = 'red', label="Good Data")
    plt.plot(transmittance_bad, "o", color='blue', label="Bad Data") 
    plt.xlabel("Round")
    plt.ylabel("Transmittance")
    plt.legend()
    plt.grid()
    plt.show()

def main():
    folder = "./qosst_bob/synchronization/data/"
    split_syncho(folder)

if __name__ == "__main__":
    main()