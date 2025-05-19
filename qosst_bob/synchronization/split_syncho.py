from qosst_bob.data import ExcessNoiseResults
import os
import shutil
import concurrent.futures
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
def copy_files(indices, target_dir, acq_dict, acq_folder):
    tasks = []
    for idx in indices:
        for fname in acq_dict.get(idx, []):
            src = os.path.join(acq_folder, fname)
            dst = os.path.join(target_dir, fname)
            tasks.append((src, dst))
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(lambda args: shutil.copy(*args), tasks)

def split_syncho(folder : str, threshold: float = 0.01):
    """
    Split the synchronisation results into two parts and save them.

    Parameters
    ----------
   
    """
    # Find the only .npy file in the folder
    npy_files = [f for f in os.listdir(folder) if f.endswith('.npy')]
    if len(npy_files) != 1:
        raise FileNotFoundError("Expected exactly one .npy file in the folder.")
    results_excess_noise = os.path.join(folder, npy_files[0])
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
    print("Saving split data...")
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
    
    # Plot the split data
    plt.figure()
    plt.plot(transmittance_good, "o", color = 'red', label="Good Data")
    plt.plot(transmittance_bad, "o", color='blue', label="Bad Data") 
    plt.xlabel("Round")
    plt.ylabel("Transmittance")
    plt.legend()
    plt.grid()
    plt.show()

    # Save the corresponding acquisitions
    good_indices = np.where(indices == 0)[0]
    bad_indices = np.where(indices == 1)[0]
    print("Acquisitions failed : ", bad_indices)

    acq_folder = os.path.join(folder, 'export')
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
    print("Copying acquisitions...")
    good_dir = os.path.join(folder, 'split_data', 'acquisitions', 'good')
    bad_dir = os.path.join(folder, 'split_data', 'acquisitions', 'bad')
    os.makedirs(good_dir, exist_ok=True)
    os.makedirs(bad_dir, exist_ok=True)

    # Copy acquisitions
    copy_files(good_indices, good_dir, acq_dict, acq_folder)
    copy_files(bad_indices, bad_dir, acq_dict, acq_folder)
    if not os.path.exists(folder + 'split_data/acquisitions/'):
        os.makedirs(folder + 'split_data/acquisitions/')




def main():
    folder = "./qosst_bob/synchronization/data/"
    split_syncho(folder)

if __name__ == "__main__":
    main()