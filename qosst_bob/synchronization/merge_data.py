"""
Script to merge the acquisitions and then analyse the stability of the performances
"""

import os
import glob
import numpy as np

def merge(folder: str, total_num_symbols: int = int(1e7)):
    """
    Merge the acquisitions and then analyse the stability of the performances.
    """
    # Find all Alice symbols files and sort them
    alice_symbols_files = sorted(glob.glob(os.path.join(folder, "acq*_alice_symbols-*")))
    signal_files = sorted(glob.glob(os.path.join(folder, "acq*_signal-*")))
    alice_photon_number_files = sorted(glob.glob(os.path.join(folder, "acq*_n-*")))

    num_acquisitions = len(alice_symbols_files)
    if num_acquisitions == 0:
        print("No acquisitions found.")
        return

    # Merge by packets
    num_packets = total_num_symbols // 
    for i in range(0, num_acquisitions, 10):
        alice_symbols_merged = []
        signal_merged = []
        packet_files = alice_symbols_files[i:i+10]
        packet_signals = signal_files[i:i+10]

        for alice_file, signal_file in zip(packet_files, packet_signals):
            alice_symbols = np.load(alice_file)
            signal = np.load(signal_file)
            alice_symbols_merged.append(alice_symbols)
            signal_merged.append(signal)

        # Concatenate arrays
        alice_symbols_merged = np.concatenate(alice_symbols_merged)
        signal_merged = np.concatenate(signal_merged)

        # Save merged arrays
        packet_idx = i // 10
        np.save(os.path.join(folder, f"merged_alice_symbols_packet{packet_idx}.npy"), alice_symbols_merged)
        np.save(os.path.join(folder, f"merged_signal_packet{packet_idx}.npy"), signal_merged)
        print(f"Saved merged packet {packet_idx} with {len(packet_files)} acquisitions.")

def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python merge_data.py <folder>")
        return 1
    merge(sys.argv[1])
    return 0

if __name__ == "__main__":
    main()