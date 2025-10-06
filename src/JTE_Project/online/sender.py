"""
EMG Producer - File-based Communication
----------------------------------------
Writes EMG chunks to files in a folder.
Consumer monitors folder and processes files.
"""

import numpy as np
import time
import os
import pickle
from pathlib import Path


def loadMiniANTEMGData(file_str, f_samp):
    """Load EMG data from ANT EMG system .txt file."""
    emg_data_raw = np.loadtxt(file_str)
    emg_data = emg_data_raw[:-1, :-2]
    emg_data = emg_data.T
    n_samples = emg_data.shape[1]
    time_axis = np.arange(0, n_samples / f_samp, step=1 / f_samp)
    return emg_data, time_axis


def send_emg_chunks_to_folder(emg_file, output_folder, f_samp=500, chunk_size=500, delay=0.5):
    """
    Load EMG file and save chunks to folder.

    Parameters
    ----------
    emg_file : str
        Path to EMG file
    output_folder : str
        Folder where chunks will be saved
    f_samp : int
        Sampling frequency
    chunk_size : int
        Samples per chunk
    delay : float
        Delay between chunks in seconds
    """
    print("=" * 70)
    print("PRODUCER - File-based EMG Sender")
    print("=" * 70)

    # Create output folder
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)

    # Clear old files
    for old_file in output_path.glob("chunk_*.pkl"):
        old_file.unlink()

    print(f"Output folder: {output_folder}")
    print(f"Cleared old chunk files\n")

    # Load EMG data
    emg_data, time_axis = loadMiniANTEMGData(emg_file, f_samp)
    n_channels, n_samples = emg_data.shape

    print(f"Loaded: {emg_file}")
    print(f"Shape: {emg_data.shape}")
    print(f"Total samples: {n_samples}")
    print(f"Chunk size: {chunk_size}")

    # Calculate chunks
    n_chunks = n_samples // chunk_size
    remainder = n_samples % chunk_size
    total_chunks = n_chunks + (1 if remainder > 0 else 0)

    print(f"Total chunks: {total_chunks}")
    print("=" * 70 + "\n")

    # Send full chunks
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size

        # Extract chunk
        emg_chunk = emg_data[:, start_idx:end_idx]
        time_chunk = time_axis[start_idx:end_idx]

        # Metadata
        metadata = {
            'chunk_index': i,
            'total_chunks': total_chunks,
            'start_sample': start_idx,
            'end_sample': end_idx,
            'time_start': float(time_chunk[0]),
            'time_end': float(time_chunk[-1])
        }

        # Save chunk to file
        emg_tuple = (emg_chunk, time_chunk, metadata)
        chunk_file = output_path / f"chunk_{i:04d}.pkl"

        with open(chunk_file, 'wb') as f:
            pickle.dump(emg_tuple, f)

        print(f"📤 Saved chunk {i+1}/{total_chunks}: {chunk_file.name}")

        if delay > 0:
            time.sleep(delay)

    # Send remainder if exists
    if remainder > 0:
        start_idx = n_chunks * chunk_size
        emg_chunk = emg_data[:, start_idx:]
        time_chunk = time_axis[start_idx:]

        metadata = {
            'chunk_index': n_chunks,
            'total_chunks': total_chunks,
            'start_sample': start_idx,
            'end_sample': n_samples,
            'time_start': float(time_chunk[0]),
            'time_end': float(time_chunk[-1])
        }

        emg_tuple = (emg_chunk, time_chunk, metadata)
        chunk_file = output_path / f"chunk_{n_chunks:04d}.pkl"

        with open(chunk_file, 'wb') as f:
            pickle.dump(emg_tuple, f)

        print(f"📤 Saved remainder chunk: {chunk_file.name}")

    # Write completion marker
    done_file = output_path / "DONE.txt"
    with open(done_file, 'w') as f:
        f.write(f"All {total_chunks} chunks sent")

    print("\n" + "=" * 70)
    print("✓ All chunks saved!")
    print("=" * 70 + "\n")


if __name__ == "__main__":

    emg_file = "F:/SMT_MASTERPROJEKT/biosignal_toolbox/data/jte/emg/BU62D/24072025_BU62D_0g_complex_1.txt"
    output_folder = "./emg_chunks"

    send_emg_chunks_to_folder(
        emg_file=emg_file,
        output_folder=output_folder,
        f_samp=500,
        chunk_size=500,
        delay=0.5
    )

    print("Producer finished. Consumer can now process the files.")
