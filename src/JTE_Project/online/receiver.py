"""
EMG Consumer - File-based Communication
----------------------------------------
Monitors folder for new EMG chunk files and processes them.
"""

from process_emg import OnlineEMGPredictor
import numpy as np
import time
import pickle
from pathlib import Path


def process_emg_from_folder(input_folder, config_filename='pipeline_mlp_to_cnn.yaml'):
    """
    Monitor folder and process EMG chunks as they arrive.

    Parameters
    ----------
    input_folder : str
        Folder to monitor for chunk files
    config_filename : str
        Configuration file for predictor
    """
    print("=" * 70)
    print("CONSUMER - File-based EMG Receiver")
    print("=" * 70)

    input_path = Path(input_folder)

    # Create folder if it doesn't exist
    input_path.mkdir(exist_ok=True)

    print(f"Monitoring folder: {input_folder}")
    print()

    # Initialize predictor
    print("Initializing predictor...")
    predictor = OnlineEMGPredictor(config_filename=config_filename)
    print()

    print("=" * 70)
    print("CONSUMER - Waiting for chunk files...")
    print("=" * 70 + "\n")

    chunks_processed = 0
    processed_files = set()
    all_predictions = []

    # Monitor folder
    while True:
        # Get all chunk files
        chunk_files = sorted(input_path.glob("chunk_*.pkl"))

        # Process new files
        for chunk_file in chunk_files:
            if chunk_file.name in processed_files:
                continue

            try:
                # Load chunk
                with open(chunk_file, 'rb') as f:
                    emg_tuple = pickle.load(f)

                emg_data, time_axis, metadata = emg_tuple
                chunks_processed += 1

                # Display chunk info
                print("\n" + "-" * 70)
                print(f"📥 PROCESSING: {chunk_file.name}")
                print("-" * 70)
                print(f"  Shape: {emg_data.shape}")
                print(f"  Time: {metadata['time_start']:.3f}s - {metadata['time_end']:.3f}s")
                print(f"  Chunk: {metadata['chunk_index']+1}/{metadata['total_chunks']}")

                # Process chunk
                start_time = time.time()
                emg_dat_corr = emg_data, time_axis
                predictions = predictor.run(emg_dat_corr)
                process_time = time.time() - start_time

                # Store predictions
                all_predictions.append(predictions)

                # Display results
                print(f"\n✓ Prediction completed in {process_time:.3f}s")
                print(f"  Predictions shape: {predictions.shape}")
                print(f"  Summary:")
                print(f"    Elbow:          min={predictions[:, 0].min():>7.3f}, "
                      f"max={predictions[:, 0].max():>7.3f}, "
                      f"mean={predictions[:, 0].mean():>7.3f}")
                print(f"    Shoulder Front: min={predictions[:, 1].min():>7.3f}, "
                      f"max={predictions[:, 1].max():>7.3f}, "
                      f"mean={predictions[:, 1].mean():>7.3f}")
                print(f"    Shoulder Side:  min={predictions[:, 2].min():>7.3f}, "
                      f"max={predictions[:, 2].max():>7.3f}, "
                      f"mean={predictions[:, 2].mean():>7.3f}")
                print("-" * 70)

                # Mark as processed
                processed_files.add(chunk_file.name)

            except Exception as e:
                print(f"\n❌ Error processing {chunk_file.name}: {e}")
                import traceback
                traceback.print_exc()

        # Check if done
        done_file = input_path / "DONE.txt"
        if done_file.exists():
            # Wait a bit to make sure all files are processed
            time.sleep(1)

            # Check if all files processed
            final_chunk_files = sorted(input_path.glob("chunk_*.pkl"))
            if all(f.name in processed_files for f in final_chunk_files):
                print("\n" + "=" * 70)
                print("✓ DONE marker found and all chunks processed")
                print("=" * 70)
                break

        # Wait before checking again
        time.sleep(0.5)

    # Final summary
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Total chunks processed: {chunks_processed}")

    if all_predictions:
        final_predictions = np.vstack(all_predictions)
        print(f"Total predictions: {final_predictions.shape}")

        # Save predictions
        output_file = "predictions_output.csv"
        np.savetxt(
            output_file,
            final_predictions,
            delimiter=',',
            header='Elbow,Shoulder_Front,Shoulder_Side',
            comments=''
        )
        print(f"\n✓ Predictions saved to: {output_file}")

    print("=" * 70 + "\n")


if __name__ == "__main__":

    input_folder = "./emg_chunks"

    process_emg_from_folder(
        input_folder=input_folder,
        config_filename='pipeline_mlp_to_cnn.yaml'
    )

    print("Consumer finished.")
