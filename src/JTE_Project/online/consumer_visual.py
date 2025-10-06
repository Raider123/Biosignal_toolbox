"""
EMG Consumer with Real-time Visualization
------------------------------------------
Monitors folder for EMG chunks and displays predictions in real-time.
Shows live plots for Elbow, Shoulder Front, and Shoulder Side torques.
"""

from process_emg import OnlineEMGPredictor
import numpy as np
import time
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading


class EMGVisualConsumer:
    """
    Consumer with real-time visualization of predictions.
    """

    def __init__(self, input_folder, config_filename='pipeline_mlp_to_cnn.yaml'):
        """
        Initialize consumer with visualization.

        Parameters
        ----------
        input_folder : str
            Folder to monitor for chunk files
        config_filename : str
            Configuration file for predictor
        """
        self.input_folder = input_folder
        self.input_path = Path(input_folder)
        self.input_path.mkdir(exist_ok=True)

        # Initialize predictor
        print("=" * 70)
        print("CONSUMER - Initializing with Real-time Visualization")
        print("=" * 70 + "\n")

        self.predictor = OnlineEMGPredictor(config_filename=config_filename)

        # Data storage
        self.chunks_processed = 0
        self.processed_files = set()
        self.all_predictions = []
        self.time_points = np.array([])  # Changed to numpy array

        # For visualization
        self.fig = None
        self.axes = None
        self.lines = []

        # Threading control
        self.running = True
        self.processing_thread = None

    def setup_plot(self):
        """Setup the real-time plot with 3 subplots."""
        print("Setting up visualization...\n")

        # Create figure with 3 subplots
        self.fig, self.axes = plt.subplots(3, 1, figsize=(12, 8))
        self.fig.suptitle('Real-time Joint Torque Predictions', fontsize=14, fontweight='bold')

        # Setup each subplot
        joint_names = ['Elbow', 'Shoulder Front', 'Shoulder Side']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        for i, (ax, name, color) in enumerate(zip(self.axes, joint_names, colors)):
            ax.set_ylabel(f'{name} Torque', fontsize=10)
            ax.set_xlabel('Time (s)', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 10)  # Initial window
            ax.set_ylim(-5, 5)  # Will auto-adjust

            # Create line object
            line, = ax.plot([], [], color=color, linewidth=1.5, label=name)
            self.lines.append(line)
            ax.legend(loc='upper right')

        plt.tight_layout()

    def update_plot(self):
        """Update the plot with current predictions."""
        if not self.all_predictions:
            return

        # Stack all predictions
        all_preds = np.vstack(self.all_predictions)
        n_total = all_preds.shape[0]

        # Create/update time axis
        if len(self.time_points) == 0:
            # First time: create initial time points
            self.time_points = np.arange(n_total) * 0.05  # 50ms per prediction
        else:
            # Extend time points if we have new predictions
            n_existing = len(self.time_points)
            if n_total > n_existing:
                last_time = self.time_points[-1]
                n_new = n_total - n_existing
                new_times = last_time + np.arange(1, n_new + 1) * 0.05
                self.time_points = np.concatenate([self.time_points, new_times])

        time_axis = self.time_points[:n_total]

        # Update each line
        for i, line in enumerate(self.lines):
            line.set_data(time_axis, all_preds[:, i])

            # Auto-adjust y-limits with some margin
            y_data = all_preds[:, i]
            if len(y_data) > 0:
                y_min, y_max = y_data.min(), y_data.max()
                y_range = y_max - y_min
                y_margin = max(y_range * 0.1, 0.5)  # At least 0.5 margin
                self.axes[i].set_ylim(y_min - y_margin, y_max + y_margin)

            # Auto-adjust x-limits to show last 10 seconds
            if len(time_axis) > 0:
                if time_axis[-1] > 10:
                    self.axes[i].set_xlim(time_axis[-1] - 10, time_axis[-1])
                else:
                    self.axes[i].set_xlim(0, max(10, time_axis[-1] + 1))

        # Refresh canvas
        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except:
            pass  # Ignore errors if window was closed

    def process_chunks(self):
        """Process chunks in background thread."""
        print("=" * 70)
        print("CONSUMER - Monitoring folder and processing chunks...")
        print("=" * 70 + "\n")

        while self.running:
            # Get all chunk files
            chunk_files = sorted(self.input_path.glob("chunk_*.pkl"))

            # Process new files
            for chunk_file in chunk_files:
                if not self.running:
                    break

                if chunk_file.name in self.processed_files:
                    continue

                try:
                    # Load chunk
                    with open(chunk_file, 'rb') as f:
                        emg_tuple = pickle.load(f)

                    emg_data, time_axis_emg, metadata = emg_tuple
                    self.chunks_processed += 1

                    # Process chunk
                    print(f"📥 Processing chunk {self.chunks_processed}: {chunk_file.name} "
                          f"({metadata['chunk_index']+1}/{metadata['total_chunks']})")

                    start_time = time.time()
                    emg_obj = emg_data, time_axis_emg
                    predictions = self.predictor.run(emg_obj)
                    process_time = time.time() - start_time

                    # Store predictions
                    self.all_predictions.append(predictions)

                    print(f"   ✓ Predicted {predictions.shape[0]} windows in {process_time:.3f}s\n")

                    # Mark as processed
                    self.processed_files.add(chunk_file.name)

                except Exception as e:
                    print(f"❌ Error processing {chunk_file.name}: {e}\n")
                    import traceback
                    traceback.print_exc()

            # Check if done
            done_file = self.input_path / "DONE.txt"
            if done_file.exists():
                # Wait a bit for remaining files
                time.sleep(1)

                # Check if all files processed
                final_chunk_files = sorted(self.input_path.glob("chunk_*.pkl"))
                if all(f.name in self.processed_files for f in final_chunk_files):
                    print("\n" + "=" * 70)
                    print("✓ All chunks processed!")
                    print("=" * 70 + "\n")
                    self.running = False
                    break

            # Wait before checking again
            time.sleep(0.3)

    def run(self):
        """Run consumer with visualization."""
        # Setup plot
        self.setup_plot()

        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_chunks, daemon=True)
        self.processing_thread.start()

        # Enable interactive mode
        plt.ion()
        plt.show()

        print("🎨 Visualization started!")
        print("💡 Close the plot window to stop\n")

        # Update loop
        try:
            while self.running:
                self.update_plot()
                plt.pause(0.1)  # Update every 100ms

                # Check if window was closed
                if not plt.fignum_exists(self.fig.number):
                    print("\n🛑 Plot window closed")
                    self.running = False
                    break

        except KeyboardInterrupt:
            print("\n\n🛑 Interrupted by user")
            self.running = False

        finally:
            # Wait for processing thread
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=2.0)

            # Keep plot open for a moment
            if plt.fignum_exists(self.fig.number):
                print("\n💡 Plot will stay open. Close it to exit completely.")
                plt.ioff()
                plt.show()

            # Save final results
            self.save_results()

            print("\n✓ Consumer finished\n")

    def save_results(self):
        """Save final predictions to file."""
        if not self.all_predictions:
            return

        final_predictions = np.vstack(self.all_predictions)

        output_file = "predictions_output.csv"
        np.savetxt(
            output_file,
            final_predictions,
            delimiter=',',
            header='Elbow,Shoulder_Front,Shoulder_Side',
            comments=''
        )

        print(f"\n💾 Predictions saved to: {output_file}")
        print(f"   Total predictions: {final_predictions.shape[0]}")
        print(f"   Chunks processed: {self.chunks_processed}")


# ! ************************************************
# ! Main Execution
# ! ************************************************

if __name__ == "__main__":

    input_folder = "./emg_chunks"

    # Create and run visual consumer
    consumer = EMGVisualConsumer(
        input_folder=input_folder,
        config_filename='pipeline_mlp_to_cnn.yaml'
    )

    consumer.run()
