"""
EMG Consumer with Real-time Visualization
------------------------------------------
Liest EMG-Daten direkt aus einer TXT-Datei,
verarbeitet sie in überlappenden Fenstern (Sliding Window)
und zeigt Echtzeit-Vorhersagen für Gelenkmomente.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path
from process_emg import OnlineEMGPredictor


class EMGFileStreamVisualConsumer:
    """
    EMG Consumer, der direkt eine EMG.TXT-Datei verarbeitet
    und Echtzeit-Vorhersagen mit Live-Visualisierung anzeigt.
    """

    def __init__(self, emg_file, config_filename='pipeline_mlp_to_cnn.yaml', f_samp=500):
        """
        Parameters
        ----------
        emg_file : str
            Pfad zur EMG.TXT-Datei
        config_filename : str
            Konfigurationsdatei für den Prädiktor
        f_samp : int
            Abtastrate in Hz
        """
        self.emg_file = Path(emg_file)
        self.f_samp = f_samp

        print("=" * 70)
        print("CONSUMER - Initializing (File Stream + Visualization)")
        print("=" * 70 + "\n")

        # Predictor laden
        self.predictor = OnlineEMGPredictor(config_filename=config_filename)

        # Datenstrukturen für Verlauf
        self.all_predictions = []

        # Visualisierungsobjekte
        self.fig = None
        self.axes = None
        self.lines = []

        self.elapsed_times = []  # speichert die Zeitachse
        self.current_time = 0.0  # Start bei 0 s

    # ---------------------------------------------------------------------
    # Daten laden und vorbereiten
    # ---------------------------------------------------------------------
    def load_emg_file(self):
        """Lädt EMG-Daten aus einer TXT-Datei."""
        print(f"📂 Lade EMG-Datei: {self.emg_file}")
        emg_data_raw = np.loadtxt(self.emg_file)
        emg_data = emg_data_raw[:-1, :-2].T  # Kanäle x Samples
        n_samples = emg_data.shape[1]
        time_axis = np.arange(0, n_samples / self.f_samp, step=1 / self.f_samp)
        print(f"   → Kanäle: {emg_data.shape[0]}, Samples: {n_samples}\n")
        return emg_data, time_axis

    # ---------------------------------------------------------------------
    # Plot Setup
    # ---------------------------------------------------------------------
    def setup_plot(self):
        """Erstellt die Live-Visualisierung mit 3 Subplots."""
        print("🎨 Setup der Visualisierung...\n")

        self.fig, self.axes = plt.subplots(3, 1, figsize=(12, 8))
        self.fig.suptitle('Joint-Torque-Estimation (Realtime)', fontsize=14, fontweight='bold')

        joint_names = ['Elbow', 'Shoulder Front', 'Shoulder Side']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        for i, (ax, name, color) in enumerate(zip(self.axes, joint_names, colors)):
            ax.set_ylabel(f'{name} Torque', fontsize=10)
            ax.set_xlabel('Zeit (s)', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 10)
            ax.set_ylim(-5, 5)
            line, = ax.plot([], [], color=color, linewidth=1.5, label=name)
            self.lines.append(line)
            ax.legend(loc='upper right')

        plt.tight_layout()

    # ---------------------------------------------------------------------
    # Plot aktualisieren
    # ---------------------------------------------------------------------
    def update_plot(self, elapsed_times):
        """Aktualisiert den Plot basierend auf der verstrichenen Zeit."""
        # Validierung der Eingabedaten
        if not self.all_predictions or not elapsed_times:
            return

        try:
            all_preds = np.vstack(self.all_predictions)
        except ValueError as e:
            print(f"Fehler beim Stacken der Predictions: {e}")
            return

        # WICHTIG: Zeitstempel für jede Prediction replizieren
        predictions_per_timestamp = all_preds.shape[0] // len(elapsed_times)

        if predictions_per_timestamp > 1:
            # Jeden Zeitstempel entsprechend oft wiederholen
            time_plot = np.repeat(elapsed_times, predictions_per_timestamp)
        else:
            time_plot = np.array(elapsed_times)

        # Längenprüfung nach Replikation
        if len(time_plot) != all_preds.shape[0]:
            print(f"Warnung: Längen stimmen nicht überein - Times: {len(time_plot)}, Predictions: {all_preds.shape[0]}")
            # Fallback: Kürze auf die kürzere Länge
            min_len = min(len(time_plot), all_preds.shape[0])
            time_plot = time_plot[:min_len]
            all_preds = all_preds[:min_len]

        # Sicherstellen, dass Daten vorhanden sind
        if all_preds.shape[0] == 0 or time_plot.shape[0] == 0:
            return

        print("ELAPSED TIMES:", elapsed_times)
        print(f"Time plot shape: {time_plot.shape}, Predictions shape: {all_preds.shape}")

        for i, line in enumerate(self.lines):
            # Daten für die aktuelle Linie extrahieren
            y_data = all_preds[:, i]

            # WICHTIG: set_data OHNE relim() aufrufen
            line.set_data(time_plot, y_data)

            # Dynamische Y-Achsen-Anpassung
            if y_data.size > 0:
                y_min, y_max = y_data.min(), y_data.max()
                y_range = y_max - y_min

                # Verhindert Division durch Null bei konstanten Werten
                if y_range > 1e-6:
                    y_margin = y_range * 0.1
                else:
                    y_margin = 0.5

                self.axes[i].set_ylim(y_min - y_margin, y_max + y_margin)

            # X-Achse: letzte 10 Sekunden anzeigen
            if len(time_plot) > 0:
                if time_plot[-1] > 10:
                    self.axes[i].set_xlim(time_plot[-1] - 10, time_plot[-1])
                else:
                    self.axes[i].set_xlim(0, max(10, time_plot[-1] + 1))

        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception as e:
            print(f"Fehler beim Aktualisieren der Canvas: {e}")

    # ---------------------------------------------------------------------
    # Sliding-Window Verarbeitung
    # ---------------------------------------------------------------------
    def process_file_in_sliding_windows(self, window_size=500, step_size=20):
        """Verarbeitet EMG-Daten in überlappenden Fenstern und aktualisiert Live-Plot."""
        self.step_size = step_size

        emg_data, time_axis = self.load_emg_file()
        n_channels, n_samples = emg_data.shape
        n_windows = (n_samples - window_size) // step_size + 1

        print("=" * 70)
        print(f"Starte Sliding-Window-Verarbeitung:")
        print(f"   Fenstergröße = {window_size}, Schrittweite = {step_size}")
        print(f"   Gesamtfenster: {n_windows}")
        print("=" * 70 + "\n")

        self.setup_plot()
        plt.ion()
        plt.show()

        for i in range(n_windows):
            start_idx = i * step_size
            end_idx = start_idx + window_size
            emg_chunk = emg_data[:, start_idx:end_idx]
            time_chunk = time_axis[start_idx:end_idx]

            emg_obj = (emg_chunk, time_chunk)

            time_step = step_size / self.f_samp
            self.current_time += time_step
            self.elapsed_times.append(self.current_time)

            try:
                preds = self.predictor.run(emg_obj)
                self.all_predictions.append(preds)
            except Exception as e:
                print(f"❌ Fehler bei Fenster {i}: {e}")
                continue

            if i % 5 == 0 or i == n_windows - 1:
                self.update_plot(elapsed_times=self.elapsed_times)
                plt.pause(0.05)

        print("\n✓ Alle Fenster verarbeitet.")
        #self.save_results()

        plt.ioff()
        plt.show()

    # ---------------------------------------------------------------------
    # Ergebnisse speichern
    # ---------------------------------------------------------------------
    def save_results(self):
        """Speichert alle vorhergesagten Momente in eine CSV-Datei."""
        if not self.all_predictions:
            print("⚠️ Keine Vorhersagen zum Speichern.")
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

        print(f"\n💾 Ergebnisse gespeichert unter: {output_file}")
        print(f"   Gesamtanzahl Fenster: {final_predictions.shape[0]}")
        print("=" * 70 + "\n")


# ---------------------------------------------------------------------
# MAIN AUSFÜHRUNG
# ---------------------------------------------------------------------
if __name__ == "__main__":
    emg_file = "F:/SMT_MASTERPROJEKT/biosignal_toolbox/data/jte/emg/BU62D/24072025_BU62D_0g_complex_1.txt"

    consumer = EMGFileStreamVisualConsumer(
        emg_file=emg_file,
        config_filename='pipeline_mlp_to_cnn.yaml',
        f_samp=500
    )

    consumer.process_file_in_sliding_windows(window_size=500, step_size=20)
