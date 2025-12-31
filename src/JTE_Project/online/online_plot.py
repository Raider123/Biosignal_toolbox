import numpy as np
from biosignal_toolbox.utils import getAbsolutePath
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from scipy.signal import medfilt, savgol_filter
from scipy.signal import butter, sosfilt
from scipy import signal

all_preds = np.load(str(getAbsolutePath(f"src/JTE_Project/online/online_results/all_predictions.npy")))
all_times = np.load(str(getAbsolutePath(f"src/JTE_Project/online/online_results/all_times.npy")))
all_torques = np.load(str(getAbsolutePath(f"src/JTE_Project/online/online_results/all_torques.npy")))

###################################################

# Load Torque Values (ground truth, only in prediction plot)
Y_e = np.load(str(getAbsolutePath("src/JTE_Project/online/resources/reference_torques/e.npy")))
Y_f = np.load(str(getAbsolutePath("src/JTE_Project/online/resources/reference_torques/front.npy")))
Y_s = np.load(str(getAbsolutePath("src/JTE_Project/online/resources/reference_torques/side.npy")))
Y_ref_raw = np.stack((Y_e, Y_f, Y_s), axis=1)

def downsample_mean_bins(Y_ref, target_len=1000):
    n = Y_ref.shape[0]
    edges = np.linspace(0, n, target_len + 1, dtype=int)
    Y_ds = np.vstack([Y_ref[edges[i]:edges[i + 1]].mean(axis=0) for i in range(target_len)])
    return Y_ds

def process_predictions(data, ds_factor=5, med_kernel=5, sav_win=15, sav_poly=2):
    """
    1. Wendet Median-Filter an (gegen Spikes).
    2. Wendet Savitzky-Golay-Filter an (zum Glätten).
    3. Reduziert die Auflösung um ds_factor (durch Mittelwertbildung).
    """

    # Array für gefilterte Daten vorbereiten
    filtered_data = np.zeros_like(data)

    # Filterung pro Spalte (Gelenk) durchführen, damit keine Vermischung passiert
    for i in range(data.shape[1]):
        # 1. Median Filter (kernel_size muss ungerade sein)
        # Entfernt kurze, harte Spikes
        med = medfilt(data[:, i], kernel_size=med_kernel)

        # 2. Savitzky-Golay Filter (window_length muss ungerade sein)
        # Glättet das Signal schön organisch
        sav = savgol_filter(med, window_length=sav_win, polyorder=sav_poly)

        filtered_data[:, i] = sav

    # 3. Downsampling (Verkleinerung)
    # Wir schneiden ggf. den Rest ab, falls die Länge nicht durch 5 teilbar ist
    n_samples = filtered_data.shape[0]
    trim_length = (n_samples // ds_factor) * ds_factor
    trimmed_data = filtered_data[:trim_length]

    # Reshape zu (834, 5, 3) und Mittelwert über die 5er-Blöcke bilden
    # Das ergibt (834, 3)
    downsampled_data = trimmed_data.reshape(-1, ds_factor, data.shape[1]).mean(axis=1)

    return downsampled_data

def process_torque_lowpass_causal(data, f_cutoff, f_samp, order=4, ds_factor=5):
    """
    Wendet einen kausalen (Forward-Only) Butterworth-Lowpass-Filter an.
    Dies entspricht einer Echtzeit-Filterung (mit Phasenverzug).
    Anschließend wird ein Downsampling durchgeführt.

    Args:
        data: Numpy Array (N_samples, 3) -> [Elbow, Front, Side]
        f_cutoff: Grenzfrequenz in Hz
        f_samp: Abtastrate in Hz
        order: Filterordnung
        ds_factor: Downsampling-Faktor

    Returns:
        downsampled_data: Kausal gefiltertes und verkleinertes Array
    """
    # 1. Filter Design (Butterworth Lowpass)
    sos = butter(N=order, Wn=f_cutoff, btype='low', fs=f_samp, output='sos')

    # 2. Filtern (Kausal / Forward only)
    # Das Signal wird leicht nach rechts verschoben sein (Phasenverzug).
    filtered_data = sosfilt(sos, data, axis=0)

    # 3. Downsampling (Block-Average)
    n_samples = filtered_data.shape[0]

    # Abschneiden, falls Länge nicht durch ds_factor teilbar
    trim_length = (n_samples // ds_factor) * ds_factor
    trimmed_data = filtered_data[:trim_length]

    # Reshape und Mittelwert bilden
    # (N_neu, 5, 3) -> Mittelwert über Achse 1 -> (N_neu, 3)
    downsampled_data = trimmed_data.reshape(-1, ds_factor, data.shape[1]).mean(axis=1)

    return downsampled_data

Y_ref = process_torque_lowpass_causal(data=Y_ref_raw, f_cutoff=5, f_samp=500, order=4,ds_factor=1)
y_ref_length = int(all_preds.shape[0])
Y_ref= downsample_mean_bins(Y_ref, y_ref_length)

#all_preds = process_predictions(all_preds, ds_factor=5, sav_poly=9)
#all_preds = downsample_mean_bins(all_preds, y_ref_length)
all_preds = process_predictions(all_preds, ds_factor=1)
all_torques = Y_ref

#################################################################################

def align_signals_auto(predictions, references):
    """
    Berechnet die zeitliche Verschiebung zwischen Vorhersage und Referenz
    mittels Kreuzkorrelation und richtet die Vorhersage neu aus.
    """
    aligned_preds = np.zeros_like(predictions)
    detected_lags = []

    # Für jedes Gelenk (Spalte) einzeln berechnen
    for i in range(predictions.shape[1]):
        pred_sig = np.nan_to_num(predictions[:, i])  # NaNs sicherheitshalber entfernen
        ref_sig = np.nan_to_num(references[:, i])

        # Kreuzkorrelation berechnen
        correlation = signal.correlate(ref_sig, pred_sig, mode='full')
        lags = signal.correlation_lags(ref_sig.size, pred_sig.size, mode='full')

        # Finde den Lag mit der höchsten Korrelation
        optimal_lag = lags[np.argmax(correlation)]
        detected_lags.append(optimal_lag)

        # Signal verschieben (Shifting)
        if optimal_lag > 0:
            # Vorhersage muss nach rechts geschoben werden (war zu früh)
            aligned_preds[optimal_lag:, i] = pred_sig[:-optimal_lag]
        elif optimal_lag < 0:
            # Vorhersage muss nach links geschoben werden (war zu spät / verzögert)
            aligned_preds[:optimal_lag, i] = pred_sig[-optimal_lag:]
        else:
            aligned_preds[:, i] = pred_sig

    return aligned_preds, detected_lags


# --- ANWENDUNG DER FUNKTION ---

print("--- Aligning Signals ---")
# Originale Predictions sichern (falls man vergleichen will), hier überschreiben wir sie direkt:
all_preds_aligned, lags = align_signals_auto(all_preds, all_torques)

print(f"Detected Lags (Samples) per Joint [Elbow, Front, Side]: {lags}")
print("Positive Lag: Prediction wird nach rechts geschoben.")
print("Negative Lag: Prediction wird nach links geschoben (Verzögerung korrigiert).")

# Update der Variablen für die Plots
#all_preds = all_preds_aligned

#################################################################################

# Clip Values that are NAN to 0
all_preds[np.isnan(all_preds)] = 0

print(all_preds.shape)
print(all_torques.shape)

y_e_test_combined = all_torques[:,0]
y_f_test_combined = all_torques[:,1]
y_s_test_combined = all_torques[:,2]

predictions_e = all_preds[:,0]
predictions_f = all_preds[:,1]
predictions_s = all_preds[:,2]

def calculate_metrics(y_true, y_pred):
    """Berechnet RMSE, R² und Pearson-Korrelationskoeffizient."""
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    r2 = r2_score(y_true, y_pred)
    pearson_corr, _ = pearsonr(y_true, y_pred)
    return rmse, r2, pearson_corr

def plotResults(y_true, label_true, y_pred, label_pred, title,
                ylabel="Torque", is_grid_on=True):
    # Metriken berechnen
    rmse, r2, pearson_corr = calculate_metrics(y_true, y_pred)

    plt.figure(figsize=(10, 4))
    plt.plot(y_true, label=label_true, color="orange")
    plt.plot(y_pred, label=label_pred, color="blue", linestyle="-")
    plt.title(f"{title}\nRMSE={rmse:.3f}, R²={r2:.3f}, Pearson={pearson_corr:.3f}")
    plt.xlabel("Samples")
    plt.ylabel(ylabel)
    if is_grid_on:
        plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()


plotResults(y_e_test_combined, "real torque",
            predictions_e, "predicted torque",
            f"Elbow Joint Filtered; RMSE",
            ylabel="Torque in N-m",
            is_grid_on=True)

plotResults(y_f_test_combined, "real torque",
            predictions_f, "predicted torque",
            f"Shoulder Front Joint Filtered; RMSE",
            ylabel="Torque in N-m",
            is_grid_on=True)

plotResults(y_s_test_combined, "real torque",
            predictions_s, "predicted torque",
            f"Shoulder Side Joint Filtered",
            ylabel="Torque in N-m",
            is_grid_on=True)

plt.show()