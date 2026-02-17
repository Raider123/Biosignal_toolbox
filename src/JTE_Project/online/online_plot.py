import numpy as np
from biosignal_toolbox.utils import getAbsolutePath
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from scipy.signal import medfilt, savgol_filter
from scipy.signal import butter, sosfilt
from scipy import signal

model_type = 'mlp'
subject_id = 'BU62D'
current_weight = '1850g'
current_move = 'grasp'
set_num = '3'

all_preds = np.load(str(getAbsolutePath(f"src/JTE_Project/online/online_results/{model_type}/{subject_id}/all_predictions_{current_weight}_{current_move}_{set_num}.npy")))
Y_ref_raw = np.load(str(getAbsolutePath(f"src/JTE_Project/online/online_results/{model_type}/{subject_id}/all_torques_{current_weight}_{current_move}_{set_num}.npy")))

# Clip Values that are NAN to 0
all_preds[np.isnan(all_preds)] = 0
y_ref_length = int(all_preds.shape[0])

################################################### Torque Preprocessing and Prediction Postprocessing etc.

def downsample_mean_bins(Y_ref, target_len=1000):
    n = Y_ref.shape[0]
    edges = np.linspace(0, n, target_len + 1, dtype=int)
    Y_ds = np.vstack([Y_ref[edges[i]:edges[i + 1]].mean(axis=0) for i in range(target_len)])
    return Y_ds

def process_predictions(data, ds_factor=1, med_kernel=5, sav_win=15, sav_poly=2):
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

def process_torque_lowpass_causal(data, f_cutoff, f_samp, order=4, ds_factor=1):
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
Y_ref= downsample_mean_bins(Y_ref, y_ref_length)
all_torques = Y_ref

all_preds = process_predictions(all_preds, ds_factor=1, med_kernel=9, sav_win= 15, sav_poly=3)

################################################################ Lag Reduction and Cropping

def crop_edges(arr, n_start, d_end):
    """
    Schneidet Zeilen vom Anfang und Ende eines Arrays ab.

    Parameters:
    - arr: Das Input-Array
    - n_start: Anzahl der Zeilen, die am Anfang weggeschnitten werden
    - d_end: Anzahl der Zeilen, die am Ende weggeschnitten werden

    Returns:
    - Das beschnittene Array
    """
    # Wir berechnen den End-Index explizit basierend auf der Länge.
    # Das ist sicherer als negatives Slicing (arr[n:-d]), falls d_end mal 0 sein sollte.
    end_index = arr.shape[0] - d_end

    # Slicing syntax: [start_index : end_index]
    return arr[n_start:end_index]

def shift_samples(arr, n, fill_value=0.0):
    """
    Verschiebt den Inhalt des Arrays um n Zeilen.

    Parameters:
    - arr: Input Array (400, 3)
    - n: Anzahl Samples (positiv = nach unten/später, negativ = nach oben/früher)
    - fill_value: Womit die Lücke gefüllt wird (0.0 oder np.nan)
    """
    result = np.empty_like(arr)

    if n > 0:
        # Verschieben nach unten (Daten am Anfang werden leer)
        result[:n] = fill_value  # Lücke füllen
        result[n:] = arr[:-n]  # Daten verschieben
    elif n < 0:
        # Verschieben nach oben (Daten am Ende werden leer)
        result[n:] = fill_value  # Lücke füllen
        result[:n] = arr[-n:]  # Daten verschieben
    else:
        return arr

    return result

#all_torques = shift_samples(all_torques, -18, 0) # -11/-10 bei 1100g complex 2
if current_move == 'complex':
    all_torques = shift_samples(all_torques, -5, 0)
else:
    all_torques = shift_samples(all_torques, -4, 0)
all_preds = crop_edges(all_preds, 10, 50)
all_torques = crop_edges(all_torques, 10, 50)

print(all_preds.shape)
print(all_torques.shape)
#################################################################################

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