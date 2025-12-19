# * Dieses Skript ist für die reine Inferenz (Vorhersage) auf einem einzelnen Set von Dateien gedacht.
# * Es lädt ein trainiertes TCN-Modell und MVC Werte, um Gelenkmomente aus sEMG zu schätzen.

# ! ************************************************
# ! Imports
# ! ************************************************

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib
import os
import warnings
from copy import deepcopy
from scipy.signal import medfilt, savgol_filter

# Eigene Libraries (Pfade müssen im PYTHONPATH sein oder relativ stimmen)
from biosignal_toolbox.eeg_lib import EEGData
from biosignal_toolbox.emg_lib import EMGData
from biosignal_toolbox.ML_lib import MLModel
from biosignal_toolbox.utils import customWarningFormat, loadConfig, getAbsolutePath

warnings.formatwarning = customWarningFormat

# ! ************************************************
# ! User Settings & File Inputs
# ! ************************************************

# Lade Konfiguration (für Sampling-Raten, Filter-Parameter etc.)
config_filename = 'pipeline_jte_bu62d.yaml'
cfg = loadConfig(filename=config_filename)

# PFADE ZU DEN GESPEICHERTEN MODELLEN (Müssen vom Training existieren)
# model_dir = "saved_models/" # Alternativ relativer Pfad
path_to_model = getAbsolutePath('src/JTE_Project/online/resources/trained_models/tcn_model.keras')
path_to_mvc = str(getAbsolutePath("src/JTE_Project/online/resources/mvc/channelwise_mvc.npy"))  # Gespeicherte MVC Werte

# DEFINITION DER EINGABEDATEIEN (Hier die Dateien für die Inferenz eintragen)
# Beispielpfade - bitte durch echte Dateipfade ersetzen
input_emg_file = str(getAbsolutePath("src/JTE_Project/online/resources/publisher/24072025_BU62D_1100g_complex_2.txt"))
input_torque_e = str(getAbsolutePath("src/JTE_Project/online/resources/reference_torques/e.npy"))
input_torque_sf = str(getAbsolutePath("src/JTE_Project/online/resources/reference_torques/front.npy"))
input_torque_ss = str(getAbsolutePath("src/JTE_Project/online/resources/reference_torques/side.npy"))

print(f"Lade Config: {config_filename}")
print(f"Inferenz auf Datei: {input_emg_file}")

# ! ************************************************
# ! 1. Daten Laden
# ! ************************************************

# EMG Daten laden
# Hinweis: Wir übergeben eine Liste mit einem File, da die Lib Listen erwartet
EMG_Data = EMGData(format="ANTmini",
                   filenames=[input_emg_file],
                   data_path="",  # Leer lassen, wenn voller Pfad oben angegeben
                   f_samp=cfg.preprocess_param.f_samp,
                   channel_names=cfg.preprocess_param.channel_names_emg)

# Torque Daten laden (Ground Truth für Vergleich)
Quali_Data_Elbow = EEGData(format="NumpyQualisys", filenames=[input_torque_e],
                           data_path="", f_samp=cfg.preprocess_param.f_samp,
                           channel_names=cfg.preprocess_param.channel_names_quali, add_marker_channel=True)

Quali_Data_Front = EEGData(format="NumpyQualisys", filenames=[input_torque_sf],
                           data_path="", f_samp=cfg.preprocess_param.f_samp,
                           channel_names=cfg.preprocess_param.channel_names_quali, add_marker_channel=True)

Quali_Data_Side = EEGData(format="NumpyQualisys", filenames=[input_torque_ss],
                          data_path="", f_samp=cfg.preprocess_param.f_samp,
                          channel_names=cfg.preprocess_param.channel_names_quali, add_marker_channel=True)

# ! ************************************************
# ! 2. Pre-Processing (Identisch zum Training)
# ! ************************************************

print("Starte Pre-Processing...")

# 2.1 Bandpass Filter
sos_hp = EMG_Data.designFilter(f_high=cfg.preprocess_param.f_cutoff_hpf,
                               f_low=cfg.preprocess_param.f_cutoff_lpf,
                               order=cfg.preprocess_param.filter_order,
                               filter_type="scipy_butter", return_type="sos")
EMG_Data.filterData_offline(filter_method=cfg.preprocess_param.filter_method, sos=sos_hp)

# 2.2 Variance Filter
width = cfg.preprocess_param.var_filter_width
ring_buffer = np.zeros(width)
EMG_Data.applyVarianceFilter_data(ring_buffer=ring_buffer, width=width, index=0)

# 2.3 Input Normalisation (MVC)
# Wir versuchen, die MVC-Werte vom Training zu laden, um konsistent zu bleiben.
if os.path.exists(path_to_mvc):
    print("Lade gespeicherte MVC-Werte...")
    loaded_mvc = np.load(path_to_mvc)
    # Reshape falls nötig (manchmal (8,) manchmal (8,1))
    if loaded_mvc.ndim == 1:
        loaded_mvc = loaded_mvc.reshape(-1, 1)

    if cfg.preprocess_param.normalisation_method == 'overall_mvc':
        EMG_Data.normalizeContinuousData(mvc=np.max(loaded_mvc))
    elif cfg.preprocess_param.normalisation_method == 'channel_wise_mvc':
        EMG_Data.normalizeContinuousData(mvc=loaded_mvc)
else:
    print("WARNUNG: Keine gespeicherten MVC-Werte gefunden. Nutze Max aus aktuellem File (kann ungenau sein).")
    channelwise_mvc = np.max(np.abs(EMG_Data.data), axis=1).reshape(-1, 1)
    EMG_Data.normalizeContinuousData(mvc=channelwise_mvc)

# 2.4 Low Pass Filter (Smoothing)
sos_lp = EMG_Data.designFilter(f_low=cfg.preprocess_param.f_cutoff_sm_lpf,
                               order=cfg.preprocess_param.sm_filter_order,
                               filter_type="scipy_butter", return_type="sos")

EMG_Data.filterData_offline(filter_method=cfg.preprocess_param.filter_method, sos=sos_lp)

# Filter auch auf Torques anwenden (für sauberen Vergleich)
Quali_Data_Elbow.filterData_offline(filter_method=cfg.preprocess_param.filter_method, sos=sos_lp)
Quali_Data_Front.filterData_offline(filter_method=cfg.preprocess_param.filter_method, sos=sos_lp)
Quali_Data_Side.filterData_offline(filter_method=cfg.preprocess_param.filter_method, sos=sos_lp)

# 2.5 Activation Function (optional)
if cfg.preprocess_param.use_activation_fncn:
    EMG_Data.calculateActivationForceFunction(d=cfg.preprocess_param.act_delay,
                                              b1=cfg.preprocess_param.act_beta1,
                                              b2=cfg.preprocess_param.act_beta2,
                                              g=cfg.preprocess_param.act_gamma,
                                              nonlinear_shape_factor=cfg.preprocess_param.act_A)

# ! ************************************************
# ! 3. Windowing & Feature Setup
# ! ************************************************

print("Windowing Data...")
# Windowing EMG
emg_indices, _ = EMG_Data.windowContinuousData(startmarkernumber=1, stopmarkernumber=2,
                                               window_size=cfg.preprocess_param.window_size_x,
                                               window_step=cfg.preprocess_param.window_step,
                                               start_index_offset=0, start_channel_pick=0, end_channel_pick=8)

# Windowing Torques
Quali_Data_Elbow.windowContinuousData(startmarkernumber=1, stopmarkernumber=2,
                                      window_size=cfg.preprocess_param.window_size_y,
                                      window_step=cfg.preprocess_param.window_step,
                                      start_index_offset=0, start_channel_pick=0, end_channel_pick=1)
Quali_Data_Front.windowContinuousData(startmarkernumber=1, stopmarkernumber=2,
                                      window_size=cfg.preprocess_param.window_size_y,
                                      window_step=cfg.preprocess_param.window_step,
                                      start_index_offset=0, start_channel_pick=0, end_channel_pick=1)
Quali_Data_Side.windowContinuousData(startmarkernumber=1, stopmarkernumber=2,
                                     window_size=cfg.preprocess_param.window_size_y,
                                     window_step=cfg.preprocess_param.window_step,
                                     start_index_offset=0, start_channel_pick=0, end_channel_pick=1)

# Längenabgleich der Fenster
min_windows = min(EMG_Data.getWindows().shape[3], Quali_Data_Elbow.getWindows().shape[3])
EMG_Data.windows = EMG_Data.windows[..., :min_windows]
Quali_Data_Elbow.windows = Quali_Data_Elbow.windows[..., :min_windows]
Quali_Data_Front.windows = Quali_Data_Front.windows[..., :min_windows]
Quali_Data_Side.windows = Quali_Data_Side.windows[..., :min_windows]

# ! ************************************************
# ! 4. Feature Extraction (Raw Timepoints for TCN)
# ! ************************************************

# Für TCN nutzen wir meist Raw Timepoints als "Features"
window_size_ms = cfg.preprocess_param.window_size_x * 1000 / EMG_Data.f_samp
feature_indices_windows_x = np.array([0, window_size_ms])
EMG_Data.featureExtractionFromWindows(feature_type="timepoints", feature_indices_windows=feature_indices_windows_x)

X_infer = EMG_Data.getFeatures()  # Shape: (N_windows, Features)

# Target Features extrahieren (Ground Truth für Plot)
window_size_ms_y = cfg.preprocess_param.window_size_y * 1000 / Quali_Data_Elbow.f_samp
# Logik analog zum Training (Mean, Mid oder End)
feature_indices_windows_y = np.array([0, window_size_ms_y])
use_mean_bool = True if cfg.preprocess_param.target_feature_select == 'mean' else False

Quali_Data_Elbow.featureExtractionFromWindows(feature_type="timepoints",
                                              feature_indices_windows=feature_indices_windows_y, use_mean=use_mean_bool)
Quali_Data_Front.featureExtractionFromWindows(feature_type="timepoints",
                                              feature_indices_windows=feature_indices_windows_y, use_mean=use_mean_bool)
Quali_Data_Side.featureExtractionFromWindows(feature_type="timepoints",
                                             feature_indices_windows=feature_indices_windows_y, use_mean=use_mean_bool)

Y_ground_truth = np.concatenate(
    [Quali_Data_Elbow.getFeatures(), Quali_Data_Front.getFeatures(), Quali_Data_Side.getFeatures()], axis=1)

# ! ************************************************
# ! 5.Reshaping
# ! ************************************************

# Reshape für TCN: (Batch, TimeSteps, Channels)
neurons_inp = X_infer.shape[1]
n_channels = 8
n_timpoints = int(neurons_inp / n_channels)
X_infer_cnn = X_infer.reshape((-1, n_timpoints, n_channels))

print(f"Input Shape für Modell: {X_infer_cnn.shape}")

# ! ************************************************
# ! 6. Inferenz (Model Prediction)
# ! ************************************************

print(f"Lade Modell von: {path_to_model}")
if not os.path.exists(path_to_model):
    raise FileNotFoundError("Modell nicht gefunden! Bitte Pfad prüfen.")

tcn_model = tf.keras.models.load_model(path_to_model, compile=False)

print("Starte Vorhersage...")
preds = tcn_model.predict(X_infer_cnn)
# preds ist Liste: [elbow_out, front_out, side_out]
preds_combined = np.concatenate([preds[0], preds[1], preds[2]], axis=1)

# ! ************************************************
# ! 7. Inverse Scaling & Post-Processing
# ! ************************************************

final_predictions = preds_combined
final_ground_truth = Y_ground_truth


# Post-Filtering (Median / SavGol)
if cfg.post_train_param.filter_type == 'median':
    for i in range(3):
        final_predictions[:, i] = medfilt(final_predictions[:, i], kernel_size=cfg.post_train_param.filter_size)

if getattr(cfg.post_train_param, 'savgol_window_len', None) is not None:
    for i in range(3):
        final_predictions[:, i] = savgol_filter(final_predictions[:, i],
                                                cfg.post_train_param.savgol_window_len,
                                                cfg.post_train_param.savgol_poly_order)


# ! ************************************************
# ! 8. Evaluation & Plotting
# ! ************************************************

def calculate_metrics_infer(y_true, y_pred):
    from sklearn.metrics import r2_score
    from scipy.stats import pearsonr
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2 = r2_score(y_true, y_pred)
    pearson_corr, _ = pearsonr(y_true.flatten(), y_pred.flatten())
    return rmse, r2, pearson_corr


def plot_inference(y_true, y_pred, title):
    rmse, r2, rho = calculate_metrics_infer(y_true, y_pred)
    plt.figure(figsize=(10, 4))
    plt.plot(y_true, label="Ground Truth (Qualisys)", color="orange", alpha=0.8)
    plt.plot(y_pred, label="TCN Prediction", color="blue", linewidth=1.5)
    plt.title(f"{title}\nRMSE={rmse:.3f}, R²={r2:.3f}, Pearson={rho:.3f}")
    plt.xlabel("Windows / Time")
    plt.ylabel("Torque (Nm)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


print("################ RESULTS ################")

# Elbow
plot_inference(final_ground_truth[:, 0], final_predictions[:, 0], title="Elbow Torque Inference")

# Shoulder Front
plot_inference(final_ground_truth[:, 1], final_predictions[:, 1], title="Shoulder Front Torque Inference")

# Shoulder Side
plot_inference(final_ground_truth[:, 2], final_predictions[:, 2], title="Shoulder Side Torque Inference")

print("Inferenz abgeschlossen.")