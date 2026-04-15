# JTE_Project

## Joint Torque Estimation from Surface EMG Signals

This repository contains the complete codebase for a master's thesis project on **real-time joint torque estimation (JTE)** of the upper limb from surface electromyography (sEMG) signals. The system estimates torques for three degrees of freedom — **elbow flexion/extension**, **shoulder frontal flexion**, and **shoulder lateral abduction** — using deep learning models (MLP and TCN) in both offline and pseudo-online settings.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Directory Structure](#directory-structure)
3. [Module Descriptions](#module-descriptions)
   - [offline/](#offline)
   - [online/](#online)
   - [Qualisys_Torque/](#qualisys_torque)
4. [Data Flow & Pipeline](#data-flow--pipeline)
5. [EMG Preprocessing Chain](#emg-preprocessing-chain)
6. [Model Architectures](#model-architectures)
7. [Dependencies](#dependencies)
8. [Usage](#usage)
9. [Configuration](#configuration)
10. [Key Design Decisions](#key-design-decisions)

---

## Project Overview

The project implements a complete pipeline from raw biosignal acquisition to torque prediction:

1. **Motion capture processing** — Raw Qualisys marker data is converted into joint torques via inverse dynamics, serving as the ground-truth labels.
2. **Offline training** — EMG signals are preprocessed (bandpass filtering, variance filtering, MVC normalisation, low-pass smoothing, neural activation force modelling) and paired with the computed torques to train MLP and TCN regression models.
3. **Pseudo-online inference** — A ZeroMQ-based publisher replays recorded EMG data sample-by-sample, and a subscriber processes the stream in real-time–like batches, performs feature extraction, runs model inference (with XLA-compiled TensorFlow graphs), and applies post-processing filters.
4. **Evaluation & visualisation** — Predicted torques are compared against ground-truth references using RMSE, R², and Pearson correlation coefficient (PCC).

Experiments are parameterised by **subject ID**, **held weight** (0 g, 1100 g, 1850 g), **movement type** (grasp, complex), and **set number**.

---

## Directory Structure

```
JTE_Project/
│
├── offline/                         # Offline model training and evaluation
│   ├── saved_offline_models/        # Serialised Keras models and artefacts from training
│   ├── jte_mlp_pipeline.py          # MLP offline pipeline (channel-wise MVC per condition)
│   ├── pipeline_mlp_offline.py      # MLP offline pipeline (global MVC across all conditions)
│   ├── pipeline_tcn_offline.py      # TCN offline pipeline (dual-input architecture)
│   └── plot_pred_mean.py            # Multi-seed prediction averaging and visualisation
│
├── online/                          # Pseudo-online (real-time) inference system
│   ├── online_results/              # Saved predictions, torques, and timing logs (.npy)
│   ├── resources/                   # Runtime resources for inference
│   │   ├── mvc/                     #   Channel-wise MVC normalisation arrays
│   │   ├── trained_models/          #   Deployed Keras models for online use
│   │   └── y_scaler/               #   Fitted scikit-learn MinMaxScaler objects (.pkl)
│   ├── emg_publisher.py             # ZeroMQ EMG data publisher (replay script)
│   ├── online_plot.py               # Post-hoc plotting and evaluation of online results
│   ├── pipeline_mlp_online.py       # Pseudo-online inference pipeline (MLP variant)
│   └── pipeline_tcn_online.py       # Pseudo-online inference pipeline (TCN variant)
│
└── Qualisys_Torque/                 # Inverse dynamics / torque computation from motion capture
    ├── output_torque_processing.py  # Torque processing, windowing, and feature extraction
    ├── plot_torques.py              # Batch torque visualisation and export to PNG
    ├── subject_parameter_calc.py    # Segment length and COM calculation from mocap markers
    └── torque_calculation.py        # Batch torque computation via inverse dynamics (COM method)
```

---

## Module Descriptions

### `offline/`

Contains the full **offline training pipelines** for both model architectures. All scripts load hyperparameters and file paths from a YAML configuration file.

| File | Description |
|---|---|
| **`jte_mlp_pipeline.py`** | MLP offline pipeline that computes **channel-wise MVC normalisation independently for each weight–movement condition**. For each condition pair the script: loads EMG and torque data, applies the full preprocessing chain (bandpass filter → variance filter → per-condition MVC normalisation → low-pass smoothing → neural activation force), extracts time-domain features (RMS, waveform length, slope sign change), frequency-domain features (multitaper PSD band powers), time-frequency features (Morlet wavelet coefficients), and differential (peak-detection) features. Features from all conditions are concatenated with one-hot–encoded weight/movement vectors, output torques are scaled per condition via `MinMaxScaler`, and a multi-output MLP (`AAN_Model`) is trained with weighted Huber loss. Post-prediction median and Savitzky-Golay filters are applied before evaluation. Per-condition Y-scalers are saved for later online use. |
| **`pipeline_mlp_offline.py`** | MLP offline pipeline that computes **a single global MVC across all weight–movement conditions simultaneously**. All EMG recordings are loaded and concatenated before MVC computation, preventing per-condition data leakage. Uses the same preprocessing, feature extraction, PCA dimensionality reduction, StandardScaler normalisation, and training procedure as `jte_mlp_pipeline.py`, but with a unified normalisation reference. Includes optional PCA and StandardScaler steps on input features. |
| **`pipeline_tcn_offline.py`** | **TCN offline pipeline** with a dual-input architecture. The TCN branch receives raw windowed EMG segments of shape `(batch, timesteps, 8 channels)` processed through dilated causal convolutions with residual connections. A second static input branch receives concatenated peak-detection features (temporal derivatives) and one-hot–encoded condition vectors. The pipeline is split into three phases: (1) loading and linear filtering (bandpass, variance), (2) global MVC calculation on the training split only, and (3) normalisation, smoothing, activation force modelling, windowing, and feature extraction. The TCN model uses multi-task output heads (one per joint) with weighted Huber or MSE loss. Supports multiple random seeds for statistical evaluation and saves predictions per seed. |
| **`plot_pred_mean.py`** | Loads saved prediction results (`.npy`) from multiple random seeds, computes per-joint mean and standard deviation of RMSE, R², and Pearson correlation, and generates publication-ready comparison plots of predicted vs. ground-truth torques with confidence bands across seeds. |
| `saved_offline_models/` | Directory storing trained Keras model files (`.keras`), channel-wise MVC arrays, and Y-scaler objects produced during offline experiments. |

### `online/`

Implements the **pseudo-online (simulated real-time) inference system** using a ZeroMQ publisher–subscriber architecture.

| File | Description |
|---|---|
| **`emg_publisher.py`** | **EMG replay node.** Reads a recorded 8-channel EMG text file and publishes each sample row over a ZeroMQ PUB socket (`tcp://127.0.0.1:5555`) at approximately the original sampling rate (~500 Hz, one sample every ~2 ms). Configurable for subject ID, weight, movement type, and set number. Used to simulate a live EMG device for the subscriber pipelines. Supports both continuous looping and single-pass modes. |
| **`pipeline_mlp_online.py`** | **MLP pseudo-online inference pipeline.** Subscribes to the EMG ZeroMQ stream and accumulates samples into batches of 50 (~100 ms at 500 Hz). For each batch the pipeline: updates the EMG ring buffer (250 samples), applies bandpass filtering, maintains a rolling variance filter buffer, applies variance filtering, MVC normalisation, low-pass smoothing, neural activation force modelling, converts the buffer to analysis windows, extracts the full feature set (timepoints, RMS, waveform length, slope sign change, frequency band powers, Morlet wavelet coefficients, differential features), concatenates a one-hot condition vector, and runs inference through an XLA-compiled (`jit_compile=True`) TensorFlow graph with a warm-up pass. A causal Savitzky-Golay post-processing filter smooths predictions in real time. Optionally renders a live 3-panel prediction plot. Saves all predictions, ground-truth torques, and elapsed times as `.npy` files. Includes a 30-second socket timeout for graceful shutdown. |
| **`pipeline_tcn_online.py`** | **TCN pseudo-online inference pipeline.** Similar architecture to the MLP pipeline but uses the **dual-input TCN model**: (1) raw windowed EMG segments of shape `(batch, 50, 8)` fed to the TCN branch, and (2) a static feature vector composed of peak-detection (differential) features and one-hot condition encoding. Uses `extract_sliding_features_pd()` to create multiple overlapping windows from the ring buffer with associated difference features for higher temporal resolution. Employs a pre-compiled concrete TensorFlow function (`get_concrete_function()`) with explicit `TensorSpec` input signatures and XLA JIT compilation for maximum CPU inference speed. Supports both per-condition Y-scaler inverse transform and raw prediction modes. |
| **`online_plot.py`** | **Post-hoc evaluation script.** Loads saved prediction and torque `.npy` files from `online_results/`, applies a causal (forward-only) Butterworth low-pass filter to the reference torques for fair comparison, downsamples via mean binning, applies median and Savitzky-Golay smoothing to predictions, compensates for processing lag via configurable sample shifting (movement-type dependent), crops edge artefacts, and produces per-joint comparison plots annotated with RMSE, R², and Pearson correlation metrics. |
| `online_results/` | Storage for per-subject, per-model output files organised as `online_results/<model_type>/<subject_id>/`: `all_predictions_<weight>_<move>_<set>.npy`, `all_torques_<weight>_<move>_<set>.npy`, `all_times_<weight>_<move>_<set>.npy`. |
| `resources/` | Runtime artefacts required for inference: pre-computed channel-wise MVC arrays (`channelwise_mvc.npy`), deployed Keras model files (`.keras`), and fitted `MinMaxScaler` objects saved as nested dictionaries via joblib (`Y_scaler.pkl`). |

### `Qualisys_Torque/`

Handles the computation of **ground-truth joint torques** from Qualisys motion capture data via inverse dynamics, as well as their post-processing for use in the training and online pipelines.

| File | Description |
|---|---|
| **`torque_calculation.py`** | **Batch torque computation script.** Iterates over all Qualisys `.tsv` files in a subject's data directory, parses filenames to extract subject ID, weight, movement type, and set number, then calls `MotionData.calculateTorque()` using the centre-of-mass (COM) method. Requires subject-specific parameters (body weight, biological sex, hand length) from a configuration dictionary. Computed torques for elbow, shoulder front, and shoulder side are saved as `.npy` files organised by subject and weight (e.g., `results/<subject>/<weight>/quali_torque_<joint>_..._.npy`). |
| **`subject_parameter_calc.py`** | **Segment parameter calculation from motion capture markers.** Loads a single Qualisys `.tsv` file, identifies shoulder, elbow, and wrist marker positions, and computes upper arm length, forearm length, and hand length via Euclidean distances averaged over time. Derives centre-of-mass (COM) positions for each segment (upper arm, forearm, hand) as percentages of segment length using anthropometric tables (Winter/Dempster) parameterised by biological sex. Outputs a formatted table of segment lengths and COM distances for use in the torque calculation. Subject parameters (body weight, sex, hand length) are specified in a configuration block at the top of the script. |
| **`output_torque_processing.py`** | **Torque post-processing and feature extraction for the training pipeline.** Provides the `process_joint_torques()` function that mirrors the torque processing steps from the offline training scripts: loads three pre-computed torque `.npy` files (elbow, shoulder front, shoulder side) as `EEGData` objects, applies low-pass Butterworth smoothing, windows the continuous data with configurable window size and step, extracts timepoint features (mean, midpoint, or endpoint per window as configured), and returns a merged target feature array of shape `(n_windows, 3)`. Also includes standalone utility code for loading EMG file metadata and MVC arrays. |
| **`plot_torques.py`** | **Batch torque visualisation and export.** Iterates over all Qualisys files in a subject directory, loads the corresponding pre-computed torque `.npy` files for each joint, and generates time-series plots using the motion capture time axis. Plots are saved as `.png` files organised by subject and weight in a `plots/` directory. Each plot shows the torque curve with grid lines, labelled axes, and a descriptive title including subject, weight, movement type, joint name, and set number. |

---

## Data Flow & Pipeline

```
┌─────────────────────┐     ┌──────────────────────┐
│  Qualisys Motion    │     │  EMG Recording        │
│  Capture (.tsv)     │     │  (8 ch sEMG, 500 Hz) │
└────────┬────────────┘     └────────┬──────────────┘
         │                           │
         ▼                           ▼
┌─────────────────────┐     ┌──────────────────────────────┐
│ Qualisys_Torque/    │     │ EMG Preprocessing:            │
│ - Subject params    │     │ 1. Bandpass filter (20-450Hz) │
│ - Inverse dynamics  │     │ 2. Variance filter            │
│ - Torque labels     │     │ 3. MVC normalisation          │
│   (3 joints × N)    │     │ 4. Low-pass smoothing         │
└────────┬────────────┘     │ 5. Neural activation force    │
         │                  │ 6. Windowing + features       │
         │                  └────────┬─────────────────────┘
         │                           │
         ▼                           ▼
    ┌────────────────────────────────────────────┐
    │         Offline Training                   │
    │  jte_mlp_pipeline.py      (MLP, per-cond)  │
    │  pipeline_mlp_offline.py  (MLP, global)    │
    │  pipeline_tcn_offline.py  (TCN, dual-in)   │
    │                                            │
    │  Outputs:                                  │
    │  → Trained .keras model                    │
    │  → Channel-wise MVC (.npy)                 │
    │  → Fitted Y-scaler (.pkl)                  │
    │  → Prediction results per seed (.npy)      │
    └──────────────────┬─────────────────────────┘
                       │
                       ▼
    ┌────────────────────────────────────────────┐
    │       Pseudo-Online Inference              │
    │                                            │
    │  emg_publisher.py ───ZMQ (PUB/SUB)───►     │
    │       pipeline_mlp_online.py               │
    │       pipeline_tcn_online.py               │
    │                                            │
    │  Outputs:                                  │
    │  → Per-timestep predictions (.npy)         │
    │  → Processing timing logs (.npy)           │
    └──────────────────┬─────────────────────────┘
                       │
                       ▼
    ┌────────────────────────────────────────────┐
    │       Evaluation & Plotting                │
    │                                            │
    │  online_plot.py      (online results)      │
    │  plot_pred_mean.py   (offline multi-seed)  │
    │                                            │
    │  Metrics: RMSE, R², Pearson r              │
    └────────────────────────────────────────────┘
```

---

## EMG Preprocessing Chain

Both offline and online pipelines apply the following signal processing steps in order:

| Step | Method | Purpose |
|---|---|---|
| 1. Bandpass filter | 4th-order Butterworth (20–450 Hz) | Remove DC offset, motion artefacts, and high-frequency noise |
| 2. Variance filter | Sliding window variance (width=20) | Envelope extraction / rectification |
| 3. MVC normalisation | Channel-wise maximum voluntary contraction | Scale each channel to [0, 1] relative to maximum activation |
| 4. Low-pass smoothing | Butterworth low-pass filter | Smooth the activation envelope |
| 5. Neural activation force | Recursive discrete model with nonlinear shaping | Transform EMG envelope into physiologically plausible muscle activation |
| 6. Windowing | Sliding windows (50 samples, configurable step) | Segment continuous data for feature extraction / model input |

---

## Model Architectures

### MLP (Multi-Layer Perceptron)

- **Input:** Flattened feature vector (time-domain, frequency-domain, time-frequency, and differential features) concatenated with one-hot–encoded weight and movement condition vectors.
- **Architecture:** `AAN_Model` — configurable hidden layers with activation functions specified via YAML config.
- **Output:** 3 neurons (elbow, shoulder front, shoulder side torques).
- **Loss:** Weighted Huber loss with per-joint weights derived from variance, smooth variance, or manual specification.

### TCN (Temporal Convolutional Network)

- **Input 1 (temporal):** Raw windowed EMG segments of shape `(batch, timesteps, 8)`.
- **Input 2 (static):** Concatenation of peak-detection features (temporal derivatives of windowed EMG) and one-hot condition vectors.
- **Architecture:** Stacked residual blocks with dilated causal 1D convolutions (exponentially increasing dilation: 1, 2, 4, 8, ...), ReLU activations, LayerNorm, SpatialDropout1D, and residual skip connections. Last-timestep readout followed by dense layers.
- **Output:** Multi-task — three separate output heads (one `Dense(1)` per joint).
- **Loss:** Per-head Huber or MSE loss with configurable per-joint loss weights.

---

## Dependencies

| Package | Purpose |
|---|---|
| `tensorflow` (≥ 2.x) | Deep learning model training and XLA-accelerated inference |
| `numpy` | Numerical array operations |
| `scipy` | Signal processing (Butterworth filters, Savitzky-Golay, median filter) |
| `pandas` | EMG data file loading |
| `scikit-learn` | Scaling (`MinMaxScaler`, `StandardScaler`), PCA, metrics (`r2_score`), one-hot encoding |
| `matplotlib` | Plotting and visualisation |
| `joblib` | Serialisation of scaler objects |
| `pyzmq` | ZeroMQ messaging for publisher–subscriber communication |
| `biosignal_toolbox` | Custom/internal library for EMG/EEG preprocessing, feature extraction, online streaming (`OnlineEMG`, `EMGData`, `EEGData`), motion capture processing (`MotionData`), ML model wrapper (`MLModel`), neural network architectures (`AAN_Model`), and utilities (`loadConfig`, `getAbsolutePath`, `plotResults`) |

---

## Usage

### 1. Compute Ground-Truth Torques

First, determine the anthropometric segment parameters for your subject by editing the configuration block in `subject_parameter_calc.py` (body weight, sex, hand length) and running it on a representative trial.

```bash 
python Qualisys_Torque/subject_parameter_calc.py
python Qualisys_Torque/torque_calculation.py
python Qualisys_Torque/plot_torques.py
```

### 2. Train Models Offline

```bash
# MLP with per-condition MVC normalisation
python offline/jte_mlp_pipeline.py

# MLP with global MVC normalisation
python offline/pipeline_mlp_offline.py

# TCN with dual-input architecture
python offline/pipeline_tcn_offline.py
```

Trained models are saved to `offline/saved_offline_models/`. For online use, copy the relevant `.keras` model, `channelwise_mvc.npy`, and `Y_scaler.pkl` to the corresponding subdirectories in `online/resources/`.

### 3. Evaluate Offline Results Across Seeds

```bash
python offline/plot_pred_mean.py
```

### 4. Run Pseudo-Online Inference

Open **two terminals**:

**Terminal 1 — Start the EMG publisher:**
```bash
python online/emg_publisher.py
```

**Terminal 2 — Start the inference pipeline:**
```bash
# For MLP:
python online/pipeline_mlp_online.py

# For TCN:
python online/pipeline_tcn_online.py
```

Results (predictions, reference torques, timing logs) are saved automatically to `online/online_results/<model_type>/<subject_id>/`.

### 5. Evaluate Online Results

```bash
python online/online_plot.py
```

This produces per-joint comparison plots with RMSE, R², and Pearson correlation metrics, including lag compensation and post-processing filter application.

---

## Configuration

Both offline and online pipelines are parameterised via YAML configuration files loaded through `biosignal_toolbox.utils.loadConfig`. The following parameters must be configured:

### Experiment Parameters
- **`subject_id`** / **`subject_code`** — Subject identifier (e.g., `'BU62D'`)
- **`weights`** — List of held weight conditions (e.g., `['0g', '1100g', '1850g']`)
- **`mov_type`** — List of movement types (e.g., `['grasp', 'complex']`)
- **`set_num`** — List of repetition/set numbers (e.g., `['1', '2', '3']`)

### Preprocessing Parameters
- `f_samp` — Sampling frequency (500 Hz)
- `f_cutoff_hpf`, `f_cutoff_lpf` — Bandpass filter cutoff frequencies
- `f_cutoff_sm_lpf` — Low-pass smoothing filter cutoff
- `var_filter_width` — Variance filter window width
- `normalisation_method` — `'channel_wise_mvc'` or `'overall_mvc'`
- `use_activation_fncn` — Enable/disable neural activation force modelling
- `act_delay`, `act_beta1`, `act_beta2`, `act_gamma`, `act_A` — Activation function coefficients
- `window_size_x`, `window_size_y`, `window_step` — Windowing parameters

### Model Parameters
- `neurons_h1`–`h4`, `act_h1`–`h4` — MLP hidden layer sizes and activations
- `n_epochs`, `batch_size`, `learning_rate` — Training hyperparameters
- `loss_fcn` — Loss function (`'huber'`, `'mse'`)
- `huber_weight_method` — Per-joint loss weighting strategy (`'var'`, `'smooth_var'`, `'manual'`, `'dynamic_huber'`)
- `train_test_split`, `validation_split` — Data split ratios
- `is_early_stop`, `patience`, `monitor` — Early stopping configuration

### Online-Specific Parameters
- `buffer_size` — EMG ring buffer length (default: 250 samples)
- `batch_size` — Samples per processing batch (default: 50 samples)
- `show_prediction_plot` — Enable/disable live visualisation
- `use_yscaler` — Enable/disable Y-scaler inverse transform

---

## Key Design Decisions

- **ZeroMQ PUB/SUB** decouples data acquisition from processing, enabling modular testing and future integration with live EMG hardware.
- **XLA-compiled inference** (`tf.function(jit_compile=True)`) and concrete function tracing with explicit `TensorSpec` signatures eliminate Python overhead during prediction, achieving sub-30 ms loop times on CPU.
- **Causal filtering only** in the online pipeline ensures no future information leakage, maintaining real-time validity.
- **One-hot condition encoding** as an auxiliary model input allows a single model to generalise across multiple weight and movement conditions.
- **Per-condition Y-scaling** (in `jte_mlp_pipeline.py`) handles the varying torque ranges across different load conditions, while global MVC (in `pipeline_mlp_offline.py` and `pipeline_tcn_offline.py`) prevents normalisation data leakage.
- **Multi-phase data processing** in the TCN pipeline (load → global MVC → normalise) explicitly prevents MVC computation from leaking test-set statistics.
- **Post-processing filters** (Savitzky-Golay + median filtering) smooth prediction jitter while preserving temporal dynamics, applied causally in the online setting.
- **Dual-input TCN architecture** separates temporal pattern learning (causal convolutions on raw EMG) from static contextual information (condition encoding and differential features), improving generalisation.
```