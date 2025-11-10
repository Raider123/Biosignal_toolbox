import numpy as np
from biosignal_toolbox.utils import getAbsolutePath
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

directory = 'tcn_zerophase_no_standard'

all_preds = np.load(getAbsolutePath(f"src/JTE_Project/online/online_results/all_predictions.npy"))
all_times = np.load(getAbsolutePath(f"src/JTE_Project/online/online_results/all_times.npy"))
all_torques = np.load(getAbsolutePath(f"src/JTE_Project/online/online_results/all_torques.npy"))

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