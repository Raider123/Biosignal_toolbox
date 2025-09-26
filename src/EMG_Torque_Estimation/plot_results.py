import numpy as np
import matplotlib.pyplot as plt

# plt.style.use("seaborn-v0_8-white")
plt.style.use("seaborn-v0_8-ticks")

# Global font and figure settings
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,          
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "axes.labelweight": "bold",
    "axes.labelsize": 12,
    "axes.linewidth": 2.5,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "lines.linewidth": 1.5,
    "lines.markersize": 3,
    "figure.figsize": (6.6, 4.0),  # column-width of figure(in inches)
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

from biosignal_toolbox.utils import customWarningFormat, loadConfig, getAbsolutePath, plotResults, createOutputDir, createReadme


config_filename = 'emg_torque_estimation_jte_oneHotEncoding.yaml'
cfg = loadConfig(filename=config_filename)


plots_path = getAbsolutePath(cfg.filepath.fig_save_path)
plots_path = plots_path / "BU62D/250926_155316_plot1"
filename = "250926_155002_grasp_complex_0gpred_results.npz"

results = np.load(plots_path / filename)
pred_results = results['pred_results']
Y_ref = results['ref_target']

joint_names = ['Elbow', 'Shoulder_Front', 'Shoulder Side']
time_axis = np.arange(0, len(Y_ref[:,0]), 1)*0.05
#? time: first 10 sec
idx_5s = time_axis <=5
time_axis_5s = time_axis[idx_5s]

for j in range(len(joint_names)):
    # Mean and std across seeds
    mean_pred = pred_results[:, idx_5s, j].mean(axis=0)
    std_pred = pred_results[:, idx_5s, j].std(axis=0)

    plt.figure()
    
    plt.plot(time_axis_5s, Y_ref[idx_5s, j], label='Ref. torque', color='red')
    plt.plot(time_axis_5s, mean_pred, label='Predicted Torque', color='blue')
    plt.fill_between(time_axis_5s, mean_pred - std_pred, mean_pred + std_pred, color='blue', alpha=0.3)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Joint Torque (N m)')
    plt.legend(frameon=True)

    plt.tight_layout()

plt.show()

