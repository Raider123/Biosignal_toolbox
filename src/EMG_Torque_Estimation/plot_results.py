import numpy as np
import matplotlib.pyplot as plt
from biosignal_toolbox.utils import loadConfig, getAbsolutePath, plotResults, setPltParams


config_filename = 'emg_torque_estimation_jte_oneHotEncoding.yaml'
cfg = loadConfig(filename=config_filename)

plots_path = getAbsolutePath(cfg.filepath.fig_save_path)
plots_path = plots_path / "BU62D/hri_all_weights_contEnc"
filename = "250928_155659_grasp_complex_0g_1100g_1850gpred_results.npz"

results = np.load(plots_path / filename)
pred_results = results['pred_results']
Y_ref = results['ref_target']

joint_names = ['Elbow', 'Shoulder_Front', 'Shoulder Side']
time_axis = np.arange(0, len(Y_ref[:,0]), 1)*0.05
#? time: from 18 sec to 28 sec
idx = (time_axis <=28) & (time_axis >=18)
time_axis_new = time_axis[idx]

setPltParams(style="ticks")
for j in range(len(joint_names)):
    # Mean and std across seeds
    mean_pred = pred_results[:, idx, j].mean(axis=0)
    std_pred = pred_results[:, idx, j].std(axis=0)

    plotResults(time_axis=time_axis_new, data_ref=Y_ref[idx,j], data_pred=mean_pred, label_ref='Ref. Torque', label_pred='Predicted Torque', is_band_plot=True, mean_inp=mean_pred, std_inp=std_pred)

plt.show()

