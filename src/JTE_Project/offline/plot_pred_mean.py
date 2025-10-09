import numpy as np
import matplotlib.pyplot as plt
from biosignal_toolbox.utils import getAbsolutePath, plotResults
from biosignal_toolbox.ML_lib import MLModel

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
    "lines.linewidth": 2,
    "lines.markersize": 3,
    "figure.figsize": (6.6, 4.0),  # column-width of figure(in inches)
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

params = ["all_no_features_test"]
data_paths = [f"results/jte/BU62D/predictions/{param}" for param in params]
ref_data_path = [f"{data_path}/ref_data.npy" for data_path in data_paths]
seed_arr = [1,7,25,45,70]

ref = np.load(ref_data_path[0])
ref_e = ref[:,0]
ref_sf = ref[:,1]
ref_ss = ref[:,2]

predictions_e = []
predictions_sf = []
predictions_ss = []

r2_e_arr = []
r2_sf_arr = []
r2_ss_arr = []

rmse_e_arr = []
rmse_sf_arr = []
rmse_ss_arr = []

rho_e_arr = []
rho_sf_arr = []
rho_ss_arr = []

save_dir = getAbsolutePath(f"plots/BU62D/predictions")

plots_e = []
plots_sf = []
plots_ss = []

for data_path in data_paths:
    predictions_e = []
    predictions_sf = []
    predictions_ss = []

    for seed in seed_arr:
        numpy_file_path = getAbsolutePath(f"{data_path}/pred_results_seed{seed}.npy")
        data = np.load(numpy_file_path)

        print(data[:,0].shape)

        predictions_e.append(data[:,0])
        predictions_sf.append(data[:,1])
        predictions_ss.append(data[:,2])


        r2_e, rmse_e, rho_e = MLModel.calculateEvalMetrics(ref_e, data[:,0],
                                                                                is_Pearson=True)
        r2_sf, rmse_sf, rho_sf = MLModel.calculateEvalMetrics(ref_sf, data[:,1],
                                                                                is_Pearson=True)
        r2_ss, rmse_ss, rho_ss = MLModel.calculateEvalMetrics(ref_ss, data[:,2],
                                                                                is_Pearson=True)
        
        r2_e_arr.append(r2_e)
        rmse_e_arr.append(rmse_e)
        rho_e_arr.append(rho_e)

        r2_sf_arr.append(r2_sf)
        rmse_sf_arr.append(rmse_sf)
        rho_sf_arr.append(rho_sf)

        r2_ss_arr.append(r2_ss)
        rmse_ss_arr.append(rmse_ss)
        rho_ss_arr.append(rho_ss)
       
    plots_e.append(predictions_e)
    plots_sf.append(predictions_sf)
    plots_ss.append(predictions_ss)

plotResults(ref_e, "Ref. Torque",
            plots_e, "Predicted Torque",
            f"Elbow Joint Filtered\nRMSE: {np.mean(rmse_e_arr):.4f} ± {np.std(rmse_e_arr):.4f} N-m | R²: {np.mean(r2_e_arr):.4f} ± {np.std(r2_e_arr):.4f} | PCC: {np.mean(rho_e_arr):.4f} ± {np.std(rho_e_arr):.4f}",
            ylabel="Joint Torque (N m)",
            is_grid_on=False,
            is_list=True,
            is_multiple=True,
            plot_len=-1, start_time=0)
fullpath = save_dir / f"elbow_{params}.png"
fullpath.parent.mkdir(parents=True, exist_ok=True)
#plt.savefig(fullpath)

plotResults(ref_sf, "Ref. Torque",
            plots_sf, "Predicted Torque",
            f"Shoulder Front Joint Filtered\nRMSE: {np.mean(rmse_sf_arr):.4f} ± {np.std(rmse_sf_arr):.4f} N-m | R²: {np.mean(r2_sf_arr):.4f} ± {np.std(r2_sf_arr):.4f} | PCC: {np.mean(rho_sf_arr):.4f} ± {np.std(rho_sf_arr):.4f}",
            ylabel="Joint Torque (N m)",
            is_grid_on=False,
            is_list=True,
            is_multiple=True,
            plot_len=-1, start_time=0)
fullpath = save_dir / f"front_shoulder_{params}.png"
fullpath.parent.mkdir(parents=True, exist_ok=True)
#plt.savefig(fullpath)

plotResults(ref_ss, "Ref. Torque",
            plots_ss, "Predicted Torque",
            f"Shoulder Side Joint Filtered\nRMSE: {np.mean(rmse_ss_arr):.4f} ± {np.std(rmse_ss_arr):.4f} N-m | R²: {np.mean(r2_ss_arr):.4f} ± {np.std(r2_ss_arr):.4f} | PCC: {np.mean(rho_ss_arr):.4f} ± {np.std(rho_ss_arr):.4f}",
            ylabel="Joint Torque (N m)",
            is_grid_on=False,
            is_list=True,
            is_multiple=True,
            plot_len=-1, start_time=0)
fullpath = save_dir / f"side_shoulder_{params}.png"
fullpath.parent.mkdir(parents=True, exist_ok=True)
#plt.savefig(fullpath)
plt.show()
