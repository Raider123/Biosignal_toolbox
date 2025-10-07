import numpy as np
from biosignal_toolbox.utils import loadConfig, getAbsolutePath


config_filename = 'emg_torque_estimation_jte_oneHotEncoding.yaml'
cfg = loadConfig(filename=config_filename)

plots_path = getAbsolutePath(cfg.filepath.fig_save_path)
plots_path = plots_path / "BU62D/hri_all_weights_contEnc"
filename = "250928_155659_grasp_complex_0g_1100g_1850geval_metrics.npz"

results = np.load(plots_path / filename)
rmse_elbow = results["rmse_elbow_post"]
rmse_front = results["rmse_front_post"]
rmse_side = results["rmse_side_post"]

r2_elbow = results["r2_elbow_post"]
r2_front = results["r2_front_post"]
r2_side = results["r2_side_post"]

rho_elbow = results["rho_elbow_post"]
rho_front = results["rho_front_post"]
rho_side = results["rho_side_post"]

print(f"Elbow RMSE stats: Mean: {np.mean(rmse_elbow):.4f}  Std. : {np.std(rmse_elbow):.4f}")
print(f"Front RMSE stats: Mean: {np.mean(rmse_front):.4f}  Std. : {np.std(rmse_front):.4f}")
print(f"Side RMSE stats: Mean: {np.mean(rmse_side):.4f}  Std. : {np.std(rmse_side):.4f}\n")

print(f"Elbow R2 stats: Mean: {np.mean(r2_elbow):.4f}  Std. : {np.std(r2_elbow):.4f}")
print(f"Front R2 stats: Mean: {np.mean(r2_front):.4f}  Std. : {np.std(r2_front):.4f}")
print(f"Side R2 stats: Mean: {np.mean(r2_side):.4f}  Std. : {np.std(r2_side):.4f}\n")

print(f"Elbow rho stats: Mean: {np.mean(rho_elbow):.4f}  Std. : {np.std(rho_elbow):.4f}")
print(f"Front rho stats: Mean: {np.mean(rho_front):.4f}  Std. : {np.std(rho_front):.4f}")
print(f"Side rho stats: Mean: {np.mean(rho_side):.4f}  Std. : {np.std(rho_side):.4f}\n")
