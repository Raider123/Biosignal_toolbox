import numpy as np
import matplotlib.pyplot as plt
from biosignal_toolbox.utils import getAbsolutePath, plotResults
from biosignal_toolbox.ML_lib import MLModel
from pathlib import Path

def parseFilename(self):
    stem = Path(self.filename).stem
    return stem.split("_")


data_path = "results/jte/BU62D/predictions/test"
ref_data_path = "results/jte/BU62D/predictions/test/ref_data.npy"
seed_arr = [1,7,25,45,70]

ref = np.load(ref_data_path)
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


for seed in seed_arr:
    numpy_file_path = getAbsolutePath(f"{data_path}/pred_results_seed{seed}.npy")
    data = np.load(numpy_file_path)

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

plotResults(ref_e, "real torque",
            predictions_e, "predicted torque",
            f"Elbow Joint Filtered; RMSE: {np.mean(rmse_e_arr):.2f} ± {np.std(rmse_e_arr):.3f} N-m | R²: {np.mean(r2_e_arr):.3f} ± {np.std(r2_e_arr):.3f} | PCC: {np.mean(rho_e_arr):.3f} ± {np.std(rho_e_arr):.3f}",
            ylabel="Torque in N-m",
            is_grid_on=True,
            is_list=True)

plotResults(ref_sf, "real torque",
            predictions_sf, "predicted torque",
            f"Shoulder Front Joint Filtered; RMSE: {np.mean(rmse_sf_arr):.2f} ± {np.std(rmse_sf_arr):.3f} N-m | R²: {np.mean(r2_sf_arr):.3f} ± {np.std(r2_sf_arr):.3f} | PCC: {np.mean(rho_sf_arr):.3f} ± {np.std(rho_sf_arr):.3f}",
            ylabel="Torque in N-m",
            is_grid_on=True,
            is_list=True)

plotResults(ref_ss, "real torque",
            predictions_ss, "predicted torque",
            f"Shoulder Side Joint Filtered; RMSE: {np.mean(rmse_ss_arr):.2f} ± {np.std(rmse_ss_arr):.3f} N-m | R²: {np.mean(r2_ss_arr):.3f} ± {np.std(r2_ss_arr):.3f} | PCC: {np.mean(rho_ss_arr):.3f} ± {np.std(rho_ss_arr):.3f}",
            ylabel="Torque in N-m",
            is_grid_on=True,
            is_list=True)

plt.show()
