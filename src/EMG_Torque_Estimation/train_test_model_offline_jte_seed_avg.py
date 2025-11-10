
#* This script loads the saved extracted input and target features, trains the model and tests it to get the prediction results on 5 different seeds to get the average performance.

#! ************************************************
#! Imports
#! ************************************************

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from joblib import load
import random

#own libs 
from biosignal_toolbox.ML_lib import MLModel
from biosignal_toolbox.models.AANModel import AAN_Model
from biosignal_toolbox.utils import customWarningFormat, loadConfig, getAbsolutePath, plotResults, createOutputDir, createReadme

import warnings
warnings.formatwarning = customWarningFormat

#! ************************************************
#! User Parameters and Data Loading
#! ************************************************

#? load config file
config_filename = 'emg_torque_estimation_jte.yaml'
cfg = loadConfig(filename=config_filename)

#? init early stopping 
if cfg.model_param.is_early_stop:
    early_callback = tf.keras.callbacks.EarlyStopping(monitor=cfg.model_param.monitor, min_delta=cfg.model_param.min_delta, patience=cfg.model_param.patience, verbose=cfg.model_param.verbose, baseline=cfg.model_param.baseline, restore_best_weights=cfg.model_param.restore_best_weights, start_from_epoch=cfg.model_param.start_from_epoch)
else:
    early_callback = None

#? filename suffix
timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
filename_suffix = timestamp

for i in range(len(cfg.data_param.mov_type)):
    filename_suffix = filename_suffix + '_' + cfg.data_param.mov_type[i]

for i in range(len(cfg.data_param.weights)):
    filename_suffix = filename_suffix + '_' + cfg.data_param.weights[i]

#? Load saved features
feature_dir = getAbsolutePath(cfg.filepath.save_features_path)
feat_inp = np.load(feature_dir / cfg.filepath.saved_features[0])
X_train = feat_inp['X_train']
X_test = feat_inp['X_test']
X_val = feat_inp['X_val']

Y_train = feat_inp['Y_train']
Y_test = feat_inp['Y_test']
Y_val = feat_inp['Y_val']

scaler_inp = load(feature_dir / cfg.filepath.saved_features[1])

#? Uncomment if you want to load features from individual condition features
# scaler_idx = 1
# Y_scaler_dict = scaler_inp["Y_scaler_dict"]
# Y_scaler_info = scaler_inp["Y_scaler_info"]

#? Uncomment if you want to load features from entire dataset together
scaler_idx = 0
Y_scaler = scaler_inp

#! ************************************************
#! Train, Load, or Test Model
#! ************************************************

# --- set global seed ---
seed_arr = [1, 7, 25, 45, 70]
# Shape (seed, samples, joint)
pred_results = np.zeros((len(seed_arr), Y_test.shape[0], 3))

r2_e_pre_arr = []
r2_sf_pre_arr = []
r2_ss_pre_arr = []

rho_e_pre_arr = []
rho_sf_pre_arr = []
rho_ss_pre_arr = []

rmse_e_pre_arr = []
rmse_sf_pre_arr = []
rmse_ss_pre_arr = []

r2_e_arr = []
r2_sf_arr = []
r2_ss_arr = []

rho_e_arr = []
rho_sf_arr = []
rho_ss_arr = []

rmse_e_arr = []
rmse_sf_arr = []
rmse_ss_arr = []

for idx in range(len(seed_arr)):
    np.random.seed(seed_arr[idx])
    random.seed(seed_arr[idx])
    tf.random.set_seed(seed_arr[idx])
    tf.config.experimental.enable_op_determinism()
    print(f"Seed: {seed_arr[idx]}")

    neurons_inp = X_train.shape[1]
    #? Init model with norm layer
    train_model = AAN_Model(neurons_inp=neurons_inp, 
                            neurons_h1=cfg.model_param.neurons_h1, 
                            act_h1=cfg.model_param.act_h1, 
                            neurons_h2=cfg.model_param.neurons_h2, 
                            act_h2=cfg.model_param.act_h2,
                            neurons_h3=cfg.model_param.neurons_h3, 
                            act_h3=cfg.model_param.act_h3,
                            neurons_h4=cfg.model_param.neurons_h4, 
                            act_h4=cfg.model_param.act_h4,
                            neuron_out=cfg.model_param.neurons_out,
                            act_out=cfg.model_param.act_out)

    MLP_model = MLModel(model = train_model, type= "keras")

    #? Set the weights and deltas for weighted huber
    if cfg.model_param.huber_weight_method == 'var':
        var_torques = np.var(Y_train, axis=0, ddof=1)
        weights_inp = 1.0 / (var_torques ** 0.5 + 1e-6)
        # weights_inp = weights_inp / np.sum(weights_inp)
        max_weight = np.percentile(weights_inp, 95)
        min_weight = np.percentile(weights_inp, 5)
        weights_inp = np.clip(weights_inp, min_weight, max_weight)
        weights_inp = weights_inp / np.mean(weights_inp)
    elif cfg.model_param.huber_weight_method == 'smooth_var':
        weights_inp = MLP_model.getSmoothVarWeights(y_train=Y_train,
                                    clip_percentile=[5,75],
                                    window_len=5,
                                    poly_order=2)
    elif cfg.model_param.huber_weight_method == 'manual':
        weights_inp = [5,5,1]
    elif cfg.model_param.huber_weight_method == 'dynamic_huber':
        weights_inp = [1,1,1]
    else:
        raise ValueError(f"Wrong Huber weight method chosen {cfg.model_param.huber_weight_method}... Please choose between 'var','smooth_var', 'manual', and 'dynamic_huber'!!")

    MLP_model.setHuberWeights(weights_inp=weights_inp)
    MLP_model.setHuberDeltas(deltas_inp=cfg.model_param.huber_deltas)

    #? Train model
    print("Training MLP model for elbow joint...")
    save_model_path = cfg.filepath.save_model_path + filename_suffix
    # print(save_model_path)
    MLP_model.trainModel(save_trained_model=cfg.model_param.is_save_model, 
                        model_filename=save_model_path, 
                        train_epochs=cfg.model_param.n_epochs, 
                        batch_size=cfg.model_param.batch_size, 
                        class_weights=None, 
                        x_train=X_train, 
                        y_train=Y_train, 
                        x_val=X_val,
                        y_val=Y_val,
                        loss_fcn=cfg.model_param.loss_fcn, 
                        optimizer=cfg.model_param.optimizer, 
                        metrics=cfg.model_param.metrics, 
                        show_train_results=cfg.model_param.show_train_results, 
                        callbacks=early_callback)
    print("MLP training done!!\n")

    #? Predict and get results 
    print("Predicting joint torques...")
    MLP_model.predictTarget(data=X_test, 
                            labels=Y_test, 
                            classification=False, 
                            show_results=False, 
                            show_pred_time=False, 
                            eval_type=cfg.post_train_param.eval_type)

    perf_results_MLP_scaled = MLP_model.getPredictionScores()

    #? Rescaling output
    if scaler_idx:
        Y_ref = []
        perf_results_MLP = []

        for wgt, mov, start_idx, end_idx in Y_scaler_info:
            Y_ref_scaled = Y_test[start_idx:end_idx]
            Y_pred_scaled = perf_results_MLP_scaled[start_idx:end_idx]

            scaler = Y_scaler_dict[wgt][mov]
            Y_ref.append(scaler.inverse_transform(Y_ref_scaled))
            perf_results_MLP.append(scaler.inverse_transform(Y_pred_scaled))

        Y_ref = np.concatenate(Y_ref, axis=0)
        perf_results_MLP = np.concatenate(perf_results_MLP, axis=0)
    else:
        perf_results_MLP = Y_scaler.inverse_transform(perf_results_MLP_scaled)
        Y_ref = Y_scaler.inverse_transform(Y_test)

    MLP_model.setPredictionScores(perf_results_MLP)

    print("Pre-filtering Eval Metrics!!")
    r2_elbow_pre, rmse_elbow_pre, rho_elbow_pre = MLModel.calculateEvalMetrics(Y_ref[:,0], perf_results_MLP[:,0], is_Pearson=True)
    r2_front_pre, rmse_front_pre, rho_front_pre = MLModel.calculateEvalMetrics(Y_ref[:,1], perf_results_MLP[:,1], is_Pearson=True)
    r2_side_pre, rmse_side_pre, rho_side_pre = MLModel.calculateEvalMetrics(Y_ref[:,2], perf_results_MLP[:,2], is_Pearson=True)
    print("\n")
    #! ************************************************
    #! Post-prediction Filtering
    #! ************************************************
    #? Median filter for removing spikes/outliers
    MLP_model.applyFilter_prediction(method=cfg.post_train_param.filter_type,
                                    window_length=cfg.post_train_param.filter_size)
    #? Savitsky Golay filter
    MLP_model.applyFilter_prediction(method="savgol",
                                    window_length=cfg.post_train_param.savgol_window_len,
                                    poly_order=cfg.post_train_param.savgol_poly_order)

    perf_results_MLP = MLP_model.getPredictionScores()
    # print(perf_results_MLP.shape)

    pred_results[idx, :, 0] = perf_results_MLP[:,0]
    pred_results[idx, :, 1] = perf_results_MLP[:,1]
    pred_results[idx, :, 2] = perf_results_MLP[:,2]

    #? Calculate the model eval metrics on the filtered predicted values
    print("Post-filtering Eval Metrics!!")
    r2_elbow, rmse_elbow, rho_elbow = MLModel.calculateEvalMetrics(Y_ref[:,0], perf_results_MLP[:,0], is_Pearson=True)
    r2_front, rmse_front, rho_front = MLModel.calculateEvalMetrics(Y_ref[:,1], perf_results_MLP[:,1], is_Pearson=True)
    r2_side, rmse_side, rho_side = MLModel.calculateEvalMetrics(Y_ref[:,2], perf_results_MLP[:,2], is_Pearson=True)

    #? Append pre-filtering metrics into the arrays
    r2_e_pre_arr.append(r2_elbow_pre)
    rho_e_pre_arr.append(rho_elbow_pre)
    rmse_e_pre_arr.append(rmse_elbow_pre)

    r2_sf_pre_arr.append(r2_front_pre)
    rho_sf_pre_arr.append(rho_front_pre)
    rmse_sf_pre_arr.append(rmse_front_pre)

    r2_ss_pre_arr.append(r2_side_pre)
    rho_ss_pre_arr.append(rho_side_pre)
    rmse_ss_pre_arr.append(rmse_side_pre)

    #? Append post-filtering metrics into the arrays
    r2_e_arr.append(r2_elbow)
    rho_e_arr.append(rho_elbow)
    rmse_e_arr.append(rmse_elbow)

    r2_sf_arr.append(r2_front)
    rho_sf_arr.append(rho_front)
    rmse_sf_arr.append(rmse_front)

    r2_ss_arr.append(r2_side)
    rho_ss_arr.append(rho_side)
    rmse_ss_arr.append(rmse_side)



print("The results across seed are...\n")
print(f"Elbow R2: {r2_e_arr}")
print(f"Front R2: {r2_sf_arr}")
print(f"Side R2: {r2_ss_arr}\n")

print(f"Elbow R2 stats: Mean: {np.mean(r2_e_arr)}  Std. : {np.std(r2_e_arr)}")
print(f"Front R2 stats: Mean: {np.mean(r2_sf_arr)}  Std. : {np.std(r2_sf_arr)}")
print(f"Side R2 stats: Mean: {np.mean(r2_ss_arr)}  Std. : {np.std(r2_ss_arr)}\n")

print(f"Elbow RMSE stats: Mean: {np.mean(rmse_e_arr)}  Std. : {np.std(rmse_e_arr)}")
print(f"Front RMSE stats: Mean: {np.mean(rmse_sf_arr)}  Std. : {np.std(rmse_sf_arr)}")
print(f"Side RMSE stats: Mean: {np.mean(rmse_ss_arr)}  Std. : {np.std(rmse_ss_arr)}\n")

print(f"Elbow Pearson stats: Mean: {np.mean(rho_e_arr)}  Std. : {np.std(rho_e_arr)}")
print(f"Front Pearson stats: Mean: {np.mean(rho_sf_arr)}  Std. : {np.std(rho_sf_arr)}")
print(f"Side Pearson stats: Mean: {np.mean(rho_ss_arr)}  Std. : {np.std(rho_ss_arr)}")


#? Plotting the filtered prediction results
time_axis = np.arange(0, len(Y_ref[:,0]), 1)*0.05
#? time: first 10 sec
idx = time_axis <=10
time_axis_new = time_axis[idx]

joint_names = ['Elbow', 'Shoulder_Front', 'Shoulder Side']

for j in range(len(joint_names)):
    # ax = axes[j]
    # Mean and std across seeds
    mean_pred = pred_results[:, idx, j].mean(axis=0)
    std_pred = pred_results[:, idx, j].std(axis=0)

    plt.figure(figsize=(8,4))
    
    plt.plot(time_axis_new, Y_ref[idx, j], label='Ref. torque', color='black')
    plt.plot(time_axis_new, mean_pred, label='Predicted Torque', color='blue')
    plt.fill_between(time_axis_new, mean_pred - std_pred, mean_pred + std_pred, color='blue', alpha=0.3)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Joint Torque (N m)')
    plt.legend()

    plt.tight_layout()

#? Showing the plots
plt.show()

#? Check if the save dir exists. If not create one
#? Create a readme.txt and include all parameters in it
if cfg.post_train_param.is_save_plot:
    choice = input("Do you want to save the plots? (Y/N)").strip().lower()
    if choice in ["y", "yes"]:
        dir_path = createOutputDir(param_obj=cfg, suffix_str="plot")
        createReadme(param_obj=cfg,
                     dir_path=dir_path)
        
        filename = filename_suffix + "eval_metrics.npz"
        np.savez_compressed(dir_path / filename,
                    rmse_elbow_pre=rmse_e_pre_arr,
                    rmse_front_pre=rmse_sf_pre_arr,
                    rmse_side_pre=rmse_ss_pre_arr,
                    r2_elbow_pre=r2_e_pre_arr,
                    r2_front_pre=r2_sf_pre_arr,
                    r2_side_pre=r2_ss_pre_arr,
                    rho_elbow_pre=rho_e_pre_arr,
                    rho_front_pre=rho_sf_pre_arr,
                    rho_side_pre=rho_ss_pre_arr,
                    rmse_elbow_post=rmse_e_arr,
                    rmse_front_post=rmse_sf_arr,
                    rmse_side_post=rmse_ss_arr,
                    r2_elbow_post=r2_e_arr,
                    r2_front_post=r2_sf_arr,
                    r2_side_post=r2_ss_arr,
                    rho_elbow_post=rho_e_arr,
                    rho_front_post=rho_sf_arr,
                    rho_side_post=rho_ss_arr)
        
        filename = filename_suffix + "pred_results.npz"
        np.savez_compressed(dir_path / filename,
                    pred_results=pred_results,
                    ref_target=Y_ref)

        print(f"✅ Saved all files in {dir_path}")
        # time_axis = np.arange(0, len(Y_ref[:,0]), 1)*0.05
        # #? time: first 10 sec
        # idx = time_axis <=10
        # time_axis_new = time_axis[idx]

        # for j in range(len(joint_names)):
        #     # ax = axes[j]
        #     # Mean and std across seeds
        #     mean_pred = pred_results[:, idx, j].mean(axis=0)
        #     std_pred = pred_results[:, idx, j].std(axis=0)

        #     plt.figure(figsize=(8,8))
            
        #     plt.plot(time_axis_new, Y_ref[idx, j], label='Ref. torque', color='black')
        #     plt.plot(time_axis_new, mean_pred, label='Predicted Torque', color='blue')
        #     plt.fill_between(time_axis_new, mean_pred - std_pred, mean_pred + std_pred, color='blue', alpha=0.3)
            
        #     plt.xlabel('Time (s)')
        #     plt.ylabel('Joint Torque (N m)')
        #     plt.legend()
        
        #     plt.tight_layout()
        #     plt.savefig(dir_path / f"band_plot_{joint_names[j]}.png")

else:
    print("❌ Plots not saved.")