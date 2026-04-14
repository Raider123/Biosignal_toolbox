
#* This script loads the saved extracted input and target features, trains the model and tests it to get the prediction results.

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
config_filename = 'emg_torque_estimation_jte_individual.yaml'
cfg = loadConfig(filename=config_filename)

#? init early stopping 
if cfg.model_param.is_early_stop:
    early_callback = tf.keras.callbacks.EarlyStopping(monitor=cfg.model_param.monitor, 
                                                      min_delta=cfg.model_param.min_delta, 
                                                      patience=cfg.model_param.patience, 
                                                      verbose=cfg.model_param.verbose, 
                                                      baseline=cfg.model_param.baseline, 
                                                      restore_best_weights=cfg.model_param.restore_best_weights, 
                                                      start_from_epoch=cfg.model_param.start_from_epoch)
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

Y_scaler = load(feature_dir / cfg.filepath.saved_features[1])

#! ************************************************
#! Train, Load, or Test Model
#! ************************************************

# --- set global seed ---
seed = 7
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

neurons_inp = X_train.shape[1]
#? Init model with norm layer
train_model_e = AAN_Model(neurons_inp=neurons_inp, 
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

train_model_sf = AAN_Model(neurons_inp=neurons_inp, 
                        neurons_h1=cfg.model_param.neurons_h1, 
                        act_h1=cfg.model_param.act_h1, 
                        neurons_h2=cfg.model_param.neurons_h2, 
                        act_h2=cfg.model_param.act_h2,
                        neurons_h3=64, 
                        act_h3=cfg.model_param.act_h3,
                        neurons_h4=32, 
                        act_h4=cfg.model_param.act_h4,
                        neuron_out=1,
                        act_out=cfg.model_param.act_out)

train_model_ss = AAN_Model(neurons_inp=neurons_inp, 
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

MLP_model_e = MLModel(model = train_model_e, type= "keras")
MLP_model_sf = MLModel(model = train_model_sf, type= "keras")
MLP_model_ss = MLModel(model = train_model_ss, type= "keras")

#? Set the weights and deltas for weighted huber
if cfg.model_param.huber_weight_method == 'var':
    var_torques = np.var(Y_train, axis=0, ddof=1)
    weights_inp = 1.0 / (var_torques ** 1)
    weights_inp = weights_inp / np.sum(weights_inp)
    max_weight = np.percentile(weights_inp, 95)
    min_weight = np.percentile(weights_inp, 5)
    weights_inp = np.clip(weights_inp, min_weight, max_weight)
elif cfg.model_param.huber_weight_method == 'manual':
    weights_inp = [5,5,1]
elif cfg.model_param.huber_weight_method == 'dynamic_huber':
    weights_inp = [1,1,1]
else:
    raise ValueError(f"Wrong Huber weight method chosen {cfg.model_param.huber_weight_method}... Please choose between 'var', 'manual', and 'dynamic_huber'!!")


#? Train model
print("Training MLP model for elbow joint...")
MLP_model_e.setHuberWeights(weights_inp=weights_inp[0])
MLP_model_e.setHuberDeltas(deltas_inp=cfg.model_param.huber_deltas[0])

save_model_path = cfg.filepath.save_model_path + filename_suffix + "_elbow"
# print(save_model_path)

MLP_model_e.trainModel(save_trained_model=cfg.model_param.is_save_model, 
                     model_filename=save_model_path, 
                     train_epochs=cfg.model_param.n_epochs, 
                     batch_size=cfg.model_param.batch_size, 
                     class_weights=None, 
                     x_train=X_train, 
                     y_train=Y_train[:,0], 
                     x_val=X_val,
                     y_val=Y_val[:,0],
                     loss_fcn=cfg.model_param.loss_fcn, 
                     optimizer=cfg.model_param.optimizer, 
                     metrics=cfg.model_param.metrics, 
                     show_train_results=cfg.model_param.show_train_results, 
                     callbacks=early_callback)

print("Training MLP model for shoulder front joint...")
MLP_model_sf.setHuberWeights(weights_inp=weights_inp[1])
MLP_model_sf.setHuberDeltas(deltas_inp=cfg.model_param.huber_deltas[1])

save_model_path = cfg.filepath.save_model_path + filename_suffix + "_front"
# print(save_model_path)

MLP_model_sf.trainModel(save_trained_model=cfg.model_param.is_save_model, 
                     model_filename=save_model_path, 
                     train_epochs=cfg.model_param.n_epochs, 
                     batch_size=cfg.model_param.batch_size, 
                     class_weights=None, 
                     x_train=X_train, 
                     y_train=Y_train[:,1], 
                     x_val=X_val,
                     y_val=Y_val[:,1],
                     loss_fcn=cfg.model_param.loss_fcn, 
                     optimizer=cfg.model_param.optimizer, 
                     metrics=cfg.model_param.metrics, 
                     show_train_results=cfg.model_param.show_train_results, 
                     callbacks=early_callback)

print("Training MLP model for shoulder side joint...")
MLP_model_ss.setHuberWeights(weights_inp=weights_inp[2])
MLP_model_ss.setHuberDeltas(deltas_inp=cfg.model_param.huber_deltas[2])

save_model_path = cfg.filepath.save_model_path + filename_suffix + "_side"
# print(save_model_path)

MLP_model_ss.trainModel(save_trained_model=cfg.model_param.is_save_model, 
                     model_filename=save_model_path, 
                     train_epochs=cfg.model_param.n_epochs, 
                     batch_size=cfg.model_param.batch_size, 
                     class_weights=None, 
                     x_train=X_train, 
                     y_train=Y_train[:,2], 
                     x_val=X_val,
                     y_val=Y_val[:,2],
                     loss_fcn=cfg.model_param.loss_fcn, 
                     optimizer=cfg.model_param.optimizer, 
                     metrics=cfg.model_param.metrics, 
                     show_train_results=cfg.model_param.show_train_results, 
                     callbacks=early_callback)

print("MLP training done!!\n")

#? Predict and get results 
print("Predicting joint torques...")
MLP_model_e.predictTarget(data=X_test, 
                          labels=Y_test[:,0], 
                          classification=False, 
                          show_results=False, 
                          show_pred_time=False, 
                          eval_type=cfg.post_train_param.eval_type)

MLP_model_sf.predictTarget(data=X_test, 
                          labels=Y_test[:,1], 
                          classification=False, 
                          show_results=False, 
                          show_pred_time=False, 
                          eval_type=cfg.post_train_param.eval_type)

MLP_model_ss.predictTarget(data=X_test, 
                          labels=Y_test[:,2], 
                          classification=False, 
                          show_results=False, 
                          show_pred_time=False, 
                          eval_type=cfg.post_train_param.eval_type)

perf_results_MLP_e_scaled = MLP_model_e.getPredictionScores()
perf_results_MLP_sf_scaled = MLP_model_sf.getPredictionScores()
perf_results_MLP_ss_scaled = MLP_model_ss.getPredictionScores()

perf_results_MLP_scaled = np.concatenate([perf_results_MLP_e_scaled, perf_results_MLP_sf_scaled, perf_results_MLP_ss_scaled], axis=1)
# perf_results_MLP_scaled = np.concatenate([perf_results_MLP_e_scaled, perf_results_MLP_sf_scaled], axis=1)
#? Rescaling output
Y_ref = Y_scaler.inverse_transform(Y_test)
perf_results_MLP = Y_scaler.inverse_transform(perf_results_MLP_scaled)

perf_results_MLP_e = perf_results_MLP[:,0]
perf_results_MLP_sf = perf_results_MLP[:,1]
perf_results_MLP_ss = perf_results_MLP[:,2]

MLP_model_e.setPredictionScores(perf_results_MLP_e.reshape(-1,1))
MLP_model_sf.setPredictionScores(perf_results_MLP_sf.reshape(-1,1))
MLP_model_ss.setPredictionScores(perf_results_MLP_ss.reshape(-1,1))

print("Pre-filtering Eval Metrics!!")
_,_ = MLModel.calculateEvalMetrics(Y_ref[:,0], perf_results_MLP_e)
_,_ = MLModel.calculateEvalMetrics(Y_ref[:,1], perf_results_MLP_sf)
_,_ = MLModel.calculateEvalMetrics(Y_ref[:,2], perf_results_MLP_ss)
print("\n")
#! ************************************************
#! Post-prediction Filtering
#! ************************************************
#? Median filter for removing spikes/outliers
MLP_model_e.applyFilter_prediction(method=cfg.post_train_param.filter_type,
                                 window_length=cfg.post_train_param.filter_size)
MLP_model_sf.applyFilter_prediction(method=cfg.post_train_param.filter_type,
                                 window_length=cfg.post_train_param.filter_size)
MLP_model_ss.applyFilter_prediction(method=cfg.post_train_param.filter_type,
                                 window_length=cfg.post_train_param.filter_size)
#? Savitsky Golay filter
MLP_model_e.applyFilter_prediction(method="savgol",
                                 window_length=cfg.post_train_param.savgol_window_len,
                                 poly_order=cfg.post_train_param.savgol_poly_order)
MLP_model_sf.applyFilter_prediction(method="savgol",
                                 window_length=cfg.post_train_param.savgol_window_len,
                                 poly_order=cfg.post_train_param.savgol_poly_order)
MLP_model_ss.applyFilter_prediction(method="savgol",
                                 window_length=cfg.post_train_param.savgol_window_len,
                                 poly_order=cfg.post_train_param.savgol_poly_order)

perf_results_MLP_e = MLP_model_e.getPredictionScores()
perf_results_MLP_sf = MLP_model_sf.getPredictionScores()
perf_results_MLP_ss = MLP_model_ss.getPredictionScores()
# print(perf_results_MLP.shape)

#? Calculate the model eval metrics on the filtered predicted values
print("Post-filtering Eval Metrics!!")
r2_elbow, rmse_elbow = MLModel.calculateEvalMetrics(Y_ref[:,0], perf_results_MLP_e)
r2_front, rmse_front = MLModel.calculateEvalMetrics(Y_ref[:,1], perf_results_MLP_sf)
r2_side, rmse_side = MLModel.calculateEvalMetrics(Y_ref[:,2], perf_results_MLP_ss)

#? Plotting the filtered prediction results
plotResults(data_ref=Y_ref[:,0],
            label_ref="real torque", 
            data_out=perf_results_MLP_e, 
            label_out="predicted torque", 
            title=f"Elbow; RMSE: {rmse_elbow}N-m   R2: {r2_elbow}", 
            ylabel="Torque in N-m", 
            is_grid_on=True)

plotResults(data_ref=Y_ref[:,1],
            label_ref="real torque", 
            data_out=perf_results_MLP_sf, 
            label_out="predicted torque", 
            title=f"Shoulder Front; RMSE: {rmse_front}N-m   R2: {r2_front}", 
            ylabel="Torque in N-m", 
            is_grid_on=True)

plotResults(data_ref=Y_ref[:,2],
            label_ref="real torque", 
            data_out=perf_results_MLP_ss, 
            label_out="predicted torque", 
            title=f"Shoulder Side; RMSE: {rmse_side}N-m   R2: {r2_side}", 
            ylabel="Torque in N-m", 
            is_grid_on=True)

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
        
        figs = [("elbow", Y_ref[:, 0], perf_results_MLP_e,
         f"Elbow; RMSE: {rmse_elbow}N-m   R2: {r2_elbow}"),
        ("front", Y_ref[:, 1], perf_results_MLP_sf,
         f"Shoulder Front; RMSE: {rmse_front}N-m   R2: {r2_front}"),
        ("side", Y_ref[:, 2], perf_results_MLP_ss,
         f"Shoulder Side; RMSE: {rmse_side}N-m   R2: {r2_side}"),]
        
        for name, ref, out, title in figs:
            plt.figure()
            plotResults(data_ref=ref,
            label_ref="real torque", 
            data_out=out, 
            label_out="predicted torque", 
            title=title, 
            ylabel="Torque in N-m", 
            is_grid_on=True)

            plt.savefig(dir_path / f"test_{name}.png")
            plt.close()
        print(f"✅ Saved all plots in {dir_path}")
else:
    print("❌ Plots not saved.")
        