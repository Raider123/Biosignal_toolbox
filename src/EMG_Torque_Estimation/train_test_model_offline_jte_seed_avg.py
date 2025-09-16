
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
config_filename = 'emg_torque_estimation_jte.yaml'
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
seed_arr = [1, 7, 25, 45, 70]
results_arr_e = []
results_arr_sf = []
results_arr_ss = []
for seed in seed_arr:
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

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
    var_torques = np.var(Y_train, axis=0, ddof=1)
    weights_inp = 1.0 / (var_torques ** 1)
    weights_inp = weights_inp / np.sum(weights_inp)
    max_weight = np.percentile(weights_inp, 95)
    min_weight = np.percentile(weights_inp, 5)
    weights_inp = np.clip(weights_inp, min_weight, max_weight)

    # weights_inp = [5,5,1]
    MLP_model.setHuberWeights(weights_inp=weights_inp)
    MLP_model.setHuberDeltas(deltas_inp=[0.25, 0.35, 0.2])

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
    perf_results_MLP = Y_scaler.inverse_transform(perf_results_MLP_scaled)
    Y_ref = Y_scaler.inverse_transform(Y_test)

    MLP_model.setPredictionScores(perf_results_MLP)

    #! ************************************************
    #! Post-prediction Filtering
    #! ************************************************
    #? Median filter for removing spikes/outliers
    MLP_model.applyFilter_prediction(method="mean",
                                    window_length=11)
    #? Savitsky Golay filter
    MLP_model.applyFilter_prediction(method="savgol",
                                    window_length=25,
                                    poly_order=3)

    perf_results_MLP = MLP_model.getPredictionScores()
    # print(perf_results_MLP.shape)

    #? Calculate the model eval metrics on the filtered predicted values
    print("Post-filtering Eval Metrics!!")
    r2_elbow, rmse_elbow = MLModel.calculateEvalMetrics(Y_ref[:,0], perf_results_MLP[:,0])
    r2_front, rmse_front = MLModel.calculateEvalMetrics(Y_ref[:,1], perf_results_MLP[:,1])
    r2_side, rmse_side = MLModel.calculateEvalMetrics(Y_ref[:,2], perf_results_MLP[:,2])

    #? Append all r2 scores
    results_arr_e.append(r2_elbow)
    results_arr_sf.append(r2_front)
    results_arr_ss.append(r2_side)

print("The results across seed are...\n")
print(f"Elbow R2: {results_arr_e}")
print(f"Front R2: {results_arr_sf}")
print(f"Side R2: {results_arr_ss}")

print(f"Elbow R2 stats: Mean: {np.mean(results_arr_e)}  Std. : {np.std(results_arr_e)}")
print(f"Front R2 stats: Mean: {np.mean(results_arr_sf)}  Std. : {np.std(results_arr_sf)}")
print(f"Side R2 stats: Mean: {np.mean(results_arr_ss)}  Std. : {np.std(results_arr_ss)}")