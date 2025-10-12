import zmq
import time
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import butter
from collections import deque
from biosignal_toolbox.emg_lib import OnlineEMG
from biosignal_toolbox.ML_lib import MLModel
from tensorflow.keras.models import load_model

class LiveEstimation:

   def __init__(self):
       self.property = {}
       self.configure_properties()
       print("Loaded properties!")

       # EMG 8 channel names
       self.channel_names = ['BP1', 'BP2', 'BP3', 'BP4', 'BP5', 'BP6', 'BP7', 'BP8']

       # Initialise HPF sos
       self.sos_hpf = butter(N=2, Wn=self.property["f_cutoff_hpf"], btype='highpass', analog=False, output='sos',
                             fs=self.property["f_samp"])
       self.sos_hpf_idx = 0

       # Initialise LPF sos
       self.sos_lpf = butter(N=2, Wn=self.property["f_cutoff_lpf"], btype='lowpass', analog=False, output='sos',
                             fs=self.property["f_samp"])
       self.sos_lpf_idx = 0

       # Create old_online EMG object
       self.EMG_live = OnlineEMG(stream_type="data",
                                 channel_names=self.channel_names, n_samples=self.property["buffer_size"],
                                 f_samp=self.property["f_samp"])

       print("Created EMG_live object!!")

       #  TCN Model
       self.load_model(
           'F:/SMT_MASTERPROJEKT/biosignal_toolbox/src/JTE_Project/offline/saved_online_models/tcn_mtl.keras')

       print("Loaded TCN Model")

       emg_context = zmq.Context()
       self.emg_socket = emg_context.socket(zmq.SUB)
       self.emg_socket.connect("tcp://127.0.0.1:5555")
       print("EMG Subscriber is active")
       self.emg_socket.setsockopt_string(zmq.SUBSCRIBE, "")

       my_context_sj0 = zmq.Context()
       self.socket_sj0 = my_context_sj0.socket(zmq.PUB)
       self.socket_sj0.bind("tcp://*:7012")
       time.sleep(0.05)
       print("Publisher SJ0 active")

       my_context_sj2 = zmq.Context()
       self.socket_sj2 = my_context_sj2.socket(zmq.PUB)
       self.socket_sj2.bind("tcp://*:7011")
       time.sleep(0.05)
       print("Publisher SJ2 active")

       my_context_elbow = zmq.Context()
       self.socket_elbow = my_context_elbow.socket(zmq.PUB)
       self.socket_elbow.bind("tcp://*:7010")
       time.sleep(0.05)
       print("Publisher Elbow active")

       # Sus - nachgucken
       self.sos_bp = self.EMG_live.designFilter(f_low=245.0, f_high=20, order=2, filter_type="scipy_butter",
                                                return_type="sos")
       self.sos_lp = self.EMG_live.designFilter(f_low=5.0, f_high=None, order=2, filter_type="scipy_butter",
                                                return_type="sos")


       print("Finished configuring EMG receiver")

       self.update_loop()

   def add_property(self, name, default_value):
       self.property[name] = default_value

   def configure_properties(self):
       self.add_property("buffer_size", 500)
       self.add_property("feature_size", 20)
       self.add_property("f_samp", 500)
       self.add_property("n_channels", 8)
       self.add_property("f_cutoff_hpf", 15)
       self.add_property("f_cutoff_lpf", 10)
       self.add_property("var_filter_width", 20)
       self.add_property("mvc", 2.7579163508176626e-06)
       self.add_property("delay", 25)
       self.add_property("beta1", 0.25)
       self.add_property("beta2", 0.05)
       self.add_property("gamma", 0.7)
       self.add_property("A", -1.5)

   def load_model(self, model_path=None):
       """
       Load the trained TCN model.

       Parameters
       ----------
       model_path : str or Path, optional
           Path to the saved model. If None, uses default path from config.
       """
       if model_path is None:
           save_model_path = (
                   Path(__file__).parent.parent / "saved_online_models"
           )
           model_path = save_model_path / "tcn_mtl.keras"

       print(f"Loading model from: {model_path}")
       self.model = load_model(model_path, compile=False)
       print("Model loaded successfully!\n")

       return True

   def start_device_stream(self):
        # Start EMG live stream (unused in Pseudoonline)
        self.EMG_live.startANTEegoStreaming(path_to_so_file=self.property["path_to_so_file"])

        return True

   def update_loop(self):
       '''
        # t_start = time.perf_counter()
        # get a new data chunk
        #chunk = self.EMG_live.getChunk(return_chunk=True)
        # self.log(np.array(chunk).shape)

        # update the ring buffer
        #self.EMG_live.updateBuffer(show_data_shape=False, channel_indices=[0, 1, 2, 3, 4, 5, 6, 7])

        # # high pass filter
        # self.EMG_live.highPassFilterOnline(cutoff_freq=self.property["f_cutoff_hpf"], order=2, fs=self.property["f_samp"], type="butter", sos=self.sos_hpf, counter=self.sos_hpf_idx)
        # self.sos_hpf_idx = 1

        # # variance filter
        # self.EMG_live.applyVarianceFilterOnline(ring_buffer=np.zeros(self.property['var_filter_width']), width=self.property['var_filter_width'], index=0)

        # # normalisation
        # self.EMG_live.normalizeContinuousDataOnline(mvc=self.property["mvc"])

        # # low pass filter
        # self.EMG_live.lowPassFilterOnline(cutoff_freq=self.property["f_cutoff_lpf"], order=2, fs=self.property["f_samp"], type="butter", sos=self.sos_lpf, counter=self.sos_lpf_idx)
        # self.sos_lpf_idx = 1

        # # neural activation force
        # self.EMG_live.calculateActivationForceOnline(d=self.property["delay"], b1=self.property["beta1"], b2=self.property["beta2"], g=self.property["gamma"], nonlinear_shape_factor=self.property["A"])

        # convert filtered data into window
        self.EMG_live.bufferToWindows(num_non_data_channels=2)

        # ! Niklas
        self.EMG_live.filterWindows(sos=self.sos_bp, apply_method="zero_phase_sos", padtype="even")

        self.EMG_live.windows = np.abs(self.EMG_live.windows)
        self.EMG_live.filterWindows(sos=self.sos_lp, apply_method="zero_phase_sos", padtype="even")
        max_val = [2.67320054e-04, 1.74150060e-04, 6.26772180e-05, 6.30003855e-05, 1.17669616e-04, 6.39598476e-05,
                   1.36005774e-05, 2.12251047e-04]
        for i in range(self.EMG_live.windows.shape[1]):
            self.EMG_live.windows[:, i, :, :] = self.EMG_live.windows[:, i, :, :] / max_val[i]
        self.EMG_live.calculateActivationForceFunctionWindows(d=50, b1=0.75, b2=0.05, g=0.1, nonlinear_shape_factor=-1.5)
        # !

        # get data buffer size to calculate feature window indices
        db = self.EMG_live.getDataBuffer()
        feature_indices_windows = np.arange(db.shape[2] - self.property["feature_size"], db.shape[2], step=1)

        # extract features from window
        self.EMG_live.featureExtractionFromWindows(feature_type="timepoints",
                                                   feature_indices_windows=feature_indices_windows)

        # inp_emg = self.EMG_live.getFeatures()

        # calculate mean absolute value (MAV)
        inp_emg = self.EMG_live.calculateMAVFromFeatures(len(self.channel_names))

        # predict elbow torque
        self.MLP_model_e.predictTarget(data=inp_emg, classification=False, show_results=False, show_pred_time=False,
                                       eval_type="old_online")

        # predict front torque
        self.MLP_model_f.predictTarget(data=inp_emg, classification=False, show_results=False, show_pred_time=False,
                                       eval_type="old_online")

        # predict side torque
        self.MLP_model_s.predictTarget(data=inp_emg, classification=False, show_results=False, show_pred_time=False,
                                       eval_type="old_online")

        if len(self.torque_out_e) == 3:
            _ = self.torque_out_e.popleft()
            _ = self.torque_out_f.popleft()
            _ = self.torque_out_s.popleft()

        self.torque_out_e.append(self.MLP_model_e.getPredictionScores())
        self.torque_out_f.append(self.MLP_model_f.getPredictionScores())
        self.torque_out_s.append(self.MLP_model_s.getPredictionScores())

        # if len(self.std_e) == 5:
        #     _ = self.std_e.popleft()
        #     _ = self.std_f.popleft()
        #     _ = self.std_e.popleft()

        # self.std_e.append(self.MLP_model_e.getPredictionScores())
        # self.std_f.append(self.MLP_model_f.getPredictionScores())
        # self.std_s.append(self.MLP_model_s.getPredictionScores())

        # self.log("Sending predicted torques [elbow, front, side]!!")
        self.port["predicted_human_torque"].write({
            'names': ['shoulder_joint_0', 'shoulder_joint_1', 'shoulder_joint_2', 'elbow_joint'],
            'elements': [{
                'position': float("nan"),
                'speed': float("nan"),
                'effort': np.mean(self.torque_out_s),
                # 'effort': 0.0,
                # 'raw': self.torque_out_s[-1],
                'raw': round((self.torque_out_s[-1] - self.torque_out_s[0])[0, 0], 2),
                'acceleration': float("nan")
            }, {
                'position': float("nan"),
                'speed': float("nan"),
                'effort': 0.0,
                'raw': 0.0,
                'acceleration': float("nan")
            },
                {
                    'position': float("nan"),
                    'speed': float("nan"),
                    'effort': np.mean(self.torque_out_f),
                    # 'raw': self.torque_out_f[-1],
                    'raw': round((self.torque_out_f[-1] - self.torque_out_f[0])[0, 0], 2),
                    'acceleration': float("nan")
                },
                {
                    'position': float("nan"),
                    'speed': float("nan"),
                    'effort': np.mean(self.torque_out_e),
                    # 'raw': self.torque_out_e[-1],
                    'raw': round((self.torque_out_e[-1] - self.torque_out_e[0])[0, 0], 2),
                    'acceleration': float("nan")
                }],
            'time': {'microseconds': 0}
        })

        # t_end = time.perf_counter()
        # self.log(f"Time for hook calc: {t_end - t_start}")
        self.socket_sj0.send_string(str(round((self.torque_out_s[-1] - self.torque_out_s[0])[0, 0], 2)))
        self.socket_sj2.send_string(str(round((self.torque_out_f[-1] - self.torque_out_f[0])[0, 0], 2)))
        self.socket_elbow.send_string(str(round((self.torque_out_e[-1] - self.torque_out_e[0])[0, 0], 2)))

        # self.save_traj_elbow.append(round((self.torque_out_e[-1] - self.torque_out_e[0])[0,0],2))
        # self.save_traj_front.append(round((self.torque_out_f[-1] - self.torque_out_f[0])[0,0],2))
        # self.save_traj_side.append(round((self.torque_out_s[-1] - self.torque_out_s[0])[0,0],2))
        '''

       while True:
           # Read emg data from the stream
           data_arr_str = self.emg_socket.recv_string()
           data_arr_np = np.array(data_arr_str)

           # Manually setting the data (otherwise use start_hook())
           self.EMG_live.setChunk(data_arr_np, chunk_type="numpy")

           # Update the internal ring buffer
           self.EMG_live.updateBuffer(num_channels = 8)

           # Temp - get data from the buffer
           #latest_data = self.EMG_live.getDataBuffer()
           #print("Buffer-Shape: ", latest_data.shape)

           self.EMG_live.ensureLoopFrequency(print_loop_time=False)

           # # high pass filter
           self.EMG_live.highPassFilter(cutoff_freq=self.property["f_cutoff_hpf"], order=2, fs=self.property["f_samp"], filter_type="butter", sos=self.sos_hpf, counter=self.sos_hpf_idx, mode='old_offline')
           self.sos_hpf_idx = 1

           # # variance filter
           self.EMG_live.applyVarianceFilter_data(mode = "old_online", ring_buffer=np.zeros(self.property['var_filter_width']), width=self.property['var_filter_width'], index=0)

           # # normalisation
           self.EMG_live.normalizeContinuousData(mvc=self.property["mvc"], mode = "old_online")

           # # low pass filter
           self.EMG_live.lowPassFilter(mode='old_offline', cutoff_freq=self.property["f_cutoff_lpf"], order=2, fs=self.property["f_samp"], filter_type="butter", sos=self.sos_lpf, counter=self.sos_lpf_idx)
           self.sos_lpf_idx = 1

           # # neural activation force
           self.EMG_live.calculateActivationForceFunction(mode='old_offline', d=self.property["delay"], b1=self.property["beta1"], b2=self.property["beta2"], g=self.property["gamma"], nonlinear_shape_factor=self.property["A"])

           # convert filtered data into window
           self.EMG_live.bufferToWindows(num_non_data_channels=0)

           # Window related Filtering
           self.EMG_live.filterWindows(sos=self.sos_bp, apply_method="zero_phase_sos", padtype="even")

           self.EMG_live.windows = np.abs(self.EMG_live.windows)
           self.EMG_live.filterWindows(sos=self.sos_lp, apply_method="zero_phase_sos", padtype="even")
           '''
           max_val = [2.67320054e-04, 1.74150060e-04, 6.26772180e-05, 6.30003855e-05, 1.17669616e-04, 6.39598476e-05,
                      1.36005774e-05, 2.12251047e-04]
           for i in range(self.EMG_live.windows.shape[1]):
               self.EMG_live.windows[:, i, :, :] = self.EMG_live.windows[:, i, :, :] / max_val[i]
           
           self.EMG_live.calculateActivationForceFunctionWindows(d=50, b1=0.75, b2=0.05, g=0.1,
                                                              nonlinear_shape_factor=-1.5)
           '''

           # get data buffer size to calculate feature window indices
           db = self.EMG_live.getDataBuffer()
           feature_indices_windows = np.arange(db.shape[2] - self.property["feature_size"], db.shape[2], step=1)

           # extract features from window
           self.EMG_live.featureExtractionFromWindows(feature_type="timepoints",
                                                      feature_indices_windows=feature_indices_windows)

           inp_emg = self.EMG_live.getFeatures()
           print(inp_emg.shape)

           # calculate mean absolute value (MAV)
           #inp_emg = self.EMG_live.calculateMAVFromFeatures(len(self.channel_names))






if __name__ == "__main__":

 live_estimation_obj = LiveEstimation()
