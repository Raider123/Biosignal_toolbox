import pyrock
from pyrock.task.task_base import TaskBase, run_task, expose

import orocos.RTT.corba as RTT
from orogen.base import Time as CTime

import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.signal import butter
from collections import deque
from biosignal_toolbox.emg_lib import OnlineEMG
from biosignal_toolbox.ML_lib import MLModel
from biosignal_toolbox.models.AANModel import AAN_Model

import zmq

import warnings
warnings.filterwarnings('ignore')


class Task(TaskBase):
    def __init__(self, **kwargs):
        super(Task, self).__init__(**kwargs)

        # add properties
        self.add_property(name="buffer_size", 
                          description="Size of ring buffer (samples)", 
                          type_name="int", 
                          default_value=500)
        self.add_property(name="feature_size", 
                          description="Size of extracted features (samples)", 
                          type_name="int", 
                          default_value=20)
        self.add_property(name="f_samp", 
                          description="Sampling frequency of EMG (Hz)", 
                          type_name="int", 
                          default_value=500)
        self.add_property(name="n_channels", 
                          description="Number of EMG channels", 
                          type_name="int", 
                          default_value=8)
        self.add_property(name="f_cutoff_hpf", 
                          description="Cutoff frequency for high-pass filter (Hz)", 
                          type_name="int", 
                          default_value=15)
        self.add_property(name="f_cutoff_lpf", 
                          description="Cutoff frequency for low-pass filter (Hz)", 
                          type_name="int", 
                          default_value=10)
        self.add_property(name="var_filter_width", 
                          description="Width of buffer for variance filter", 
                          type_name="int", 
                          default_value=20)
        self.add_property(name="mvc", 
                          description="Maximum Voluntary Contraction of the muscles (V)", 
                          type_name="float", 
                          default_value=2.7579163508176626e-06)
        self.add_property(name="delay", 
                          description="Electromechanical delay (samples)", 
                          type_name="int", 
                          default_value=25)
        self.add_property(name="beta1", 
                          description="Coefficient for force activation difference equation", 
                          type_name="float", 
                          default_value=0.25)
        self.add_property(name="beta2", 
                          description="Coefficient for force activation difference equation", 
                          type_name="float", 
                          default_value=0.05)
        self.add_property(name="gamma", 
                          description="Coefficient for force activation difference equation", 
                          type_name="float", 
                          default_value=0.7)
        self.add_property(name="A", 
                          description="Linearity factor for force activation difference equation", 
                          type_name="float", 
                          default_value=-1.5)
        self.add_property(name="path_to_so_file", 
                          description="Path to the .so file for ANT EEGO amplifier", 
                          type_name="/std/string", 
                          default_value="/home/dfki.uni-bremen.de/kschari/kc_ws/repos/eego-sdk-pybind11/build/python3")  
        self.add_property(name="model_path", 
                          description="Path to the saved ML models", 
                          type_name="/std/string", 
                          default_value=os.environ["AUTOPROJ_CURRENT_ROOT"] + "/bundles/recupera/data/subject_data/saved_ML_models/0g/")

        # add output ports
        self.add_port(name="predicted_human_torque", 
                      port_type=RTT.COutput, 
                      type_name="/base/samples/Joints")

    def configure_hook(self):
 
        # EMG 8 channel names
        self.channel_names = ['BP1', 'BP2', 'BP3', 'BP4', 'BP5', 'BP6', 'BP7', 'BP8']

        # Initialise HPF sos
        self.sos_hpf = butter(N=2, Wn=self.property["f_cutoff_hpf"], btype='highpass', analog=False, output='sos', fs=self.property["f_samp"])
        self.sos_hpf_idx = 0

        # Initialise LPF sos
        self.sos_lpf = butter(N=2, Wn=self.property["f_cutoff_lpf"], btype='lowpass', analog=False, output='sos', fs=self.property["f_samp"])
        self.sos_lpf_idx = 0

        # Create old_online EMG object
        self.EMG_live = OnlineEMG(stream_type = "data", n_channels=self.property["n_channels"], channel_names=self.channel_names, n_samples=self.property["buffer_size"], f_samp=self.property["f_samp"])

        self.log("Created EMG_live object!!")

        #  MLP model
        model_e = AAN_Model()
        self.MLP_model_e = MLModel(model=model_e, type="keras")

        model_f = AAN_Model()
        self.MLP_model_f = MLModel(model=model_f, type="keras")

        model_s = AAN_Model()
        self.MLP_model_s = MLModel(model=model_s, type="keras")

        self.log("Created AAN models for all joints!!")

        # Raw predicted torque output list
        self.torque_out_e = deque([])
        self.torque_out_f = deque([])
        self.torque_out_s = deque([])

        # Standard Deviation
        self.std_e = deque([])
        self.std_f = deque([])
        self.std_s = deque([])

        my_context_sj0 = zmq.Context()
        self.socket_sj0 = my_context_sj0.socket(zmq.PUB)
        self.socket_sj0.bind("tcp://*:7012")
        time.sleep(0.05)

        my_context_sj2 = zmq.Context()
        self.socket_sj2 = my_context_sj2.socket(zmq.PUB)
        self.socket_sj2.bind("tcp://*:7011")
        time.sleep(0.05)

        my_context_elbow = zmq.Context()
        self.socket_elbow = my_context_elbow.socket(zmq.PUB)
        self.socket_elbow.bind("tcp://*:7010")
        time.sleep(0.05)

        #! Niklas
        self.sos_bp = self.EMG_live.designFilter(f_low = 245.0, f_high = 20, order = 2, filter_type = "scipy_butter", return_type = "sos")
        self.sos_lp = self.EMG_live.designFilter(f_low = 5.0, f_high = None, order = 2, filter_type = "scipy_butter", return_type = "sos")
        
        self.save_traj_elbow = []
        self.save_traj_front = []
        self.save_traj_side  = []


        return True

    def start_hook(self):
        # Load saved MLP model for elbow
        self.MLP_model_e.loadModel(filename="FW28D_AAN_model_elbow_try_3", path=self.property["model_path"])

        # Load saved MLP model for front
        self.MLP_model_f.loadModel(filename="FW28D_AAN_model_front_try_3", path=self.property["model_path"])

        # Load saved MLP model for side
        self.MLP_model_s.loadModel(filename="FW28D_AAN_model_side_try_3", path=self.property["model_path"])

        self.log("Loaded all saved MLP models!")

        # Start EMG live stream
        self.EMG_live.startANTEegoStreaming(path_to_so_file=self.property["path_to_so_file"])

        return True

    def update_hook(self):
        # t_start = time.perf_counter()
        # get a new data chunk 
        chunk = self.EMG_live.getChunk(return_chunk=True)
        # self.log(np.array(chunk).shape)

        # update the ring buffer
        self.EMG_live.updateBuffer(show_data_shape = False, channel_indices=[0, 1, 2, 3, 4, 5, 6, 7])

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

        #! Niklas
        self.EMG_live.filterWindows(sos = self.sos_bp, apply_method = "zero_phase_sos", padtype = "even")

        self.EMG_live.windows = np.abs(self.EMG_live.windows)
        self.EMG_live.filterWindows(sos = self.sos_lp, apply_method = "zero_phase_sos", padtype = "even")
        max_val = [2.67320054e-04, 1.74150060e-04, 6.26772180e-05, 6.30003855e-05, 1.17669616e-04, 6.39598476e-05, 1.36005774e-05, 2.12251047e-04]
        for i in range(self.EMG_live.windows.shape[1]):
                self.EMG_live.windows[:,i,:,:] = self.EMG_live.windows[:,i,:,:]/max_val[i]
        self.EMG_live.calculateActivationForceFunctionWindows(d=50, b1=0.75, b2=0.05, g=0.1, nonlinear_shape_factor=-1.5)
        #!

        # get data buffer size to calculate feature window indices
        db = self.EMG_live.getDataBuffer()
        feature_indices_windows = np.arange(db.shape[2]-self.property["feature_size"], db.shape[2], step = 1)

        # extract features from window
        self.EMG_live.featureExtractionFromWindows(feature_type = "timepoints", feature_indices_windows = feature_indices_windows)

        # inp_emg = self.EMG_live.getFeatures()

        # calculate mean absolute value (MAV)
        inp_emg = self.EMG_live.calculateMAVFromFeatures(len(self.channel_names))

        # predict elbow torque
        self.MLP_model_e.predictTarget(data = inp_emg, classification=False, show_results = False, show_pred_time = False, eval_type = "old_online")
        
        # predict front torque
        self.MLP_model_f.predictTarget(data = inp_emg, classification=False, show_results = False, show_pred_time = False, eval_type = "old_online")
        
        # predict side torque
        self.MLP_model_s.predictTarget(data = inp_emg, classification=False, show_results = False, show_pred_time = False, eval_type = "old_online")

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
            'names': ['shoulder_joint_0','shoulder_joint_1','shoulder_joint_2','elbow_joint'],
            'elements': [{
                'position': float("nan"),
                'speed': float("nan"),
                'effort': np.mean(self.torque_out_s),
                # 'effort': 0.0,
                # 'raw': self.torque_out_s[-1],
                'raw': round((self.torque_out_s[-1] - self.torque_out_s[0])[0,0], 2),
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
                'raw': round((self.torque_out_f[-1] - self.torque_out_f[0])[0,0], 2),
                'acceleration': float("nan")
            },
            {
                'position': float("nan"),
                'speed': float("nan"),
                'effort': np.mean(self.torque_out_e),
                # 'raw': self.torque_out_e[-1],
                'raw': round((self.torque_out_e[-1] - self.torque_out_e[0])[0,0],2),
                'acceleration': float("nan")
            }],
            'time': {'microseconds': 0}
        })
        
        # t_end = time.perf_counter()
        # self.log(f"Time for hook calc: {t_end - t_start}")
        self.socket_sj0.send_string(str(round((self.torque_out_s[-1] - self.torque_out_s[0])[0,0], 2)))
        self.socket_sj2.send_string(str(round((self.torque_out_f[-1] - self.torque_out_f[0])[0,0], 2)))
        self.socket_elbow.send_string(str(round((self.torque_out_e[-1] - self.torque_out_e[0])[0,0],2)))  

        # self.save_traj_elbow.append(round((self.torque_out_e[-1] - self.torque_out_e[0])[0,0],2))
        # self.save_traj_front.append(round((self.torque_out_f[-1] - self.torque_out_f[0])[0,0],2))
        # self.save_traj_side.append(round((self.torque_out_s[-1] - self.torque_out_s[0])[0,0],2))

    def stop_hook(self):
        # self.traj_elbow = np.array(self.save_traj_elbow)
        # np.save("../traj_elbow_1.npy", self.traj_elbow)

        # self.traj_front = np.array(self.save_traj_front)
        # np.save("../traj_front_1.npy", self.traj_front)

        # self.traj_side = np.array(self.save_traj_side)
        # np.save("../traj_side_1.npy", self.traj_side)
        # self.log("Saved trajectories")
        return True

    def cleanup_hook(self):
        return True
    
    def error_hook(self):
        return True

    # create an operation callable via omniORB
    # @expose(return_type='int', arg_types=['/std/string'])
    # def dummy_operation(self, arg):
    #     self.log(arg)
    #     return 42
