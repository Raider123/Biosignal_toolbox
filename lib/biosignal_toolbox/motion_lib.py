# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import numpy as np
import warnings 
#from datetime import datetime
import os
import mne
import csv 
import pandas as pd 
from scipy.interpolate import interp1d
from biosignal_toolbox.time_series_lib import Timeseries


# *********************************************************************************
# ************************* Methods ***********************************************
# *********************************************************************************

class MotionData(Timeseries):
    """
    This class includes useful methods for the processing and visualization of Motion data such as from Qualisys motion tracking.

    Parameters
    ----------
    Timeseries : class
        The base timeseries class that includes most of the data processing methods for biosignals (e.g. filters for EMG and EEG etc.)
    """

    def __init__(self, format="qualisys_tsv", data_path=None, filename=None, f_samp=None, channel_names=None, header_rows=11, columns_to_skip=2): 

        """
        The constructor of the MotionData class. 
            
        Parameters
        ----------
        format: str, optional
            The format in which the data is loaded", by default "qualisys_tsv"
        data_path: str, optional
            The path where the data is stored, by default None
        filename: list of str, optional
            A list of filenames to loaded, currently only one set can be loaded at a time (single element in the list), by default None
        f_samp: int, optional
            The sampling rate of the EMG system in Hz, by default None
        channel_names: list of str, optional
            A list of strings with the channel names/marker names etc., by default None 
        header_rows: int, optional
            The number of lines in header of file, by default 11
        columns_to_skip: int, optional
            The number of columns in data to skip for the x axis of first channel to start, by default 2

        Attributes
        ----------
        raw_obj: mne raw object
            The mne raw object that is used to create the object. Only required for format type "RawObj".
        fsamp: float
            The sampling rate of the EMG system in Hz
        channel_names : list
            A list of channel names as strings, if not known from the data format.
        data: numpy ndarray
             The channel wise (raw) data as numpy array (shape: n_channel, n_sampels). 
        epochs: numpy ndarray
            The epoched data as numpy ndarray with shape (n_trials, n_channels, n_sampels).
        windows: numpy ndarray
            A numpy array with windowed data (shape: n_trials, n_channels, n_sampels, n_windows).
        events: numpy ndarray 
            The events (also called markers) in the data. The shape is: (indices, 0, eventnumber).
    
        Author
        ------
        Author: Niklas Kueper \n
        Last changed: 25.08.2025 (by Kartik Chari)
        """  
        
        # parameter 
        self.raw_obj = None
        self.f_samp = f_samp
        self.channel_names = channel_names
        # data structures 
        self.data = None
        self.epochs = None
        self.windows = None
        self.events = None # not provided by loaded data yet 
        
        # load data
        if(format == "qualisys_tsv"):
            warnings.warn("only one (first) dataset can be loaded currently! Ignoring if more than one filename is included in the list ... ")
            self.loadQualisysData(data_path, filename, header_rows, columns_to_skip)

            self.createMNERaw() #create mne raw object 

        else: 
            warnings.warn("No other data formats are currenty supported, terminating ... ")


        # print("data shape:", self.data.shape)
        super().__init__(f_samp=self.f_samp, channel_names = self.channel_names, raw_obj=self.raw_obj, events=self.events, data=self.data, epochs=self.epochs, windows=self.windows)


    def createMNERaw(self):
        """
        This method is used (internally) to create mne raw objects. 
        
        Author
        ------
        Author: Niklas Kueper \n
        Last changed: 17.04.2024 (by Niklas Kueper)
        """
        
        # create mne object 
        sfreq = self.f_samp  # Sampling frequency
        data = self.data # (channel, sampels)
        #times = np.arange(0, data.shape[1], 1/sfreq)  # 
        ch_types = ['misc'] * len(self.channel_names) # 
        info = mne.create_info(ch_names=self.channel_names, sfreq=sfreq, ch_types=ch_types)
        #scalings = {'emg': 1}
        raw = mne.io.RawArray(data[0:len(self.channel_names), :], info) # only pass the actual EMG channel 
        self.raw_obj = raw
        self.data = self.raw_obj.get_data() # data as numpy array in shape (channels, sampels)

    def loadQualisysData(self, data_path, filename, header_rows, columns_to_skip): 
        """
        This method loads the qualisys data into a numpy array and also extracts important information from header.

        Author
        ------
        Author: Niklas Kueper \n
        Last changed: 25.08.2025 (by Kartik Chari)
        """

        tsv_file = open(os.path.join(data_path, filename))
        qualisys_file = list(csv.reader(tsv_file, delimiter="\t"))
        self.f_samp = float(qualisys_file[3][1])
        qualisys_data = np.array(qualisys_file[header_rows:]) 
        qualisys_data = qualisys_data[:, columns_to_skip:]
        marker_names = qualisys_file[9][1:]
        tsv_file.close()
        
        channel_names = []
        # set axis parameters 
        for marker in marker_names:
            for axis in ["x", "y", "z"]:
                channel_names.append(marker+"_"+axis)
            
        self.channel_names = channel_names
        
        self.time_axis = np.arange(0, qualisys_data.shape[0], 1/self.f_samp)
        self.data = qualisys_data.T
         
    def interpQualisysData(self, kind='linear'):
        """
        This method detects zeroes and Nan in the qualisys data and replaces them with smooth linear interpolation

        Parameters
        ----------
        kind: str, optional
            Specifies kind of interpolation to use. The string can be one of 'linear', 'nearest', 'nearest-up’, 'zero', 'slinear', 'quadratic', 'cubic', 'previous', or 'next'. 'zero', 'slinear', 'quadratic', by default 'linear'.

        Author
        ------
        Author: Kartik Chari \n
        Last changed: 25.08.2025 (by Kartik Chari)
        """
        for row in range(self.data.shape[0]):
            # copy of selected data row
            data_arr = self.data[row].copy()
            # ref index of array elements
            idx = np.arange(len(data_arr))
            # find valid data points indices
            mask = (data_arr != 0) & ~np.isnan(data_arr)
            valid_idx = np.where(mask)[0]

            # if all values are missing or all values are present, return the array as it is
            if len(valid_idx) == 0 or len(valid_idx) == len(data_arr):
                continue
            # try using interp1d if there are missing datapoints. In case of ValueError, use linear interpolation
            try:
                interp_func = interp1d(x=valid_idx, y=data_arr[valid_idx], kind=kind, bounds_error=False, fill_value="extrapolate")
                data_arr[~mask] = interp_func(idx[~mask])
            except ValueError:
                warnings.warn("Not enough datapoints for the selected kind of interpolation requested. Shifting to linear instead.")
                interp_func = interp1d(x=valid_idx, y=data_arr[valid_idx], kind='linear', bounds_error=False, fill_value="extrapolate")
                data_arr[~mask] = interp_func(idx[~mask])
            
            self.data[row] = data_arr
    
    def calculateTorque(self, body_weight_kg=80, obj_weight_g=0):
        """
        This method calculates torque values for elbow and shoulder joints. The shoulder joint torque is decomposed into front shoulder and side shoulder torque using projection method.

        Parameters
        -----
        body_weight: int, optional
            Weight of the whole body in kg, by default 80
        obj_weight: int, optional
            Weight of the object held in hand in g, by default 0

        Author
        -----
        Author: Kartik Chari \n
        Last changed: 25.08.2025 (by Kartik Chari)
        """
        # joint sequence
        self.joint_names = np.array(["elbow", "shoulder_front", "shoulder_side"])
        # initialise arrays
        self.torque_out = np.empty((3,0))

        torque_elbow = 0
        torque_shoulder = 0
        torque_shoulder_front = 0
        torque_shoulder_side = 0

        #? get indices for relevant channels
        self.sr_idx = [self.channel_names.index(ch) for ch in ['s_r_x', 's_r_y']]
        self.sl_idx = [self.channel_names.index(ch) for ch in ['s_l_x', 's_l_y']]
        self.er_idx = [self.channel_names.index(ch) for ch in ['e_r_x', 'e_r_y']]
        self.wr_idx = [self.channel_names.index(ch) for ch in ['w_r_x', 'w_r_y']]
        
        # calculate side shoulder angle of right arm in rad.
        self.side_shoulder_ang_rad = self.calculateSideShoulderAngle_rad()
        # calculate perpendicular dist between elbow and load in m
        forearm_perp_dist_m = self.calculateForearmPerpDist_m()
        # calculte perpendicular distance between shoulder and elbow in m
        upperarm_perp_dist_m = self.calculateUpperArmPerpDist_m()
        # calculate total perpendicular distance between shoulder and load in m
        total_arm_perp_dist_m = upperarm_perp_dist_m + forearm_perp_dist_m
        # estimate forearm weight from body weight
        forearm_weight_kg = body_weight_kg * 0.016
        # estimate full arm weight from body weight
        arm_weight_kg = body_weight_kg * 0.05

        #? torque calculation
        # elbow torque = mass * g * perp_dist
        torque_elbow = float((obj_weight_g/1000) + forearm_weight_kg) * 9.81 * forearm_perp_dist_m
        # total shoulder torque
        torque_shoulder = float((obj_weight_g/1000) + arm_weight_kg) * 9.81 * total_arm_perp_dist_m
        #? project total shoulder force into axes of saggital and frontal planes
        #* T_{front} = |t_{total}| cos(theta)
        torque_shoulder_front = torque_shoulder * np.cos(self.side_shoulder_ang_rad)
        #* T_{side} = |t_{total}| cos(phi)
        if self.side_shoulder_ang_rad <= np.pi/2:
            torque_shoulder_side = torque_shoulder * np.cos(np.pi/2 - self.side_shoulder_ang_rad)
        else:
            torque_shoulder_side = torque_shoulder * np.cos(-np.pi/2 + self.side_shoulder_ang_rad)
        
        self.torque_out[0] = np.append(self.torque_out[0],torque_elbow)
        self.torque_out[1] = np.append(self.torque_out[1],torque_shoulder_front)
        self.torque_out[2] = np.append(self.torque_out[2],torque_shoulder_side)

    def calculateSideShoulderAngle_rad(self):
        """
        This method calculates the angle between the imaginery x axis and the right arm (shoulder -> elbow) in rad.

        Returns
        -----
        float
            side shoulder angle in rad.

        Author
        -----
        Author: Kartik Chari \n
        Last changed: 26.08.2025 (by Kartik Chari)
        """
        # calculate shoulder to elbow vector -> right arm
        s_e_xy = np.array([self.data[self.er_idx[0],:] - self.data[self.sr_idx[0],:], self.data[self.er_idx[1],:] - self.data[self.sr_idx[1],:]])
        # calculate shoulder to shoulder ref vector -> project right to left shoulder
        s_rl_xy = np.array([self.data[self.sl_idx[0],:] - self.data[self.sr_idx[0],:], self.data[self.sl_idx[1],:] - self.data[self.sr_idx[1],:]])
        # y-axis parallel to ground (right arm)
        y_axis_normalised, _ = self.normalise_vector(np.array([s_rl_xy[0], s_rl_xy[1], 0]))
        # z-axis global up
        z_axis = np.array([0, 0, 1])
        # x-axis -> cross product between y and z axes
        x_axis_normalised, _ = self.normalise_vector(np.cross(y_axis_normalised, z_axis))
        x_axis_normalised_xy = np.array([x_axis_normalised[0], x_axis_normalised[1]])
        # calculate angle between sagttal axis and arm
        cos_angle = np.clip(np.dot(s_e_xy, x_axis_normalised_xy), -1.0, 1.0)
        return np.acos(cos_angle)
    
    def calculateElbowAngle_rad(self):
        """
        This method calculates the angle between the upperarm and forearm vectors in rad.

        Returns
        -----
        float
            elbow angle in rad.

        Author
        -----
        Author: Kartik Chari \n
        Last changed: 26.08.2025 (by Kartik Chari)
        """
        # calculate elbow to shoulder vector -> right arm
        e_s_xy = np.array([self.data[self.sr_idx[0],:] - self.data[self.er_idx[0],:], self.data[self.sr_idx[1],:] - self.data[self.er_idx[1],:]])
        e_s_xy_normalised, self.upperarm_euclidean_dist_m = self.normalise_vector(e_s_xy)
        # calculate elbow to wrist vector -> right arm
        e_w_xy = np.array([self.data[self.wr_idx[0],:] - self.data[self.er_idx[0],:], self.data[self.wr_idx[1],:] - self.data[self.er_idx[1],:]])
        e_w_xy_normalised, self.forearm_euclidean_dist_m = self.normalise_vector(e_w_xy)
        # calculate elbow angle
        cos_angle = np.clip(np.dot(e_s_xy_normalised, e_w_xy_normalised), -1, 1)
        return np.acos(cos_angle)

    def calculateForearmPerpDist_m(self):
        """
        This method calculates the perpendicular distance between elbow and load in hand in m.

        Returns
        -----
        float
            perpendicular between elbow and load in hand in m.

        Author
        -----
        Author: Kartik Chari \n
        Last changed: 26.08.2025 (by Kartik Chari)
        """
        # calculate the elbow angle in rad and euclidean lengths of forearm and upper arm in m. (arm lengths are self attributes)
        self.elbow_angle_rad = self.calculateElbowAngle_rad()
        # calculate the side shoulder angle in rad. Check if it is already created before and only call the function otherwise
        if not hasattr(self, 'side_shoulder_ang_rad'):
            self.side_shoulder_ang_rad = self.calculateSideShoulderAngle_rad()
        # return the forearm perpendicular dist in m
        return self.forearm_euclidean_dist_m * np.sin(self.elbow_angle_rad - self.side_shoulder_ang_rad)

    def calculateUpperArmPerpDist_m(self):
        """
        This method calcualtes the perpendicular distance between shoulder and elbow in m.

        Returns
        -------
        float
            perpendicular distance between shoulder and elbow in m.

        Author
        -----
        Author: Kartik Chari \n
        Last changed: 26.08.2025 (by Kartik Chari)
        """
        # calculate the side shoulder angle in rad. Check if it is already created before and only call the function otherwise
        if not hasattr(self, 'side_shoulder_ang_rad'):
            self.side_shoulder_ang_rad = self.calculateSideShoulderAngle_rad()
        # return the upperarm perpendicular dist in m
        return self.upperarm_euclidean_dist_m * np.sin(self.side_shoulder_ang_rad)

    @staticmethod
    def normalise_vector(inp_vec):
        """
        This method returns unit vector in the direction of input vector and also handles zero safely.

        Parameters
        -----
        inp_vec: array
            Input array to be normalised
        
        Author
        -----
        Author: Kartik Chari \n
        Last changed: 26.08.2025 (by Kartik Chari)
        """
        norm = np.linalg.norm(inp_vec)
        if norm == 0:
            warnings.warn("zero vector! Not normalising!")
            return inp_vec
        return inp_vec / norm, norm