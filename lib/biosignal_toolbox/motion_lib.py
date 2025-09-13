
#! ************************************************
#! Imports
#! ************************************************
import numpy as np
from numpy.typing import ArrayLike
import warnings 
from typing import List
import mne, csv, re
from pathlib import Path
from scipy.interpolate import interp1d
from biosignal_toolbox.utils import getAbsolutePath
from biosignal_toolbox.time_series_lib import Timeseries

#! ************************************************
#! Methods
#! ************************************************

class MotionData(Timeseries):
    """
    This class includes useful methods for the processing and visualization of Motion data such as from Qualisys motion tracking.

    Parameters
    ----------
    Timeseries : class
        The base timeseries class that includes most of the data processing methods for biosignals (e.g. filters for EMG and EEG etc.)
    """

    def __init__(self, format: str = "qualisys_tsv", data_path: str = None, filename: List[str] = None, f_samp: int = None, channel_names: List[str] = None, header_rows: int = 11, columns_to_skip: int = 2) -> None: 

        """
        The constructor of the MotionData class. 
            
        Parameters
        ----------
        format : str, optional
            The format in which the data is loaded", by default "qualisys_tsv"
        data_path : str, optional
            The path where the data is stored, by default None
        filename : list of str, optional
            A list of filenames to loaded, currently only one set can be loaded at a time (single element in the list), by default None
        f_samp : int, optional
            The sampling rate of the EMG system in Hz, by default None
        channel_names : list of str, optional
            A list of strings with the channel names/marker names etc., by default None 
        header_rows : int, optional
            The number of lines in header of file, by default 11
        columns_to_skip : int, optional
            The number of columns in data to skip for the x axis of first channel to start, by default 2

        Attributes
        ----------
        raw_obj : mne raw object
            The mne raw object that is used to create the object. Only required for format type "RawObj"
        fsamp : float
            The sampling rate of the EMG system in Hz
        channel_names : list
            A list of channel names as strings, if not known from the data format
        data : numpy ndarray
             The channel wise (raw) data as numpy array (shape: n_channel, n_sampels)
        epochs : numpy ndarray
            The epoched data as numpy ndarray with shape (n_trials, n_channels, n_sampels)
        windows : numpy ndarray
            A numpy array with windowed data (shape: n_trials, n_channels, n_sampels, n_windows)
        events : numpy ndarray 
            The events (also called markers) in the data. The shape is: (indices, 0, eventnumber)
    
        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 25.08.2025 (by Kartik Chari)
        """  
        
        # parameter 
        self.raw_obj = None
        self.f_samp = f_samp
        self.channel_names = channel_names
        self.filename = filename
        # data structures 
        self.data = None
        self.epochs = None
        self.windows = None
        self.events = None # not provided by loaded data yet 

        self.data_path = getAbsolutePath(input_path=data_path)
        # load data
        if(format == "qualisys_tsv"):
            warnings.warn("only one (first) dataset can be loaded currently! Ignoring if more than one filename is included in the list ... ")
            self.loadQualisysData(header_rows, columns_to_skip)

            self.createMNERaw() #create mne raw object 

        else: 
            warnings.warn("No other data formats are currenty supported, terminating ... ")


        # print("data shape:", self.data.shape)
        super().__init__(f_samp=self.f_samp, channel_names = self.channel_names, raw_obj=self.raw_obj, events=self.events, data=self.data, epochs=self.epochs, windows=self.windows)


    def createMNERaw(self) -> None:
        """
        This method is used (internally) to create mne raw objects. 
        
        Author
        ------
        Author : Niklas Kueper \n
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

    def loadQualisysData(self, header_rows: int = 11, columns_to_skip: int = 2) -> None: 
        """
        This method loads the quali data into a numpy array and also extracts important information from header.

        Parameters
        ----------
        header_rows : int, optional 
            The number of lines in header of file, by default 11
        columns_to_skip : int, optional
            The number of columns in data to skip for the x axis of first channel to start, by default 2
        
        Author
        ------
        Author : Niklas Kueper \n
        Last changed : 25.08.2025 (by Kartik Chari)
        """
        tsv_file = open(self.data_path / Path(self.filename))
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
        print(self.channel_names)
        
        self.time_axis = np.arange(0, 1/self.f_samp*qualisys_data.shape[0], 1/self.f_samp)
        # print(self.time_axis)
        self.data = qualisys_data.T
         
    def interpQualisysData(self, kind: str = 'linear') -> None:
        """
<<<<<<< HEAD
        This method detects zeroes and Nan in the quali data and replaces them with smooth linear interpolation.
=======
        This method detects zeroes and Nan in the qualisys data and replaces them with smooth linear interpolation.
>>>>>>> 306e0a815e9ebbcf68f8bb2e035814839e083b34

        Parameters
        ----------
        kind : str, optional
            Specifies kind of interpolation to use. The string can be one of 'linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', or 'next'. 'zero', 'slinear', 'quadratic', and 'cubic' refer to a spline interpolation of zeroth, first, second or third order; 'previous' and 'next' simply return the previous or next value of the point; 'nearest-up' and 'nearest' differ when interpolating half-integers (e.g. 0.5, 1.5) in that 'nearest-up' rounds up and 'nearest' rounds down, by default 'linear'

        Author
        ------
        Author : Kartik Chari \n
        Last changed : 25.08.2025 (by Kartik Chari)
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
    
    def calculateTorque(self, body_weight_kg: int = None, obj_weight_g: int = None, subject_hand_length_mm:int = None, subject_biological_sex: str = "male", method: str = "com") -> None:
        """
        This method calculates torque values for elbow and shoulder joints. The shoulder joint torque is then decomposed into front shoulder and side shoulder torque using projection. It uses 2 methods - "com" and "end_to_end". The "com" method is more accurate and considers the distance to the centre of mass of the segment for torque calculation instead of the full arm length.

        Parameters
        ----------
        body_weight_kg : int, optional
            Weight of the whole body in kg, by default None
        obj_weight_g : int, optional
            Weight of the object held in hand in g, by default None
        subject_hand_length_mm : int, optional
            Length of the hand between the wrist and 3rd knuckle in mm, by default None
        subject_biological_sex : str, optional
            Biological sex of the subject. The available options are "male" and "female", by default "male"
        method : str, optional
            The method to use for torque calculation. Options available are "com" and "end_to_end", by default "com"

        Raises
        ------
        ValueError
            Raised if obj_weight_g is None
        ValueError
            Raised if subject_hand_length_mm is None
<<<<<<< HEAD
        ValueError
            Raised if self.channel_names don't match either shoulder_r_x or s_r_x style
=======
>>>>>>> 306e0a815e9ebbcf68f8bb2e035814839e083b34
        
        Author
        -----
        Author : Kartik Chari \n
        Last changed : 03.09.2025 (by Kartik Chari)
        """
        
        if body_weight_kg is None:
            body_weight_kg = 80
            warnings.warn("Body weight not provided for Torque calculation. Using 80 kg for calculations!!")
        if obj_weight_g is None:
            raise ValueError("Please provide object weight in g for Torque calculation!!")
        if subject_hand_length_mm is None:
            raise ValueError("Please provide hand length in mm for Torque calculation!!")
        
        # joint sequence
        self.joint_names = np.array(["elbow", "shoulder_front", "shoulder_side"])
        # initialise arrays
        self.torque_out = np.empty((3,0))

        torque_elbow = 0
        torque_shoulder = 0
        torque_shoulder_front = 0
        torque_shoulder_side = 0

        #? get indices for relevant channels
<<<<<<< HEAD
        if 'shoulder_r_x' in self.channel_names:
            self.sr_idx = [self.channel_names.index(ch) for ch in ['shoulder_r_x', 'shoulder_r_y', 'shoulder_r_z']]
            self.sl_idx = [self.channel_names.index(ch) for ch in ['shoulder_l_x', 'shoulder_l_y', 'shoulder_l_z']]
            self.er_idx = [self.channel_names.index(ch) for ch in ['elbow_r_x', 'elbow_r_y', 'elbow_r_z']]
            self.wr_idx = [self.channel_names.index(ch) for ch in ['wrist_r_x', 'wrist_r_y', 'wrist_r_z']]
        elif 's_r_x' in self.channel_names:
            self.sr_idx = [self.channel_names.index(ch) for ch in ['s_r_x', 's_r_y', 's_r_z']]
            self.sl_idx = [self.channel_names.index(ch) for ch in ['s_l_x', 's_l_y', 's_l_z']]
            self.er_idx = [self.channel_names.index(ch) for ch in ['e_r_x', 'e_r_y', 'e_r_z']]
            self.wr_idx = [self.channel_names.index(ch) for ch in ['w_r_x', 'w_r_y', 'w_r_z']]
        else:
            raise ValueError("Channel names don't match our library pattern -> 'shoulder_r_x' or 's_r_x'!!")
=======
        self.sr_idx = self.getChannelIndex(joint_name='shoulder', hand='right')
        self.sl_idx = self.getChannelIndex(joint_name='shoulder', hand='left')
        self.er_idx = self.getChannelIndex(joint_name='elbow', hand='right')
        self.wr_idx = self.getChannelIndex(joint_name='wrist', hand='right')
>>>>>>> 306e0a815e9ebbcf68f8bb2e035814839e083b34

        #? calculate perpendicular distances
        # calculte perpendicular distance between shoulder and elbow in mm
        upperarm_perp_dist_mm = self.calculateUpperArmPerpDist_mm(subject_biological_sex=subject_biological_sex,
                                                                  method=method)
        # calculte perpendicular distance between elbow and wrist in mm
        forearm_perp_dist_mm = self.calculateForearmPerpDist_mm(subject_biological_sex=subject_biological_sex,
                                                                method=method)
        # calculte perpendicular distance between wrist and object in hand in mm
        hand_perp_dist_mm = self.calculateHandPerpDist_mm(subject_biological_sex=subject_biological_sex, 
                                                          subject_hand_length_mm=subject_hand_length_mm, 
                                                          method=method)
        #? calculate segment weights
        # upperarm weight
        upperarm_weight_kg = self.getSegmentWeight_kg(body_weight_kg=body_weight_kg, 
                                                      segment_name="upperarm", 
                                                      subject_biological_sex=subject_biological_sex)
        # forearm weight
        forearm_weight_kg = self.getSegmentWeight_kg(body_weight_kg=body_weight_kg, 
                                                     segment_name="forearm", 
                                                     subject_biological_sex=subject_biological_sex)
        # hand weight
        hand_weight_kg = self.getSegmentWeight_kg(body_weight_kg=body_weight_kg, 
                                                  segment_name="hand", 
                                                  subject_biological_sex=subject_biological_sex)
        #? End to End segment length method
        if method == "end_to_end":
            # calculate total perpendicular distance between shoulder and load in mm
            whole_arm_perp_dist_mm = upperarm_perp_dist_mm + forearm_perp_dist_mm + hand_perp_dist_mm
            # full arm weight
            whole_arm_weight_kg = upperarm_weight_kg + forearm_weight_kg + hand_weight_kg
            #? torque calculation
            # elbow torque = mass * g * perp_dist
            torque_elbow = (float(forearm_weight_kg + hand_weight_kg + (obj_weight_g/1000)) 
                            * 9.81 
                            * (forearm_perp_dist_mm + hand_perp_dist_mm)/1000)
            # total shoulder torque
            torque_shoulder = (float((obj_weight_g/1000) + whole_arm_weight_kg) 
                               * 9.81 
                               * whole_arm_perp_dist_mm/1000)
        #? COM method
        elif method == "com":
            #? elbow torque
            forearm_perp_dist_end_to_end_mm = self.calculateForearmPerpDist_mm(subject_biological_sex=subject_biological_sex,
                                                                               method="end_to_end")
            tau_forearm_e = (forearm_weight_kg 
                           * 9.81 
                           * forearm_perp_dist_mm/1000)
            tau_hand_e = (float((obj_weight_g/1000) + hand_weight_kg) 
                        * 9.81 
                        * (forearm_perp_dist_end_to_end_mm + hand_perp_dist_mm)/1000)
            
            torque_elbow = tau_forearm_e + tau_hand_e

            #? shoulder torque
            upperarm_perp_dist_end_to_end_mm = self.calculateUpperArmPerpDist_mm(subject_biological_sex=subject_biological_sex, 
                                                                                 method="end_to_end")
            tau_upperarm_s = (upperarm_weight_kg 
                              * 9.81 
                              * upperarm_perp_dist_mm/1000)
            tau_forearm_s = (tau_forearm_e + (forearm_weight_kg 
                                              * 9.81 
                                              * upperarm_perp_dist_end_to_end_mm/1000))
            tau_hand_s = (tau_hand_e + (float((obj_weight_g/1000) + hand_weight_kg) 
                                        * 9.81 
                                        * upperarm_perp_dist_end_to_end_mm/1000))
            
            torque_shoulder = tau_upperarm_s + tau_forearm_s + tau_hand_s

        #? project total shoulder force into axes of saggital and frontal planes
        #* calculate the side shoulder angle in rad. 
        # Check if it is already created before and only call the function otherwise
        if not hasattr(self, 'side_shoulder_ang_rad'):
            self.side_shoulder_ang_rad = self.calculateSideShoulderAngle_rad()
        #* T_{front} = |t_{total}| cos(theta)
        torque_shoulder_front = torque_shoulder * np.cos(self.side_shoulder_ang_rad)
        #* T_{side} = |t_{total}| cos(phi)
        torque_shoulder_side = torque_shoulder * np.sin(self.side_shoulder_ang_rad)
        
        self.torque_out = np.hstack((self.torque_out, 
                                     np.array([np.array(torque_elbow), 
                                               np.array(torque_shoulder_front), 
                                               np.array(torque_shoulder_side)])))
        print("Torques calculated!!")

    def calculateSideShoulderAngle_rad(self) -> float:
        """
        This method calculates the angle between the imaginery x axis (shoulder -> ground) and the right arm (shoulder -> elbow) in rad.

        Returns
        -------
        float
            Side shoulder angle in rad
        
        Raises
        ------
        ValueError
            Raised if shapes of point1 and point2 do not match

        Author
        ------
        Author : Kartik Chari \n
        Last changed : 26.08.2025 (by Kartik Chari)
        """
        # calculate shoulder to elbow vector -> right arm
        s_e_vector = np.array([self.data[self.er_idx[0],:] - self.data[self.sr_idx[0],:], 
                               self.data[self.er_idx[1],:] - self.data[self.sr_idx[1],:],
                               self.data[self.er_idx[2],:] - self.data[self.sr_idx[2],:]])
        s_e_vector_normalised = self.normalise_vector(s_e_vector)

        # calculate shoulder to shoulder ref vector -> project right to left shoulder
        s_rl_vector = np.array([self.data[self.sl_idx[0],:] - self.data[self.sr_idx[0],:], 
                                self.data[self.sl_idx[1],:] - self.data[self.sr_idx[1],:],
                                self.data[self.sl_idx[2],:] - self.data[self.sr_idx[2],:]])
        # y-axis parallel to ground (right arm)
        s_rl_vector_normalised = self.normalise_vector(s_rl_vector)

        # check if the 2 points have the same shape
        if s_e_vector_normalised.shape != s_rl_vector_normalised.shape:
            raise ValueError(f"Shapes of point1 {s_e_vector_normalised.shape} and point2 {s_rl_vector_normalised.shape} do not match!!")
        # check shape and choose axis
        if s_rl_vector_normalised.shape[0] == 3 and s_rl_vector_normalised.shape[1] > 1:
            axis = 0
        else:
            axis = 1
        
        # calculate angle between horizontal axis and arm
        cos_angle = np.clip(np.sum(s_e_vector_normalised * s_rl_vector_normalised, axis=axis), -1.0, 1.0)
        return np.arccos(cos_angle) - (np.pi/2)
    
    def calculateElbowAngle_rad(self) -> float:
        """
        This method calculates the angle between the upperarm and forearm vectors in rad.

        Returns
        ------
        float
            Elbow angle in rad

        Author
        ------
        Author : Kartik Chari \n
        Last changed : 26.08.2025 (by Kartik Chari)
        """
        # calculate elbow to shoulder vector -> right arm
        e_s_vector = np.array([self.data[self.sr_idx[0],:] - self.data[self.er_idx[0],:], 
                               self.data[self.sr_idx[1],:] - self.data[self.er_idx[1],:],
                               self.data[self.sr_idx[2],:] - self.data[self.er_idx[2],:]])
        e_s_vector_normalised = self.normalise_vector(e_s_vector)

        # calculate elbow to wrist vector -> right arm
        e_w_vector = np.array([self.data[self.wr_idx[0],:] - self.data[self.er_idx[0],:], 
                               self.data[self.wr_idx[1],:] - self.data[self.er_idx[1],:],
                               self.data[self.wr_idx[2],:] - self.data[self.er_idx[2],:]])
        e_w_vector_normalised = self.normalise_vector(e_w_vector)

        # check if the 2 points have the same shape
        if e_s_vector_normalised.shape != e_w_vector_normalised.shape:
            raise ValueError(f"Shapes of point1 {e_s_vector_normalised.shape} and point2 {e_w_vector_normalised.shape} do not match!!")
        # check shape and choose axis
        if e_w_vector_normalised.shape[0] == 3 and e_w_vector_normalised.shape[1] > 1:
            axis = 0
        else:
            axis = 1
        
        # calculate elbow angle
        cos_angle = np.clip(np.sum(e_s_vector_normalised * e_w_vector_normalised, axis=axis), -1, 1)
        return np.arccos(cos_angle)

    def calculateUpperArmPerpDist_mm(self, subject_biological_sex: str = "male", method: str = "com") -> float:
        """
        This method calculates the perpendicular distance between shoulder and elbow in mm.

        Parameters
        ----------
        subject_biological_sex : str, optional
            Biological sex of the subject. The available options are "male" and "female", by default "male"
        method : str, optional
            The method to use for torque calculation. Options available are "com" and "end_to_end", by default "com"

        Returns
        -------
        float
            Perpendicular distance between shoulder and elbow in mm

        Raises
        ------
        ValueError
            Raised if method is outside ["com", "end_to_end"]
        
        Author
        ------
        Author : Kartik Chari \n
        Last changed : 02.09.2025 (by Kartik Chari)    
        """
        if not hasattr(self, 'side_shoulder_ang_rad'):
            self.side_shoulder_ang_rad = self.calculateSideShoulderAngle_rad()
        if not hasattr(self, 'upperarm_euclidean_dist_mm'):
            self.upperarm_euclidean_dist_mm = self.getEuclideanDistance_mm(joint_idx1=self.sr_idx, 
                                                                           joint_idx2=self.er_idx)
        # check which method to use for calculating the forearm perp dist
        if method == "end_to_end":
            return self.upperarm_euclidean_dist_mm * np.sin(self.side_shoulder_ang_rad)
        elif method == "com":
            upperarm_com = (self.getCOM_percent(segment_name="upperarm", subject_biological_sex=subject_biological_sex) 
                            / 100 
                            * self.upperarm_euclidean_dist_mm)
            return upperarm_com * np.sin(self.side_shoulder_ang_rad)
        else:
            raise ValueError(f"Invalid method selected: {method}. Please choose either \"end_to_end\" or \"com\"!!")
    
    def calculateForearmPerpDist_mm(self, subject_biological_sex: str = "male", method: str = "end_to_end") -> float:
        """
        This method calculates the perpendicular distance between elbow and wrist in mm.

        Parameters
        ----------
        subject_biological_sex : str, optional
            Biological sex of the subject. The available options are "male" and "female", by default "male"
        method : str, optional
            The method to use for torque calculation. Options available are "com" and "end_to_end", by default "com"

        Returns
        -------
        float
            Perpendicular between elbow and wrist in mm
        
        Raises
        ------
        ValueError
            Raised if method is outside ["com", "end_to_end"]

        Author
        ------
        Author : Kartik Chari \n
        Last changed : 02.09.2025 (by Kartik Chari)
        """
        # check if the attribute has been created; if not call the resp. function
        if not hasattr(self, 'elbow_angle_rad'):
            self.elbow_angle_rad = self.calculateElbowAngle_rad()
        if not hasattr(self, 'side_shoulder_ang_rad'):
            self.side_shoulder_ang_rad = self.calculateSideShoulderAngle_rad()
        if not hasattr(self,'forearm_euclidean_dist_mm'):
            self.forearm_euclidean_dist_mm = self.getEuclideanDistance_mm(joint_idx1=self.er_idx, 
                                                                          joint_idx2=self.wr_idx)
        # check which method to use for calculating the forearm perp dist
        if method == "end_to_end":
            return self.forearm_euclidean_dist_mm * np.sin(self.elbow_angle_rad - self.side_shoulder_ang_rad)
        elif method == "com":
            forearm_com = (self.getCOM_percent(segment_name="forearm", subject_biological_sex=subject_biological_sex) 
                           / 100 
                           * self.forearm_euclidean_dist_mm)
            return  forearm_com * np.sin(self.elbow_angle_rad - self.side_shoulder_ang_rad)
        else:
            raise ValueError(f"Invalid method selected: {method}. Please choose either \"end_to_end\" or \"com\"!!")
        
    def calculateHandPerpDist_mm(self, subject_hand_length_mm: int = 0, subject_biological_sex: str = "male", method: str = "end_to_end") -> float:
        """
        This method calculates the perpendicular distance between wrist and load in hand in mm.

        Parameters
        ----------
        subject_biological_sex : str, optional
            Biological sex of the subject. The available options are "male" and "female", by default "male"
        method : str, optional
            The method to use for torque calculation. Options available are "com" and "end_to_end", by default "com"

        Returns
        -------
        float
            Perpendicular between wrist and load in hand in mm
        
        Raises
        ------
        ValueError
            Raised if method is outside ["com", "end_to_end"]

        Author
        ------
        Author : Kartik Chari \n
        Last changed : 02.09.2025 (by Kartik Chari)
        """
        # check if the attribute has been created; if not call the resp. function
        if not hasattr(self, 'elbow_angle_rad'):
            self.elbow_angle_rad = self.calculateElbowAngle_rad()
        if not hasattr(self, 'side_shoulder_ang_rad'):
            self.side_shoulder_ang_rad = self.calculateSideShoulderAngle_rad()

        # check which method to use for calculating the forearm perp dist
        if method == "end_to_end":
            return subject_hand_length_mm / 2 * np.sin(self.elbow_angle_rad - self.side_shoulder_ang_rad)
        elif method == "com":
            hand_com = (self.getCOM_percent(segment_name="hand", subject_biological_sex=subject_biological_sex) 
                        / 100 
                        * subject_hand_length_mm)
            return hand_com * np.sin(self.elbow_angle_rad - self.side_shoulder_ang_rad)
        else:
            raise ValueError(f"Invalid method selected: {method}. Please choose either \"end_to_end\" or \"com\"!!")
    
    def getEuclideanDistance_mm(self, joint_idx1: int, joint_idx2: int) -> float:
        """
        This method calculates the euclidean distance between 2 joints

        Parameters
        ----------
        joint_idx1 : int
            Index of the first joint
        joint_idx2 : int
            Index of the second joint

        Returns
        -------
        float
            Euclidean dist between 2 joints

        Raises
        ------
        ValueError
            Raised if any of the two joint indices are not provided
        ValueError
            Raised if shapes of point1 and point2 do not match
        
        Author
        ------
        Author : Kartik Chari \n
        Last changed : 03.09.2025 (by Kartik Chari)
        """
        if not joint_idx1 or not joint_idx2:
            raise ValueError("Please provide 2 joint indices to get the length of the vector!!")
        
        point1 = np.array([self.data[joint_idx1[0],:], 
                           self.data[joint_idx1[1],:],
                           self.data[joint_idx1[2],:]])
        
        point2 = np.array([self.data[joint_idx2[0],:], 
                           self.data[joint_idx2[1],:],
                           self.data[joint_idx2[2],:]])

        # check if the 2 points have the same shape
        if point1.shape != point2.shape:
            raise ValueError(f"Shapes of point1 {point1.shape} and point2 {point2.shape} do not match!!")

        # check shape and choose axis
        if point1.shape[0] == 3 and point1.shape[1] > 1:
            axis = 0
        else:
            axis = 1

        return np.linalg.norm(point2-point1, axis=axis)
    
    @staticmethod
    def getSegmentWeight_kg(body_weight_kg: int = None, segment_name: str = None, subject_biological_sex: str ="male") -> float:
        """
        This static method first creates a dictionary with the relation between body weight and upperarm segment weights for both males and females. Then it returns the weight of the requested segment in kg.

        Reference
        ---------
        P.  de  Leva,  "Adjustments  to  Zatsiorsky-Seluyanov's  segment  inertia parameters," Journal  of Biomechanics, vol. 29, no.  9, pp. 1223-1230, 1996, doi: 10.1016/0021-9290(95)00178-6. 

        Parameters
        ----------
        body_weight_kg : int, optional
            Weight of the whole body in kg, by default 80
        segment_name : str, optional
            Name of the segment whose weight is requested. The available options are "forearm", "upperarm", "hand", by default ""
        subject_biological_sex : str, optional
            Biological sex of the subject. The available options are "male" and "female", by default ""

        Returns
        -------
        float
            Weight of the segment in kg
        
        Raises
        ------
        ValueError
            Raised if the segment_name is not in the REF_DICT
        
        Author
        ------
        Author : Kartik Chari \n
        Last changed : 03.09.2025 (by Kartik Chari)
        """
        REF_DICT = {
            'male': {
                'upperarm': 0.0271,     #2.71% of body weight
                'forearm':  0.0162,     #1.62% of body weight
                'hand':     0.0061      #0.61% of body weight
            },
            'female': {
                'upperarm': 0.0255,     #2.55% of body weight
                'forearm':  0.0138,     #1.38% of body weight
                'hand':     0.0056      #0.56% of body weight
            }
        }

        #? convert input str into lower cases
        subject_biological_sex = subject_biological_sex.strip().lower()
        segment_name = segment_name.strip().lower()

        if subject_biological_sex not in REF_DICT:
            warnings.warn(f"Invalid sex: {subject_biological_sex}. Must be 'male' or 'female'. Using 'male' for calculations now!!")
        if segment_name not in REF_DICT[subject_biological_sex]:
            raise ValueError(f"Invalid segment: {segment_name}. one of {list(REF_DICT[subject_biological_sex].keys())}!!")
        if body_weight_kg is None:
            body_weight_kg = 80
            warnings.warn("Body weight not provided for Torque calculation. Using 80 kg for calculations!!")
        
        return body_weight_kg * REF_DICT[subject_biological_sex][segment_name]
    
    @staticmethod
    def getCOM_percent(segment_name: str = None, subject_biological_sex: str = "male") -> float:
        """
        This static method refers to the reference dictionary and returns the distance of the COM of the arm segment from the origin joint in percentage. For upperarm, origin is shoulder joint. For forearm, origin is elbow. For hand, origin is wrist.

        Reference
        ---------
        P.  de  Leva,  "Adjustments  to  Zatsiorsky-Seluyanov's  segment  inertia parameters," Journal  of Biomechanics, vol. 29, no.  9, pp. 1223-1230, 1996, doi: 10.1016/0021-9290(95)00178-6.

        Parameters
        ----------
        segment_name : str, optional
            Name of the segment whose weight is requested. The available options are "forearm", "upperarm", by default None
        subject_biological_sex : str, optional
            Biological sex of the subject. The available options are "male" and "female", by default "male"

        Returns
        -------
        float
            Distance of COM of segment from origin in percentage 

        Raises
        ------
        ValueError
            Raised if the segment_name is not in the REF_DICT
        
        Author
        ------
        Author : Kartik Chari \n
        Last changed : 02.09.2025 (by Kartik Chari)
        """
        REF_DICT = {
            'male': {
                'upperarm': 57.72,     #% of forearm dist. from elbow
                'forearm':  45.74,     #% of forearm dist. from elbow
                'hand':     79.00      #% of forearm dist. from elbow
            },
            'female': {
                'upperarm': 57.54,     #% of forearm dist. from elbow
                'forearm':  45.59,     #% of forearm dist. from elbow
                'hand':     74.74      #% of forearm dist. from elbow
            }
        }

        #? convert input str into lower cases
        subject_biological_sex = subject_biological_sex.strip().lower()
        segment_name = segment_name.strip().lower()

        if subject_biological_sex not in REF_DICT:
            warnings.warn(f"Invalid sex: {subject_biological_sex}. Must be 'male' or 'female'. Using 'male' for calculations now!!")
        if segment_name not in REF_DICT[subject_biological_sex]:
            raise ValueError(f"Invalid segment: {segment_name}. one of {list(REF_DICT[subject_biological_sex].keys())}!!")
        
        return REF_DICT[subject_biological_sex][segment_name]
    
    @staticmethod
    def normalise_vector(inp_vec: ArrayLike) -> ArrayLike:
        """
        This method returns unit vector in the direction of input vector and also handles zero safely.

        Parameters
        ----------
        inp_vec: array_like
            Input array to be normalised
        
        Returns
        -------
        array_like
            Normalised unit vector
        
        Author
        -----
        Author : Kartik Chari \n
        Last changed : 03.09.2025 (by Kartik Chari)
        """
        inp_vec = np.array(inp_vec)
        # check shape and choose axis
        if inp_vec.shape[0] == 3 and inp_vec.shape[1] > 1:
            axis = 0
        else:
            axis = 1

        norm = np.linalg.norm(inp_vec, axis=axis)
        if (norm == 0).any():
            warnings.warn("zero vector! Not normalising!")
            return inp_vec
        return inp_vec / norm
    
<<<<<<< HEAD
    def saveTorques_npy(self, save_torques=False, save_dir=None, joints_to_save=['all']):
=======
    def getChannelIndex(self, joint_name: str = None, hand: str = None) -> list:
        """
        This methods outputs the channel indices for the requested joint and hand as a list [x_idx, y_idx, z_idx] with the help of regex.

        Parameters
        ----------
        joint_name : str, optional
            Name of the joint, by default None
        hand : str, optional
            Name of the hand/side out of 'right' or 'left', by default None

        Returns
        -------
        list
            List of x,y,z indices of the requested joint and side

        Raises
        ------
        ValueError
            Raised if joint_name or hand is None
        ValueError
            Raised if hand.lower() is not 'right' or 'left'
        
        Author
        -----
        Author : Kartik Chari \n
        Last changed : 04.09.2025 (by Kartik Chari)
        """
        if joint_name is None or hand is None:
            raise ValueError("Please enter joint_name and hand to proceed!!")
        if hand.lower() not in ['left', 'right']:
            raise ValueError("Invalid hand arg... hand must be \"left\" or \"right\"!!")
        idx_list = []
        # short hand for hand
        hand_code = 'r' if hand.lower() == 'right' else 'l'
        # short prefixes for joints
        short_prefixes = {'shoulder': 's', 'elbow': 'e', 'wrist': 'w'}
        short_joint_code = short_prefixes[joint_name]
        
        for axis in ['x', 'y', 'z']:
            # regex pattern to ensure self.channel_name contains joint_name and hand anywhere in any style
            match_pattern = (
                    rf"(^({hand}|{hand_code})_?({joint_name}|{short_joint_code})_?{axis}$)|"   # side_joint_axis
                    rf"(^({joint_name}|{short_joint_code})_?({hand}|{hand_code})_?{axis}$)|"   # joint_side_axis
                    rf"(^({hand}|{hand_code})({joint_name}|{short_joint_code}){axis}$)|"       # camelCase style
                    rf"(^({joint_name}|{short_joint_code})({hand}|{hand_code}){axis}$)"                                    # axis at end
                    )
            for ch_idx, ch_name in enumerate(self.channel_names):
                if re.search(match_pattern, ch_name, flags=re.IGNORECASE):
                    idx_list.append(ch_idx)
                    break
            else:
                idx_list.append(None)

        return idx_list

    
    def saveTorques_npy(self, save_torques: bool = False, save_dir: str = None, joints_to_save: List[str] = ['all']) -> None:
>>>>>>> 306e0a815e9ebbcf68f8bb2e035814839e083b34
        """
        This method save the calculated torques into numpy (.npy) files.
        It first checks whether the specified dir exists. If it does not, it will create one and then save.

        Parameters
        ----------
        save_torques: bool, optional
            Boolean to choose whether to save the calculated torques into a .npy file, by default False.
        save_dir: str, optional
            Path to the dir where the files need to be saved relative to the projec root, by default None.
<<<<<<< HEAD
        joints_to_save: lsit of str, optional
=======
        joints_to_save: list of str, optional
>>>>>>> 306e0a815e9ebbcf68f8bb2e035814839e083b34
            The joints whose torques must be saved out of 'elbow','shoulder_front', and 'shoulder_side',  by default ['all'].

        Author
        ------
        Author: Kartik Chari \n
        Last changed: 27.08.2025 (by Kartik Chari)
        """
        if save_torques:
            if save_dir == None:
                warnings.warn("No Path for saving torque files specified! Using temp!")
                save_dir = "temp/"
            # get components of filename
            filename_comp = self.parseFilename()
            # form the file suffix
            file_suffix = "_".join(filename_comp) + ".npy"
            # get absolute path of the save dir
            save_dir = getAbsolutePath(input_path=save_dir)

            if 'all' in joints_to_save:
                joints_to_save = self.joint_names
<<<<<<< HEAD
            if 'all' in joints_to_save:
                joints_to_save = self.joint_names
            # iterate over each joint name and save the torque in file
            for _, joint in enumerate(joints_to_save):
            for _, joint in enumerate(joints_to_save):
=======
            # iterate over each joint name and save the torque in file
            for _, joint in enumerate(joints_to_save):
>>>>>>> 306e0a815e9ebbcf68f8bb2e035814839e083b34
                if joint in self.joint_names:
                    # get index of the joint name
                    joint_idx = np.where(self.joint_names == joint)[0][0]
                    # full path of the file being saved
                    fullpath = save_dir / f"quali_torque_{joint}_{file_suffix}"
                    fullpath.parent.mkdir(parents=True, exist_ok=True)
                    np.save(fullpath, self.torque_out[joint_idx])
    
    def parseFilename(self):
        """
        This method parses the filename into its different components.

        Returns
        -------
        list of str
            list of individual components of filename

        Author
        ------
        Author: Kartik Chari \n
        Last changed: 27.08.2025 (by Kartik Chari)
        """
        # remove the extension
        stem = Path(self.filename).stem
        # split by underscore
        return stem.split("_")
