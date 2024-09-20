import sys 
import warnings 
import numpy as np 
from time import perf_counter
import time 
import matplotlib.pyplot as plt
import mne
from scipy import signal as sig
from scipy.signal import convolve
from scipy.fft import fft, fftfreq
from tensorflow.keras.utils import to_categorical
import mne_features.univariate as mne_feat
from mne.preprocessing import ICA
import copy 
from mne.preprocessing import Xdawn
from pybv import write_brainvision
import warnings
import zmq
from PyEMD import EMD

# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************


class Timeseries():
    """
    This class includes useful methods to process and analyse timeseries data. 

    """
    def __init__(self, raw_obj = None, events = None, channel_names = None, f_samp = None, data = None, epochs = None, windows = None):

        """
        This class includes useful methods and paramters for the (pre)processing and viusalization of timeseries data. It is mainly dependend on numpy, mne, scipy and additional utils (e.g. keras preprocessing)). 

        Parameters
        ----------
        raw_obj :  mne raw object, optional
            The mne raw object that is used to create the object. Only required for format type "RawObj".
        events : numpy ndarray 
            The events (also called markers) in the data. The shape is: (indices, 0, eventnumber). 
        channel_names : list of str
            A list of channel names as strings, if not known from the data format. Not required for format "Brainvision" (see EEG class). 
        f_samp: float
            The sampling rate of the data, not required for type "Brainvision" (see EEG class).  
        data : numpy ndarray, optional 
            The channel wise (raw) data as numpy array (shape: n_channel, n_sampels), currently fully optional (not used by any format).
        epochs : numpy ndarray, optional
            The epoched data as numpy ndarray with shape (n_trials, n_channels, n_sampels). 
        windows : numpy ndarray, optional
            A numpy array with windowed data (shape: n_trials, n_channels, n_sampels, n_windows), only required for format "Live".  

        Attributes
        ----------
        raw_obj : instance of class Raw (mne) 
            The mne raw object to be used for full mne support, please see the mne wiki for further information.
        events : numpy ndarray 
            The events (also called markers) in the data. The shape is: (indices, 0, eventnumber).  
        __channel_names : list of str
            A list of channel names as strings, if not known from the data format. Not required for format "Brainvision" (see EEG class).
        __fsamp : float 
            The sampling rate of the data in Hz. 
        time_axis_epochs: 1D numpy array 
            The time axis of the data after the epoching step. 
        units : str 
            The units of the data (i.e. "uV" or "V")
        epochs : numpy ndarray 
            The epoched data as numpy ndarray with shape: (n_trials, n_channels, n_sampels). 
        epoch_obj : instance of class Epochs (mne)
            The mne epochs object to be used for full mne support, see the mne wiki for more information. 
        average_epochs : numpy ndarray 
            The averaged epochs as numpy ndarray with shape: (n_channels, n_sampels), (deprecated, to be removed in future). 
        windows : numpy ndarray 
            The windowed data as numpy array with shape: (n_trials, n_channels, n_sampels, n_windows). 
        window_names : list of str
            The names (indentifier) of each window as a list of strings (has same size as n_windows). 
        feature_vec : numpy ndarray 
            A feature vector as numpy array, can have different shapes depending on the later used ML model (see ML class). 
        calib_means : float 
            A calibration value containing the mean values of the data instance (e.g. of the training data). 
        calib_stds : float 
            A calibration value containing the standard deviation values of the data instance (e.g. of the training data). 

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 18.04.2024 (by Niklas Kueper)
        """
        
        # provided data formats 
        self.raw_obj = raw_obj
        self.data = data#
        self.windows = windows 
        self.window_names = None 
        self.epochs = epochs 

        # parameter 
        #basic params 

        self.__channel_names = channel_names
        self.__fsamp = f_samp
        self.time_axis_epochs = None
        self.units = None
        # epoching 
        self.epochs = None 
        self.epoch_obj =None
        self.average_epochs = None
        #events
        self.events = events
        # features 
        self.feature_vec = None
        self.calib_means = None
        self.calib_stds = None

        # for filtering 
        self.zi = None

        # mne objects 
        if(self.raw_obj): 
            self.mne_info = raw_obj.info
        else: 
            self.mne_info = None


    def rereferencingEpoching(self, marker_number, error_number, channel_list, inverse_keep_channel, t1, t2, reref_channels = [], apply_filter=False, f_highpass = None, f_lowpass= None, apply_baseline_correction = False,  t0_baseline = None, t1_baseline= None, apply_ica = False, n_ica_comp = 20, exclude_ica_comp = [0, 1]): 
        """
        Apply referencing and epoching with given filters to a raw_obj mne instance

        Parameters
        ----------
        marker_number
            The markernumber of the used event for creating the epochs
        error_number
            The markernumber of the trials with an error which are excluded from the evaluation
        channel_list
            As list of EEG-channels that are either kept or dropped from evaluation depending on the "inverse_keep_channel" flag
        inverse_keep_channel
            If False, all channel in "channel_list" are kept, otherwise the specified channels are dropped
        t1
            Start time of the epocs in ms
        t2
            End time of the epochs in ms
        reref_channels, optional
            A list of channels that are used for rereferencing. If the list is empty, the original ref-channel is used, if ["average"] an average an average reference is applied , by default []
        apply_filter, optional
            Boolean flag that should be True if a filter should be applied, by default False
        f_highpass, optional
            The highpass cutoff frequency in Hz (only used when apply_filter is True), by default None
        f_lowpass, optional
            The lowpass cutoff frequency in Hz (only used when apply_filter is True), by default None
        apply_baseline_correction, optional
            Boolean flag that is set to True if baseline correction should be applied (mean value between t0_baseline and t1_baseline is used as correction), by default False
        t0_baseline, optional
            Specified start time for the baseline correction, by default None
        t1_baseline, optional
            Specified end time for the baseline correction, by default None
        apply_ica, optional
            Boolean flag that should be True if a simple ICA removal should be applied, by default False
        n_ica_comp, optional
            Number of components the ICA should calculate, by default 20
        exclude_ica_comp, optional
            The components that are excluded if the ICA flag is True, by default [0, 1]
            
        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 28.11.2022 (by Niklas Kueper)  

        """        
        
        # rereferencing 
        rereferenced_eeg_raw_obj = self.raw_obj.copy()
        event_id_used = marker_number
        
        if not (reref_channels):
            print("no reref channels specified, using original ref")
            print("")
        else: 
            if (reref_channels[0] == "average"):
                print("using average reference over all electrodes")
                print("")
                rereferenced_eeg_raw_obj, ref_data = mne.set_eeg_reference(rereferenced_eeg_raw_obj, ref_channels='average' ,copy=True)
            else:
                print("using custom electrodes for rereferencing")
                rereferenced_eeg_raw_obj, ref_data = mne.set_eeg_reference(rereferenced_eeg_raw_obj, ref_channels=reref_channels ,copy=True)
        
        raw_obj_eeg_rereferenced = rereferenced_eeg_raw_obj.copy()
        #drop channel

        #apply filter 
        if (apply_filter): 
            filtered_eeg_rereferenced = raw_obj_eeg_rereferenced.filter(f_highpass,f_lowpass)
        else: 
            filtered_eeg_rereferenced = raw_obj_eeg_rereferenced


        
        #extract events
        include_events = self.events
        plot_onset_indices = np.where(include_events[:,2] == marker_number)[0] # S100 marker is leaving plate 
        exclude_indices = np.where(include_events[:,2] == error_number)[0] # S3 marker should be excluded 
        
        #plot_onset_indices = plot_onset_indices[0:-1] # cut of last movement , might be after experiment --> not used anymore, leads to confusion ! 
        include_mask = np.ones(plot_onset_indices.shape)

        #search for correct indizes without S3 errors 
        for i in range(0, len(plot_onset_indices)): 
            for j in range(0, len(exclude_indices)):
                if ((exclude_indices[j]-1) == plot_onset_indices[i] or (exclude_indices[j]-2) == plot_onset_indices[i]):
                    include_mask[i] = 0

        #Epochs with an S3 error are excluded 
        plot_onset_indices_correct = plot_onset_indices[include_mask.astype(bool)]

        #create new event matrix 
        used_include_events = np.zeros((len(plot_onset_indices_correct),3))
        used_include_events = include_events[plot_onset_indices_correct,:]

        #eeg_epochs = mne.Epochs(filtered_eeg_rereferenced, events = used_include_events, event_id = event_id_used,tmin=t1, baseline=None, tmax=t2, preload=True, reject_by_annotation = True)
        if not(channel_list): 
            if(apply_baseline_correction): 
                eeg_epochs = mne.Epochs(filtered_eeg_rereferenced, events = used_include_events,event_id = marker_number, tmin=t1, baseline=(t0_baseline, t1_baseline), tmax=t2, preload=True, reject_by_annotation = True)
            else: 
                eeg_epochs = mne.Epochs(filtered_eeg_rereferenced, events = used_include_events, event_id = marker_number, tmin=t1, baseline=None, tmax=t2, preload=True, reject_by_annotation = True)
                
        else: #drop specified channels if False 

            # keep or drop specified channels 
            if(inverse_keep_channel == True): 
                filtered_eeg_rereferenced.drop_channels(channel_list)
            else: 
                filtered_eeg_rereferenced.pick_channels(channel_list)


            if(apply_baseline_correction): 
                eeg_epochs = mne.Epochs(filtered_eeg_rereferenced, events = used_include_events,tmin=t1,event_id = marker_number, tmax=t2, baseline=(t0_baseline, t1_baseline), preload=True, reject_by_annotation = True)
            else: 
                eeg_epochs = mne.Epochs(filtered_eeg_rereferenced, events = used_include_events,tmin=t1, event_id = marker_number, tmax=t2, baseline=None, preload=True, reject_by_annotation = True)
        

        # if ica should be used 
        if(apply_ica): 
            ica = ICA(n_components=n_ica_comp) 
            ica.fit(eeg_epochs)
            ica.apply(eeg_epochs, exclude = exclude_ica_comp)
            self.ica = ica # save the ica 
        else: 
            self.ica = None

        # Get remaining channel names  
        self.__channel_names = filtered_eeg_rereferenced.ch_names
        #self.obj_filtered = filtered_eeg_rereferenced.copy()
        
        self.epoch_obj = eeg_epochs.copy() # the object of epochs from mne 
        
        #get data out as numpy array for further processing 
        self.epochs = eeg_epochs.get_data()#units = "uV") 
        # change here to float32 ? 
        self.average_epochs = np.mean(self.epochs, axis = 0)
        self.event_id = event_id_used 
        
        #generate a time axis for the epochs 
        self.time_axis_epochs = np.arange(t1,t2+1/self.__fsamp, step = 1/self.__fsamp) #build time axis (epoch)

    def showICAcomponents(self): 
        """
        Show the ica components if ica was done before. 
        """
        if(self.ica): 
            print(self.epoch_obj.info)
            self.ica.plot_components(inst = self.epoch_obj)
            #plt.show()
        else: 
            warnings.warn("no ica was performed, not showing components ...")


    def mneRawMethod(self, method_name = None, **kwargs): 
            """
            Apply a mne raw objects method and update the parameters of the current object. This method should be used instead of writing new methods that just execute an existing method of an mne raw object (like raw.show()). 
            Please the the wiki of mne for further informations about the methods. 

            Parameters
            ----------
            method_name : str, optional
                The name of the method to be executed passed as string, by default None
            **kwargs : optional
                The arguments passed to the method (e.g. plot = True, array = [0, 1, 2])
            
            Author
            ------
            Author : Niklas Kueper \n
            Last changed: 15.12.2023 (by Niklas Kueper)
            """

            if hasattr(self.raw_obj, method_name) and callable(getattr(self.raw_obj, method_name)):
                method = getattr(self.raw_obj, method_name)
                method(**kwargs) # call method with params 
            else:
                print("mne raw object does not have the expected method.")
            
            self.updatefromRawObject() # update class internal params from mne raw object

    def updatefromRawObject(self): 
        """
        This method can and should be used to update the class internal parameters and variables. Please make sure that this method is called whenever external methods (like mne raw methods) are executed to update the data. 

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 12.11.2023 (by Niklas Kueper)
        
        """

        # update parameter 
        #basic params 
        self.__channel_names = self.raw_obj.ch_names
        self.__fsamp = self.raw_obj.info['sfreq']
        self.data = self.raw_obj.get_data() # data as numpy array in shape (channels, sampels) 
        #events epochs_filter
        self.events, a = mne.events_from_annotations(self.raw_obj)

    # def updateRawObject(self): 

    #     self.raw_obj = mne.io.RawArray(self.data, self.mne_info)

    def reverseWindows(self): 
        """
        Reverse the windowed data along the sample dimension (invert the time axis). 

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 18.05.2024 (by Niklas Kueper)
        """

        temp_wind = self.windows[:, :, ::-1, :] # reverse along sample axis 
        self.windows = temp_wind

        
    def designFilter(self, f_low = None, f_high= None, order = 2, filter_type = "scipy_butter", Q = 30, show_response = False, alpha = 0.999, return_type = "ba", rp = 0.175478486150103, rs = 60.0, beta = 3.0): 
        """
        Design a digital fir or iir filter (notch, bandpass, highpass or lowpass). 

        Parameters
        ----------
        f_low : float, optional
            The lowpass frequency of the filter (the high frequency boundary), by default None
        f_high : float, optional
            The highpass frequency of the filter (the low frequency boundary), by default None
        order : int, optional
            The order of the filter, by default 2
        filter_type : str, optional
            The type of the filter as string (common biosignal filters), can be "scipy_butter", "scipy_bessel", "dc_notch", "dc_removal" or "allpass", by default "scipy_butter"
        Q : int, optional
            The sharpness factor of a notch filter (only required for "dc_notch"), see more information about scipys iirnotch method for further details, by default 30
        show_response : bool, optional
            If True, the filter response is shown (frequency and phase response), by default False
        alpha : float, optional
            The alpha value (sharpness) of the dc removal filter (only required for "dc_removal" filter), by default 0.98
        return_type : str, optional
            The type of the filter coefficients to be returned, can be "ba" or "sos". "sos should be preferred to avoid instabilites, but not all filtering methods support the "sos" type, by default "ba"

        Returns
        -------
        tuple of numpy arrays or numpy array 
            The returned filter parameters (ba or sos). 
        
        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 15.12.2023 (by Niklas Kueper)
        """
        
        # calc individual coeffs 
        if(filter_type == "dc_notch"):
            b, a = sig.iirnotch(f_high, Q, fs=self.__fsamp)

        if(filter_type == "scipy_butter"): # prefer this one 
            if(f_high and f_low): 

                #self.__fsamp = 500.0
                # print(self.__fsamp)
                # print(f_high)
                # print(f_low)
                b, a = sig.iirfilter(order, [f_high, f_low], btype='bandpass', ftype='butter', output='ba', fs=self.__fsamp)
                sos = sig.iirfilter(order, [f_high, f_low], btype='bandpass', ftype='butter', output='sos', fs=self.__fsamp)
            elif(f_high):
                sos = sig.iirfilter(order, f_high, btype='highpass', ftype='butter', output='sos', fs=self.__fsamp)
                #zi = sig.lfilter_zi(b, a)
            elif(f_low): 
                sos = sig.iirfilter(order, f_low, btype='lowpass', ftype='butter', output='sos', fs=self.__fsamp)
                #zi = sig.lfilter_zi(b, a)
        
        if(filter_type == "scipy_bessel"): # prefer this one 
            if(f_high and f_low): 
                sos = sig.iirfilter(order, [f_high, f_low], btype='bandpass', ftype='bessel', output='sos', fs=self.__fsamp)
            
            elif(f_high):
                sos= sig.iirfilter(order, f_high, btype='highpass', ftype='bessel', output='sos', fs=self.__fsamp)
                #zi = sig.lfilter_zi(b, a)
            elif(f_low): 
                sos = sig.iirfilter(order, f_low, btype='lowpass', ftype='bessel', output='sos', fs=self.__fsamp)
                #zi = sig.lfilter_zi(b, a)

        if(filter_type == "scipy_ellip"): # prefer this one 
            if(f_high and f_low): 
                sos = sig.iirfilter(order, [f_high, f_low], btype='bandpass', ftype='ellip', rp=rp, rs = rs,  output='sos', fs=self.__fsamp)

            elif(f_high):
                sos= sig.iirfilter(order, f_high, btype='highpass', ftype='ellip', output='sos',rp=rp,rs = rs, fs=self.__fsamp)
                #zi = sig.lfilter_zi(b, a) rs
            elif(f_low): 
                sos = sig.iirfilter(order, f_low, btype='lowpass', ftype='ellip', output='sos', rp=rp,rs = rs, fs=self.__fsamp)
                #zi = sig.lfilter_zi(b, a)

        if(filter_type == "dc_removal"): 
            a = [1, -1 * alpha]
            b = [1, -1]
        
        if(filter_type == "allpass"):
            scale = 0.8
            b = [ 1 *scale, -alpha *scale] 
            a = [-alpha*scale, 1*scale]

        if(filter_type == "fir_hann"): 
            if(f_low): 
                b = sig.firwin(order, f_low / (self.__fsamp / 2), window='hann')
                a = [1.0]
        if(filter_type == "fir_hamming"): 
            if(f_low): 
                b = sig.firwin(order, f_low / (self.__fsamp / 2), window='hamming')
                a = [1.0]

        if(filter_type == "fir_kaiser"): 
            if(f_low): 
                b = sig.firwin(order, f_low / (self.__fsamp / 2), window = ('kaiser', beta))
                a = [1.0]

        if(show_response): 
            w, h = sig.freqz(b, a, worN=2024)
            plt.subplot(2, 1, 1)

            x = (w/np.pi)*(self.__fsamp/2)

            db = 20*np.log10(np.maximum(np.abs(h), 1e-5))
            plt.plot(x, db)
            plt.ylim(-75, 5)
            plt.grid(True)
            plt.yticks([0, -20, -40, -60])
            plt.ylabel('Gain [dB]')
            plt.title('Frequency Response')
            plt.subplot(2, 1, 2)
            plt.plot(x, (np.angle(h)))
            plt.grid(True)
            plt.yticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi],[r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
            plt.ylabel('Phase [rad]')
            plt.xlabel('Frequency[Hz]')
            plt.show()

            # group delay 
            w, gd = sig.group_delay((b, a), fs = self.__fsamp)
            plt.title('Digital filter group delay')
            plt.plot(w, gd)
            plt.ylabel('Group delay [samples]')
            plt.xlabel('Frequency [Hz]')
            plt.show()

        # return filter coeffs 
        if(return_type == "ba"): 
            return b, a
        
        elif(return_type == "sos"): 
            return sos 
        

    def applyVarianceFilter(self,  n_var = 20, apply_to_structures = "raw"): 

        """
        This function applies a variance filter to the EMG data

        Parameters
        ----------
        n_var : int
            The filter length of the variance filter
        apply_to_structures : str
            The data structures to which the methods should be applied, can be "raw", "epochs" or "windows". 

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 05.02.2024 (by Niklas Kueper)
        """    
        
        if (apply_to_structures == "windows"): 
            filtered_windows = np.zeros(self.windows.shape)

            for trial_idx in range(0, filtered_windows.shape[0]): 
                for channel_idx in range(0, filtered_windows.shape[1]):
                    for index in range(0, filtered_windows.shape[2]): 
                        for window_idx in range(0, filtered_windows.shape[3]): 

                            if (index < n_var): 
                                filtered_windows[trial_idx, channel_idx, index, window_idx] = 0 # just set values to zero if filterlength is not reached yet 
                            else: 
                                filtered_windows[trial_idx, channel_idx, index, window_idx] = np.var(self.windows[trial_idx, channel_idx, index-n_var:index, window_idx])

            self.windows = filtered_windows

        elif(apply_to_structures == "raw"): # channels, sampels 
            
            # signal init 
            emg_filtered = np.zeros(self.data.shape)
                
            for channel_idx in range(0, emg_filtered.shape[0]):  
                for sample_idx in range(0, emg_filtered.shape[1]): 
                    
                    if not (sample_idx < n_var):
                        emg_filtered[channel_idx, sample_idx] = np.var(self.data[channel_idx, sample_idx-n_var:sample_idx])

            self.data = emg_filtered
            # update raw object 
            self.raw_obj = mne.io.RawArray(self.data, self.mne_info)


        else: # apply to epochs: trials, channels, sampels 
            filtered_epochs = np.zeros(self.epochs.shape)

            for trial_idx in range(0, filtered_epochs.shape[0]):
                for channel_idx in range(0, filtered_epochs.shape[1]):  
                    for sample_idx in range(0, filtered_epochs.shape[2]): 

                        if not (sample_idx < n_var):
                            filtered_epochs[trial_idx, channel_idx, sample_idx] = np.var(self.epochs[trial_idx, channel_idx, sample_idx-n_var:sample_idx])

            self.epochs = filtered_epochs

    def applyMovingAverageFilter(self,  n = 20, apply_to_structures = "windows"): 

        """
        This function applies a variance filter to the EMG data

        Parameters
        ----------
        n : int
            The filter length of the moving average filter
        apply_to_structures : str
            The data structures to which the methods should be applied, can be "raw", "epochs" or "windows". 
        
        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 30.04.2024 (by Niklas Kueper)
        """    

        moving_avg_kernel = np.ones(n) / n
        
        if (apply_to_structures == "windows"): 
            filtered_windows = np.zeros(self.windows.shape)

            for trial_idx in range(0, filtered_windows.shape[0]): 
                for channel_idx in range(0, filtered_windows.shape[1]):
                    for window_idx in range(0, filtered_windows.shape[3]): 
                        
                        filtered_windows[trial_idx, channel_idx, :, window_idx] = convolve(self.windows[trial_idx, channel_idx, :, window_idx], moving_avg_kernel, mode='same') # just set values to zero if filterlength is not reached yet 

            self.windows = filtered_windows

        elif(apply_to_structures == "raw"): # channels, sampels 
            
            # signal init 
            emg_filtered = np.zeros(self.data.shape)
                
            for channel_idx in range(0, emg_filtered.shape[0]):  
                for sample_idx in range(0, emg_filtered.shape[1]): 
                    
                    if not (sample_idx < n):
                        emg_filtered[channel_idx, sample_idx] = np.mean(self.data[channel_idx, sample_idx-n:sample_idx])

            self.data = emg_filtered
            # update raw object 
            self.raw_obj = mne.io.RawArray(self.data, self.mne_info)


        else: # apply to epochs: trials, channels, sampels 
            filtered_epochs = np.zeros(self.epochs.shape)

            for trial_idx in range(0, filtered_epochs.shape[0]):
                for channel_idx in range(0, filtered_epochs.shape[1]):  
                    for sample_idx in range(0, filtered_epochs.shape[2]): 

                        if not (sample_idx < n):
                            filtered_epochs[trial_idx, channel_idx, sample_idx] = np.var(self.epochs[trial_idx, channel_idx, sample_idx-n:sample_idx])

            self.epochs = filtered_epochs

    
    def mneRawToBrainvision(self, folder, filename, meas_date = None, resolution = 0.1, unit = "µV"):

        """
        This method can be used to store the data from an mne raw object in the brainvision format. 

        Parameters
        ----------
        folder : str
            The target folder where the data should be stored in the brainvision format, should be the path to the created data folder of the toolbox. 
        filename : str
            The filename of the data to be stored. 
        meas_date : datetime object, str, optional
            The measurement date on on which the data was recorded, by default None
        resolution : float, optional
            The resolution of the data in the measurement unit, by default 0.1
        unit : str, optional
            The units of the data to be stored as string, can be "uV" or "V", by default "µV"

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 28.11.2022 (by Niklas Kueper)
        """

        events=copy.deepcopy(self.events) 
        events_bv = events[:, [0, 2]]
        write_brainvision(data=self.raw_obj.get_data(), sfreq=self.__fsamp, ch_names=self.__channel_names, fname_base=filename, folder_out=folder, events=events_bv, meas_date = meas_date, resolution = resolution, unit = unit)
        print("stored data in brainvision format")

    def getRawObject(self):
        """
        This method can be used to get the current mne raw object. 

        Returns
        -------
        mne raw object
            The current mne raw object.  

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 12.11.2023 (by Niklas Kueper)
        """

        return self.raw_obj
    

    
    def getEpochs(self):
        """
        This method returns the epoched an possibly processed data as numpy array. 

        Returns
        -------
        epochs : numpy ndarray 
            The numpy ndarray including the epoched data with shape: (n_trials, n_channels, n_sampels). 

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 15.04.2024 (by Niklas Kueper)
        """

        return self.epochs
    
    def getEvents(self): 
        """
        This method returns the events (i.e. markers) as numpy array. The events can mark experimental events or stimuli or are used to help with evaluating the data. 

        Returns
        -------
        numpy ndarray
            The events as numpy ndarray with shape: (indices, 0, eventnumbers). 

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 05.11.2023 (by Niklas Kueper)
        """

        return self.events
    
    def getSamplingRate(self): 
        """
        This method returns the sampling rate of the data in Hz. 

        Returns
        -------
        float
            The sampling rate of the data as float value in Hz. 
        
        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 05.11.2023 (by Niklas Kueper)
        """

        return self.__fsamp

    def getCalibStats(self): 
        """
        This function returns calibration parameters that where calculated for the dataset (see calcCalibStats for more information). 

        Returns
        -------
        tuple
            A tuple containing:
            calib_means : float
                The mean values for each channel as 1D numpy array. 
            calib_stds : float 
                The standard deviations for each channel as 1D numpy array. 
            calib_mins : float 
                The minimum values for each channel as 1D numpy array. 
            calib_maxs : float 
                The maximum values for each channel as 1D numpy array. 

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 05.11.2023 (by Niklas Kueper)
        """

        return self.calib_means, self.calib_stds, self.calib_mins, self.calib_maxs
    
    def setCalibStats(self, means, stds, mins, maxs): 
        """ 
        This method can be used to set some basic statistical parameters for the dataset (e.g. mean and std) for each data channel. Can be used for example to calibrate or normalize a train data set afte setting the parameters. 

        Parameters
        ----------
        means : numpy array
            A 1D numpy array containing the mean values of each channel. 
        stds : numpy array 
            A 1D numpy array containing the standard deviation values of each channel. 
        mins : numpy array 
            A 1D numpy array containing the minimum values of each channel. 
        maxs : numpy array 
            A 1D numpy array containing the maximum values of each channel. 
        
        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 05.11.2023 (by Niklas Kueper)
        """

        self.calib_means = means
        self.calib_stds = stds
        self.calib_mins = mins
        self.calib_maxs = maxs

    def calcCalibStats(self, feature_times = None): 
        """
        This method can be used to calculate statistical parameters for the windowed data (training data) like mean and standard deviations for each channel. 

        Parameters
        ----------
        feature_times : 1D numpy array or list, optional
            A numpy array containing 2 values, which define the start and endpoint (times) of the data in the window to be selected for the calculation procedure, by default None
        
        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 01.10.2023 (by Niklas Kueper)
        """

        # shape: trials, channels, sampels, windows
        calib_means = np.zeros(self.windows.shape[1]) # channel wise 
        calib_stds = np.zeros(self.windows.shape[1])
        calib_maxs = np.zeros(self.windows.shape[1])
        calib_mins = np.zeros(self.windows.shape[1])

        if(feature_times): 
            start_point_index = int((feature_times[0]/1000)*self.__fsamp)
            stop_point_index = int((feature_times[1]/1000)*self.__fsamp)

        for channel_idx in range(0, self.windows.shape[1]):   # channel wise 
            if(feature_times):
                channel_data = self.windows[:, channel_idx, start_point_index:stop_point_index, :].flatten() # use only the features time points for window norm 
            else: 
                channel_data = self.windows[:, channel_idx, :, :].flatten() # use hole window 

            # calc descriptive values 
            mean = (np.max(channel_data) +np.min(channel_data))/2 
            std = np.std(channel_data)
            max = np.max(channel_data)
            min = np.min(channel_data)
            # calib values 
            calib_means[channel_idx] = mean
            calib_stds[channel_idx] = std
            calib_maxs[channel_idx] = max 
            calib_mins[channel_idx] = min
        # set the matrices for calibration 
        self.calib_means = calib_means
        self.calib_stds = calib_stds
        self.calib_mins = calib_mins
        self.calib_maxs = calib_maxs

    def setChannelNames(self, ch_names): 
        """
        This method can be used to set the channel names of the data if they were not specified before.

        Parameters
        ----------
        ch_names : list of str
            A list of channels names. 

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 05.11.2023 (by Niklas Kueper)
        """

        self.__channel_names = list(ch_names)
        self.raw_obj.ch_names = list(ch_names)

    def getChannelNames(self): 
        """
        This method returns the channel names of the data. 

        Returns
        -------
        list of str
            A list of channel names. 
        
        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 05.11.2023 (by Niklas Kueper)
        """

        return self.__channel_names
    
    def simpleICAFilteringEpochs(self, n_components = 20, exclude_components = [0, 1]): 
        """
        This method can be used to fit and apply a ICA on epoched data (on mne epochs object). 

        Parameters
        ----------
        n_components : int, optional
            The number of components of the ICA, by default 20
        exclude_components : list of int, optional
            The components of the ICA to be excluded from the data. Most probably the first channels referring to eyeblinks or saccades, by default [0, 1]
        
        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 17.05.2024 (by Niklas Kueper)
        """

        ica = ICA(n_components=n_components, method="infomax", random_state=97) 
        ica.fit(self.epoch_obj)
        ica.apply(self.epoch_obj, exclude = exclude_components)
        self.ica = ica # save for later 
        self.epochs = self.epoch_obj.get_data()

    def applyBaselineCorrectionToEpochs(self, t0_baseline, t1_baseline): 
        """
        Apply mne's baseline correction to epochs object. (updates epochs numpy array as well)

        Parameters
        ----------
        t0_baseline : float
            The start time (first time) of the baseline correction interval. 
        t1_baseline : float
            The end time (second time) of the baseline correction interval. 

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 17.05.2024 (by Niklas Kueper)
        """

        self.epoch_obj.apply_baseline(baseline=(t0_baseline, t1_baseline))
        self.epochs = self.epoch_obj.get_data()

    def filterRawData(self,  b = None, a = [1], sos = None, apply_method = "zero_phase_sos", padtype = "even"):
        """
        This method is used to apply designed filters to raw data with the shape (channels, sampels). 
        
        Parameters
        ----------
        b : 1D numpy array, optional
            The filter coefficients (numerator), by default None
        a : 1D numpy array, optional
            The filter coefficients (denominator), by default [1] (fir)
        sos : array_like, optional
            The filter coefficients in sos format, by default None
        apply_method : str, optional
            The filtering method to be applied, can be "zero_phase_sos", "gustav", "zero_phase_ba" or "forward_filter". Please have a look at the documentation of the filter implemetations for more information (i.e. scipy and mne docu), by default "zero_phase_sos"
        padtype : str, optional
            The padding type used when zero phase filtering is applied, by default "even"
        """

        self.updatefromRawObject() # get the current data and 

        for channel_idx in range(0, self.data.shape[0]): 
            # perform zero phase forward backward filtering with gustafson method to reduce artifacts  
            
            #filtered_window = sig.lfilter(b, a, current_wind[channel_idx, :].copy())#, method ="gust") # forward backward filtering with gustafson method 
            if(apply_method == "zero_phase_sos"): 
                filtered_channel = sig.sosfiltfilt(sos, self.data[channel_idx, :], padlen = len(self.data[channel_idx, :])-1, padtype =padtype)#, padtype ="even") # normal filtering with padding 
                self.data[channel_idx, :] = filtered_channel
            
            elif(apply_method == "zero_phase_ba"): 
                filtered_channel = sig.filtfilt(b, a, self.data[channel_idx, :], padlen = len(self.data[channel_idx, :])-1, padtype =padtype) # normal filtering with padding
                self.data[channel_idx, :] = filtered_channel

            elif(apply_method == "forward_sos_filter"): 
                filtered_channel = sig.sosfilt(sos,  self.data[channel_idx, :].copy())
                self.data[channel_idx, :] = filtered_channel 

            elif(apply_method == "forward_ba_filter"): 
                filtered_channel = sig.lfilter(b, a, self.data[channel_idx, :].copy())
                #  self.zi = filtered_window

                self.data[channel_idx, :] = filtered_channel 
        
        # update the data of the raw object 
        self.raw_obj._data = self.data
    
    def filterWindows(self, b = None, a = [1], sos = None, apply_method = "zero_phase_sos", mne_filter_type = None, f_high = None, f_low = None, order = None, fir_design = None, padtype = "even"): # under change 
        """
        Apply a designed digital filter to the windowed data (window wise for each channel). Please be careful in selection appropriately designed filters, especially because they are applied on small data chunks (windows)!
        Therefore, consider that artifacts might occur depending on the selected method and parameters. 

        Parameters
        ----------
        b : 1D numpy array, optional
            The filter coefficients (numerator), by default None
        a : 1D numpy array, optional
            The filter coefficients (denominator), by default [1] (fir)
        sos : array_like, optional
            The filter coefficients in sos format, by default None
        apply_method : str, optional
            The filtering method to be applied, can be "zero_phase_sos", "gustav", "zero_phase_ba" or "forward_filter". Please have a look at the documentation of the filter implemetations for more information (i.e. scipy and mne docu), by default "zero_phase_sos"
        mne_filter_type : str, optional
            The type of the mne filter to be applied, can be "mne_fir", "mne_iir", only required when using an mne filter , by default None
        f_high : float, optional
            The highpass filter frequency when applying the mne filter, by default None
        f_low : float, optional
            The lowpass filter frequency when applying the mne filter, by default None, by default None
        order : int, optional
            The order of the mne filter to be applied, only required for mne filter, by default None
        fir_design : str, optional
            The fir design method as string for mne filters (e.g. "hamming", see mne documentation for more information), by default None
                
        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 15.12.2023 (by Niklas Kueper)
        """
        

        # shape: trials, channels, sampels, windows
        
        for trial_idx in range(0, self.windows.shape[0]): 
                for window_idx in range(0, self.windows.shape[3]): 
                    
                    current_wind = self.windows[trial_idx, :, :, window_idx]

                    # currently the best 
                    if(mne_filter_type == "mne_fir" or mne_filter_type == "mne_iir"): # TODO: remove this filter here and add a new filtering method maybe ? Paramters do not fit 
                        if(mne_filter_type == "mne_iir"): 
                            method = "iir"
                        else: 
                            method = "fir"

                        for channel_idx in range(0, self.windows.shape[1]): # TODO: replace the function call by the parser method for mne objects 
                            #print(current_wind.shape)
                            filtered_window= mne.filter.filter_data(current_wind[channel_idx, :], sfreq = self.__fsamp, l_freq =f_high , h_freq = f_low, filter_length=order, method = method, fir_design = fir_design, verbose = "CRITICAL") # pad = "symmetric"
                            self.windows[trial_idx, :, :, window_idx] = filtered_window

                    else: 
                        for channel_idx in range(0, self.windows.shape[1]):
                            
                            # perform zero phase forward backward filtering with gustafson method to reduce artifacts  
                            if(apply_method == "gustav"): 
                                if(padtype == "own"): 
                                    N = len(current_wind[channel_idx, :])
                                    padded_wind = np.pad(current_wind[channel_idx, :].copy(), (0, N),  mode='symmetric', reflect_type='even')
                                    filtered_window = sig.filtfilt(b, a, padded_wind, method ="gust") # forward backward filtering with gustafson method
                                    filtered_window_cut = filtered_window[0:N]
                                    self.windows[trial_idx, channel_idx, :, window_idx] = filtered_window_cut

                                else: 
                                    filtered_window = sig.filtfilt(b, a, current_wind[channel_idx, :].copy(), method ="gust") # forward backward filtering with gustafson method
                                    self.windows[trial_idx, channel_idx, :, window_idx] = filtered_window
                            
                            #filtered_window = sig.lfilter(b, a, current_wind[channel_idx, :].copy())#, method ="gust") # forward backward filtering with gustafson method 
                            elif(apply_method == "zero_phase_sos"): 

                                    if(padtype == "own"): 
                                        N = len(current_wind[channel_idx, :])
                                        padded_wind = np.pad(current_wind[channel_idx, :].copy(), (0, N),  mode='symmetric', reflect_type='even')
                                        filtered_window = sig.sosfiltfilt(sos, padded_wind, padlen = len(current_wind[channel_idx, :])-1, padtype ="even")
                                        filtered_window_cut = filtered_window[0:N]
                                        self.windows[trial_idx, channel_idx, :, window_idx] = filtered_window_cut

                                    else: 
                                        filtered_window = sig.sosfiltfilt(sos, current_wind[channel_idx, :], padlen = len(current_wind[channel_idx, :])-1, padtype =padtype)#, padtype ="even") # normal filtering with padding 
                                        self.windows[trial_idx, channel_idx, :, window_idx] = filtered_window
                            
                            elif(apply_method == "zero_phase_ba"): 
                                filtered_window = sig.filtfilt(b, a, current_wind[channel_idx, :], padlen = len(current_wind[channel_idx, :])-1, padtype =padtype) # normal filtering with padding
                                self.windows[trial_idx, channel_idx, :, window_idx] = filtered_window

                            elif(apply_method == "forward_sos_filter"): 
                                filtered_window = sig.sosfilt(sos, current_wind[channel_idx, :].copy())
                                self.windows[trial_idx, channel_idx, :, window_idx] = filtered_window 

                            elif(apply_method == "forward_ba_filter"): 
                                # get initial filter state 

                                # if(a ==[1.0]): # if fir filter do not calc initial state 
                                #     filtered_window = sig.lfilter(b, a, current_wind[channel_idx, :].copy()) 
                                # else: 
                                #     if not(self.zi): 
                                #         z0 = sig.lfilter_zi(b, a)
                                #         self.zi = z0 *current_wind[channel_idx, 0]


                                filtered_window = sig.lfilter(b, a, current_wind[channel_idx, :].copy())
                                #  self.zi = filtered_window

                                self.windows[trial_idx, channel_idx, :, window_idx] = filtered_window 
                                         
    def emdFilterWindows(self, component_used = -1):
        emd = EMD(max_imf=2) 

        # shape: trials, channels, sampels, windows
        for trial_idx in range(0, self.windows.shape[0]): 
                for window_idx in range(0, self.windows.shape[3]): 
                        for channel_idx in range(0, self.windows.shape[1]):
                            data_to_filter = self.windows[trial_idx, channel_idx, :, window_idx]
                            imf = emd.emd(data_to_filter)
                            self.windows[trial_idx, channel_idx, :, window_idx] = imf[component_used, :]
                        

    def fft_bandpass_filter(self, eeg_data, fs, lowcut, highcut):
        """
        Apply an FFT-based bandpass filter to EEG data.

        Parameters:
        eeg_data : numpy.ndarray
            Input EEG data (2D array: channels x time).
        fs : float
            Sampling frequency of the EEG data.
        lowcut : float
            Lower cutoff frequency of the bandpass filter.
        highcut : float
            Upper cutoff frequency of the bandpass filter.

        Returns:
        filtered_eeg : numpy.ndarray
            Filtered EEG data (2D array: channels x time).
        """
        
        # Number of samples in the signal
        n_samples = eeg_data.shape[-1]
        print(n_samples)

        # FFT of the signal
        eeg_fft = np.fft.fft(eeg_data, axis=-1)

        # Frequency bins
        freqs = np.fft.fftfreq(n_samples, d=1/fs)
        print(freqs)
        
        # Create the bandpass filter mask
        bandpass_mask = (np.abs(freqs) >= lowcut) & (np.abs(freqs) <= highcut)

        # Apply the bandpass filter
        eeg_fft_filtered = eeg_fft * bandpass_mask

        # Inverse FFT to get the filtered signal back in time domain
        filtered_eeg = np.fft.ifft(eeg_fft_filtered, axis=-1).real

        return filtered_eeg
    
    def FFTBandpassWindows(self, lowcut = None, highcut= None): 

        # Apply the FFT bandpass filter
        for trial_idx in range(0, self.windows.shape[0]): 
            for window_idx in range(0, self.windows.shape[3]): 

                filtered_window = self.fft_bandpass_filter(self.windows[trial_idx, :, :, window_idx], self.__fsamp, lowcut, highcut)
                self.windows[trial_idx, :, :, window_idx] = filtered_window


    def minMaxNormWindows(self): 
        """
        Channel wise min-max normalization method is appllied to windows that the minimum values is zero and maximum value one. 

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 24.11.2023 (by Niklas Kueper)
        """
        
        normed_wind = np.zeros(self.windows.shape)
        for channel in range(0, self.windows.shape[0]): 
            wind_channel = self.windows[channel, :] + (-1 *np.min(self.windows[channel, :]))
            wind_channel = wind_channel / np.max(wind_channel)
            normed_wind[channel, :] = wind_channel


        for trial_idx in range(0, self.windows.shape[0]): 
            for window_idx in range(0, self.windows.shape[3]): 
                for channel_idx in range(0, self.windows.shape[1]): 
                            
                    current_wind = copy.deepcopy(self.windows[trial_idx, channel_idx, :, window_idx])
                    current_wind = current_wind + (-1 *np.min(current_wind))
                    current_wind = current_wind / np.max(current_wind)
                    
                    self.windows[trial_idx, channel_idx, :, window_idx] = current_wind

    def cutWindows(self, n_samples_start = 25, n_samples_end = 25): 
        """
        This function can be used to cut the length of the windowed data (sample dimension).
        The method can also be used for reshaped windows (see reshapeWindowsForCNNnets)

        Parameters
        ----------
        n_samples_start : int
            The number of samples to cut at the start of the windows.
        n_samples_end : int
            The number of samples to cut at the end of the windows.

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 18.05.2024 (by Niklas Kueper)
        """
        
        # if(self.windows.shape[2] < n_samples_start+n_samples_end):   # trials, channels, sampels, windows 
        #     raise Exception("number of cutted samples exceeds window length, not performing the cutting ... ")
        # else: 
        if(n_samples_end and n_samples_start): 
            self.windows = self.windows[:, :, int(n_samples_start):int(-1*n_samples_end), :]
        elif (not n_samples_end and n_samples_start): 
            self.windows = self.windows[:, :, int(n_samples_start):, :]
        elif (n_samples_end and not n_samples_start): 
            self.windows = self.windows[:, :, 0:int(-1*n_samples_end), :]

    def getDataFromChannels(self, channel_names, average = True, windowed_data = False, epoched_data = False): 
        
        """
        This function returns data from the channels

        Parameters
        ----------
        channel_names : str
            The channel names (list) of the data tha should be returned
        average : bool, optional
            If True the data should be averaged after windowing of epoching(deprecated, averaged should be implemented seperately), by default True
        windowed_data : bool, optional
            If True the windowed data of the specified channels is returned, by default False
        epoched_data : bool, optional
            If True the epoched data of the specified channels is returned, by default False

        Returns
        -------
        2D-numpy array
            data_channels
                Numpy array with the data from the selected channels. Can be either in the format of windowd data (if windowed_data = True), epoched data (if epoched_data = True or raw data in format (channels, samples)) 

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 22.11.2023 (by Niklas Kueper)
        """                
        
        if(len(channel_names) > 1): # for more then one channel
            channel_idxs = []
            for channel_name in channel_names: 
                channel_idx = self.__channel_names.index(channel_name)
                channel_idxs.append(channel_idx)
        else: 
            channel_idxs = self.__channel_names.index(channel_names[0])


        if (windowed_data): 
            # shape: trials, channel, sampels, windows 

            if(average): # if average should be returned 
                
                return np.mean(self.windows[:, channel_idxs, :, :], axis = 0)
            else: 
                return self.windows[:, channel_idxs, :, :]

        elif(epoched_data): # data is epoched  

            if(average):
                self.average_epochs = np.mean(self.epochs, axis = 0) # average it first 
                data_channel = self.average_epochs[channel_idxs, :] # data channel with shape: (channels, sampels) for average 
            else: 
                data_channel = self.epochs[:, channel_idxs, :] # data channel with shape: (trials, channels, sampels)  

            return data_channel
        
        else: # return data from raw object 
            return self.raw_obj.get_data(picks=channel_names) # format is channels, sampels (numpy array shape)


    # TODO: move this to machine learning lib !
    def getKerasPredictionResultsLRP(self, model, epochs, n_samp_features):  #TODO: Move to ml_lib or remove duplicate 
        """
        Get the prediction results of the classifier after predicting on the test data (scores and predicted labels). 

        Parameters
        ----------
        model : The machine learning model instance, can be for example a keras or sklearn model. 
            The keras model object
        epochs : Numpy array
            The EEG-epochs as numpy array with shape (n_epochs, n_ channels, n_samples)
        n_samp_features : int
            Number of samples that are used as features for both classes, currentsl the last n_samp_features datapoints are considered as erp class and dht remaining ar from noerp class

        Returns
        -------
        tuple
            predicted_lables : Numpy array
                The predicted labels of the classifier as float values (0.0 noerp or 1.0 erp)
            true_labels : Numpy array
                The true labels in respect to the number of n_samp_features as erp labels (-n_samp_features to time 0 as erp labeled points)
            trial_prediction : Numpy array
                The prediction scores of each classified datapoint

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 22.11.2023 (by Niklas Kueper)

        """

        true_labels = []
        predicted_labels = []
        prediction_scores = []

        for trials in epochs: 
            # true labels single trial 
            erp_labels = np.zeros((trials.shape[1]))
            erp_labels[-n_samp_features:] = 1.0
            true_labels.append(erp_labels[:,])
            # single trial predictions 
            trial_prediction = model.predict(trials.T)
            trial_label = [0 if score <0.5 else 1 for score in trial_prediction]
            trial_label = np.array(trial_label)
            predicted_labels.append(trial_label)
            prediction_scores.append(trial_prediction)
            
        #flatten the trial labels
        true_labels = np.array(true_labels)
        predicted_labels = np.array(predicted_labels)
        prediction_scores = np.array(prediction_scores)

        return predicted_labels, true_labels, prediction_scores

    # TODO: move this to machine learning lib !
    def calcMovingAveragePredictionScores(self, trial_prediction_test, n_samp = 20): # 
        """
        This function calculates the moving average prediction scores

        Parameters
        ----------
        trial_prediction_test : numpy Ndarray 
            The numpy array with shape (trials, channel, sampels). 
        n_samp : int,  by default 20
            The length of the moving average window (samples over with the average is calculated). 

        Returns
        -------
        Numpy array
            processed_trial_predictions : float
                A numpy ndarray containing the postprocessed prediction scores (same shape as trial_prediction_test (input)). 

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 05.02.2024 (by Niklas Kueper)
        """   

        processed_trial_predictions = np.zeros(trial_prediction_test.shape)

        trial_idx = 0
        for trial in trial_prediction_test: 
            for index in range(0, len(trial)): 
                if(index < n_samp): 
                    processed_trial_predictions[trial_idx, index] = 0 # what to to when buffer not full ? 
                else: 
                    processed_trial_predictions[trial_idx, index] = np.mean(trial[index-n_samp:index])

            trial_idx = trial_idx +1

        return processed_trial_predictions

    def getTimeAxisEpochs(self): 
        """
        This method returns the calculated time axis for the epoched data as 1D numpy array. 
        
        Returns
        -------
        numpy array (1D)
            The time axis as 1D numpy array containing the time axis values (in seconds) in relation to the event used for epoching. 
        
        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 09.01.2024 (by Niklas Kueper)
        """

        if (self.epochs is None): 
            warnings.warn("Data not epoched yet, please do this before using this method, terminating ...")
        else: # if epochs exist already  
            return self.time_axis_epochs
        

    def splitTrainTestEpochs(self, n_test_epochs = 5):
        """
        Split the epochs (data type epochs) into two seperate epoched data instances. This can be used for example to split the epochs into training and testing epochs as suggested by the name. 

        Parameters
        ----------
        n_test_epochs : int, optional
            The number of epochs to be splitted (e.g. for testing a classifier), by default 5

        Returns
        -------
        Tuple of data objects of class EEGData (train and test)
            The returned data objects of class EEGData, each including the specified number of trials. 

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 05.02.2024 (by Niklas Kueper)
        """        

        epochs_train = self.epochs[0:-n_test_epochs, :, :] 
        epochs_test = self.epochs[-n_test_epochs:, :, :]

        print("train shape", epochs_train.shape)
        print("test shape", epochs_test.shape)

        # copy objects 
        data_obj_train = copy.deepcopy(self)
        data_obj_test = copy.deepcopy(self)

        # epochs returning 
        data_obj_train.epochs = epochs_train
        data_obj_test.epochs = epochs_test

        return data_obj_train, data_obj_test


    def timeShiftingLinearSpatialFilter(self, erp_value_type = "min", replace_epochs = False, max_sample_diff = 200): 
        """
        A linear spatial filter with one remaining channel that additionally applies a timeshift to each individual channel according to the a minimum or maximum ERP magnitude. 

        Parameters
        ----------
        erp_value_type : str, optional
            The erp value type for calculating the reference channel for the time shift. Can be "min" or "max" in order to find the channel with the maximum or minimum value for further calculations, by default "min"
        replace_epochs : bool, optional
            A boolean flag indicating if the current epochs for the calculation should be replaced, by default False
        max_sample_diff : int, optional
            The maximum sample difference between channels that is still allowed for applying the time shift. Possible time shifts above this value will not be considered by setting it to the maximum value, by default 200

        Returns
        -------
        Numpy ndarray 
            A numpy array with the shape of epochs (trials, channels, sampels) after applying the spatial filter. 

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 05.02.2024 (by Niklas Kueper)
        """        
        
        if(erp_value_type == "min"): 
    
            min_vals = np.min(self.average_epochs, axis = 1)
            min_indices = np.argmin(self.average_epochs, axis = 1)

            source_index = min_indices[np.argmin(min_vals)]
            
            indices_diffs = min_indices - source_index

            coefficients = min_vals *(-1)
            norm_coefficients =  (coefficients/np.sum(coefficients))
            weighted_epochs = np.zeros(self.epochs.shape)

            resulting_channel_epochs = 0
            
            for trial_idx in range(0, self.epochs.shape[0]): 
                for channel_idx in range(0, self.epochs.shape[1]): 
                    weighted_epochs[trial_idx, channel_idx, :] = self.epochs[trial_idx, channel_idx, :] * norm_coefficients[channel_idx]

            # calc actual resulting channel
            resulting_channel_epochs = np.zeros((weighted_epochs.shape[0], weighted_epochs.shape[2]))
            for trial_idx in range(0, weighted_epochs.shape[0]): 
                weighted_trial = weighted_epochs[trial_idx, :, :] # channels, sampels
                for channel_idx in range(0, self.epochs.shape[1]): 
                    
                    weighted_trial_channel = weighted_trial[channel_idx, :] # sampels
                    
                    if ((indices_diffs[channel_idx] > 0) and (max_sample_diff > indices_diffs[channel_idx])): # if there is a delay after the source channel 

                        weighted_trial_channel_temp = copy.deepcopy(weighted_trial_channel)
                        weighted_trial_channel_temp = np.pad(weighted_trial_channel_temp, pad_width = (0, np.abs(indices_diffs[channel_idx])), mode='edge') # append array at end 
                        weighted_trial_channel_temp = weighted_trial_channel_temp[indices_diffs[channel_idx]:]
                        
                    elif ((indices_diffs[channel_idx] < 0) and (max_sample_diff*(-1) < indices_diffs[channel_idx])):
                        
                        weighted_trial_channel_temp = copy.deepcopy(weighted_trial_channel)
                        weighted_trial_channel_temp = np.pad(weighted_trial_channel_temp, pad_width = (np.abs(indices_diffs[channel_idx]), 0), mode='edge') # append array at end 
                        weighted_trial_channel_temp = weighted_trial_channel_temp[0: len(weighted_trial_channel_temp)-np.abs(indices_diffs[channel_idx])]
                    
                    else: 
                        weighted_trial_channel_temp = weighted_trial_channel

                    weighted_epochs[trial_idx, channel_idx, :] = weighted_trial_channel_temp

                resulting_channel_epochs[trial_idx, :] = np.sum(weighted_epochs[trial_idx, :, :], axis = 0) 

            if (replace_epochs): 
                self.epochs = weighted_epochs*self.epochs.shape[1] # calc weighted epochs with same scaling as before 

                self.average_epochs = np.mean(self.epochs, axis = 0)
            
            resulting_channel_epochs  = np.expand_dims(resulting_channel_epochs, axis=1)
                
            return resulting_channel_epochs

    
    def reshapeWindowsForCNNnets(self): 
        """
        This function reshapes the windows from multiple trials to be fitted for the CNN networks like EEGNet.
        This method is only required if windows are processed for more than one trial and windwo (do not use for single window processing)

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 22.11.2023 (by Niklas Kueper)  
        """        

        reshaped_EEG_windows= np.zeros((self.windows.shape[0]*self.windows.shape[3], self.windows.shape[1], self.windows.shape[2], 1)) # (n_trials * n_windows, n_channels, n_sampels, 1). 
        

        # get the features in one dim for all trials and windows 
        for channel_idx in range(0, self.windows.shape[1]):
            for sample_idx in range(0, self.windows.shape[2]): 

                reshaped_EEG_windows[:, channel_idx, sample_idx, 0] = self.windows[:, channel_idx, sample_idx, :].flatten()

        self.windows = reshaped_EEG_windows

    def labelsToCategorical(self, num_classes = 2): 
        """
        This method converts the class labels into the one hot encoding style. 

        Parameters
        ----------
        num_classes : int, optional
            Total number of classes. If None, this would be inferred as max(y) + 1, by default 2

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 05.02.2024 (by Niklas Kueper)
        """        

        y = to_categorical(self.labels, num_classes)

        self.labels = y

        
    def getWindows(self): 
        """
        This function returns the windowed data as a numpy array

        Returns
        -------
        Numpy array
            windows
                The windowed data as numpy array with shape: (n_trials, n_channels, n_sampels, n_windows).

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 05.02.2024 (by Niklas Kueper)
        """  

        return self.windows
    
    def getWindowNames(self): 
        """
        This funcion returns the identifier of each window

        Returns
        -------
        window_names : str
            The names (indentifier) of each window as a list of strings (has same size as n_windows).

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 05.02.2024 (by Niklas Kueper)
        """    

        return self.window_names
    
    def getFeatures(self): 
        """
        This functions returns the featuer vectors als floats

        Returns
        -------
        feature_vec : float
            A feature vector as numpy array, can have different shapes depending on the later used ML model (see ML class).

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 05.02.2024 (by Niklas Kueper)
        """  

        return self.feature_vec

    def getLabels(self): 
        """
        This functions returns the labels as a numpy array with floats

        Returns
        -------
        labels : float
            Numpy array of the labels
        
        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 05.02.2024 (by Niklas Kueper)
        """  

        return self.labels
    
    def onlineLRPWindowPredictionPostprocessing(self, window_wise_predicts, high_tresh, low_tresh, short_samp, long_samp): 
        """
        Apply an online capable postprocessing for the detection of LRP, where a linear function decides for the LRP class over which time a defined probability has to be reached for the detection of the positive class. 

        Parameters
        ----------
        window_wise_predicts : Numpy ndarray
            A numpy array with the predictions made on each window. Has the shape (n_trials, n_windows). 
        high_tresh : float 
            The higher treshold where the linear decision function ends. 
        low_tresh : float
            The lower treshold where the linear decision function starts. 
        short_samp : int
            The short sampels reflect how short the latest part or smallest chunk of data of interest is (i.e. only the late part or motor potential of an LRP).
        long_samp : int
            The long sampels reflecting how long the maximum ERP to be detected possibly is.  

        Returns
        -------
        Numpy ndarray 
            A numpy ndarray with shape (n_trials, n_windows) including the classification decision in the form of class labels (e.g. 1.0 or 0.0 for binary classification)
            
        Meta information: 
            Author: Niklas Kueper 
            Last changed: 05.02.2024 (by Niklas Kueper)
        """

        classified_windows = np.zeros((window_wise_predicts.shape[0], window_wise_predicts.shape[2])) # output shape (n_trials, n_windows)

        for trial_idx in range(0, window_wise_predicts.shape[0]): 
            for window_idx in range(0, window_wise_predicts.shape[2]): 
                
                # get prediction scores of current trial and window 
                current_predicts = window_wise_predicts[trial_idx, :, window_idx] # one second window data 

                tested_sampel_range = np.arange(short_samp, long_samp, step = -1)
                #print("sampel range", tested_sampel_range)
                tresh_step = -1*(high_tresh-low_tresh)/len(tested_sampel_range) # from high to low tresh (short sampels to long sampels)
                tested_tresh_range = np.arange(high_tresh, low_tresh, step = tresh_step)
                #print("Tresh range", tested_tresh_range)

                for index in range(0, len(tested_sampel_range)): 
                    mean_val = np.mean(current_predicts[tested_sampel_range[index]:]) 

                    if (mean_val > tested_tresh_range[index]): 
                        classified_windows[trial_idx, window_idx] = 1.0
                        break

        return classified_windows


    def windowEEGEpochs(self, window_size = 1000, window_step = 50, no_channel_dim = False):
        """
        This function cuts (overlapping) windows from continues EEG-signals (currently only for postprocessing without channel deimension)

        Parameters
        ----------
        window_size : int, optional
            The size of the windows in ms to be cutout, by default 1000
        window_step : int, optional
            The stepsize of the sliding window (slinding step) in ms, by default 50
        no_channel_dim : bool, optional
            Set to True if the EEG data (epochs) have no channel dimension, by default False
        
        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 05.02.2024 (by Niklas Kueper)
        """ 


        #if (with_channel_dim == False): 

        window_size_samp = int((window_size/1000) * self.__fsamp) 
        window_step_samp = int((window_step/1000) * self.__fsamp) 

        if(no_channel_dim): # support old way to postprocess
            epochs_arr_cut = self.epochs[:, 1:]
            num_of_windows =  int((epochs_arr_cut.shape[1]-window_size_samp)/window_step_samp)+1
        else: 
            epochs_arr_cut = self.epochs[:, :, 1:] # trials, channels, sampels 
            num_of_windows =  int((epochs_arr_cut.shape[2]-window_size_samp)/window_step_samp)+1


        #init window arrays with shape (trials, channels, sampel of window, windownumber)
        if(no_channel_dim): 
            wind_arr = np.zeros((epochs_arr_cut.shape[0], window_size_samp, num_of_windows))
        else:
            wind_arr = np.zeros((epochs_arr_cut.shape[0],epochs_arr_cut.shape[1], window_size_samp, num_of_windows))

        wind_names = []

        for win_nr in range(0, num_of_windows): 
            wind_start_idx = win_nr*window_step_samp
            wind_end_idx = window_size_samp+wind_start_idx

            if(no_channel_dim): 
                wind_name = "bis"+str(int((((epochs_arr_cut.shape[1]-wind_end_idx)*-1)/self.__fsamp) *1000))
            else: 
                wind_name = "bis"+str(int((((epochs_arr_cut.shape[2]-wind_end_idx)*-1)/self.__fsamp) *1000))
            
            wind_names.append(wind_name) # a list of all window names
            #create arrays for windows 

            if(no_channel_dim): 
                wind_arr[:, :, win_nr] = win_nr
                wind_arr[:, :, win_nr] = epochs_arr_cut[:, wind_start_idx:wind_end_idx]
            else:
                wind_arr[:, :, :, win_nr] = win_nr
                wind_arr[:, :, :, win_nr] = epochs_arr_cut[:, :, wind_start_idx:wind_end_idx]


        self.windows = wind_arr 
        #self.num_windows = num_of_windows
        self.window_names = wind_names


    def windowSelection(self, selected_windows): 
        """
        This function selects windwos and extract them from all windows segmented by specifying the window names

        Parameters
        ----------
        selected_windows : list of str
            The names of the windows (given after windowing) which are selected for further processing
        
        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 05.02.2024 (by Niklas Kueper)
        """        
       
        # interate over specified window names and extract the windows 
        indices_selected_winds = []
        for current_window_name in selected_windows: 
            indices_selected_winds.append(self.window_names.index(current_window_name))

        selected_windows_arr = self.windows[:, :, :, indices_selected_winds] 

        self.windows = selected_windows_arr
        self.window_names = selected_windows

    def windowStandardization(self, norm = False, use_min_max_norm = False):
        """
        This method can be used to standardize windowed time series data. 

        Parameters
        ----------
        norm : bool, optional
            Boolean flag wheather to normalize the window after standadizing the data, by default False
        use_min_max_norm : bool, optional
            If True, a minimum maximum normalization is applied to each window. Otherwise (default) the z-transform is applied, by default False

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 05.02.2024 (by Niklas Kueper)
        """        
        
        # shape: trials, channels, sampels, windows 
        for trial_idx in range(0, self.windows.shape[0]): 
            for channel_idx in range(0, self.windows.shape[1]): 
                for window_idx in range(0, self.windows.shape[3]): 
                    # get current window 
                    current_wind = self.windows[trial_idx, channel_idx, :, window_idx]
                    
                    if(use_min_max_norm): 
                        
                        current_wind_norm = current_wind - self.calib_mins[channel_idx]
                        current_wind_norm = current_wind_norm/((self.calib_mins[channel_idx]*-1)+self.calib_maxs[channel_idx])


                    else: 
                        # apply z-transform 
                        current_wind_norm = current_wind - self.calib_means[channel_idx] 
                        current_wind_norm_out = current_wind_norm/self.calib_stds[channel_idx]  
                        
                        if(norm): 
                        #current_wind_norm = current_wind+(-1*min)-1 # -1 is min 
                            current_wind_norm_out = current_wind_norm/np.max(current_wind_norm)


                    self.windows[trial_idx, channel_idx, :, window_idx] = current_wind_norm_out # 1 is max 

    def WindowMedianCorrection(self, ratio_len = 0.1):
        """
        Apply a median correction to each windowed timeseries data. 

        Parameters
        ----------
        ratio_len : float, optional
            The percentage of the window which is used for the median correction starting from the first point of each window (i.e. 0.1 reflects to the first 10% of the window used to calculated the median value to be corrected for), by default 0.1

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 05.02.2024 (by Niklas Kueper)
        """         

        end_idx = int(self.windows.shape[2]*ratio_len)

        # shape: trials, channels, sampels, windows 
        for trial_idx in range(0, self.windows.shape[0]): 
            for channel_idx in range(0, self.windows.shape[1]): 
                for window_idx in range(0, self.windows.shape[3]): 
                    # get current window 
                    current_wind = copy.deepcopy(self.windows[trial_idx, channel_idx, :, window_idx])

                    median_val = np.median(current_wind[0:end_idx])

                    #print("median value is ", median_val)
                    current_wind_med_corr = current_wind -median_val
                    
                    self.windows[trial_idx, channel_idx, :, window_idx] = current_wind_med_corr # 1 is max 


    def WindowMeanCorrection(self): 
        """
        Apply a mean correction on windowed time series data. 

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 05.02.2024 (by Niklas Kueper)
        """        

        # shape: trials, channels, sampels, windows 
        for trial_idx in range(0, self.windows.shape[0]): 
            for channel_idx in range(0, self.windows.shape[1]): 
                for window_idx in range(0, self.windows.shape[3]): 
                    # get current window 
                    current_wind = copy.deepcopy(self.windows[trial_idx, channel_idx, :, window_idx])

                    mean_val = np.mean(current_wind)

                    current_wind_mean_corr = current_wind -mean_val
                    
                    self.windows[trial_idx, channel_idx, :, window_idx] = current_wind_mean_corr # 1 is max 

    #TODO: Move this to ML lib 
    def calcTestAccAndRates(self, prediction_labels, true_labels):
        """
        This function returns the metrics from classification output of the test data. Currently the accuracy, balaned accuracy, tnr and tpr are calculated

        Parameters
        ----------
        prediction_labels : numpy array
            The prediced labels as 1D numpy array (flatten the array if it has more dimensions)
        true_labels : numpy array
            The true labels as 1D numpy array (flatten the array if it has more dimensions)

        Returns
        -------
        tuple
            tnr : float
                True negative rate
            tpr : float
                True positive rate
            acc : float
                Accuracy
            ba : float
                Balanced accuracy

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 28.11.2022 (by Niklas Kueper)
        """

        prediction_labels = prediction_labels.astype(int)
        true_labels = true_labels.astype(int)
        
        negatives = np.sum(np.abs(true_labels-1))
        positives = np.sum(true_labels)
        
        n = len(true_labels)
        tps= 0 
        tns = 0
        for index in range(0, n): 
            if (prediction_labels[index] == 0 and true_labels[index] == 0):
                tns = tns+1
            elif (prediction_labels[index] == 1 and true_labels[index] == 1): 
                tps = tps+1
        tnr = tns/negatives
        tpr = tps/positives
        acc = ((tns+tps)/n) #in percent
        ba = (tnr+tpr)/2

        return tnr, tpr, acc, ba
    
    def xDAWNSpatialfilter(self, n_components = 2, processing_type="fit_apply", return_filter = True, markernumber = 100, xd = None): 
        """
        This function implements an Xdawn filter algorithm. You can choose between only fit, only apply or both.

        Parameters
        ----------
        n_components : int, optional
            The number of components to decompose the signals, by default 2
        processing_type : str, optional
            Determans how you would like to use the function. If "fit" is parsed the Xdawn filter will be only fitted, if "apply" is parsed the Xdawn filter will be only applyed, "filter_apply" does both, by default "fit_apply"
        return_filter : bool, optional
            Returns the, by default True
        xd : instance of Xdawn, optional
            Parsing an instance of Xdawn for "apply" only purpose, by default None

        Returns
        -------
        xd : instance of Xdawn 

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 05.02.2024 (by Niklas Kueper)
        """        

        if (processing_type == "fit_apply"): # assuming this is only for training data or the epochs it should be fitted on 

            # Xdawn instance
            xd = Xdawn(n_components=n_components)
            
            # Fit xdawn
            xd.fit(self.epoch_obj)
            # apply 
            epochs_denoised = xd.apply(self.epoch_obj)
            self.epochs = epochs_denoised[markernumber].get_data()
            

            if(return_filter): 
                return xd 

        elif(processing_type == "fit"): # only fit 

            # Xdawn instance
            xd = Xdawn(n_components=n_components)
            
            # Fit xdawn
            xd.fit(self.epoch_obj)

            if(return_filter): 
                return xd 
        
        elif(processing_type == "apply"): # apply only (pass fitted filter !) 
            # apply only 
            epochs_denoised = xd.apply(self.epoch_obj)
            self.epochs = epochs_denoised[markernumber].get_data() 


    def applyxDAWNToWindows(self, xd, n_components = 2):
        """
        This function applies an already trained Xdawn filter to windowed timeseries data. 

        Parameters
        ----------
        xd : instance of Xdawn
            This is an instance of an Xdawn filter object
        n_components : int, optional
            The number of components to decompose the signals, by default 2

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 05.02.2024 (by Niklas Kueper)
        """         
        
        new_windows = np.zeros((self.windows.shape[0], n_components, self.windows.shape[2], self.windows.shape[3])) # reduced channel dim

        for window_idx in range(0, self.windows.shape[3]): 
            windows_filtered= xd.transform(self.windows[:, :, :, window_idx])
            new_windows[:, :, :, window_idx] = windows_filtered

        self.windows = new_windows # replace old windows 

    # TODO: move this to ML lib !
    def calcTrialMetric(self, predict_scores, pos_class_start_time, f_samp, decision_bound, num_class_instances): # TODO: Sample rate does not have to be a parameter here
        """
        This function seperates, calculates and returns the prediction of posivie and negative classes based on the specified bounds

        Parameters
        ----------
        predict_scores : Numpy ndarray 
            A numpy ndarray containing the prediction scores (shape: (n_trials, n_predictions)). 
        pos_class_start_time : float
            The time in ms where the positive class starts or is defined (binary classification only) in each trial. 
        f_samp : int
            The samplerate in Hz
        decision_bound : list
            A list specifying the bounds for positive class predictions
        num_class_instances : int
            The number of positive class instances

        Returns
        -------
        float
            ba : Balanced accuracy
            tnr : True negative rate
            tpr : True positive rate

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 05.02.2024 (by Niklas Kueper)
        """       

        pos_class_start_samp = int((pos_class_start_time/1000) * f_samp)
        tns = 0 
        tps = 0 
        fns = 0 
        fps = 0

        for trial in predict_scores: 
            # seperate the predictions of both classes 
            pos_class_predicts = trial[pos_class_start_samp:]
            neg_class_predicts = trial[0:len(trial)+pos_class_start_samp]

            # calc the number of times the score is over the decision bound  
            pos_class_predicts_over_bound = np.sum(pos_class_predicts > decision_bound) 
            neg_class_predicts_over_bound = np.sum(neg_class_predicts > decision_bound) 

            # calc metrics on numbers of times 
            if(pos_class_predicts_over_bound > num_class_instances): # at leat one pos class instance detected = true prediction 
                tps = tps +1  
            else: 
                fns = fns+1 # no pos class instance detected = falsely predicted negative class 

            if(neg_class_predicts_over_bound < num_class_instances): # no pos class detected, all true 
                tns = tns +1
            else: 
                fps = fps +1 # wrongly detected positive class 

            # calc rates and metric 
            tnr = tns/(tns+fps) 
            tpr = tps/(tps+fns)
            ba = (tnr+tpr)/2

        return ba, tnr, tpr 
    

    def OnechannelFFT(self, one_channel_data, plot = False, window = "hamming", beta = 1, title = None):
        """
        This function calculates the FFT for one channel of the timeseries data

        Parameters
        ----------
        one_channel_data : Numpy array
            The data for one channel
        plot : bool, optional
            If True the FFT spectrum of the data is plotted, by default False
        window : str, optional
            The windowing function that can be applied before the FFT is calculated (see usable scipy window functions), by default "hamming"
        beta : int, optional
            Shape parameter, determines trade-off between main-lobe width and side lobe level. As beta gets large, the window narrows, by default 1
        title : str, optional
            The Title of the plot, by default None

        Returns
        -------
        Numpy array
            xf : float
            The frequency axis
            yfn : float
            The FFT magnitude values of the frequency spectrum

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 22.11.2023 (by Niklas Kueper)
        """         

        N = len(one_channel_data)
        # sample spacing
        dT = 1.0/self.__fsamp
        x = np.linspace(0.0, N*dT, N, endpoint=False)
        
        # apply window before calculating fft 
        if(window == "hamming"): 
            window_fct = sig.windows.hamming(N)
            y = window_fct*one_channel_data
        elif (window == "hann"): 
            window_fct = sig.windows.hann(N)
            y = window_fct*one_channel_data

        elif (window == "kaiser"): 
            window_fct = sig.windows.kaiser(N, beta = beta)
            y = window_fct*one_channel_data

        else: 
            y = one_channel_data


        yf = fft(y)

        xf = fftfreq(N, dT)[:N//2]
        yfn = 2.0/N * np.abs(yf[0:N//2])


        if (plot): 
            fig = plt.figure()
            plt.plot(xf, yfn)
            plt.ylabel("|H|")
            if(title): 
                plt.title(title)
            else: 
                plt.title("FFT Spectrum")
            plt.xlabel("Frequency in Hz")
            plt.show()

        return xf, yfn
     
    
    def detrendWindows(self):
        """
        This funcition uses detrending to remove linear trends from each window of the data.
        Self.windwos array will contain detrended time series data for each trial, window, and channel

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 05.02.2024 (by Niklas Kueper)
        """  

        # get the features in one dim for all trials and windows 
        for trial_idx in range(0, self.windows.shape[0]):
            for window_idx in range(0, self.windows.shape[3]):
                for channel_idx in range(0, self.windows.shape[1]):
                    current_window = self.windows[trial_idx, channel_idx, :, window_idx]
                    current_window_corr = sig.detrend(current_window)
                    self.windows[trial_idx, channel_idx, :, window_idx] = current_window_corr


    def featureExtractionFromWindows(self,  feature_type = "timepoints", feature_indices_windows = None, use_mean = False, N = 1, add_neightbour_diffs  = False, neighbours_list = [("C1", "CZ")], psd_method = "multitaper"): 
        """
        Apply method to extract time or frequency features from time series data. See feature_types parameter for the types of features that are supported. 

        Parameters
        ----------
        feature_type : str, optional
            _description_, by default "timepoints"
        feature_indices_windows : Numpy array, optional
            Numpy array with time feature indices inside the window in ms, by default None
        use_mean : bool, optional
            If True, the mean of the timepoints is calculated as features, by default False
        N : int, optional
            The numbers of samples when calculating mean features, by default 1
        add_neightbour_diffs : bool, optional
            If True, the differences between channel features are added as additional features (e.g. for neighbour channels), by default False
        neighbours_list : list, optional
            A list of tuples specifying the neighbour channels for adding the neighbour features (only used when add_neighbour_diffs == True), by default [("C1", "CZ")]
        psd_method : str, optional
            The method to be used for calculating psd features (see compute_pow_freq_bands of mne_features for detailled information), by default "multitaper"

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 08.03.2024 (by Niklas Kueper)
        """        

        # (n_trials, n_channels, n_sampels, n_windows).
        #print("windows shape", self.windows.shape)

        if(feature_type == "timepoints"): 
            
            feature_times_indices = ((feature_indices_windows/1000)*self.__fsamp).astype(int)

            # init stuff 
            if(use_mean): 
                x_train_features = np.zeros((self.windows.shape[0], self.windows.shape[3], N*self.windows.shape[1]))

            else: 
                x_train_features = np.zeros((self.windows.shape[0], self.windows.shape[3], int(len(feature_times_indices)*self.windows.shape[1]))) # shape: trials, windows, features

            if(add_neightbour_diffs): 
                 x_train_features_add = np.zeros((self.windows.shape[0], self.windows.shape[3], len(feature_times_indices)*len(neighbours_list))) # shape: trials, windows, features

            mean_feat_buffer = np.zeros((self.windows.shape[1], N))


            # get the features in one dim for all trials and windows 
            for trial_idx in range(0, self.windows.shape[0]):
                for window_idx in range(0, self.windows.shape[3]): 
                    
                    if(use_mean):
                        # shape channel, sampels
                        #  
                        current_wind = self.windows[trial_idx, :, feature_times_indices[0]:feature_times_indices[-1], window_idx] # use the first and las value only 
                        k = int(current_wind.shape[1]/N) 
                        

                        for idx in range(0, N): 
                            mean_feat_buffer[:, idx] = np.mean(current_wind[:, (idx*k):((idx*k) +k)], axis = 1) 

                        x_train_features[trial_idx, window_idx, :] = mean_feat_buffer.flatten() # use mean of timepoints

                    else: 
                        #print("feature_times_indices", feature_times_indices) 
                        x_train_features[trial_idx, window_idx, :] = self.windows[trial_idx, :, feature_times_indices, window_idx].flatten()

                    # neighbour diff features 
                    if (add_neightbour_diffs): # if you want to add local feature diffs 
                        features_add = np.zeros((x_train_features.shape [0], x_train_features.shape[1], len(feature_times_indices), len(neighbours_list))) # if use mean values only one dim 
                        i = 0

                        for channel_tup in neighbours_list: # loop over all indice values and calc diff of features 
                            idx1 = self.__channel_names.index(channel_tup[0])
                            idx2 = self.__channel_names.index(channel_tup[1])

                            # shape: trials, windows, features 
                            current_wind_feat = self.windows[trial_idx, :, feature_times_indices, window_idx].T # get current window features: channel, sampels

                            features_add[trial_idx, window_idx, :, i] = (current_wind_feat[idx1, :] - current_wind_feat[idx2, :])
                            i = i+1

                        # flatten the feature dims 
                        x_train_features_add[trial_idx, window_idx, :] = features_add[trial_idx, window_idx, :, :].flatten()

                    
            # flatten the trials and windows as train instances 
            x_train = np.zeros((x_train_features.shape[0]*x_train_features.shape[1], x_train_features.shape[2]))
            #print(x_train.shape)


            # train data for neighbour condition 
            if(add_neightbour_diffs): 
                x_train_add = np.zeros((x_train_features_add.shape[0]*x_train_features_add.shape[1], x_train_features_add.shape[2]))

                for feature_idx in range(0, x_train_features_add.shape[2]):
                    x_train_add[:, feature_idx] = x_train_features_add[:, :, feature_idx].flatten()
        
            # flatten data 
            for feature_idx in range(0, x_train_features.shape[2]):
                x_train[:, feature_idx] = x_train_features[:, :, feature_idx].flatten() #(n_train, features)
            

        elif(feature_type == "meanfreqs" or feature_type == "medianfreqs"): #fix this 


            # (n_trials, n_channels, n_sampels, n_windows).
            num_of_freq_bands = 5 

            if(add_neightbour_diffs): 
                 x_train_features_add = np.zeros((self.windows.shape[0], self.windows.shape[3], num_of_freq_bands*len(neighbours_list))) # shape: trials, windows, features

            x_train_features = np.zeros((self.windows.shape[0], self.windows.shape[3], num_of_freq_bands*self.windows.shape[1])) # shape: trials, windows, features

            #print(x_train_features.shape)
            features = np.zeros((num_of_freq_bands, self.windows.shape[1]))
            
            # get the features in one dim for all trials and windows 
            for trial_idx in range(0, self.windows.shape[0]):
                for window_idx in range(0, self.windows.shape[3]):
                    for channel_idx in range(0, self.windows.shape[1]):
                         
                        xf, yf = self.OnechannelFFT(self.windows[trial_idx, channel_idx,:, window_idx], self.__fsamp) # do fft of sliced data

                        normed_freqs = (xf*yf)/np.sum(yf) # norm the frequencies

                        if(feature_type == "meanfreqs"): 
                            freqs_arr = np.array([np.mean(normed_freqs[0:4]), np.mean(normed_freqs[4:8]), np.mean(normed_freqs[8:14]), np.mean(normed_freqs[14:30]), np.mean(normed_freqs[30:40])]) #  mean band frequencies
                        else: 
                            freqs_arr  = np.array([np.median(normed_freqs[0:4]), np.median(normed_freqs[4:8]), np.median(normed_freqs[8:14]), np.median(normed_freqs[14:30]), np.median(normed_freqs[30:40])]) #  mean band frequencies

                        features[:, channel_idx] = freqs_arr # freq band features for each channels 

                    if (add_neightbour_diffs): # if you want to add local feature diffs 
                        features_add = np.zeros((features.shape[0], len(neighbours_list)))
                        i = 0

                        for channel_tup in neighbours_list: # loop over all indice values and calc diff of features 
                            idx1 = self.__channel_names.index(channel_tup[0])
                            idx2 = self.__channel_names.index(channel_tup[1])
                            features_add[:, i] = features[:, idx1] -features[:, idx2] # calculate local feature diffs 
                            i = i+1

                        # flatten the feature dims 
                        x_train_features_add[trial_idx, window_idx, :] = features_add.flatten()

                    x_train_features[trial_idx, window_idx, :] = features.flatten()


            # flatten the trials and windows as train instances 
            x_train = np.zeros((x_train_features.shape[0]*x_train_features.shape[1], x_train_features.shape[2]))
            #print(x_train.shape)

            for feature_idx in range(0, x_train_features.shape[2]):
                x_train[:, feature_idx] = x_train_features[:, :, feature_idx].flatten()

            # train data for neighbour condition 
            if(add_neightbour_diffs): 
                x_train_add = np.zeros((x_train_features_add.shape[0]*x_train_features_add.shape[1], x_train_features_add.shape[2]))

                for feature_idx in range(0, x_train_features_add.shape[2]):
                    x_train_add[:, feature_idx] = x_train_features_add[:, :, feature_idx].flatten()


        elif(feature_type == "freqBandPower"): 
            
            #windows (n_trials, n_channels, n_sampels, n_windows).
            
            freq_bands = np.array([0.5, 4., 8., 13., 30., 100.]) # default that is used 
            #freq_bands = np.array([0.5, 1.0, 2.5, 4., 5.5, 6.5, 8. ,9.5, 11.5, 13., 16., 25., 30., 40.]) # default that is used
            num_of_freq_bands = len(freq_bands)-1

            if(add_neightbour_diffs): 
                 x_train_features_add = np.zeros((self.windows.shape[0], self.windows.shape[3], num_of_freq_bands*len(neighbours_list))) # shape: trials, windows, features

            x_train_features = np.zeros((self.windows.shape[0], self.windows.shape[3], num_of_freq_bands*self.windows.shape[1])) # shape: trials, windows, features

            #print(x_train_features.shape)
            features = np.zeros((num_of_freq_bands, self.windows.shape[1])) # 5, 34
            
            # get the features in one dim for all trials and windows 
            for trial_idx in range(0, self.windows.shape[0]):
                for window_idx in range(0, self.windows.shape[3]):
                         
                    current_wind = self.windows[trial_idx, :, :, window_idx] # shape channel, sampels
                    features = mne_feat.compute_pow_freq_bands(sfreq = self.__fsamp, data =current_wind, freq_bands=freq_bands, normalize = False, psd_method = psd_method)
                    x_train_features[trial_idx, window_idx, :] = features

            # flatten the trials and windows as train instances 
            x_train = np.zeros((x_train_features.shape[0]*x_train_features.shape[1], x_train_features.shape[2]))
            #print(x_train.shape)

            for feature_idx in range(0, x_train_features.shape[2]):
                x_train[:, feature_idx] = x_train_features[:, :, feature_idx].flatten()

            # train data for neighbour condition 
            if(add_neightbour_diffs): 
                x_train_add = np.zeros((x_train_features_add.shape[0]*x_train_features_add.shape[1], x_train_features_add.shape[2]))

                for feature_idx in range(0, x_train_features_add.shape[2]):
                    x_train_add[:, feature_idx] = x_train_features_add[:, :, feature_idx].flatten()


        # how to proceed with features (both conditions)
        if(add_neightbour_diffs): 
            self.feature_vec = np.concatenate((x_train, x_train_add), axis = 1)
        else: 
            self.feature_vec = x_train 


    def featureExtractionReshapedWindows(self,  feature_type = "timepoints", feature_indices_windows = None, psd_method = "multitaper"):
        """
        Apply method to extract time or frequency features from time series data. See feature_types parameter for the types of features that are supported. 
        It is the similar to featureExtractionFromWindows but operates on reshaped windows for CNN nets. Neighbour channel diffs and mean features are no longer supported (not used). 
        
        Parameters
        ----------
        feature_type : str, optional
            _description_, by default "timepoints"
        feature_indices_windows : Numpy array, optional
            Numpy array with time feature indices inside the window in ms, by default None
        psd_method : str, optional
            The method to be used for calculating psd features (see compute_pow_freq_bands of mne_features for detailled information), by default "multitaper"

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 24.04.2024 (by Niklas Kueper)
        """        

        # (n_trials, n_channels, n_sampels, n_windows).
        #print("windows shape", self.windows.shape)

        if(feature_type == "timepoints"): 
            
            feature_times_indices = ((feature_indices_windows/1000)*self.__fsamp).astype(int)
            # windows have now shape: (trials, channels, sampels, windows)
            shape_windows = self.windows.shape

            #x_train_features = copy.deepcopy(self.windows).reshape((shape_windows[0]*shape_windows[3], shape_windows[1], shape_windows[2])) # is now only (windows, channels, sampels)

            x_train_extracted = copy.deepcopy(self.windows[:, :, feature_times_indices, 0]) 
            x_train = x_train_extracted.reshape((x_train_extracted.shape[0], x_train_extracted.shape[1]*x_train_extracted.shape[2])) # join channels and sampels together 
            
            # set this ? 
            self.feature_vec = x_train 
            
        elif(feature_type == "freqBandPower"): 

            #windows (n_trials, n_channels, n_sampels, n_windows).

            freq_bands = np.array([0.5, 4., 8., 13., 30., 100.]) # default that is used 
            num_of_freq_bands = len(freq_bands)-1

            x_train_features = copy.deepcopy(self.windows[:, :,:, 0]) # make a copy 
            shape_features = x_train_features.shape
            #x_train_extracted = np.zeros((shape_features[0], num_of_freq_bands, shape_features[1])) # is now shape: (instances, features, channels) 
            x_train = np.zeros((shape_features[0], shape_features[1]*num_of_freq_bands)) # final shape is (instances, features)
            
            # get the features in one dim for all trials and windows 
            for window_idx in range(0, shape_features[0]): # over all training instances 
                
                features_window = mne_feat.compute_pow_freq_bands(sfreq = self.__fsamp, data =x_train_features[window_idx,:,:], freq_bands=freq_bands, normalize = False, psd_method = psd_method)
                x_train[window_idx, :] = features_window.flatten()

        self.feature_vec = x_train 


    def addFeatures(self, x):
        """
        This function add features to the feature_vec by concatinating them

        Parameters
        ----------
        x : numpy array
            The features that should be added. An array of floats with the same shape as feature_vec

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 08.03.2024 (by Niklas Kueper)
        """ 

        features = np.concatenate((self.feature_vec, x), axis = 1)
        self.feature_vec = features 

    def printFeatureShape(self):
        """
        This function prints the shape of feature_vec

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 08.03.2024 (by Niklas Kueper)
        """   

        print("feature shape: ", self.feature_vec.shape)

    def setWindowLabels(self, label_list):
        """
        Set /encode the class labels of segmented windows for the classifiction task. 
        The labels are reshaped to generate the same shape as after the feature extraction or after using the reshapeWindows

        Parameters
        ----------
        label_list : list of floats
            A list of labels that correspond to the window class labels (e.g. [0.0, 0.0, 1.0, 1.0])

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 22.07.2023 (by Niklas Kueper)
        """
        
        y = np.zeros((self.windows.shape[0], self.windows.shape[3])) # shapes trials, windows 
        for trial_idx in range(0, y.shape[0]): # over trials 
            y[trial_idx, :] = np.array(label_list)  # shape: trials, window labels

        y = y.flatten() # flatten the labels
        y_temp = np.zeros((y.shape[0], 1))
        y_temp[:, 0] = y
        y = y_temp

        self.labels = y 
    
    def setWindows(self, windows): 
        """
        Set the windows (numpy) array of the class 

        Parameters
        ----------
        windows : Numpy ndarray
            A numpy ndarray with shape: (trials, channels, sampels, windows)
        """

        self.windows = windows


    def dtwFeatureVecWindows(self):
        """
        This method generates a feature vector for the dtw algorithm based on windowed data. 

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 08.03.2024 (by Niklas Kueper)
        """  

        # windows in trials, channel, sampels, windows 
        self.feature_vec = np.zeros((int(self.windows.shape[0]*self.windows.shape[3]), self.windows.shape[1], self.windows.shape[2]))

        for channel_idx in range(0, self.windows.shape[1]): 
            for sample_idx in range(0, self.windows.shape[2]): 
                self.feature_vec[:, channel_idx, sample_idx] = self.windows[:, channel_idx, sample_idx, :].flatten()
    


    def calcEEGWindowOnset(self, window_predicts, num_pos_windows):
        """
        The function returns a numpy array, indicating the onset of positive labels for each trial and window.

        Parameters
        ----------
        window_predicts : Numpy array
            Representing predicted labels for each trial and window
        num_pos_windows : int
            The number of consecutive positive windows required to trigger an onset

        Returns
        -------
        Numpy array
            onset_window_predicts : indicating the onset of positive labels for each trial and window

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 08.03.2024 (by Niklas Kueper)
        """         

        onset_window_predicts = np.zeros(window_predicts.shape)
        trial_idx = 0

        for trial in window_predicts:
            count_pos_windows = 0 

            for wind_idx in range(0, window_predicts.shape[1]): 
                if(trial[wind_idx] > 0.5): # window has pos label 
                    count_pos_windows = count_pos_windows+1
                
                if (count_pos_windows >= num_pos_windows): 
                    onset_window_predicts[trial_idx, wind_idx] = 1.0 
                    count_pos_windows = 0
                    break

            trial_idx = trial_idx+1

        return onset_window_predicts


    def calcTrialMetricWindows(self, predict_labels, bounds, num_class_instances):
        """
        This function seperates, calculates and returns the prediction of posivie and negative classes based on the specified bounds


        Parameters
        ----------
        predict_labels : list   
            A list of predicted labels for each trial
        bounds : list
            A list specifying the bounds for positive class predictions
        num_class_instances : int
            The number of positive class instances

        Returns
        -------
        float
            ba : Balanced accuracy
            tnr : True negative rate
            tpr : True positive rate

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 08.03.2024 (by Niklas Kueper)
        """ 

        tns = 0 
        tps = 0 
        fns = 0 
        fps = 0

        for trial in predict_labels: 
            # seperate the predictions of both classes 
            pos_class_predicts = trial[bounds[0]:bounds[1]]
            neg_class_predicts = trial[0:bounds[0]]
            
            # calc the number of times the score is over the decision bound  
            pos_class_predicts_over_bound = np.sum(pos_class_predicts) 
            neg_class_predicts_over_bound = np.sum(neg_class_predicts) 

            # calc metrics on numbers of times 
            if(pos_class_predicts_over_bound >= num_class_instances): # at leat one pos class instance detected = true prediction 
                tps = tps +1  
            else: 
                fns = fns+1 # no pos class instance detected = falsely predicted negative class 

            if(neg_class_predicts_over_bound < num_class_instances): # no pos class detected, all true 
                tns = tns +1
            else: 
                fps = fps +1 # wrongly detected positive class 

            # calc rates and metric 
            tnr = tns/(tns+fps) 
            tpr = tps/(tps+fns)
            ba = (tnr+tpr)/2

        return ba, tnr, tpr 


    def applyRelabelling(self, predicted_labels, determine_labels, searching_bounds):
        """
        This function apllies the relabelling method to the classification output in order to get the "true ground truth" labels. This function schould be carefully use since it creates new ground truth labels for the evaluation of the classifier!

        Parameters
        ----------
        predicted_labels : Numpy array
            The predicted labels as 1D-Numpy array (flatten the arry if it has more dimensions)
        determine_labels : int
            The amount of negative classes that are counted from the right side(end of each epoch/trial to specify the "label change point")
        searching_bounds : list
            A list with boundaries([lower bound, upper bound]) in which the label change point for the relabelling is searched (e.g. for LRP the numbers of the window for -1000 ms and 0 ms)

        Returns
        -------
        Numpy array
            new_true_labels : Containing the new ground truth labels (0.0 neg class; 1.0 pos class) with shape (n_trials, n_samples)

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 28.11.2022 (by Niklas Kueper)
        """    

        new_true_labels = np.zeros(predicted_labels.shape)
        new_true_labels[:, -1] = 1.0 # last label has to be positive class 

        #predicted_labels_cut = predicted_labels[:, searching_bounds[0]:searching_bounds[1]] # cut to searching bounds where the label change point is searched 
        
        trial = 0 

        for predicted_trial_label in predicted_labels: 
            # reset for every trial 
            neg_class_counter = 0
            index =  searching_bounds[1]-1 # do this like that for now but change afterwards,  upper bound could lay in the middle ! 

            while index > searching_bounds[0]: # go from the back to front until the minimum point 
                # count number of neg classes 
                if (predicted_trial_label[index] == 0): 
                    neg_class_counter = neg_class_counter+1
                else: 
                    neg_class_counter = 0
                
                if(neg_class_counter >= determine_labels): 
                    new_true_labels[trial, index+determine_labels:] = 1.0 # set true labels after label change point
                    break # stop when change point was found
                
                index = index -1

            # no consecutive determine label numbers of neg class found 
            if(index <= searching_bounds[0] and neg_class_counter < determine_labels): 
                new_true_labels[trial, searching_bounds[0]:] = 1.0

            trial = trial +1

        return new_true_labels


    def calcWindowMetrics(self, wind_arr, evaluation_time_per_window, window_step, f_samp_eeg, n_samp_features, use_relabelling, determine_labels, searching_bounds):
        """
        This function calculates the metrics of a window wise classification output

        Parameters
        ----------
        wind_arr : Numpy array
            The windowes class predictions with shape (n_trials, n_samples, n_windows)
        evaluation_time_per_window : int
            The time in ms at the end of each window for which the window metric is calculated (-evaluation_time_per_window to 0 ms are used)
        window_step : int
            The slinding step size of the windows in ms (standard is 50 ms)
        f_samp_eeg : int
            The sampling rate of the EEG-data in Hz
        n_samp_features : int
            The number of samples that are used as features (-n_samp_features: 0 of each epoch)
        use_relabelling : bool
            If True, the relabelling method is applied to calculate the metrics. This method should be trated with care since it effects the classification perfomance!
        determine_labels : int
            The number of concecutive negative classes that are counted when estimating the label change point of both classes
        searching_bounds : list
            A list of the lower and upper bound of window numbers ([lower bound, upper bound]) where the label change point is searched

        Returns
        -------
        float
            tnr : True negative rate
            tpr : True positive rate
            acc : Accuracy
            ba : Balanced accuracy
            window_predictions : The prediction values (0 - 1) for all windows and trials
        Numpy array
            window_eval_true_labels : float
                The true labels that are specified or the new true labels when relabelling is used

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 29.11.2022 (by Niklas Kueper)
        """         

        window_step_samp = int((window_step/1000) * f_samp_eeg) 
        evaluation_samp_per_window = int((evaluation_time_per_window/1000) * f_samp_eeg) 
        window_predictions_samp = np.sum(wind_arr[:, -evaluation_samp_per_window:, :], axis = 1)
        
        # convert the label of each sampels to windowwise labels 
        window_predictions = (window_predictions_samp > evaluation_samp_per_window/2).astype(float)
        
        
        window_eval_true_labels = np.zeros(window_predictions.shape)
        num_erp_windows = int(np.round(n_samp_features/window_step_samp)) 
        trial_erp_label = window_eval_true_labels[0, :] 
        trial_erp_label[-num_erp_windows:] = 1.0

        for trial_nr in range(0, window_eval_true_labels.shape[0]): 
            window_eval_true_labels[trial_nr, :] = trial_erp_label

        #get metrics for window evaluation 
        if (use_relabelling == False): 
            tnr, tpr, acc, ba = calcTestAccAndRates(window_predictions.flatten(), window_eval_true_labels.flatten())
        else: 
            relabelled_true_labels = applyRelabelling(window_predictions, determine_labels, searching_bounds)
            tnr, tpr, acc, ba = calcTestAccAndRates(window_predictions.flatten(), relabelled_true_labels.flatten())
            window_eval_true_labels = relabelled_true_labels # return the relabelled true labels instead 

        return tnr, tpr, acc, ba, window_predictions, window_eval_true_labels
    
    def calcSampleLossFromSameSamples(self, num_allowed_samples = 5):
        """
        A helping method for calculating how much samples are lost in a recording for the cometa system (wireless connection loss). 

        Returns
        -------
        numpy ndarray
            A numpy array with shape: (channels, sampels)) containing the numbers of samples that are lost for each recording channel and index. Only the total amount of lost samples are shown for each index. 

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 02.02.2024 (by Niklas Kueper)

        """

        loss_count_local = 0 
        loss_array = np.zeros(self.data.shape) 

        if (True): # change later 
            for channel_idx in range(0, self.data.shape[0]): 
                is_this_samp_loss = False
                is_last_samp_loss = False
                print("loss channel: " +str(channel_idx))

                for sample_idx in range (1, self.data.shape[1]): 
                    is_last_samp_loss = is_this_samp_loss  # update 

                    if(self.data[channel_idx, sample_idx] == self.data[channel_idx, sample_idx-1]): # same value as before 
                        loss_count_local = loss_count_local+1
                        is_this_samp_loss = True 
                    else: 
                        is_this_samp_loss = False

                    
                    if(is_last_samp_loss and not is_this_samp_loss): # the loss is over at this point 
                        loss_array[channel_idx, sample_idx] = loss_count_local
                        loss_count_local = 0 


                loss_count = np.sum(loss_array[channel_idx, :] > num_allowed_samples) # calc actual number of losses above threshold  

                print("number of data losses: ", loss_count)

            return loss_array

        else: 
            warnings.warn("not implemented for other data stages, terminating ... ")
            return None


class OnlineTimeseriesStreaming(): 

    def __init__(self, stream_type = "data", channel_names = ["1", "2", "3"], n_channels=3, n_samples= 500, dt_process_data = 0.05, f_samp = 1000.0): 
        """
        This class is used for provide and handle online streamed time series data. 

        Parameters
        ----------
        stream_type : str, optional
            The type of the stream that should be created or used. Can be "data" for timeseries data or "impedance" for receiving/sending impedance values, by default "data"
        channel_names : list, optional
            A list of channel names of the timeseries data. Might not be used in case a stream is providing the channel names automatically, by default ["1", "2", "3"]
        n_channels : int, optional
            The number of time series channels. Might not be used in case a stream is providing the channel names automatically, by default 3
        n_samples : int, optional
            The number of timeseries samples that are stored and updated in a buffer for each channel, by default 500
        dt_process_data : float, optional
            The time interval in which new data should be received/the buffer updated. Reflect to the loop frequency of the processing, by default 0.05
        f_samp : float, optional
            The sampling rate in Hz, by default 1000.0

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 13.03.2024 (by Niklas Kueper)
        """ 
    
        # init params for stream (not known directly )
        self.data_stream = None
        self.impedance_stream = None
        self.data_chunk = None
        self.impedance_chunk = None
        self.client_type = None

        # general params for streaming timeseries data 
        self.__f_samp = f_samp
        self.stream_type = stream_type
        self.n_channels = n_channels
        self.buffersize = n_samples
        self.dt_process_data = dt_process_data
        self.channel_names = channel_names
        self.data_buffer = np.zeros((1, n_channels, self.buffersize, 1)) # data buffer has shape (trials, n_channels, sampels, windows)

        self.last_loop_time = 0.0


    def startANTEegoStreaming(self, path_to_so_file):
        """
        This methods starts a data stream based on the ANT Eego SDK for receiving real time data from the device. 

        Parameters
        ----------
        path_to_so_file : str
            A string containing the path were the .so (shared object) file for the ANT SDK is stored. 

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 13.03.2024 (by Niklas Kueper)
        """      


        self.client_type = "ANTEego" # set client type for data reading 
        
        sys.path.append(path_to_so_file) # append file path of the SDK .so (shared object) file 
        
        import eego_sdk # import here because path is not clear yet 

        #starting and init 
        factory = eego_sdk.factory()
        v = factory.getVersion()
        print('version: {}.{}.{}.{}'.format(v.major, v.minor, v.micro, v.build))
        print('delaying to allow slow devices to attach...')
        time.sleep(1)

        
        amplifiers=factory.getAmplifiers()
        print("available amplifiers:", amplifiers)
        
        if(amplifiers): 
            self.amplifier = amplifiers[0] # only use  first amplifier in list (no daisy chaining/multiple amps integrated currently)
            print(type(self.amplifier))
            print("connecting to: ", self.amplifier)


            rates = self.amplifier.getSamplingRatesAvailable()
            self.ref_ranges = self.amplifier.getReferenceRangesAvailable()
            self.bip_ranges = self.amplifier.getBipolarRangesAvailable()
            self.channel_names = self.amplifier.getChannelList()

            print('  amplifier: {}'.format('{}-{:06d}-{}'.format(self.amplifier.getType(), self.amplifier.getFirmwareVersion(), self.amplifier.getSerialNumber())))
            print('  rates....... {}'.format(rates))
            print('  ref ranges.. {}'.format(self.ref_ranges))
            print('  bip ranges.. {}'.format(self.bip_ranges))
            print('  channels.... {}'.format(self.channel_names))

            
            #create stream object for getting EMG/EEG data 
            if (self.__f_samp == 1000): 
                samp_index = 2
            elif (self.__f_samp == 500): 
                samp_index = 0
            else: 
                warnings.warn("specify valid sampling rate, using 1000 Hz now ")
                samp_index =2  # default is 500 Hz

            # differentiate between data stream and impedance stream 
            if(self.stream_type == "data"): 
            
                # open stream 
                self.data_stream = self.amplifier.OpenEegStream(rates[samp_index], self.ref_ranges[0], self.bip_ranges[0])

                time.sleep(0.15) # after that it starts writing data to the buffer approx. 
                self.last_loop_time = perf_counter()
                
                # TODO: send synchronization event when starting the measurement 

                self.n_channels = len(self.data_stream.getChannelList())
                print("data stream has detecte n channels:", self.n_channels)

            elif(self.stream_type == "impedance"): 

                self.impedance_stream = self.amplifier.OpenImpedanceStream()

                print('stream:')    
                print('  channels.... {}'.format(self.impedance_stream.getChannelList()))
                print('  impedances.. {}'.format(list(self.impedance_stream.getData())))
        else: 
            warnings.warn("no amplifier found, terminating ...")
                    

    def getChunk(self, return_chunk = False): 
        """
        This method is used to read the data from a created stream for further processing. If required the data chunk can be returned. 
        Ensure that this method is called with a fixed frequency to avoid that any (internal) buffers from lsl or orther clients are overflowing.  

        Parameters
        ----------
        return_chunk : bool, optional
            If True, the read data chunk will be returned, by default False

        Returns
        -------
        list
            A list including the received data chunk. 

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 13.03.2024 (by Niklas Kueper)
        """      

        if(self.client_type == "ANTEego"): 
             
            try:
                if(self.stream_type == "data"):
                    self.data_chunk = (self.data_stream.getData()) # read EMG/EEG data out of buffer
                    
                    if(return_chunk): 
                        return np.array(self.data_chunk) 
                else: 
                    self.impedance_chunk = (self.impedance_stream.getData()) # read EMG/EEG data out of buffer
                    if(return_chunk):
                        return np.array(self.impedance_chunk) 

            except Exception as e:
                print('error: {}'.format(e))
                print(f"Is a data stream already created ? ")
    

   # same as in the EEG toolbox, remove later on 
    def updateBuffer(self, channel_indices = None, show_data_shape = False):  #current_local_time, timestamp_offset, 
        """
        This function provides the most recent data samples and timestamps in a buffer (fist val is oldest, last the newest)

        Parameters
        ----------
        chunk : list
            Current data chunk with shape (samples, channels)
        channel_indices : list, optional
            If only selected channel indices should be extraced, by default None
        check_sample_loss : bool, optional
            If True the function is checkinf for sample losses, by default True
        show_data_shape : bool, optional
            If True, the shape of the data chunk (numpy array) is shown to check the proper dimesions of incoming data, by default False 

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 13.03.2024 (by Niklas Kueper)
        """        

        #data 
        #print("type of chunk 1", type(self.data_chunk))
        if(self.data_chunk): # only to this if new data is received 
            current_chunk = (np.array(self.data_chunk).T) # chunk is sampels, channels, after transpose then channels, sampels !


            if(channel_indices): 
                current_chunk = current_chunk[channel_indices, :]

            if(show_data_shape):
                print("data chunk shape:", current_chunk.shape) # should be in channels, sampels

            # #print(current_chunk.shape)
            # current_chunk = current_chunk[0:n_channels, :] # use first n channels

            n_samples = current_chunk.shape[1] # how much new samples 

            if (n_samples > self.data_buffer.shape[2]): # print error message 
                print("Buffer overflow")

            #print("self.data_buffer.shape", self.data_buffer.shape)

            self.data_buffer = np.roll(self.data_buffer, shift = int(-1*n_samples), axis = 2) # shift array by n samples  data_buffer: shape (trials, channel, sampels, windows)
            self.data_buffer[0, :, int(-1*n_samples):, 0] = current_chunk # channels, sampels shape , update latest values in buffer  --> is this correct 

            #TODO: write this again but proper 
            # if (check_sample_loss): 
            #     # check for sample loss 
            #     sample_indices = self.data_buffer[0, -3, :, 0].astype(int) # sample indice channel
            #     for i in range(0, len(sample_indices) -1): 
            #         if sample_indices[i] + 1 != sample_indices[i+1]:
            #             warnings.warn(f"Sample loss at {i}: {sample_indices[i:i+2]}")

        else: 
            warnings.warn("no new data chunk received!")

    
    def bufferToWindows(self, num_non_data_channels = 0):
        """
        This method converts the buffered data from the data_buffer into data windows for further processing.  

        Parameters
        ----------
        num_non_data_channels : int, optional
            Number of channels to be removed, by default 3

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 13.03.2024 (by Niklas Kueper)
        """        

        self.windows = self.data_buffer[:, 0:self.n_channels-num_non_data_channels, :, :] # assuming last num_non_data_channels are appended at the end (as done by LiveAmp connector)

    def getDataBuffer(self): 
        """
        This function returns the data_buffer

        Returns
        -------
        Numpy array
            data_butter : float

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 08.03.2024 (by Niklas Kueper)
        """  

        return self.data_buffer

    def ensureLoopFrequency(self, print_loop_time = False): 
        """
        This function should be called in an infinite while loop to ensure a fixed loop frequency for the detection/classification of data. 

        Parameters
        ----------
        print_loop_time : bool, optional
            A flag if the loop time should be printed. This is especially useful to check if the data processing and classification is fast enough (< dt_process_data), by default False
        
        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 13.03.2024 (by Niklas Kueper)
        """      

        # wait for some time to ensure a "fixed" frequency to read new data from buffer 
        while((perf_counter()-self.last_loop_time) < self.dt_process_data): 
            pass

        if(print_loop_time): 
            print("loop time(ms):  ", (perf_counter() - self.last_loop_time)*1000)
        
        self.last_loop_time = perf_counter()


    # TODO: update once other options are there 
    def printStreamMetadata(self, stream_info_obj):
        """
        This function prints some basic meta data of the stream

        Parameters
        ----------
        stream_info_obj : StreamInlet object
            A pylsl StreamInlet object which contains alle the information of the stream

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 08.03.2024 (by Niklas Kueper)
        """        
        
        print("") 
        print("Meta data")
        print("Name:", stream_info_obj.name())
        print("Type:", stream_info_obj.type())
        print("Number of channels:", stream_info_obj.channel_count())
        print("Nominal sampling rate:", stream_info_obj.nominal_srate())
        print("Channel format:",stream_info_obj.channel_format())
        print("Source_id:",stream_info_obj.source_id())
        print("Version:",stream_info_obj.version())
        print("")


    # ZMQ stuff 
    #TODO: remove this in the future, it is implemented in another repository now which is marker_sync_utils 
    def startZMQServer(self, port_name):
        """
        This function creates a socket connection as a publisher to send commands
        
        Parameters
        ----------
        port_name : str
            The address string. This has the form "tcp://interface:port"

        Returns
        -------
        zmq.Socket instance
            The created socket object for the zmq connection.

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 08.03.2024 (by Niklas Kueper)
        """

        # create a socket connection as a publisher to send commands 
        my_context = zmq.Context()
        my_socket = my_context.socket(zmq.PUB)
        my_socket.bind("tcp://*:"+port_name)
        print("Publisher ready")
        return my_socket
    
    #zmq server
    #TODO: remove this in the future, it is implemented in another repository now which is marker_sync_utils 
    def establishZMQ(port_name):
        """
        This function creats a socket connection as a pubisher to send commands

        Parameters
        ----------
        port_name : str
           The address string. This has the form "tcp://interface:port"

        Returns
        -------
        zmq.Socket instance
            The created socket object for the zmq connection.

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 08.03.2024 (by Niklas Kueper)
        """  
        
        # create a socket connection as a publisher to send commands 
        my_context = zmq.Context()
        my_socket = my_context.socket(zmq.PUB)
        my_socket.bind("tcp://*:"+port_name)
        print("Publisher ready")
        return my_socket
