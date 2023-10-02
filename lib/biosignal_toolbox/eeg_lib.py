# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import numpy as np
import matplotlib.pyplot as plt
import mne
from sklearn.utils import shuffle
from scipy import signal as sig
import os 
from scipy.fft import fft, fftfreq
from tensorflow.keras.utils import to_categorical
import scipy 

# *********************************************************************************
# ************************* Methods ***********************************************
# *********************************************************************************

class EEGData:

    def __init__(self, format = "Brainvision", filenames = None, data_path = None, epochs = None, raw_obj = None, f_samp = None):
       
        
        self.raw_obj = None
        # parameter 
        #basic params 
        self.__ch_names = None
        self.__montage = None
        self.__fsamp = None
        self.time_axis_epochs = None
        # epoching 
        self.epochs = None 
        self.epoch_obj =None
        self.obj_filtered = None
        self.average_epochs = None
        #events
        self.events = None
        # windowing 
        self.windows = None 
        self.window_names = None 
        self.num_windows = None
        # features 
        self.feature_vec = None
        self.calib_means = None
        self.calib_stds = None

        if(filenames and format == "Brainvision"): 
            #create numpy array with file names 
            data_str_arr = []
            for files_str in filenames: 
                data_str_arr.append(os.path.join(data_path, files_str)) 
            data_str_arr = np.array(data_str_arr)

            self.raw_obj = self.loadBrainproductsData(data_str_arr)

            # parameter 
            #basic params 
            self.__ch_names = self.raw_obj.ch_names
            self.__fsamp = self.raw_obj.info['sfreq']
            #events
            self.events, self.event_id = mne.events_from_annotations(self.raw_obj)

        
        elif(format == "NumpyEpochs"): 
            self.epochs = epochs
            self.__fsamp = f_samp

        elif(format == "RawObj"): 
            self.raw_obj = raw_obj
            self.__fsamp = f_samp

        else: 
            print("No dataset specified ...")
        

    def getRawObject(self):
        return self.raw_obj
    
    def getEpochs(self):
        return self.time_axis_eeg_epochs, self.epochs
    
    def getEvents(self): 
        return self.events
    
    def getSamplingRate(self): 
        return self.__fsamp

    def getCalibStats(self): 
        return self.calib_means, self.calib_stds, self.calib_mins, self.calib_maxs
    
    def setCalibStats(self, means, stds, mins, maxs): 
        self.calib_means = means
        self.calib_stds = stds
        self.calib_mins = mins
        self.calib_maxs = maxs

    def setChannelNames(self, ch_names): 
        self.__ch_names = list(ch_names)

    def calcCalibStats(self, feature_times = None): 
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


    def loadBrainproductsData(self, dataset_list): 

        """
        This function can be used for loading one or more datasets in brainproducts format.
        Arguments:
            dataset_list: A list of strings with filenames of the datasets to be loaded.

        Returns:
            raw: An mne object with the loaded (concatenated) dataset(s). 

        Meta information: 
            Author: Niklas Kueper 
            Last changed: 28.11.2022 (by Niklas Kueper)
        """
        
        if (len(dataset_list) > 1): 
            raw_list = []
            for dataset in dataset_list: 
                raw1 = mne.io.read_raw_brainvision(dataset, preload = True, verbose = False)
                raw_list.append(raw1)
            raw = mne.concatenate_raws(raw_list)
        else: 
            raw = mne.io.read_raw_brainvision(dataset_list[0], preload = True, verbose = False)

        return raw

        
    def applyButterFilter(self, signal, f_lowpass = None, f_highpass = None, N = 1, type = "bandpass", padlen = 20, show_response = False): 

        """
        This function filters a signal with a simple digital butterworth lowpass filter with order N. 
        Arguments:
            signal: The signal to be filtered as onedimensional numpy array. 
            f_lowpass: The cutoff frequency of the lowpass filter. 
            N: The order of the butterworth filter. 
            type: The type of the filter as string, can be "bandpass", "lowpass" or "highpass". 
            padlen: The number of values to pad for filtering (see zero padding method). 

        Meta information: 
            Author: Niklas Kueper 
            Last changed: 21.09.2023 (by Niklas Kueper)
        """

        if(type == "bandpass"): 
            sos = sig.butter(N, [f_highpass, f_lowpass], btype=type, analog=False, output='sos', fs=self.__fsamp)
        elif(type == "lowpass"): 
            sos = sig.butter(N, f_lowpass, btype=type, analog=False, output='sos', fs=self.__fsamp)
        elif(type == "highpass"): 
            sos = sig.butter(N, f_highpass, btype=type, analog=False, output='sos', fs=self.__fsamp)

        w, h = sig.sosfreqz(sos, worN=512, whole = True)

        if(show_response): 
            plt.subplot(2, 1, 1)
            db = 20*np.log10(np.maximum(np.abs(h), 1e-5))
            plt.plot(w/np.pi, db)
            plt.ylim(-75, 5)
            plt.grid(True)
            plt.yticks([0, -20, -40, -60])
            plt.ylabel('Gain [dB]')
            plt.title('Frequency Response')
            plt.subplot(2, 1, 2)
            plt.plot(w/np.pi, np.angle(h))
            plt.grid(True)
            plt.yticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi],[r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
            plt.ylabel('Phase [rad]')
            plt.xlabel('Normalized frequency (1.0 = Nyquist)')
            plt.show()
        
        # signal shape: n_channel, n_sampels
        filtered_signal = np.zeros(signal.shape)
        for channel_idx in range(0, signal.shape[0]): 
            filtered_signal[channel_idx, :] = sig.sosfiltfilt(sos, signal[channel_idx, :], padlen=padlen, padtype='even') 

        return filtered_signal
    
    def FilterWindows(self, f_low =None, f_high = None, order = 2, filter_type = "scipy_butter", fir_design = "firwin2", Q = None, show_response = False): # under change 
        # shape: trials, channels, sampels, windows

        if(filter_type == "dc_notch"):
            b, a = sig.iirnotch(f_high, Q, fs=self.__fsamp)

        if(filter_type == "scipy_butter"): # prefer this one 
            if(f_high and f_low): 
                b, a = sig.iirfilter(order, [f_high, f_low], btype='bandpass', ftype='butter', output='ba', fs=self.__fsamp)
            elif(f_high):
                b, a = sig.iirfilter(order, f_high, btype='highpass', ftype='butter', output='ba', fs=self.__fsamp)
                #zi = sig.lfilter_zi(b, a)
            elif(f_low): 
                b, a = sig.iirfilter(order, f_low, btype='lowpass', ftype='butter', output='ba', fs=self.__fsamp)
                #zi = sig.lfilter_zi(b, a)
        
            
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
            
        
        for trial_idx in range(0, self.windows.shape[0]): 
                for window_idx in range(0, self.windows.shape[3]): 
                    
                    current_wind = self.windows[trial_idx, :, :, window_idx]
                    #data_buffer[:, int(mid_buffer):int(mid_buffer+current_wind.shape[1])] = current_wind

                    # currently the best 
                    if(filter_type == "mne_fir" or filter_type == "mne_iir"): 
                        filtered_window= mne.filter.filter_data(current_wind, sfreq = self.__fsamp, l_freq =f_high , h_freq = f_low, filter_length=order, method = filter_type, fir_design = fir_design, phase = 'zero', fir_window = "hamming", pad = "symmetric") # pad = "symmetric"
                        self.windows[trial_idx, :, :, window_idx] = filtered_window

                    elif(filter_type == "scipy_butter" or filter_type =="dc_notch"): 
                        for channel_idx in range(0, self.windows.shape[1]):
                            
                            # perform zero phase forward backward filtering with gustafson method to reduce artifacts  
                            filtered_window = sig.filtfilt(b, a, current_wind[channel_idx, :], method ="gust") # forward backward filtering with gustafson method 
   
                            self.windows[trial_idx, channel_idx, :, window_idx] = filtered_window


                    elif(filter_type == "fft_bandpass"):  # fft bandpass implementation from pySPACE 


                        filtered_window = np.zeros(current_wind.shape)
                        for channel_idx in range(0, self.windows.shape[1]): 

                            n = len(current_wind[channel_idx, :])

                            res = 0.1 # fixed resolution to 0.1 Hz 
                            fourier_transformed = scipy.fftpack.fft(current_wind[channel_idx, :], n = int(self.__fsamp/res)) # increase resolution to 0.1 Hz 

                            inverse = scipy.fftpack.ifft(fourier_transformed, n = self.__fsamp) # go back to normal samp rate size

                            #Compute the pass band indices
                            lower_bound = int(round(float(f_high) / (self.__fsamp) * len(fourier_transformed)))
                            upper_bound = int(round(float(f_low) / (self.__fsamp) * len(fourier_transformed)))


                            #Setting frequencies outside the pass band to 0
                            for i in range(0, lower_bound):
                                fourier_transformed[i] = 0
                                fourier_transformed[-i-1] = 0

                            for i in range(upper_bound,len(fourier_transformed)//2):
                                fourier_transformed[i] = 0
                                fourier_transformed[-i-1] = 0
                            
                            inverse = scipy.fftpack.ifft(fourier_transformed, n = self.__fsamp) # go back to normal samp rate size 


                            #Inverse Fourier transform and project to real component
                            self.windows[trial_idx, channel_idx, :, window_idx] = inverse


    def createActicapMontage(self, plot_montage, rename_channels, set_montage = True): 

        """
        This function can be used to create an acticap montage (used by e.g. LiveAmp64). The montage was created based on the acticap manual and an easycap template provided by mne.
        Arguments:
            plot_montage: A boolean flag if the montage info should be shown or not. if set to True, the montage will be shown. 

        Returns:
            montage: An mne montage object, that was created for the acticap layout. 

        Meta information: 
            Author: Niklas Kueper 
            Last changed: 28.11.2022 (by Niklas Kueper)
        """
        
        montage = mne.channels.make_standard_montage('easycap-M1', head_size=0.095) 
        #show easy cap Montage  

        if (rename_channels):
            montage.rename_channels({'Cz' : 'CZ','Pz' : 'PZ','Fz' : 'FZ','CPz' : 'CPZ', 'Fp1': 'FP1','Fp2': 'FP2','Oz': 'OZ','POz': 'POZ'}, allow_duplicates=False) # maybe change this ? 
        
        exclude_list = np.array(['O10','Fpz', 'Iz', 'F9', 'F10', 'P9', 'P10', 'O9', 'FCz', 'AFz']) # may change this ? 
        
        # get params of structure 
        easy_cap_ch_names = montage.ch_names
        easy_cap_dig = montage.dig
        #dev_head = montage.dev_head_t

        # find indizes to remove channels not in ActiCap 
        indizes_to_removing_channel = []
        for index in range(0, len(exclude_list)):
            for index1 in range(0, len(easy_cap_ch_names)):
                a = exclude_list[index]
                b = easy_cap_ch_names[index1]

                if(a == b):
                    indizes_to_removing_channel.append(index1)

        # seperate Standard digs and EEG digs 
        easy_cap_dig_standard = easy_cap_dig[0:3]
        easy_cap_dig_eeg = easy_cap_dig[3:]

        #delete Channels not there for acticap
        indizes_to_removing_channel_sorted = sorted(indizes_to_removing_channel, reverse= True)
        for indizes in indizes_to_removing_channel_sorted:
            #print('Indizes', indizes-count)
            del easy_cap_ch_names[indizes]
            del easy_cap_dig_eeg[indizes]

        # Adapted channel names and digitazation for acticap montage 
        easy_cap_dig_adapted = easy_cap_dig_standard.copy() # only head digits etc. 
        easy_cap_dig_adapted.extend(easy_cap_dig_eeg) # append selected channel digits 
        easy_cap_ch_names_adapted = easy_cap_ch_names.copy()

        #create Montage 
        #PlotMontage = True
        acti_cap_montage = mne.channels.DigMontage(dig=easy_cap_dig_adapted, ch_names=easy_cap_ch_names_adapted)
        if(plot_montage == True): 
            acti_cap_montage.plot()
            plt.show()
        
        self.__montage = acti_cap_montage

        if(set_montage): 
            self.obj_filtered.set_montage(self.__montage)


    def topoplot(self, times, title_str, min_val, max_val): 

        """
        This function creates and showes an topoplot at different points in time. 
        Arguments:
            mean_epochs: The average epochs over all trials in numpy format. Shape should be (channel, sampels). 
            time_axis_eeg_epoch: The time axis of the EEG-epochs as one dimensional numpy array. 
            mne_obj: The mne object of the dataset from which the information is used for the plot (e.g. channel names). 
            times: A list of times in ms at which the topolot should be created. 
            title_str: The title of the plot as string. 
            min_val: The minimum Voltage in the color scale. 
            max_val: The maximum Voltage in the color scale. 
            f_samp_eeg: Sampling rate of the EEG-data in Hz. 

        Returns:
            -
        
        Meta information: 
            Author: Niklas Kueper 
            Last changed: 04.04.2023 (by Niklas Kueper)
        """

        mean_epochs = np.mean(self.epochs, axis = 0)

        #topoplot at different times 
        n,m = mean_epochs.shape
        #start_idx = (start_time/1000)*f_samp_eeg
        #step_idx = (step_time/1000)* f_samp_eeg
        time_idx = (np.array(times)).astype(int)
        time_axis_eeg_epoch_ms = (self.time_axis_epochs *1000).astype(int) # make time axis in ms for the analysis 


        #indices_of_topoplot = np.arange(start_idx, m, step = step_idx).astype(int) # 22 er steps 
        mean_epochs.astype(float)
        
        count = 0
        fig, ax = plt.subplots(nrows=len(time_idx), figsize=(8, 20), gridspec_kw=dict(top=0.9),sharex=True, sharey=True)
        fig.subplots_adjust(hspace=0.5)
        for t_index in time_idx: 
            index = np.array(np.where(time_axis_eeg_epoch_ms == t_index))
            if (index.size == 0): # index not found 
                index = np.array(np.where(time_axis_eeg_epoch_ms == t_index+1))
            index = index[0][0] # numpy array to int value 
            cmap = 'bwr'
            im, cn = mne.viz.plot_topomap(mean_epochs[:,index], self.obj_filtered.info, cmap = cmap, axes = ax[count], show = False, image_interp = 'cubic',extrapolate='local',vlim = [min_val, max_val])
            str_time = str(t_index)+" ms"
            ax[count].set_title(title_str+str_time, color='black', fontsize=12)
            cbar =fig.colorbar(im, ax = ax[count], orientation="vertical", pad = 0.15)
            cbar.set_label("in uV")
            count = count+1
        plt.show()

    def getDataOneChannel(self, channel_name, average = True, is_windowed = False): 

        channel_idx = self.__ch_names.index(channel_name)

        if (is_windowed): 
            # shape: trials, channel, sampels, windows 

            if(average): # if average should be returned 
                
                return np.mean(self.windows[:, channel_idx, :, :], axis = 0)
            else: 
                return self.windows[:, channel_idx, :, :]

        else: # if not windowed yet 

            if(average):
                data_channel = self.average_epochs[channel_idx, :]
            else: 
                data_channel = self.epochs[:, channel_idx, :]

            return data_channel, self.time_axis_epochs


    def getKerasPredictionResultsLRP(self, model, epochs, n_samp_features): 

        """
        This function creates and showes an topoplot at different points in time. 
        Arguments:
            model: The keras model object. 
            epochs: The EEG-epochs as numpy array with shape: (n_epochs, n_channels, n_sampels)
            n_samp_features: Number of sampels that are used as features for both classes, currently the last n_samp_features datapoints are consideres as erp class and the remaining are from noerp class. 


        Returns:
            predicted_labels: The predicted labels of the classifier as float values (0.0 noerp or 1.0 erp)
            true_labels: The true labels in respect to the number of n_samp_features as erp labels (-n_samp_features to time 0 as erp labelled points)
            trial_prediction: The prediction scores of each classified datapoint. 
        
        Meta information: 
            Author: Niklas Kueper 
            Last changed: 28.11.2022 (by Niklas Kueper)
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


    def calcMovingAveragePredictionScores(self, trial_prediction_test, n_samp): 
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
    

    def rereferencingEpoching(self, marker_number, error_number,channel_list, inverse_keep_channel, reref_channels, apply_filter, f_highpass, f_lowpass, event_id_used, t1, t2, apply_baseline_correction,  t0_baseline, t1_baseline): 
        
        """
        Apply rereferencing and epoching with given parameters and filters to an raw_obj mne instance. 

        Arguments: 
            marker_number: The markernumber of the used event for creating the epochs. 
            error_number: The markernumber of trials with an error that are excluded from the evaluation. 
            channel_list: A list of EEG-channels that are either kept or dropped from evaluation depending on the "inverse_keep_channel" flag. 
            inverse_keep_channel: If False, all channel in "channel_list" are kept, otherwise the specified channel are dropped. 
            reref_channels: A list of channels that are used for rereferencing. If the list is empty, the original ref-channel is used, if ["average"] is passed an average reference is applied.
            apply_filter: Boolean flag that should be True if a filter should be applied. 
            f_highpass: The highpass cutoff frequency in Hz (only used when apply_filter is True). 
            f_lowpass: The lowpass cutoff frequency in Hz (only used when apply_filter is True). 
            event_id_used: The created mne event id (e.g. {"movement_onset": marker_number}). 
            t1: Start time of the epochs in ms.  
            t2: End time of the epochs in ms. 
            f_samp_eeg: Sampling rate of the EEG-signals in Hz. 
            apply_baseline_correction: Boolean flag that is set to True if baseline correction should be applied (mean value between t0_baseline and t1_baseline is used as correction). 
            t0_baseline: Specified start time for the baseline correction. 
            t1_baseline: Specified end time for the baseline correction. 


        Class parameters:
            erp_epochs: The epochs of the erp analysis as numpy array with shape: (n_epochs, n_channel, n_samples)
            erp_epoch_obj: The erp epochs object created by mne. 
            time_axis_eeg_batch: The created time axis as one dimensional numpy array. 
            remaining_eeg_channel_names: A list of EEG-channels that are included in the analysis (in the erp_epochs array). 
            filtered_eeg_rereferenced: Manipulated instance of an mne raw_obj object (after filtering and channel selection). 

        Meta information: 
            Author: Niklas Kueper 
            Last changed: 28.11.2022 (by Niklas Kueper)
        """

        
        # rereferencing 
        rereferenced_eeg_raw_obj = self.raw_obj.copy()
        
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
        plot_events, plot_event_dict = mne.events_from_annotations(filtered_eeg_rereferenced)
        plot_onset_indices = np.where(plot_events[:,2] == marker_number)[0] # S100 marker is leaving plate 
        exclude_indices = np.where(plot_events[:,2] == error_number)[0] # S3 marker should be excluded 
        
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
        used_plot_events = np.zeros((len(plot_onset_indices_correct),3))
        used_plot_events = plot_events[plot_onset_indices_correct,:]

        #eeg_epochs = mne.Epochs(filtered_eeg_rereferenced, events = used_plot_events, event_id = event_id_used,tmin=t1, baseline=None, tmax=t2, preload=True, reject_by_annotation = True)
        if not(channel_list): 
            if(apply_baseline_correction): 
                eeg_epochs = mne.Epochs(filtered_eeg_rereferenced, events = used_plot_events, event_id = event_id_used, tmin=t1, baseline=(t0_baseline, t1_baseline), tmax=t2, preload=True, reject_by_annotation = True)
            else: 
                eeg_epochs = mne.Epochs(filtered_eeg_rereferenced, events = used_plot_events, event_id = event_id_used, tmin=t1, baseline=None, tmax=t2, preload=True, reject_by_annotation = True)
                
        else: #drop specified channels if False 

            # keep or drop specified channels 
            if(inverse_keep_channel == True): 
                filtered_eeg_rereferenced.drop_channels(channel_list)
            else: 
                filtered_eeg_rereferenced.pick_channels(channel_list)

            if(apply_baseline_correction): 
                eeg_epochs = mne.Epochs(filtered_eeg_rereferenced, events = used_plot_events, event_id = event_id_used,tmin=t1, tmax=t2, baseline=(t0_baseline, t1_baseline), preload=True, reject_by_annotation = True)
            else: 
                eeg_epochs = mne.Epochs(filtered_eeg_rereferenced, events = used_plot_events, event_id = event_id_used,tmin=t1, tmax=t2, baseline=None, preload=True, reject_by_annotation = True)
        
        # Get remaining channel names  
        self.__ch_names = filtered_eeg_rereferenced.ch_names
        self.obj_filtered = filtered_eeg_rereferenced.copy()
        
        self.epoch_obj = eeg_epochs.copy() # the object of epochs from mne 

        #get data out as numpy array for further processing 
        self.epochs = eeg_epochs.get_data() 
        self.average_epochs = np.mean(self.epochs, axis = 0)
        
        #generate a time axis for the epochs 
        self.time_axis_epochs = np.arange(t1,t2+1/self.__fsamp, step = 1/self.__fsamp) #build time axis (epoch)


    def reshapeWindowsForCNNnets(self): 

        """
        Select windows and extract them from all windows segmented by specifying the windows names.  

        Meta information: 
            Author: Niklas Kueper 
            Last changed: 14.09.2023 (by Niklas Kueper)
        """


        reshaped_EEG_windows= np.zeros((self.windows.shape[0]*self.windows.shape[3], self.windows.shape[1], self.windows.shape[2], 1)) # (n_trials * n_windows, n_channels, n_sampels, 1). 


        # get the features in one dim for all trials and windows 
        for channel_idx in range(0, self.windows.shape[1]):
            for sample_idx in range(0, self.windows.shape[2]): 

                reshaped_EEG_windows[:, channel_idx, sample_idx, 0] = self.windows[:, channel_idx, sample_idx, :].flatten()

        self.windows = reshaped_EEG_windows


    def labelsToCategorical(self, num_classes = 2): 

        y = to_categorical(self.labels, num_classes)

        self.labels = y


    def getWindows(self): 
        return self.windows
    
    def getWindowNames(self): 
        return self.window_names
    
    def getFeatures(self): 
        return self.feature_vec


    def getLabels(self): 
        return self.labels

    def getTrainLabels(self, type = "binary"): 

        if(type == "binary"):
            return self.labels
        elif(type == "onehotencoding"): 
            return self.labels[:, 1]

    def onlineLRPWindowPredictionPostprocessing(self, window_wise_predicts, high_tresh, low_tresh, short_samp, long_samp): 

        """
        Apply an online capable postprocessing for the detection of LRP, where a linear function decides for the LRP class over which time a defined probability has to be reached for the detection of the positive class. 

        Arguments:  
            Missing ... 


        Returns:
            Missing ...  

        Meta information: 
            Author: Niklas Kueper 
            Last changed: 18.01.2023 (by Niklas Kueper)
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


    def windowEEGEpochs(self, window_size, window_step, no_channel_dim = False):

        """
        This function cuts (overlapping) windows from continues EEG-signals (currently only for postprocessing without channel dimension). 

        Arguments:
            no_channel_dim(optional): Set to True if the EEG data (epochs) have no channel dimension. 
            window_size: The size of the windows in ms to be cutout (standard: 1000). 
            window_step: The stepsize of the sliding window (sliding step) in ms (standard: 25)
            
        
        Meta information: 
            Author: Niklas Kueper 
            Last changed: 14.09.2023 (by Niklas Kueper)
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
        self.num_windows = num_of_windows
        self.window_names = wind_names




    def windowSelection(self, selected_windows): 
        """
        Select windows and extract them from all windows segmented by specifying the windows names.  

        Arguments:
            selected_windows: The names of the windows (given after windowing) which are selected for further processing. 
        
        Meta information: 
            Author: Niklas Kueper 
            Last changed: 14.09.2023 (by Niklas Kueper)
        """

        # interate over specified window names and extract the windows 
        indices_selected_winds = []
        for current_window_name in selected_windows: 
            indices_selected_winds.append(self.window_names.index(current_window_name))

        selected_windows_arr = self.windows[:, :, :, indices_selected_winds] 

        self.windows = selected_windows_arr
        self.window_names = selected_windows

    def windowStandardization(self, norm = False, use_min_max_norm = False):
        
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
                        current_wind_norm = current_wind_norm/self.calib_stds[channel_idx]  
                        
                        if(norm): 
                        #current_wind_norm = current_wind+(-1*min)-1 # -1 is min 
                            current_wind_norm = current_wind_norm/np.max(current_wind_norm)


                    self.windows[trial_idx, channel_idx, :, window_idx] = current_wind_norm # 1 is max 

                    # print("")
                    # print(np.min(current_wind_norm))
                    # print(np.mean(current_wind_norm))
                    # print(np.std(current_wind_norm))
                    # print(np.max(current_wind_norm))
                    # print("")

    def calcTestAccAndRates(self, prediction_labels, true_labels):

        """
        Get metrics from classification output of the test data. Currently the accuracy, balanced accuracy,  tnr and tpr are calculated. 

        Arguments:
            prediction_labels: The predicted labels as one dimensional numpy array (flatten the array if it has more dimensions).
            true_labels: The true labels as one dimensional numpy array (flatten the array if it has more dimensions). 

        Returns:
            tnr: True negative rate 
            tpr: True positive rate
            acc: Accuracy
            ba: Balanced accuracy
        
        Meta information: 
            Author: Niklas Kueper 
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


    def onlineWindowPredictionPostprocessing_v1(self, window_wise_predicts, short_tresh, mid_tresh, long_tresh, short_sampels, mid_sampels, long_sampels): 

        classified_windows = np.zeros((window_wise_predicts.shape[0], window_wise_predicts.shape[2])) # output shape (n_trials, n_windows)

        for trial_idx in range(0, window_wise_predicts.shape[0]): 
            for window_idx in range(0, window_wise_predicts.shape[2]): 
                
                # get prediction scores of current trial and window 
                current_predicts = window_wise_predicts[trial_idx, :, window_idx] # one second window data 
                mean_long_time_detections = np.mean(current_predicts[long_sampels:]) 
                mean_mid_time_detections = np.mean(current_predicts[mid_sampels:])
                mean_short_time_detections = np.mean(current_predicts[short_sampels:])

                # if one of both criteriums (short or long detection) is fulfilled the window gets the positive class label  
                if((mean_long_time_detections > long_tresh) or (mean_short_time_detections > short_tresh) or (mean_mid_time_detections > mid_tresh)): 
                    classified_windows[trial_idx, window_idx] = 1.0 
                else: 
                    classified_windows[trial_idx, window_idx] = 0.0
                
        return classified_windows


    def onlineWindowPredictionPostprocessing_v2(self, window_wise_predicts, high_tresh, low_tresh, short_samp, long_samp): 

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


    def onlineWindowPredictionPostprocessing_v3(self, window_wise_predicts, thresh, start_samp): 

        classified_windows = np.zeros((window_wise_predicts.shape[0], window_wise_predicts.shape[2])) # output shape (n_trials, n_windows)

        for trial_idx in range(0, window_wise_predicts.shape[0]): 
            for window_idx in range(0, window_wise_predicts.shape[2]): 
                
                # get prediction scores of current trial and window 
                current_predicts = window_wise_predicts[trial_idx, :, window_idx] # one second window data 

                mean_detections = np.mean(current_predicts[start_samp:]) 

                # if one of both criteriums (short or long detection) is fulfilled the window gets the positive class label  
                if(mean_detections > thresh): 
                    classified_windows[trial_idx, window_idx] = 1.0 
                else: 
                    classified_windows[trial_idx, window_idx] = 0.0

        return classified_windows


    def calcTrialMetric(self, predict_scores, pos_class_start_time, f_samp, decision_bound, num_class_instances): 
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
    

    def OnechannelFFT(self, one_channel_data, fsamp, plot = False): 
        """
        This function calculates the FFT for one channel of timeseries data. 
        Arguments:

        Returns:

        Meta information: 
            Author: Niklas Kueper 
            Last changed: .. 
        """

        N = len(one_channel_data)
        # sample spacing
        dT = 1.0/fsamp
        x = np.linspace(0.0, N*dT, N, endpoint=False)
        y = one_channel_data
        yf = fft(y)

        xf = fftfreq(N, dT)[:N//2]
        yfn = 2.0/N * np.abs(yf[0:N//2])


        if (plot): 
            fig = plt.figure()
            plt.plot(xf, yfn)

        return xf, yfn
    
    def detrendWindows(self): 
        # get the features in one dim for all trials and windows 
        for trial_idx in range(0, self.windows.shape[0]):
            for window_idx in range(0, self.windows.shape[3]):
                for channel_idx in range(0, self.windows.shape[1]):
                    current_window = self.windows[trial_idx, channel_idx, :, window_idx]
                    current_window_corr = sig.detrend(current_window)
                    self.windows[trial_idx, channel_idx, :, window_idx] = current_window_corr


    def featureExtractionFromWindows(self, feature_type, feature_indices_windows = None, use_mean = False, N = 1): 

        # (n_trials, n_channels, n_sampels, n_windows).
        feature_type_total = None

        if(feature_type == "fusion"):  # extract both feature types after each other 
            feature_type_total = feature_type
            feature_type = "timepoints"


        if(feature_type == "timepoints"): 

            feature_times_indices = ((feature_indices_windows/1000)*self.__fsamp).astype(int)

            #x_train = np.zeros((int(train_windows.shape[0]*train_windows.shape[3]), int(train_windows.shape[1]*train_windows.shape[2]))) # shape: train samples, features 

            if(use_mean): 
                x_train_features = np.zeros((self.windows.shape[0], self.windows.shape[3], N*self.windows.shape[1]))
            else: 
                x_train_features = np.zeros((self.windows.shape[0], self.windows.shape[3], int(len(feature_times_indices)*self.windows.shape[1]))) # shape: trials, windows, features
            

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
                        x_train_features[trial_idx, window_idx, :] = self.windows[trial_idx, :, feature_times_indices, window_idx].flatten()

                    
            # flatten the trials and windows as train instances 
            x_train = np.zeros((x_train_features.shape[0]*x_train_features.shape[1], x_train_features.shape[2]))
            #print(x_train.shape)

            for feature_idx in range(0, x_train_features.shape[2]):
                x_train[:, feature_idx] = x_train_features[:, :, feature_idx].flatten()


        if(feature_type_total == "fusion"): 
            x_train_time = x_train
            feature_type = "meanfreqs"

        elif(feature_type == "meanfreqs" or feature_type == "medianfreqs"): #fix this 


            # (n_trials, n_channels, n_sampels, n_windows).
            num_of_freq_bands = 5 
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


                    x_train_features[trial_idx, window_idx, :] = features.flatten()

                    
            # flatten the trials and windows as train instances 
            x_train = np.zeros((x_train_features.shape[0]*x_train_features.shape[1], x_train_features.shape[2]))
            #print(x_train.shape)

            for feature_idx in range(0, x_train_features.shape[2]):
                x_train[:, feature_idx] = x_train_features[:, :, feature_idx].flatten()


        if(feature_type_total == "fusion"): 

            self.feature_vec = np.concatenate((x_train_time, x_train), axis = 1)
        else: 
            self.feature_vec = x_train 

    def addFeatures(self, x): 
        print(self.feature_vec.shape)
        print(x.shape)
        features = np.concatenate((self.feature_vec, x), axis = 1)
        self.feature_vec = features 

    def printFeatureShape(self): 
        print("feature shape: ", self.feature_vec.shape)

    def setWindowLabels(self, label_list): 

        """
        Set/encode the class labels of segemented windows for the classification task.  

        Arguments:
            label_list: A list of labels that correspond to the windows class labels (e.g. [0.0, 0.0, 1.0, 1.0]). 
        
        
        Meta information: 
            Author: Niklas Kueper 
            Last changed: 22.07.2023 (by Niklas Kueper)
        """

        y = np.zeros((self.windows.shape[0], self.windows.shape[3])).astype(dtype=np.float64)
        for trial_idx in range(0, y.shape[0]): 
            y[trial_idx, :] = np.array(label_list)  # shape: trials, window labels

        y = y.flatten() # flatten the labels
        y_temp = np.zeros((y.shape[0], 1))
        y_temp[:, 0] = y
        y = y_temp

        self.labels = y 


    def calcEEGWindowOnset(self, window_predicts, num_pos_windows): 

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
        Apply the relabelling method to the classification output in order to get the "true ground truth" labels. This function should be carefully used since it creates new ground truth labels for the evaluation of the classifier! 

        Arguments:
            prediction_labels: The predicted labels as one dimensional numpy array (flatten the array if it has more dimensions).
            determine_labels: The amount of negative classes that are counted from the right side (end of each epoch/trial) to specify the "label change point". 
            searching_bounds: A list with boundaries ([lower bound, upper bound]) in which the label change point for the relabelling is searched (e.g. for LRP the numbers of the windows for -1000 ms and 0 ms). 

        Returns:
            new_true_labels: A numpy array (shape: (n_trials, n_sampels)) containing the new ground truth labels (0.0 neg class; 1.0 pos class). 

        Meta information: 
            Author: Niklas Kueper 
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
        Calculate Metrics of a window wise classification output. 

        Arguments:
            wind_arr: The windowed class predictions as numpy array with shape (n_trials, n_sampels, n_windows). 
            evaluation_time_per_window: The time in ms at the end of each window for which the window metric is calculated (-evaluation_time_per_window to 0 ms are used). 
            window_step: The sliding step size of the windows in ms (standard 50 ms). 
            f_samp_eeg: The sampling rate of the EEG-data in Hz. 
            n_samp_features: The number of sampels that are used as features (-n_samp_features:0 of each epoch). 
            use_relabelling: If this flag is set to True, the relabelling method is applied to calculate the metrics. This method should be treated with care since it effects the classification performance!
            determine_labels: The number of consecutive negative classes that are counted when estimating the label change point of both classes. 
            searching_bounds: A list of the lower and upper bound of window numbers ([lower bound, upper bound]) where the label change point is searched. 

        Returns:
            tnr: True negative rate 
            tpr: True positive rate 
            acc: Accuracy 
            ba: Balanced accuracy 
            window_predictions: The prediction values (0 - 1) for all windows and trials 
            window_eval_true_labels: The true labels that are specified or the new true labels when relabelling is used. 

        Meta information: 
            Author: Niklas Kueper 
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



   
