# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import numpy as np
import matplotlib.pyplot as plt
import mne
mne.set_log_level('WARNING')
from pathlib import Path
from mne.preprocessing import ICA 
import warnings

# own dependencies 
from biosignal_toolbox.time_series_lib import Timeseries
from biosignal_toolbox.time_series_lib import OnlineTimeseriesStreaming
from biosignal_toolbox.utils import getAbsolutePath

# *********************************************************************************
# ************************* Methods ***********************************************
# *********************************************************************************


class EEGData(Timeseries):    
    """
    This class includes useful methods and paramters for the loading and viusalization of EEG data. It is mainly dependend on the Timeseries class. 

    Parameters
    ----------
    Timeseries : class
        The base timeseries class that includes most of the data processing methods for biosignals (e.g. filters for EMG and EEG etc.)
    """

    def __init__(self, format = "Brainvision", filenames = None, data_path = None, epochs = None, raw_obj = None, f_samp = 500, channel_names = None, windows = None, data = None, add_marker_channel = False):
        """
        The constructor of the EEGData class. 
        
        Parameters
        ----------
        format : str, optional
            The format in which the data is loaded, by default "Brainvision"
        filenames : list, optional
            A list of filenames to be loaded, by default None
        data_path : str, optional
            The path where the data is stored, by default None
        epochs : numpy ndarray, optional 
            The epochs to be set as numpy ndarray, only required when format type "NumpyEpochs" is selected.
        raw_obj :  mne raw object, optional
            The mne raw object that is used to create the object. Only required for format type "RawObj".
        f_samp: float
            The sampling rate of the data, not required for type "Brainvision". 
        channel_names : list of str
            A list of channel names as strings, if not known from the data format. Not required for format "Brainvision".
        windows : numpy ndarray, optional
            A numpy array with windowed data (shape: n_trials, n_channels, n_sampels, n_windows), only required for format "Live". 
        data : numpy ndarray, optional 
            The channel wise (raw) data as numpy array (shape: n_channel, n_sampels), currently fully optional (not used by any format).


        Attributes
        ----------
        raw_obj : mne raw object
            The mne raw object that is used to create the object. Only required for format type "RawObj".
        f_samp : float
            The sampling rate of the EMG system in Hz
        channel_names : list
            A list of channel names as strings, if not known from the data format.
        data : numpy ndarray
             The channel wise (raw) data as numpy array (shape: n_channel, n_sampels). 
        epochs : numpy ndarray
            The epoched data as numpy ndarray with shape (n_trials, n_channels, n_sampels).
        windows : numpy ndarray
            A numpy array with windowed data (shape: n_trials, n_channels, n_sampels, n_windows).
        events : numpy ndarray 
            The events (also called markers) in the data. The shape is: (indices, 0, eventnumber).
        __montage : mne montage object 
            The montage object for mne (see mne documentation for more information). 

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 23.08.2024 (by Kartik Chari)
        """

        #basic params 
        self.raw_obj = raw_obj
        self.f_samp = f_samp
        self.channel_names = channel_names
        self.data = data
        self.epochs = epochs
        self.windows = windows
        self.events = None
        # EEG specific params 
        self.__montage = None
        # absolute data path
        self.data_path = getAbsolutePath(input_path=data_path)

        if(filenames and format == "Brainvision"): 
            #create numpy array with file names 
            data_str_arr = []
            for files_str in filenames: 
                data_str_arr.append(self.data_path / Path(files_str)) 
            data_str_arr = np.array(data_str_arr)

            self.raw_obj = self.loadBrainproductsData(data_str_arr)
            
            # update parameter 
            #basic params 
            self.channel_names = self.raw_obj.ch_names
            self.f_samp = self.raw_obj.info['sfreq']
            self.data = self.raw_obj.get_data() # data as numpy array in shape (channels, sampels) 
            #events epochs_filter
            self.events, self.event_ids = mne.events_from_annotations(self.raw_obj)


        elif(format == "NumpyEpochs"): 
            self.epochs = epochs
            self.f_samp = f_samp

        elif(format == "RawObj"): 
            self.raw_obj = raw_obj
            self.f_samp = f_samp

        elif(format == "Live"): 
            self.windows = windows
            self.f_samp = f_samp
            self.channel_names = channel_names

        elif(format == "Recorded_LSL_stream"): 
            if(filenames): # implement running over all files and appending data to each other 

                if(len(filenames) > 1): 
                    concat_list = []
                    for filename in filenames: 
                        concat_list.append(np.load(self.data_path / (filename+".npy")))
                    
                    data = np.concatenate(concat_list)
                else: 
                    data = np.load(self.data_path / (filenames[0]+".npy"))

            self.f_samp = f_samp
            self.channel_names = channel_names
            
            # set annotation events (markers)
            self.createAnnotationEvents(data=data)
            self.data = self.raw_obj.get_data() # data as numpy array in shape (channels, sampels) 

            # print(type(self.events)) # (events, 3)
            # print("events", self.events)

        elif(format == "NumpyQualisys"): 
            if(filenames): # implement running over all files and appending data to each other 

                if(isinstance(filenames, list)): 
                    concat_list = []
                    for filename in filenames: 
                        concat_list.append(np.load(self.data_path /(filename+".npy"),allow_pickle=True, encoding='bytes'))
                    
                    data = np.concatenate(concat_list)
                else: 
                    if add_marker_channel:
                        data = np.load(self.data_path / (filenames[0]+".npy"),allow_pickle=True, encoding='bytes').tolist()
                    else:
                        data = np.load(self.data_path / (filenames[0]+".npy"),allow_pickle=True, encoding='bytes')
                    if isinstance(data,dict):
                        if file_type == 'combined':
                            #! Access data in the same order as EMG data and concatenate the arrays into a single numpy array
                            if outer_key_order_d == [] and inner_key_order_d == []:
                                weights_order = list(data.keys())
                                type_order = list(next(iter(data.values())).keys())
                            elif outer_key_order_d == [] and inner_key_order_d != []:
                                weights_order = list(data.keys())
                                type_order = inner_key_order_d
                            elif outer_key_order_d != [] and inner_key_order_d == []:
                                weights_order = outer_key_order_d
                                type_order = list(next(iter(data.values())).keys())
                            else:
                                weights_order = outer_key_order_d
                                type_order = inner_key_order_d

                            tmp_array_of_lists = []
                            for weight in weights_order:
                                for mov_type in type_order:
                                    tmp_array_of_lists.append(data[weight][mov_type])
                            data = np.concatenate(tmp_array_of_lists)
                        elif file_type == 'individual':
                            data = data[list(data.keys())[0]][list(next(iter(data.values())).keys())[0]]
                if add_marker_channel:
                    # Adding an extra event channel at the end for quali markers
                    column_of_no_markers = -1 * np.ones((data.shape[0],1))
                    data = np.hstack((data,column_of_no_markers))
                    # Making the first and last but 5th sample (considering 20ms offset) as the boundaries for syncing
                    offset_idx = -1*(20*f_samp/1000)
                    # print(f"Quali offset: {int(offset_idx)}")
                    data[0,-1] = 1
                    data[int(offset_idx),-1] = 1
                    print(f"Data: {data.shape}")
                else:
                    pass
                        data = np.load(filename,allow_pickle=True, encoding='bytes').reshape(-1,1)
                        # print(f"Quali data shape: {data.shape}")
                        if add_marker_channel:
                            # Adding an extra event channel at the end for qualisys markers
                            column_of_no_markers = -1 * np.ones((data.shape[0],1))
                            data = np.hstack((data,column_of_no_markers))
                            # Making the first and last but 20th sample (considering 40ms offset at the end) as the boundaries for syncing
                            offset_idx = -1*(40*f_samp/1000)
                            # print(f"Quali offset: {int(offset_idx)}")
                            data[0,-1] = 1
                            data[int(offset_idx),-1] = 2
                            # print(f"Data: {data.shape}")
                            # print(f"Events: {np.where(data[:,1] == 1)[0]}")
                        
                            concat_list.append(data)
                    
                    data = np.vstack(concat_list)
                    # print(np.where(data[:,1] == 1)[0])
                        concat_list.append(np.load(self.data_path /(filename+".npy"),allow_pickle=True, encoding='bytes'))
                    
                    data = np.concatenate(concat_list)
                else: 
                    if add_marker_channel:
                        data = np.load(self.data_path / (filenames[0]+".npy"),allow_pickle=True, encoding='bytes')
                    else:
                        data = np.load(self.data_path / (filenames[0]+".npy"),allow_pickle=True, encoding='bytes')

                    if isinstance(data, list):
                        data = np.array(data)
                    if data.ndim == 1:
                        data = data[:, np.newaxis]

                    if isinstance(data,dict):
                        if file_type == 'combined':
                            #! Access data in the same order as EMG data and concatenate the arrays into a single numpy array
                            if outer_key_order_d == [] and inner_key_order_d == []:
                                weights_order = list(data.keys())
                                type_order = list(next(iter(data.values())).keys())
                            elif outer_key_order_d == [] and inner_key_order_d != []:
                                weights_order = list(data.keys())
                                type_order = inner_key_order_d
                            elif outer_key_order_d != [] and inner_key_order_d == []:
                                weights_order = outer_key_order_d
                                type_order = list(next(iter(data.values())).keys())
                            else:
                                weights_order = outer_key_order_d
                                type_order = inner_key_order_d

                            tmp_array_of_lists = []
                            for weight in weights_order:
                                for mov_type in type_order:
                                    tmp_array_of_lists.append(data[weight][mov_type])
                            data = np.concatenate(tmp_array_of_lists)
                        elif file_type == 'individual':
                            data = data[list(data.keys())[0]][list(next(iter(data.values())).keys())[0]]
                if add_marker_channel:
                    # Adding an extra event channel at the end for quali markers
                    column_of_no_markers = -1 * np.ones((data.shape[0],1))
                    data = np.hstack((data,column_of_no_markers))
                    # Making the first and last but 5th sample (considering 20ms offset) as the boundaries for syncing
                    offset_idx = -1*(20*f_samp/1000)
                    # print(f"Quali offset: {int(offset_idx)}")
                    data[0,-1] = 1
                    data[int(offset_idx),-1] = 1
                    print(f"Data: {data.shape}")
                else:
                    pass

            self.f_samp = f_samp
            self.channel_names = channel_names
            
            # set annotation events (markers)
            self.createAnnotationEvents(data=data)
            self.data = self.raw_obj.get_data() # data as numpy array in shape (channels, sampels) 
            # print(type(self.events)) # (events, 3)
            # print("events", self.events)

        else: 
            print("No dataset specified ...")
        
        # init the super class 
        super().__init__(raw_obj = self.raw_obj, events = self.events, channel_names = self.channel_names, f_samp = self.f_samp, data = self.data, epochs = self.epochs, windows = self.windows)

    def loadBrainproductsData(self, dataset_list): 
        """
        This method is used to load a dataset in the Brainvision format (.eeg, .vhdr, .vmrk). 

        Parameters
        ----------
        dataset_list : list of str
            A list of filenames to be loaded. If the list contains more than one filename, the datasets are concatenated. 

        Returns
        -------
        mne raw object
            The loaded data as mne raw object. 

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 06.07.2022 (by Niklas Kueper)
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
    
    def createAnnotationEvents(self, data=None):
        """
        This method creates mne raw object from loaded data, adds events and sets annotations from the events

        Parameters
        ----------
        data : numpy list, optional
            concatenated list of input data, by default None
        """
        if data is None:
            print("ERROR: Please provide input numpy list of data!!")
        else:
            #! Creating an mne object
            data = data.T # in form (channel, sampels)
            sfreq = self.f_samp  # Sampling frequency
            #print("markers", np.where(data[-1, :] == 64)[0])
            times = np.arange(0, data.shape[1], 1/sfreq)  # 
            ch_types = ['eeg'] * len(self.channel_names) # only EEG for now 
            ch_names = self.channel_names
            info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
            #scalings = {'eeg': 1}
            raw = mne.io.RawArray(data[0:len(ch_names), :], info) # only pass the actual EEG channel 
            self.raw_obj = raw
            #! Creating events
            event_channel = data[-1, :]
            marker_indices = np.where(event_channel > 0)[0]
            marker_numbers = event_channel[marker_indices]
            events = np.zeros((len(marker_indices), 3))
            events[:, 0] = marker_indices
            events[:, 2] = marker_numbers
            self.events = events.astype(int)
            #! Creating annotations
            annotations = mne.annotations_from_events(events = events, sfreq = self.f_samp, event_desc=None, first_samp=0, orig_time=None, verbose=None)
            self.raw_obj.set_annotations(annotations = annotations)


    def createActicapMontage(self, plot_montage = False, rename_channels=None, set_montage = True): 

        """
        This function can be used to create an acticap montage (used by e.g. LiveAmp64). The montage was created based on the acticap manual and an easycap template provided by mne.

        Parameters
        ----------
        plot_montage : bool
            A boolean flag if the montage info should be shown or not. if set to True, the montage will be shown.

        Returns
        -------
        montage : mne.channels.Montage
            An mne montage object, that was created for the acticap layout.

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 28.11.2023 (by Niklas Kueper)
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
            self.epoch_obj.set_montage(self.__montage)

    
    def topoplot(self, times, title_str = "Topoplot at selected times", min_val = -6e-06, max_val = 6e-06, save_figure = False, filename = "test.png", path = ""):
                 
        """
        This method creates a topoplot at different times in relation to an specific event. 
    
        Parameters
        ----------
        times : list of int
            The times for which the topoplot should be created (in ms). 
        title_str : str, optional
            The title of the topoplot, by default "Topoplot at selected times"
        min_val : float, optional
            The minimum value of the colorbar in the units of the data, by default -6e-06
        max_val : float, optional
            The maximum value of the colorbar in the units of the data, by default 6e-06
        save_figure : bool, optional
            Wheather to save the figure or not. 
        filename : str, optional 
            If save_figure is True, the name under which the figure is saved. 
        path : str, optional
            The path where to save the figure if save_figure is True. 
        
        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 17.05.2024 (by Niklas Kueper)
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
            im, cn = mne.viz.plot_topomap(mean_epochs[:,index], self.epoch_obj.info, cmap = cmap, axes = ax[count], show = False, image_interp = 'cubic',extrapolate='local',vlim = [min_val, max_val])
            str_time = str(t_index)+" ms"
            ax[count].set_title(title_str+str_time, color='black', fontsize=12)
            cbar =fig.colorbar(im, ax = ax[count], orientation="vertical", pad = 0.15)
            cbar.set_label("in uV")
            count = count+1
        plt.show()

        if(save_figure): 
            fig.savefig(path +filename)
        

    def icaEOGArtifactRemoval(self, n_components = 20, drop_epochs = False, threshhold = 0.05, ch_names = ["FP1", "FP2"], plot_steps = False, apply_baseline = False, baseline = (None, -0.2)):
        """
        This function automatically detects EOG artifacts in the given rereferenced EEG-signal using an ICA.
        The decisive components are marked and removed.

        Parameters
        ----------
        n_components : int, optional
            Numer of principal components thate are passed to the ICA algorithm during the fitting, by default 20
        drop_epochs : bool, optional
            If True, a threshold dtermins the dropping of bad epochs base on peak to peak value, by default False
        threshhold : float, optional
            The theshold for dropping epochs, by default 0.05
        ch_names : list, optional
            The name of the channel(s) to use for EOG peak detection, by default ["FP1", "FP2"]
        plot_steps : bool, optional
            If True the single steps of the ICA will be visualized for further analysis, by default False
        baseline : tuple, optional
            The time interval to consider as “baseline” when applying baseline correction, by default (None, -0.2)
        
        Author
        ------
        Author : Patrick Bings \n
        Last changed: 11.03.2024 (by Patrick Bings)
        """

        # set montage for plotting
        self.epoch_obj.set_montage(self.__montage)

        # dropping bad epochs based on peak to peak value if True
        if(drop_epochs):
            self.epoch_obj.drop_bad(reject = {'eeg': threshhold})

        # creating epochs around the EOG-artifacts
        eog_evoked = mne.preprocessing.create_eog_epochs(self.raw_obj, ch_name=ch_names).average()

        if(apply_baseline): 
            eog_evoked.apply_baseline(baseline=baseline)

        # creating the ICA and fitting it to the epoched raw data: 
        ica = ICA(n_components=n_components, max_iter="auto", random_state=97)
        ica.fit(self.epoch_obj)

        # exclude the right ICs
        ica.exclude = []

        # find which ICs match the EOG pattern
        eog_indices, eog_scores = ica.find_bads_eog(self.epoch_obj, ch_name=ch_names)

        ica.exclude.extend(eog_indices)
        print(ica.exclude)

        ica.apply(self.epoch_obj)

        self.epochs = self.epoch_obj.get_data()
        
        if plot_steps:
            ica.plot_components()
            ica.plot_overlay(eog_evoked, exclude=eog_indices, show=False)
            # barplot of ICA component "EOG match" scores
            ica.plot_scores(eog_scores)
            # plot diagnostics
            ica.plot_properties(self.epoch_obj, picks=eog_indices)
            #plot ICs applied to raw data, with EOG matches highlighted
            ica.plot_sources(self.epoch_obj, show_scrollbars=False)
            # plot ICs applied to the averaged EOG epochs, with EOG matches highlighted
            ica.plot_sources(eog_evoked)


class OnlineEEG(OnlineTimeseriesStreaming, EEGData): 
    """
    This class provides useful methods for doing online EMG processing and classification. It inherits processing methods from the EMGData class and methods for data streaming from OnlineTimeseriesStreaming . 
    
    Parameters
    ----------
    OnlineTimeseriesStreaming : class
        The OnlineTimeseriesStreaming includes useful methods for online data streaming and data handling. 
    EEGData : class
        The base EEGData class including all processing methods for EEG data. 
    """ 
    
    def __init__(self, stream_type = "data", channel_names = ["1", "2", "3"], n_samples= 500, dt_process_data = 0.05, f_samp = 1000.0): 
        """
        The constructor of the OnlineEEG class. 

        Author
        ------
        Author : Niklas Kueper \n
        Last changed: 20.09.2024 (by Niklas Kueper)
        """        
        #print(stream_type)
        
        
        OnlineTimeseriesStreaming.__init__(self, stream_type = stream_type, channel_names = channel_names, n_samples= n_samples, dt_process_data = dt_process_data, f_samp = f_samp)#, stream_type = stream_type, channel_names = ["1", "2", "3"], n_channels=n_channels, n_samples= n_samples, dt_process_data = dt_process_data, f_samp = f_samp)
        EEGData.__init__(self, format = "Live", f_samp = f_samp, channel_names = channel_names)

        



