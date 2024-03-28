import sys 
import warnings 
import numpy as np 
from time import perf_counter
import time 


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
        self.f_samp = f_samp
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
            if (self.f_samp == 1000): 
                samp_index = 2
            elif (self.f_samp == 500): 
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
                    self.data_chunk = list(self.data_stream.getData()) # read EMG/EEG data out of buffer
                    if(return_chunk): 
                        return self.data_chunk
                else: 
                    self.impedance_chunk = list(self.impedance_stream.getData()) # read EMG/EEG data out of buffer
                    if(return_chunk):
                        return self.impedance_chunk

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
        if(self.data_chunk): # only to this if new data is received 
            current_chunk = (np.array(self.data_chunk)) # chunk is sampels, channels, after transpose then channels, sampels !
            if(show_data_shape): 
                print("data chunk shape:", current_chunk.shape) # should be in channels, sampels 


            if(channel_indices): 
                current_chunk = current_chunk[channel_indices, :]

            # #print(current_chunk.shape)
            # current_chunk = current_chunk[0:n_channels, :] # use first n channels

            n_samples = current_chunk.shape[1] 

            if (n_samples > self.data_buffer.shape[2]): # print error message 
                print("Buffer overflow")

            
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