


from pylsl import StreamInlet, resolve_stream, StreamOutlet, StreamInfo
import numpy as np 
import keyboard
import os 
import argparse
from biosignal_toolbox.eeg_lib import EEGData, OnlineEEGUtils
import time 
import random

def main():


    #************************************************************
    # ********************** user params ************************
    #************************************************************


    f_samp_eeg = 500.0
    n_channel = 37 
    dt = 0.04
    chunksize = 20

    # set paths 
    proj_path = "/home/dfki.uni-bremen.de/nkueper/Dokumente/DFKI_Job/EXPECT/biosignal_toolbox"
    data_path = proj_path+"/data/"


    #  loading and epoching for training  
    data_list = ["BR60D_unilateral_live_2_data"] #"BR60D_unilateral_LSL_set4_1.vhdr"
    # eeg stream params 
    channel_names = ['F3', 'F1', 'FZ', 'F2', 'F4', 'FFC1h', 'FC5', 'ff3', 'ff4','FC3', 'FC1', 'FC2', 'FCC3h', 'FCC1h', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'FCC2h', 'CCP3h', 'CCP1h', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'P3', 'P1', 'PZ', 'P2', 'P4']
    
    #print(len(channel_names))
    #************************************************************
    # ***********************************************************
    #************************************************************
    

    # data_train = EEGData(format = "Recorded_LSL_stream", filenames = data_list, data_path = data_path, f_samp = f_samp_eeg, channel_names = channel_names)  
    # raw_obj = data_train.getRawObject()
    # raw_data = raw_obj.get_data()
    # print("raw data shape", raw_data.shape)

    raw_data = np.load(data_path+"BR60D_unilateral_live_2_data.npy") 
    raw_data = raw_data.T
    print("raw data shape", raw_data.shape)


    info = StreamInfo('Liveamp', 'EEG', channel_count = n_channel, nominal_srate=f_samp_eeg)

    # next make an outlet
    outlet = StreamOutlet(info)


    for i in range(0, raw_data.shape[1]): 

        print("now sending data...")
    
        # make a new random 8-channel sample; this is converted into a
        # pylsl.vectorf (the data type that is expected by push_sample)

        window_chunk = raw_data[:, i*chunksize:i*chunksize+chunksize]
        print("chunk shape", window_chunk.shape)
        window_chunk = window_chunk.T
        
        # now send it and wait for a bit
        outlet.push_chunk(window_chunk.tolist())
        time.sleep(dt)



if __name__ == '__main__':
    main()
