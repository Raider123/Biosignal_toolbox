""" 

Recording script for saving data from LSL stream as numpy array. Run this script with sudo rights because keyboard (package) requires this. 
To run the script with you current python version activated, give the full path of your python version before running the script. Type which python in the console to get the full path. 


"""


from pylsl import StreamInlet, resolve_stream
import numpy as np 
import keyboard
import os 
import argparse
from biosignal_toolbox.eeg_lib import EEGData, OnlineEEGUtils

def main():


    #************************************************************
    # ********************** user params ************************
    #************************************************************

    # set paths 
    proj_path = "/home/dfki.uni-bremen.de/nkueper/Dokumente/DFKI_Job/EXPECT/biosignal_toolbox"
    data_path = proj_path+"/data/"


    #************************************************************
    #************************************************************
    #************************************************************

    # argument parser 
    parser = argparse.ArgumentParser("Parser for accepting run-time arguments")
    parser.add_argument('-n','--name',help='File name')
    args = vars(parser.parse_args())

    if args['name'] is not None:
        filename = args['name']
    else: 
        filename = 'test'


    # first resolve an EEG stream on the lab network
    print("looking for an EEG stream...")
    streams = resolve_stream('type', 'EEG') # create data stream 

    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0]) 
    stream_info = inlet.info()

    #Create EEG utils object for helping methods 
    EEGutils = OnlineEEGUtils()
    EEGutils.printStreamMetadata(stream_info) # print stream info 

    # run continiously 
    running = False

    # uncomment if data should be recorded (not necessary for online prediction, use buffer for that)
    data_arr = []
    time_stamp_arr = []


    print("Press s to start recording ...")
    while(True): 
        if(keyboard.is_pressed("s")): 
            running = True
            break 

    print("started recording")

    while running:
       
        chunk, timestamps = inlet.pull_chunk() # get a new data chunk

        if(chunk): # if list not empty (new data)
            
            # uncomment to record ALL data received (not required for participants)
            data_arr = data_arr+chunk
            time_stamp_arr = time_stamp_arr + timestamps

            if(keyboard.is_pressed("e")): 
                running = False
    
    # vonvert and save data and timestamps 
    print("end recording")

    data_arr_np = np.array(data_arr)
    time_stamp_arr_np = np.array(time_stamp_arr) 

    print("storing data")
    if not (os.path.isfile(data_path+filename+"_data") and data_path+filename+"_timestamp"): 
        np.save(data_path+filename+"_data", data_arr_np)
        np.save(data_path+filename+"_timestamp", time_stamp_arr_np)

    else: 
        print("File already exists take care next time !")
        np.save(data_path+filename+"_data_1", data_arr_np)
        np.save(data_path+filename+"_timestamp_1", time_stamp_arr_np)


if __name__ == '__main__':
    main()
