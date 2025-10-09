import numpy as np 

# own libs
proj_path = "/home/dfki.uni-bremen.de/nkueper/Dokumente/DFKI_Job/EXPECT/biosignal_toolbox"
data_path = proj_path+"/data/"

# old_online params
buffer_size = 500  # size of ringbuffer in samples, currently set to 2500 (5 sec data times 500 Hz sampling rate)
dt_read_buffer= 0.05 # time in seconds how often the buffer is read  (updated with new incoming chunks)
num_classes = 2
print_times = False
do_class_pred = True
send_marker = True
deadtime = 1 # in seconds after exo movement done 

# subject info 
subject = "Test"
scenario_name = "intentional_unilateral"
iteration = 0
result_file_name = "live_train_results"

# model names --> always the newest model thingi 
MLP_eval_name = "current_intentional_unilateral_live_model_MLP_0"
EEGNet_eval_name = "current_intentional_unilateral_live_model_EEGNet0"

# ML params 
decision_bound = 0.7
n_count_positives = 2 # how many window have to be positive 

# MLP Net 
#features = "fusion" # which features to be used for classification, "timepoints" or "meanfreqs" or "fusion" (combine both)
#feature_indices_windows = np.arange(900, 1000, step = 2) # numpy array with time feature indices, (950, 1000) means last 100 ms of a window are used 
feature_indices_windows = np.array([900, 920, 940, 960, 965, 970, 975, 980, 985, 990, 995, 999])

# # marker params 
# usb_port = '/dev/ttyUSB0'
# Baudrate = 115200

# eeg stream params 
#channel_names = ["F5", "F3", "F1", "FZ", "F2", "F4", "F6", "FC5", "FC3", "FC1", "FC2", "FC4", "FC6", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "P5", "P3", "P1", "PZ", "P2", "P4", "P6"]
channel_names = ["FC3", "FC1", "C3", "C1", "CZ", "CP3", "CP1", "CPZ", "CCP1h", "FCC1h", "CCP3h", "FCC3h"]
num_non_data_channels = 1 # or 3 ? 

# shared array params for scores 
#sa_name1 = "shm://scores"
#sa_size1 = buffer_size
score_names = ["product"]

# EEG data params 
n_channels = len(channel_names)
f_samp_eeg = 500.0

# zmq stuff 
zmq_port = "34761"
zmq_topic = b"10"
ip_rock = "10.250.3.15"

