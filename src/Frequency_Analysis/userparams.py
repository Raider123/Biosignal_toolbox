import numpy as np 

# *********************************************************************************
# ************** User Parameters and data selection  ******************************
# *********************************************************************************

# own libs
proj_path = "/home/dfki.uni-bremen.de/nkueper/Dokumente/DFKI_Job/EXPECT/biosignal_toolbox"
#data_path = proj_path+"/data/20240626_r______Test_Shielding_cabine_outside_shielding_cabin_resting/Set2"
#data_path = proj_path+"/data/20240626_r______Test_Shielding_cabine_in_BB310_resting/Set1"
data_path = proj_path+"/data/20240626_r______Test_Shielding_cabine_in_shielding_cabin_resting/Set2"


filename = ["20240626_r______Test_Shielding_cabine_in_shielding_cabin_resting_Set2.vhdr"]
# channels_to_evaluate = ["FC6"]
title = filename[0]
#indizes = [0, -1] # indices of the window to evaluate 
scales_plot = 0.0005 # how to scale the axis 
scales_plot_freq = 0.00001
n_channels_to_show = 4

# filter settings 
f_highpass = 0.5 

