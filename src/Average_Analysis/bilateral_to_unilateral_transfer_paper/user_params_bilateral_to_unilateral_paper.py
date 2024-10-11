
# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************
import os 


# *********************************************************************************
# ************** User Parameters and data selection  ******************************
# *********************************************************************************

#proj path 
proj_path = "/home/dfki.uni-bremen.de/nkueper/Dokumente/DFKI_Job/EXPECT/biosignal_toolbox"
data_path = os.path.join(proj_path, 'data') # path where the data lays 

figure_path = "/home/dfki.uni-bremen.de/nkueper/Dokumente/DFKI_Job/Abbildungen/LRP_Plots/"

subject = "JD68"
evalname = "_unilateral_01_4Hz_ICA"

#filenames = [#"20211210_r_JV43_intentional_unilateral_set1.vhdr", "20211210_r_JV43_intentional_unilateral_set2.vhdr", "20211210_r_JV43_intentional_unilateral_set3.vhdr"] 
             #"20211216_r_RA12_intentional_unilateral_set1.vhdr", "20211216_r_RA12_intentional_unilateral_set2.vhdr", "20211216_r_RA12_intentional_unilateral_set3.vhdr"] 
             # "20211220_r_AV82_intentional_unilateral_set1.vhdr", "20211220_r_AV82_intentional_unilateral_set2.vhdr", "20211220_r_AV82_intentional_unilateral_set3.vhdr"] 
             # "20211223_r_UP28_intentional_unilateral_set1.vhdr", "20211223_r_UP28_intentional_unilateral_set2.vhdr", "20211223_r_UP28_intentional_unilateral_set3.vhdr"] 
             #  "20220104_r_ZS27_intentional_unilateral_set1.vhdr", "20220104_r_ZS27_intentional_unilateral_set2.vhdr", "20220104_r_ZS27_intentional_unilateral_set3.vhdr"] 
             # "20220105_r_JD68_intentional_unilateral_set1.vhdr", "20220105_r_JD68_intentional_unilateral_set2.vhdr", "20220105_r_JD68_intentional_unilateral_set3.vhdr"]
             # "20220107_r_QS70_intentional_unilateral_set1.vhdr", "20220107_r_QS70_intentional_unilateral_set2.vhdr", "20220107_r_QS70_intentional_unilateral_set3.vhdr"]
             #  "20211222_r_XP01_intentional_unilateral_set1.vhdr", "20211222_r_XP01_intentional_unilateral_set2.vhdr", "20211222_r_XP01_intentional_unilateral_set3.vhdr", "20211222_r_XP01_intentional_unilateral_set4.vhdr"]

filenames = [#"20211210_r_JV43_intentional_bilateral_set1.vhdr", "20211210_r_JV43_intentional_bilateral_set2.vhdr", "20211210_r_JV43_intentional_bilateral_set3.vhdr"]
              #"20211216_r_RA12_intentional_bilateral_set1.vhdr", "20211216_r_RA12_intentional_bilateral_set2.vhdr", "20211216_r_RA12_intentional_bilateral_set3.vhdr"]
              #"20211220_r_AV82_intentional_bilateral_set1.vhdr", "20211220_r_AV82_intentional_bilateral_set2.vhdr", "20211220_r_AV82_intentional_bilateral_set3.vhdr"]
              #"20211223_r_UP28_intentional_bilateral_set1.vhdr", "20211223_r_UP28_intentional_bilateral_set2.vhdr", "20211223_r_UP28_intentional_bilateral_set3.vhdr"]
              #"20220104_r_ZS27_intentional_bilateral_set1.vhdr", "20220104_r_ZS27_intentional_bilateral_set2.vhdr", "20220104_r_ZS27_intentional_bilateral_set3.vhdr"]
             "20220105_r_JD68_intentional_bilateral_set1.vhdr", "20220105_r_JD68_intentional_bilateral_set2.vhdr", "20220105_r_JD68_intentional_bilateral_set3.vhdr"]
             #"20220107_r_QS70_intentional_bilateral_set1.vhdr", "20220107_r_QS70_intentional_bilateral_set2.vhdr", "20220107_r_QS70_intentional_bilateral_set3.vhdr"]
             # "20211222_r_XP01_intentional_bilateral_set1.vhdr", "20211222_r_XP01_intentional_bilateral_set2.vhdr", "20211222_r_XP01_intentional_bilateral_set3.vhdr"]


# Filtering Params for EEG data 
f_highpass = 0.1 # in Hz 
f_lowpass = 4.0 # in Hz 
apply_filter = True # setting to False will ignore the filtering 

f_samp_eeg = 500.0
reref_channel = ["average"] # use average reference 

marker_number = 101 # markernumber that should be used for e.g. epoching (e.g.  movement onset)
error_number = 3 # number of the error marker (trials will be excluded)


# time selection for epoching of the data 
epoching_time_before_onset = -1.5 # time in seconds (start epoch)
epoching_time_after_onset = 0.0 # time in seconds (0 = movement onset)

# should baseline correction be applied ? (standard -1.5 to -1 seconds)
apply_baseline_correction = True 
t0_baseline = -1.5
t1_baseline = -1.0

# topoplot params 
plot_montage = False 
min_val = -6e-06 # Voltage values for colour scale 
max_val = 6e-06

# ica settings 
apply_ica = True
n_ica_comp = 5
 #[3] for uni JV43 01_4 Hz filter [1] for bilateral 
#[0, 1, 2] for uni RA12 [0, 1, 4] for bilateral 
#[0, 1, 3] for uni AV82 --> [0] for bilateral 
# [2, 4] for uni UP28  --> [0, 7] for bilateral 
# [0, 1, 2, 6, 9] for uni ZS27 --> many artefacts , [0, 1, 2, 3, 4, 5] for bilateral 
#[0, 4] for uni JD68 , [2] for bilateral 
# [0, 1, 2, 3, 4, 5, 8, 9] for QS70 uni --> many artefacts , [0, 2, 5] for bilateral 
# [0, 1, 2, 4, 5, 6, 7] for uni XP01 , [0, 1, 2, 6, 7, 8, 9] for bilateral 
exclude_ica_comp = [2]
show_ica_components = True


# topoplot params 
topoplot_times =  [-1000, -100] # times in ms to the event after epoching 
topoplot_title_str = "time to movement onset "

# select channel name for visualizing it 
channels_to_evaluate = ["C2", "C1"]

# eeg channel that are kept (inverse_keep_channel = False) or dropped (inverse_keep_channel = True) for further evaluations, empty list meaning all channels are kept 
inverse_keep_channel = True 
channel_list = ["x_dir", "y_dir", "z_dir"]# "FP1", "FP2", "F8", "T7", "T8", "TP9", "TP10", "P7","P8", "PO9", "O1", "OZ", "O2", "PO10", "AF7", "AF3", "AF4", "AF8", "FT9", "FT7", "FT8", "FT10", "TP7", "TP8", "PO7", "PO3", "POZ", "PO4", "PO8","F7"]
rename_channels = True 





