from mne import io
import numpy as np
import sys
import mne

vhdr_path = "SCRATCH/pySPACEcenter/storage/eeg_data/raw_eeg_data/%s" % (sys.argv[1])

# specify windows ('LRP' or 'NO' markers in .vmrk file mark the end of the corresponding window)
tmin, tmax = -1, 0. #in seconds
event_id_dict = {'Stimulus/LRP': 1, 'Stimulus/NO': 0}

raw_data = io.read_raw_brainvision(vhdr_path, preload=True, verbose=False)
raw_data.apply_function(lambda x: ((x - np.mean(x)) / np.std(x)))
raw_data.resample(128)
raw_data.filter(0, 40, method='iir')

events, event_id = mne.events_from_annotations(raw_data, event_id_dict, verbose=False)

raw_data.info["events"] = events

epochs = mne.Epochs(raw_data, events, event_id, tmin, tmax, picks=['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FT9', 'FT7', 'FC3', 'FC4', 'FT8', 'FT10', 'C5', 'C1', 'C2', 'C6', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'P5', 'P1', 'P2', 'P6', 'PO7', 'PO3', 'POz', 'PO4', 'PO8'], event_repeated='drop', proj=False, baseline=None, preload=True, verbose=False)
labels = epochs.events[:,-1]

# extract raw data. scale by 1000 due to scaling sensitivity in deep learning
X_data = epochs.get_data()*1000 # format is in (trials, channels, samples)

np.save('SCRATCH/data/%s_X_data.npy' % (vhdr_path.split("/")[-1].split(".")[0]), X_data)
np.save('SCRATCH/data/%s_labels.npy' % (vhdr_path.split("/")[-1].split(".")[0]), labels)
