import numpy as np
import sys
from EEGModels import EEGNet

from tensorflow import keras
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K

K.set_image_data_format('channels_last')

# load preprocessed data for test
data_path = 'SCRATCH/data'
X_test = np.load('%s/%s_X_data.npy' % (data_path, sys.argv[3]))
Y_test = np.load('%s/%s_labels.npy' % (data_path, sys.argv[3]))

kernels, chans, samples = 1, 64, 129


# convert labels to one-hot encodings.
Y_test       = np_utils.to_categorical(Y_test)
X_test       = X_test.reshape(X_test.shape[0], chans, samples, kernels)

# load the model
model = keras.models.load_model("SCRATCH/models/%s_%s_%s_%s_%s_%s" % (sys.argv[1].split("_")[-3], sys.argv[1].split("_")[-2], sys.argv[1].split("_")[-1], sys.argv[2].split("_")[-2], sys.argv[2].split("_")[-1], sys.argv[1].split("_")[2]))

probs       = model.predict(X_test)
preds       = probs.argmax(axis = -1)  
diff = preds-Y_test.argmax(axis=-1)
all_positives = np.where(preds == 1)[0]
all_negatives = np.where(preds == 0)[0]
fp = np.where(diff == 1)[0]
fn = np.where(diff == -1)[0]
num_tp = len(all_positives) - len(fp)
num_tn = len(all_negatives) - len(fn)

print("total nr ", len(preds))
print("nr positives ", len(all_positives))
print("nr negatives ", len(all_negatives))
print("positives ", all_positives)
print("negatives ", all_negatives)

print("ba ", (num_tp+num_tn)/(num_tp+num_tn+len(fp)+len(fn)))
print("precision ", num_tp/(num_tp+len(fp)) if (num_tp+len(fp))>0 else 0)
print("recall ", num_tp/(num_tp+len(fn)) if (num_tp+len(fn)) > 0 else 0)
print("prec+rec ", (num_tp/(num_tp+len(fp)) + num_tp/(num_tp+len(fn)))/2 if (num_tp+len(fp))>0 and (num_tp+len(fn)) > 0 else 0)
print("fn-rate ", len(fn)/(len(fn)+num_tp) if (len(fn)+num_tp) else 0)
print("fp-rate ", len(fp)/(len(fp)+num_tn) if (len(fp)+num_tn) else 0)
print("probs: ", probs);
print("preds: ", preds);

print('False positives: ', fp)
print('False negatives: ', fn)


