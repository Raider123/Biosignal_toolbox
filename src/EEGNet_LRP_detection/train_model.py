import numpy as np
import sys

import mne
from mne import io
from mne.datasets import sample

from EEGModels import EEGNet

from tensorflow.keras import utils as np_utils
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K

K.set_image_data_format('channels_last')

# load preprocessed data
data_path = 'SCRATCH/data'
X_train = np.load('%s/%s_X_data.npy' % (data_path, sys.argv[1]))
Y_train = np.load('%s/%s_labels.npy' % (data_path, sys.argv[1]))
X_validate = np.load('%s/%s_X_data.npy' % (data_path, sys.argv[2]))
Y_validate = np.load('%s/%s_labels.npy' % (data_path, sys.argv[2]))

name_shema = "%s_%s_%s_%s_%s_%s" % (sys.argv[1].split("_")[-3], sys.argv[1].split("_")[-2], sys.argv[1].split("_")[-1], sys.argv[2].split("_")[-2],sys.argv[2].split("_")[-1], sys.argv[1].split("_")[2])

kernels, chans, samples = 1, 64, 129

# convert labels to one-hot encodings.
Y_train = np_utils.to_categorical(Y_train)
Y_validate   = np_utils.to_categorical(Y_validate)

X_validate   = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
X_train   = X_train.reshape(X_train.shape[0], chans, samples, kernels)

model = EEGNet(nb_classes = 2, Chans = chans, Samples = samples, 
               dropoutRate = 0.5, kernLength = 32, F1 = 8, D = 2, F2 = 16, 
               dropoutType = 'Dropout')

model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics = [metrics.Precision(name='precision'), metrics.Recall(name='recall')])

checkpointer = ModelCheckpoint(filepath='SCRATCH/weights/%s.h5' % name_shema, verbose=1,
                               save_best_only=True)

early_stopping = EarlyStopping(monitor="val_loss", patience=10)

class_weights = {0:1, 1:2}

fittedModel = model.fit(X_train, Y_train, batch_size = 64, epochs = 300,
                        verbose = 2, validation_data=(X_validate, Y_validate),
                        callbacks=[early_stopping, checkpointer], class_weight = class_weights)

model.load_weights('SCRATCH/weights/%s.h5' % name_shema)

model.save("SCRATCH/models/%s" % name_shema)



