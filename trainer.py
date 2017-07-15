import keras
import numpy as np
import scipy as sp
from scipy import misc
import pandas as pd
import pickle
import sys
import os
import tensorflow as tf
from tqdm import tqdm
import gc

from sklearn import model_selection
from sklearn import metrics

from keras import optimizers
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.metrics import categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Turn off tensorflow output

# Constants
IMG_SIZE = 128
learn_rates = [1e-4, 1e-5]

# Convulutional Neural Network
def model_nn():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))   
    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.65))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.55))
    model.add(Dense(1, activation='sigmoid'))
    return model



print('\nStarto!\n')


# Load and process train labels
with open('../train_labels.csv', 'r') as f:
	labels_u = pd.read_csv(f)

length = len(labels_u)

labels = np.zeros((length), dtype=np.uint8)
for i in tqdm(range(length)):
	labels[i] = labels_u['invasive'][i]
del labels_u
gc.collect()

print('Loaded and processed train labels...\n')


# Load and process image data
with open('./data/train.npy', 'rb') as f:
	images_u = np.load(f)

images = np.zeros((length,IMG_SIZE,IMG_SIZE,3), dtype=np.uint8)
for i in tqdm(range(length)):
	images[i] = sp.misc.imresize(images_u[i], (IMG_SIZE,IMG_SIZE,3))
del images_u
gc.collect()

print('Loaded and processed train images...\n')


print('Start training network\n')

model = model_nn()
print(model.summary())
kf = model_selection.KFold(n_splits = 7, shuffle = True)

for train, test in kf.split(images):
	x_tr = images[train]; x_te = images[test]
	y_tr = labels[train]; y_te = labels[test]

	datagen = ImageDataGenerator(
			rotation_range = 30,
			width_shift_range = 0.2,
			height_shift_range = 0.2,
			shear_range = 0.2,
			zoom_range = 0.2,
			horizontal_flip = True,
			vertical_flip = True,
			fill_mode = 'nearest')

	for learn_rate in learn_rates:
		print('\nTraining model with learn rate: ', learn_rate, '\n')

		earlystop = keras.callbacks.EarlyStopping(
			monitor='val_loss', patience = 5, verbose=0, mode='auto')

		sgd = optimizers.SGD(lr = learn_rate, decay = 0, momentum = 0.8, nesterov = True)
		model.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics=['accuracy'])

		model.fit_generator(datagen.flow(x_tr, y_tr, batch_size=32),
	                    steps_per_epoch=256, epochs=1000,
	                    callbacks=[earlystop], validation_data=(x_te, y_te))

	model.save('./models/model.h5')
	break


print('\nEnd!\n')



