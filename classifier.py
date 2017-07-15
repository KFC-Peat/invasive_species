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


IMG_SIZE = 128

with open('../sample_submission.csv', 'r') as f:
	ss = pd.read_csv(f)

length = len(ss)

# Load and process image data
with open('./data/test.npy', 'rb') as f:
	images_u = np.load(f)

images = np.zeros((length,IMG_SIZE,IMG_SIZE,3), dtype=np.uint8)
for i in tqdm(range(length)):
	images[i] = sp.misc.imresize(images_u[i], (IMG_SIZE,IMG_SIZE,3))
del images_u
gc.collect()

print('Loaded and processed test images...\n')


model = load_model('./models/model.h5')
print('Loaded neural network model...\n')

predictions = model.predict(images)
print('Data predictions made...\n')

for i in tqdm(range(len(ss))):
	ss.loc[i,'invasive'] = predictions[i][0]

with open('./data/submission.csv', 'w') as f:
	ss.to_csv(f, index=False)

print('\nDone!\n')