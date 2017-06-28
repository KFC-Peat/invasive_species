import numpy as np
import scipy as sp
from scipy import misc
from os import listdir
import sys
import matplotlib.pyplot as plt
import csv
import random as rand


"""
### MAKE THE CSV DATA INTO NUMPY ARRAY
IMG_NUM = 1531

data_labels = np.zeros([IMG_NUM], dtype=np.uint8)

with open('../train_labels.csv', 'r') as f:
    csv_object = csv.reader(f)
    csv_list = []
    for line in csv_object:
        csv_list.append(line)

for i in range(IMG_NUM):
    data_labels[i] = int(csv_list[i+1][1])

with open('./data/labels.npy', 'wb') as f:
    np.save(f, data_labels)
"""


### MAKE THE IMAGE DATA INTO NUMPY ARRAY
IMG_NUM = 1531
IMG_SIZE = 48

data_set = np.zeros([IMG_NUM,IMG_SIZE,IMG_SIZE,3], dtype=np.uint8)

for i in range(IMG_NUM):

    fp = '../test/{}.jpg'.format(i+1)

    image = sp.misc.imread(fp)

    image_scaled = sp.misc.imresize(image,[IMG_SIZE,IMG_SIZE,3])
    data_set[i,:,:,:] = image_scaled

    if i%100 == 0:
        print(i)

with open('./data/test48.npy', 'wb') as f:
    np.save(f, data_set)