import numpy as np
import scipy as sp
from scipy import misc
from os import listdir
import sys
import matplotlib.pyplot as plt
import csv


### MAKE THE CSV DATA INTO NUMPY ARRAY
IMG_NUM = 2295

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



### MAKE THE IMAGE DATA INTO NUMPY ARRAY
IMG_NUM = 2295
IMG_SIZE = 32

data_set = np.zeros([IMG_NUM,IMG_SIZE,IMG_SIZE,3], dtype=np.uint8)

for i in range(IMG_NUM):

    fp = '../train/{}.jpg'.format(i+1)

    image = sp.misc.imread(fp)
    image_scaled = sp.misc.imresize(image,[32,32,3])
    data_set[i,:,:,:] = image_scaled

    if i%100 == 0:
        print(i)

with open('./data/image32.npy', 'wb') as f:
    np.save(f, data_set)