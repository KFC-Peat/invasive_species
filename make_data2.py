import numpy as np
import scipy as sp
from scipy import misc
from os import listdir
import sys
import matplotlib.pyplot as plt
import csv
import random as rand

### MAKE THE IMAGE DATA INTO NUMPY ARRAY
IMG_NUM = 1513

data_set = np.zeros([IMG_NUM,433,577,3], dtype=np.uint8)

for i in range(IMG_NUM):

    fp = '../test/{}.jpg'.format(i+1)

    image = sp.misc.imread(fp)

    image_scaled = sp.misc.imresize(image,[433,577,3])

    data_set[i,:,:,:] = image_scaled

    if i%100 == 0:
        print(i)

with open('./data/test.npy', 'wb') as f:
    np.save(f, data_set)