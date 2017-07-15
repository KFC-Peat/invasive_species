import numpy as np
import scipy as sp
from scipy import misc
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

### MAKE THE IMAGE DATA INTO NUMPY ARRAY

dirpath = '../test'
filelist = os.listdir(dirpath)

IMG_NUM = len(filelist)

data_set = np.zeros([IMG_NUM,433,577,3], dtype=np.uint8)

for i in tqdm(range(IMG_NUM)):

    fp = '../test/{}.jpg'.format(i+1)
    image = sp.misc.imread(fp)
    image_scaled = sp.misc.imresize(image,[433,577,3])
    data_set[i,:,:,:] = image_scaled

with open('./data/test.npy', 'wb') as f:
    np.save(f, data_set)