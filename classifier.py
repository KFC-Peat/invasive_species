import math as m
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy as sp
import scipy.misc
import sys
import tensorflow as tf
import time

from neural_net import neural_net

# Load the testing data
def data_loader():

    with open('./data/test32.npy', 'rb') as f:
        image_array = np.load(f)

    return image_array

def classify(image_array):

    IMG_NUM = np.shape(image_array)[0]
    IMG_SIZE = 32

    sess = tf.InteractiveSession()

    x = tf.placeholder(dtype=tf.float32, shape=[None, IMG_SIZE, IMG_SIZE, 3])
    y = tf.placeholder(dtype=tf.float32, shape=[None, 2])
    keep_prob = tf.placeholder(tf.float32)

    y_ = neural_net(x, y, keep_prob)

    saver = tf.train.Saver()
    filepath = './models/feature'
    saver.restore(sess, filepath)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

    print('\nLoaded neural network...\n')

    bs = 32
    batchs = IMG_NUM // bs

    image_batch = np.zeros([bs, IMG_SIZE, IMG_SIZE, 3], dtype=np.uint8)
    zeros = np.zeros([bs, 2], dtype=np.uint8)
    ones = np.zeros([bs, 2], dtype=np.uint8)
    predictions = np.zeros([IMG_NUM], dtype=np.float32)
    prob = np.zeros([bs], dtype=np.float32)

    for i in range(bs):
        ones[i,1] = 1
        zeros[i,0] = 1

    for i in range(batchs):
        image_batch[:,:,:,:] = image_array[bs*i:bs*(i+1),:,:,:]

        ce_ones = cross_entropy.eval(feed_dict={x: image_batch, y: ones, keep_prob: 1.0})
        ce_zeros = cross_entropy.eval(feed_dict={x: image_batch, y: zeros, keep_prob: 1.0})

        
        for j in range(bs):
            if ce_zeros[j] == 0:
                prob[j] = -m.log(ce_ones[j])
            else:
                prob[j] = m.log(ce_zeros[j])

        for j in range(bs):
            predictions[i*bs+j] = prob[j]

    
    #last semibatch
    image_batch[:,:,:,:] = image_array[IMG_NUM-bs:IMG_NUM,:,:,:]

    ce_ones = cross_entropy.eval(feed_dict={x: image_batch, y: ones, keep_prob: 1.0})
    ce_zeros = cross_entropy.eval(feed_dict={x: image_batch, y: zeros, keep_prob: 1.0})

    for j in range(bs):
        if ce_zeros[j] == 0:
            prob[j] = -m.log(ce_ones[j])
        else:
            prob[j] = m.log(ce_zeros[j])

    for j in range(bs):
        predictions[IMG_NUM-bs+j] = prob[j]

    
    #normalise between 0 and 1
    for j in range(len(predictions)):
        if predictions[j] >= 10:
            predictions[j] = 1
        elif predictions[j] <= -10:
            predictions[j] = 0
        else:
            predictions[j] = predictions[j] / 20 + 0.5
    
    return predictions


image_array = data_loader()

predictions = classify(image_array)

"""for i in range(len(predictions)):
    print(predictions[i])"""

with open('./data/predictions.npy', 'wb') as f:
    np.save(f, predictions)