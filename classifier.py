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
    label_batch = np.zeros([bs, 2], dtype=np.uint8)
    predictions = np.zeros([IMG_NUM], dtype=np.uint8)

    for i in range(bs):
        label_batch[i,1] = 1

    for i in range(batchs):
        image_batch[:,:,:,:] = image_array[bs*i:bs*(i+1),:,:,:]

        preds = correct_prediction.eval(feed_dict={x: image_batch, y: label_batch, keep_prob: 1.0})

        for j in range(bs):
            if preds[j]:
                predictions[i*bs+j] = 1
    
    return predictions


image_array = data_loader()

predictions = classify(image_array)

with open('./data/predictions.npy', 'wb') as f:
    np.save(f, predictions)