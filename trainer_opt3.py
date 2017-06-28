import matplotlib.pyplot as plt
import numpy as np
import os
import scipy as sp
import scipy.misc
import sys
import tensorflow as tf
import time
import random

from neural_net96 import neural_net

# Load the training data
def data_loader():

    with open('./data/train.npy', 'rb') as f:
        image_array = np.load(f)
    with open('./data/labels.npy', 'rb') as f:
        label_array = np.load(f)

    length = np.shape(label_array)[0]

    label_array_onehot = np.zeros([length,2], dtype=np.uint8)

    for i in range(length):
        if label_array[i] == 0:
            label_array_onehot[i,0] = 1
        else:
            label_array_onehot[i,1] = 1

    return image_array, label_array_onehot

image_array, label_array = data_loader()


def trainer(image_array, label_array):

    # Get data dimentions
    IMG_NUM = 2000
    IMG_SIZE = 96
    col = np.shape(image_array)[1]
    row = np.shape(image_array)[2]



    # Initialise neural network
    sess = tf.InteractiveSession()

    x = tf.placeholder(dtype=tf.float32, shape=[None, IMG_SIZE, IMG_SIZE, 3])
    y = tf.placeholder(dtype=tf.float32, shape=[None, 2])
    keep_prob = tf.placeholder(tf.float32)

    y_ = neural_net(x, y, keep_prob)

    print('Initialised neural network...\n')



    # More neural network initialisation
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()
    filepath = './models/feature'

    print('Initialised neural network P2...\n')



    # Train the network
    print('Start neural network training...\n')

    bs = 32
    bs_t = 250
    epochs = 20000
    max_batch = IMG_NUM // bs

    sess.run(tf.global_variables_initializer())

    image_batch = np.zeros([bs, IMG_SIZE, IMG_SIZE, 3], dtype=np.uint8)
    label_batch = np.zeros([bs, 2], dtype=np.uint8)

    # create testing batch
    image_batch_t = np.zeros([bs_t, IMG_SIZE, IMG_SIZE, 3], dtype=np.uint8)
    label_batch_t = np.zeros([bs_t, 2], dtype=np.uint8)

    for j in range(bs_t):
        image_batch_t[j,:,:,:] = sp.misc.imresize(image_array[IMG_NUM+j,:,:,:],[IMG_SIZE,IMG_SIZE,3])
    label_batch_t[:,:] = label_array[IMG_NUM:IMG_NUM+bs_t,:]

    for i in range(epochs):
        # create the batches
        cb = i % max_batch
        for j in range(bs):
            image_batch[j,:,:,:] = sp.misc.imresize(image_array[bs*cb+j,:,:,:],[IMG_SIZE,IMG_SIZE,3])
        label_batch[:,:] = label_array[bs*cb:bs*(cb+1),:]

        # test the current network
        if i%100 == 0:
            print(i, accuracy.eval(feed_dict={x: image_batch_t, y: label_batch_t, keep_prob: 1.0}))

        # train the network
        train_step.run(feed_dict={x: image_batch, y: label_batch, keep_prob: 0.5})

    # Training complete
    print('\nFinished training...\n\n')



image_array, label_array = data_loader()

trainer(image_array,label_array)