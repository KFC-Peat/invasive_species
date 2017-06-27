import matplotlib.pyplot as plt
import numpy as np
import os
import scipy as sp
import scipy.misc
import sys
import tensorflow as tf
import time

from neural_net import neural_net


# Load the training data
def data_loader():

    with open('./data/image32.npy', 'rb') as f:
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



# This function trains a binary neural network to determine whether a feature is present
def trainer(image_array, label_array):

    # Get data dimentions
    img_num = np.shape(image_array)[0]
    img_size = np.shape(image_array)[1]



    # Initialise neural network

    sess = tf.InteractiveSession()

    x = tf.placeholder(dtype=tf.float32, shape=[None, img_size, img_size, 3])
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
    epochs = 2000
    max_batch = img_num // bs
    sess.run(tf.global_variables_initializer())

    image_batch = np.zeros([bs, img_size, img_size, 3], dtype=np.uint8)
    label_batch = np.zeros([bs, 2], dtype=np.uint8)

    for i in range(epochs):

        cb = i % max_batch

        image_batch[:,:,:,:] = image_array[bs*cb:bs*(cb+1),:,:,:]
        label_batch[:,:] = label_array[bs*cb:bs*(cb+1),:]

        if i%100 == 0:
            print(i, accuracy.eval(feed_dict={x: image_batch, y: label_batch, keep_prob: 1.0}))

        train_step.run(feed_dict={x: image_batch, y: label_batch, keep_prob: 0.5})


    # Training complete
    print('\nFinished training...\n\n')

    saver.save(sess, filepath) # save neural net
    sess.close() # close tensorflow session



image_array, label_array = data_loader()

trainer(image_array,label_array)