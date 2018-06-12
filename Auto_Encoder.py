# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2018/6/2 9:53
# @Author  : Yuzhou Hou
# @Email   : m18010639062@163.com
# @File    : Auto_Encoder.py
# Description : 自动编码机
# --------------------------------------

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/data/", one_hot = True)


learning_rate = 0.01  #0.05，0.02，0.005
#n_hidden_1 = 512
n_hidden_1 = 256
#n_hidden_3 = 128
n_hidden_2 = 64
#n_hidden_5 = 32
n_hidden_3 = 16
n_hidden_4 = 2
n_input = 784

# Auto_Encoder is an unsupervised model
# Let the output has the same shape as the input
x = tf.placeholder(tf.float32, [None, n_input])
y = x

# The whole model are separated into encoder and decoder
# Encoder T: R^n -> R^m, where m < n, this T can be treated as dimensionality reduction
weights = {'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
           'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
           'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
           'encoder_h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
# Decoder T': R^m -> R^n, treated as reconstruction
           'decoder_h1': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_3])),
           'decoder_h2': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_2])),
           'decoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
           'decoder_h4': tf.Variable(tf.random_normal([n_hidden_1, n_input]))}

bias = {'encoder_b1': tf.Variable(tf.constant(0.0, shape=[n_hidden_1])),
        'encoder_b2': tf.Variable(tf.constant(0.0, shape=[n_hidden_2])),
        'encoder_b3': tf.Variable(tf.constant(0.0, shape=[n_hidden_3])),
        'encoder_b4': tf.Variable(tf.constant(0.0, shape=[n_hidden_4])),

        'decoder_b1': tf.Variable(tf.constant(0.0, shape=[n_hidden_3])),
        'decoder_b2': tf.Variable(tf.constant(0.0, shape=[n_hidden_2])),
        'decoder_b3': tf.Variable(tf.constant(0.0, shape=[n_hidden_1])),
        'decoder_b4': tf.Variable(tf.constant(0.0, shape=[n_input]))}


def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), bias['encoder_b1']))
    # layer_1 = tf.nn.tf.nn.tanh(tf.add(tf.matmul(x, weights['encoder_h1']), bias['encoder_b1']))
    # tanh(x) = 2 * sigmoid(2 * x) - 1
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), bias['encoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']), bias['encoder_b3']))
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['encoder_h4']), bias['encoder_b4']))
    return layer_4


def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), bias['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), bias['decoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']), bias['decoder_b3']))
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['decoder_h4']), bias['decoder_b4']))
    return layer_4


encoder_out = encoder(x)
pred = decoder(encoder_out)

loss_function = tf.reduce_mean(tf.pow(pred - y, 2))
# loss_funtion = -tf.reduce_mean(y * tf.log(pred))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss_function)
# optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_function)

train_epochs = 20
batch_size = 256
display_step = 5

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    total_batch = int(mnist.train.num_examples / batch_size)
    for epoch in range(train_epochs):
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Add a white noise N(0, 0.3), improve the robustness of our model
            batch_x = batch_x + 0.3 * np.random.randn(batch_size, 784)
            _, loss = sess.run([optimizer, loss_function], feed_dict={x: batch_x})
            if epoch % display_step == 0:
                print("Epoch: ", '%04d' % (epoch + 1), "cost = ", "{:.9f}".format(loss))
    print("finished!")

    prediction = sess.run(pred, feed_dict={x: mnist.test.images[:10]})
    test_vec = mnist.test.images[:10]
    test_vec = test_vec + 0.3 * np.random.randn(10, 784)
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(10):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(prediction[i], (28, 28)))
    plt.show()

    aa = [np.argmax(l) for l in mnist.test.labels]
    encoder_result = sess.run(encoder_out, feed_dict={x: mnist.test.images})
    plt.scatter(encoder_result[:, 0], encoder_result[:, 1], c=aa)
    plt.colorbar()
    plt.show()