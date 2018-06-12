# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2018/6/4 12:20
# @Author  : Yuzhou Hou
# @Email   : m18010639062@163.com
# @File    : Auto_Encoder_pretrain.py
# Description : Using Auto Encoder for pretraining

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/data/', one_hot = True)

# 建立网络，首先是两个自动编码机，分别是784-256-256-784(或者784-256-784)的降噪自动编码机，用原始图像加入白噪声作为输入
# 和256-128-128-256(或者256-128-256)的自动编码机，使用上一个自动编码机的编码器输出作为输入
# 将上述两个自动编码机的编码器级联并与输出层全链接，再使用比较小的学习率进行微调

n_input = 784
n_hidden_1 = 256
n_hidden_2 = 128
n_classes = 10

l1_x = tf.placeholder('float', [None, n_input])
l1_y = tf.placeholder('float', [None, n_input])
keep_prob = tf.placeholder(tf.float32)

l2_x = tf.placeholder('float', [None, n_hidden_1])
l2_y = tf.placeholder('float', [None, n_hidden_1])

l3_x = tf.placeholder('float', [None, n_hidden_2])
l3_y = tf.placeholder('float', [None, n_classes])

weights = {
    'l1_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'l1_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_1])),
    'l1_out': tf.Variable(tf.random_normal([n_hidden_1, n_input])),

    'l2_h1': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'l2_h2': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_2])),
    'l2_out': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),

    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}

bias = {
    'l1_b1': tf.Variable(tf.constant(0.0, shape=[n_hidden_1])),
    'l1_b2': tf.Variable(tf.constant(0.0, shape=[n_hidden_1])),
    'l1_out': tf.Variable(tf.zeros([n_input])),

    'l2_b1': tf.Variable(tf.zeros([n_hidden_2])),
    'l2_b2': tf.Variable(tf.zeros([n_hidden_2])),
    'l2_out': tf.Variable(tf.zeros([n_hidden_1])),

    'out': tf.Variable(tf.zeros([n_classes]))
}


def noise_l1_autoencoder(x):
    return tf.nn.sigmoid(tf.add(tf.matmul(x, weights['l1_h1']), bias['l1_b1']))


def noise_l1_autodecoder(layer_1, keep_prob):
    l1 = tf.nn.dropout(layer_1, keep_prob)
    l1_2 = tf.nn.sigmoid(tf.add(tf.matmul(l1, weights['l1_h2']), bias['l1_b2']))
    l1_2_out = tf.nn.dropout(l1_2, keep_prob)
    return tf.nn.sigmoid(tf.add(tf.matmul(l1_2_out, weights['l1_out']), bias['l1_out']))


l1_out_r = noise_l1_autoencoder(l1_x)
l1_re = noise_l1_autodecoder(l1_out_r, keep_prob)
loss_function_l1 = tf.reduce_mean(tf.pow(l1_re - l1_y, 2))
optimizer_l1 = tf.train.AdamOptimizer(0.01).minimize(loss_function_l1)


def noise_l2_autoencoder(x):
    return tf.nn.sigmoid(tf.add(tf.matmul(x, weights['l2_h1']), bias['l2_b1']))


def noise_l2_autodecoder(layer_2):
    l2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['l2_h2']), bias['l2_b2']))
    return tf.nn.sigmoid(tf.add(tf.matmul(l2, weights['l2_out']), bias['l2_out']))

# 使用了3个损失函数. 分别是两个自动编码机的损失MSE，级联以后的整个MLP的分类损失cross entropy

l2_out_r = noise_l2_autoencoder(l2_x)
l2_re = noise_l2_autodecoder(l2_out_r)
loss_function_l2 = tf.reduce_mean(tf.pow(l2_re - l2_y, 2))
optimizer_l2 = tf.train.AdamOptimizer(0.01).minimize(loss_function_l2)

l3_out = tf.add(tf.matmul(l3_x, weights['out']), bias['out'])
loss_function_l3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=l3_out, labels=l3_y))
optimizer_l3 = tf.train.AdamOptimizer(0.01).minimize(loss_function_l3)

l1_l2out = tf.nn.sigmoid(tf.add(tf.matmul(l1_out_r, weights['l2_h1']), bias['l2_b1']))
out = tf.matmul(l1_l2out, weights['out']) + bias['out']

loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=l3_y))
# 降低学习率做微调
optimizer = tf.train.AdamOptimizer(0.001).minimize(loss_function)

# 训练第一个自动编码机

epochs = 50
batch_size = 128
total_batch = int(mnist.train.num_examples / batch_size)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in range(epochs):
    cost = 0.
    for i in range(total_batch):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x_noise = batch_x + 0.3 * np.random.randn(batch_size, 784)
        _, loss_l1 = sess.run([optimizer_l1, loss_function_l1], feed_dict = {l1_x : batch_x_noise, l1_y : batch_x, keep_prob : 0.5})
        cost += loss_l1
    if (epoch + 1) % 10 == 0:
        print("Epoch %02d/%02d average cost : %.6f" %(epoch + 1, epochs, cost / total_batch))
print("finished!")
test_noisy = mnist.test.images[:10] + 0.3 * np.random.randn(10, 784)
l1_rec = sess.run(l1_re, feed_dict = {l1_x : test_noisy, keep_prob : 1.0})
f, a = plt.subplots(3, 10, figsize = (10, 3))
for i in range(10):
    a[0][i].imshow(np.reshape(test_noisy[i], (28, 28)))
    a[1][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    a[2][i].matshow(np.reshape(l1_rec[i], (28, 28)), cmap = plt.get_cmap('gray'))
plt.show()

# 训练第二个自动编码机

for epoch in range(epochs):
    cost = 0.
    for i in range(total_batch):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        l1_encoder = sess.run(l1_out_r, feed_dict = {l1_x : batch_x, keep_prob : 1.})
        # l1_encoder = l1_encoder + 0.3 * np.random.randn(batch_size, 256)
        _, loss_l2 = sess.run([optimizer_l2, loss_function_l2], feed_dict = {l2_x : l1_encoder, l2_y : l1_encoder, keep_prob : 1.})
        cost += loss_l2
    if (epoch + 1) % 10 == 0:
        print("Epoch %02d/%02d average cost : %.6f" %(epoch + 1, epochs, cost / total_batch))
print("finished!")
test_vec = mnist.test.images[:10]
l1_encoder = sess.run(l1_out_r, feed_dict = {l1_x : test_vec, keep_prob : 1.0})
l2_rec = sess.run(l2_re, feed_dict = {l2_x : l1_encoder, keep_prob : 1.0})
f, a = plt.subplots(3, 10, figsize = (10, 3))
for i in range(10):
    a[0][i].imshow(np.reshape(l1_encoder[i], (16, 16)))
    a[1][i].imshow(np.reshape(l2_rec[i], (16, 16)))
plt.show()

# 训练分类器

for epoch in range(epochs):
    cost = 0.
    for i in range(total_batch):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        l1_encoder = sess.run(l1_out_r, feed_dict = {l1_x : batch_x, keep_prob : 1.})
        l2_encoder = sess.run(l2_out_r, feed_dict = {l2_x : l1_encoder, keep_prob : 1.})
        _, loss_l3 = sess.run([optimizer_l3, loss_function_l3], feed_dict = {l3_x : l2_encoder, l3_y : batch_y})
        cost += loss_l3
    if (epoch + 1) % 10 == 0:
        print("Epoch %02d/%02d average cost : %.6f" %(epoch + 1, epochs, cost / total_batch))
print("finished!")

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(l3_y, 1)), 'float'))
acc = sess.run(accuracy, feed_dict = {l1_x : mnist.test.images, l3_y : mnist.test.labels})
print("Accuracy : ", acc)

# 级联微调

for epoch in range(epochs):
    cost = 0.
    for i in range(total_batch):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        _, loss = sess.run([optimizer, loss_function], feed_dict = {l1_x : batch_x, l3_y : batch_y})
        cost += loss
    if (epoch + 1) % 10 == 0:
        print("Epoch %02d/%02d average cost : %.6f" %(epoch + 1, epochs, cost /  total_batch))

acc = sess.run(accuracy, feed_dict = {l1_x : mnist.test.images, l3_y : mnist.test.labels})
print("Accuracy : ", acc)

sess.close()