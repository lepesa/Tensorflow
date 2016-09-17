# Simppeli CNN MNIST aineiston tunnistamiseen
# 

import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 28*28 -> 24*24*3 (5x5) conv layer -> 12x12 (2x2) pooling layer -> 128 fully connected layer -> 10 output layer 

# MNIST:n data Googlen esimerkeistä
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
inputs = tf.placeholder("float", [None, 28*28])
desired_outputs = tf.placeholder("float", [None, 10])

W_conv1 = weight_variable([5, 5, 1, 3])
b_conv1 = bias_variable([3])

x_image = tf.reshape(inputs, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_fc1 = weight_variable([14 *14 * 3, 128])
b_fc1 = bias_variable([128])

h_pool1_flat = tf.reshape(h_pool1, [-1, 14*14*3])
h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([128, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(desired_outputs * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(desired_outputs,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i%1000 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                inputs:batch[0], desired_outputs: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
            
        train_step.run(feed_dict={inputs: batch[0], desired_outputs: batch[1], keep_prob: 0.5})

    print("test accuracy %g"%accuracy.eval(feed_dict={inputs: mnist.test.images, desired_outputs: mnist.test.labels, keep_prob: 1.0}))
