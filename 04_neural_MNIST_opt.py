# MNIST numeroiden tunnistusta. Käytetään 784-800-10 neuroverkkoa + cross entropy. Tällä pitäisi pääsä ~2% virheesseen

import tensorflow as tf
import time

# MNIST:n data Googlen esimerkeistä
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.001
training_epochs = 100
batch_size = 10
n_hidden1 = 800
n_input = 784 
n_ouput = 10

inputs = tf.placeholder("float", [None, n_input])
desired_outputs = tf.placeholder("float", [None, n_ouput])

# Verkon painot ja biasit
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden1])),
    'out': tf.Variable(tf.random_normal([n_hidden1, n_ouput]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden1])),
    'out': tf.Variable(tf.random_normal([n_ouput]))
}

# Tehdään layerit, käytetään sigmoidia 
layer_hidden_output = tf.nn.sigmoid(tf.add(tf.matmul(inputs, weights['h1']), biases['b1']))
layer_output_output = tf.add(tf.matmul(layer_hidden_output, weights['out']), biases['out'])


error_function =  tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(layer_output_output, desired_outputs))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(error_function)
init = tf.initialize_all_variables()
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    start_time = time.time()

    correct_prediction = tf.equal(tf.argmax(layer_output_output, 1), tf.argmax(desired_outputs, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_step, error_function], feed_dict={inputs: batch_x,
                                                                     desired_outputs: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % 1 == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", 
                  "{:.9f}".format(avg_cost) , "seconds {:.2f}".format((time.time()-start_time)))
            print("Accuracy:", accuracy.eval({inputs: mnist.test.images, desired_outputs: mnist.test.labels}))

    print("Ready")
    print("Final accuracy:", accuracy.eval({inputs: mnist.test.images, desired_outputs: mnist.test.labels}))