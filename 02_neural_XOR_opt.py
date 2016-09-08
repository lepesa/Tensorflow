# Sama kuin 01_neural.xor, paitsi eri syntaksi ja parempi SGD optimoija
import numpy as np
import tensorflow as tf

# Parametrit: 2-4-1 verkko
learning_rate = 0.01
n_hidden1 = 4 
n_input = 2 
n_ouput = 1 

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
layer_hidden_output = tf.add(tf.matmul(inputs, weights['h1']), biases['b1'])
layer_hidden_output = tf.nn.sigmoid(layer_hidden_output)

layer_output_output = tf.add(tf.matmul(layer_hidden_output, weights['out']), biases['out'])
##layer_output_output = tf.nn.sigmoid(layer_output_output)

# Cross entropy ja monipuolisempi metodi sen käyttöön. 
error_function = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(layer_output_output, desired_outputs))

# SGD Adam
train_step = tf.train.AdamOptimizer(learning_rate).minimize(error_function)

# XOR:n neljä eri tilaa opetusta varten
training_inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
training_outputs = [[0.0], [1.0], [1.0], [0.0]]

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for i in range(10000):
        _, error = sess.run([train_step, error_function],
                            feed_dict={inputs: np.array(training_inputs),
                                       desired_outputs: np.array(training_outputs)})
        if i%1000 == 0:
            print("Epoch ",i ,", Error: ", error)

    # Tulostetaan lopputulokset. Huomaa että layer_output_output:sta 
    # ei ole otettu sigmoida kuten 01-esimerkissä koska käytämme eri
    # error_functiota
    print("{0,0} = " ,sess.run(tf.nn.sigmoid(layer_output_output), feed_dict={inputs: np.array([[0.0, 0.0]])}))
    print("{0,1} = " ,sess.run(tf.nn.sigmoid(layer_output_output), feed_dict={inputs: np.array([[0.0, 1.0]])}))
    print("{1,0} = " ,sess.run(tf.nn.sigmoid(layer_output_output), feed_dict={inputs: np.array([[1.0, 0.0]])}))
    print("{1,1} = " ,sess.run(tf.nn.sigmoid(layer_output_output), feed_dict={inputs: np.array([[1.0, 1.0]])}))
