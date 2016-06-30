'''
    Reference:
        TensorFlow tutorials
        Stanford CS231n
        Aymeric Damien TF example 
'''

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.001
training_iters = 200000
batch_size = 100
display_step = 20
img_vec_size = 784  # MNIST img 28*28
num_class = 10      # MNIST classes 0-9
#dropout = 0.75

x = tf.placeholder(tf.float32, [None, img_vec_size])
y_ = tf.placeholder(tf.float32, [None, num_class])
#   Dropout to :    prevent overfit
#                   "Ensemble Learning"
#                   A bit faster fp
keep_prob = tf.placeholder(tf.float32)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# Keep elongate the "depth" of convolution "block" by conv+relu 
# Shrink the surface area by pooling
W_dict = {
    'W_conv1': weight_variable([5, 5, 1, 32]),      # 5x5 size filter, 1 channel, 32 depth
    'W_conv2': weight_variable([5, 5, 32, 64]),     # 5x5 size filter, 32 , 64 depth
    'W_fc1': weight_variable([7 * 7 * 64, 1024]),   # fc layer, 'vectorize' 7*7*64 inputs, 1024 outputs
    'W_fc2': weight_variable([1024, num_class])     # 1024 inputs, 10 output classes
}

b_dict = {
    'b_conv1': bias_variable([32]), # match the depth of convolution "cube"
    'b_conv2': bias_variable([64]),
    'b_fc1': bias_variable([1024]),
    'b_fc2': bias_variable([num_class])
}

# Merge matmul+bias then relu 2 step in tutorial into one convolution step
def convolution(x, W, b):
    return tf.nn.relu(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') + b)

# Pooling block size n*n
def max_pooling(x, n):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='SAME')

def cnn(X, weight, bias, dropout):
    # Reshape input igm to 4-D
    X = tf.reshape(X, shape=[-1, 28, 28, 1])

    # Convolution Layer 1
    conv1 = convolution(X, weight['W_conv1'], bias['b_conv1'])
    # Max Pooling
    conv1 = max_pooling(conv1, n=2)
    # Dropout
    # conv1_drop = tf.nn.dropout(conv1, dropout)

    # Convolution Layer 2
    conv2 = convolution(conv1, weight['W_conv2'], bias['b_conv2'])
    # Max Pooling
    conv2 = max_pooling(conv2, n=2)
    # Dropout
    # conv2_drop = tf.nn.dropout(conv2, dropout)

    # Fully Connected Layer 1
    conv2flat = tf.reshape(conv2, [-1, weight['W_fc1'].get_shape().as_list()[0]]) 
    # Reshape col of conv2flat to row of W_fc1 same as: tf.reshape(conv2_drop, [-1, 7*7*64])
    fc1 = tf.nn.relu(tf.matmul(conv2flat, weight['W_fc1']) + bias['b_fc1']) # Relu activation
    fc1_drop = tf.nn.dropout(fc1, dropout)

    output = tf.nn.softmax(tf.matmul(fc1_drop, weight['W_fc2']) + bias['b_fc2'])
    return output

# Build CNN Graph
y_conv = cnn(x, W_dict, b_dict, keep_prob)

# Cost
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Eval
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

#Train and Eval in session
with tf.Session() as sess:

    sess.run(tf.initialize_all_variables())

    for i in range(20000):
      batch = mnist.train.next_batch(128) # power of 2 for parellel processing ?
      if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print "step %d, training accuracy %g"%(i, train_accuracy)
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.75}) #should be close to 0.5 for more layers

    print "test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})

