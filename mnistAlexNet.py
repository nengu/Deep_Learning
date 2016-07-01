'''
    Reference:
        TensorFlow tutorials
        Stanford CS231n
        CSDN 
'''

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.001
training_iters = 200000
batch_size = 64
display_step = 20
img_vec_size = 784 #img 28*28
num_class = 10  
dropout = 0.8 # Tried smaller value but behaved very bad

# tf Graph input
x = tf.placeholder(tf.float32, [None, img_vec_size])
y = tf.placeholder(tf.float32, [None, num_class])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# Keep elongate the "depth" of convolution "block" by conv+relu 
# Shrink the surface area by pooling
# Alex Net W & b
W_dict = {
    'wc1': weight_variable([3, 3, 1, 64]),          # 3x3 size filter, 1 channel, 64 depth
    'wc2': weight_variable([3, 3, 64, 128]),        # 3x3 size filter, 64 , 128 depth
    'wc3': weight_variable([3, 3, 128, 256]),       # 3x3 size filter
    'wc4': weight_variable([2, 2, 256, 512]),       # 2x2 size filter try to avoid 2*2 in shallow net
    'wfc1': weight_variable([2 * 2 * 512, 1024]),   # fc layer1, 'vectorize' 2*2*512 inputs, 1024 outputs
    'wfc2': weight_variable([1024, 1024]),          # fc2 1024 inputs, 1024 output classes
    'wDest': weight_variable([1024, num_class])
}

b_dict = {
    'bc1': bias_variable([64]),
    'bc2': bias_variable([128]),
    'bc3': bias_variable([256]),
    'bc4': bias_variable([512]),
    'bfc1': bias_variable([1024]),
    'bfc2': bias_variable([1024]),
    'bDest': bias_variable([num_class])
}

# Merge matmul+bias then relu 2 step in tutorial into one convolution step
def convolution(name, x, W, b):
    return tf.nn.relu(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') + b, name=name)

# Pooling block size n*n
def max_pooling(name, x, n):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='SAME', name=name)

# Parameter referencing existing AlexNet implementation 
def norm(name, poolingRes, depth_radius=4):
    return tf.nn.lrn(poolingRes, depth_radius, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)
    # Local Response Normalization with depth_radius (divide by sum)

def alexNet(X, weight, bias, dropout):
    # Reshape input igm to 4-D
    X = tf.reshape(X, shape=[-1, 28, 28, 1])

    # Convolution Layer 1
    conv1 = convolution('conv1', X, weight['wc1'], bias['bc1'])
    # Max Pooling
    pool1 = max_pooling('pool1', conv1, n=2)
    # Normalization
    norm1 = norm('norm1', pool1)
    # Dropout
    norm1 = tf.nn.dropout(norm1, dropout)

    # Convolution Layer 2
    conv2 = convolution('conv2', norm1, weight['wc2'], bias['bc2'])
    # Max Pooling
    pool2 = max_pooling('pool2', conv2, n=2)
    # Normalization
    norm2 = norm('norm2', pool2)
    # Dropout
    norm2 = tf.nn.dropout(norm2, dropout)

    # Convolution Layer 3
    conv3 = convolution('conv3', norm2, weight['wc3'], bias['bc3'])
    # Max Pooling
    pool3 = max_pooling('pool3', conv3, n=2)
    # Normalization
    norm3 = norm('norm3', pool3)
    # Dropout
    norm3 = tf.nn.dropout(norm3, dropout)

    # Convolution Layer 4
    conv4 = convolution('conv4', norm3, weight['wc4'], bias['bc4'])
    # Max Pooling
    pool4 = max_pooling('pool4', conv4, n=2)
    # Normalization
    norm4 = norm('norm4', pool4)
    # Dropout
    norm4 = tf.nn.dropout(norm4, dropout)

    # Memory Peak here
    # Fully Connected Layer 1
    fc1 = tf.reshape(norm4, [-1, weight['wfc1'].get_shape().as_list()[0]]) 
    # Reshape column of conv4(norm4) to wfc1 row number for them to connect
    fc1 = tf.nn.relu(tf.matmul(fc1, weight['wfc1']) + bias['bfc1'], name='fc1') 

    # Fully Connected Layer 2
    fc2 = tf.nn.relu(tf.matmul(fc1, weight['wfc2']) + bias['bfc2'], name='fc2') 

    # Output
    output = tf.matmul(fc2, weight['wDest']) + bias['bDest']

    return output

# GREAT REFERENCE why use softmax_cross_entropy_with_logits
# http://stackoverflow.com/questions/34240703/difference-between-tensorflow-tf-nn-softmax-and-tf-nn-softmax-cross-entropy-with

# Build AlexNet in Graph
y_alexnet = alexNet(x, W_dict, b_dict, keep_prob)

# Cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_alexnet, y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

# Eval
correct_prediction = tf.equal(tf.argmax(y_alexnet,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

#Train and Eval in session
with tf.Session() as sess:

    sess.run(init)
    epoch = 1
    # Keep training until reach max iterations
    while epoch * batch_size < training_iters:
        #batch = mnist.train.next_batch(64)
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # Train
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
        if epoch % display_step == 0:
            # batch accuracy
            train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
            print "Iter " + str(epoch*batch_size) + ", Batch Training Accuracy= " + "{:.5f}".format(train_accuracy)
        epoch += 1

    print "Final Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.})
