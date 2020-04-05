import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
np.random.seed(1234)
BATCH_SIZE = 100
LR = 0.001              # learning rate
test_x = mnist.test.images[:2000]
test_y = mnist.test.labels[:2000]

width = 28 # width of the image in pixels
height = 28 # height of the image in pixels
flat = width * height # number of pixels in one image
class_output = 10 # number of possible classifications for the problem

#define add_layer
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    baises = tf.Variable(tf.zeros([out_size])+ 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + baises
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

#define placeholder
tf_x = tf.placeholder(tf.float32, [None, 28*28]) / 255.
image = tf.reshape(tf_x, [-1, 28, 28, 1])              # (batch, height, width, channel)
tf_y = tf.placeholder(tf.int32, [None, 10])            # input y

#conv layer1
conv1 = tf.layers.conv2d(inputs=image, filters=16, kernel_size=1,
                         strides=1, padding='same', activation=tf.nn.relu)
                        # 28x28x16
pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2) #14x14x16
#conv layer2
conv2 = tf.layers.conv2d(pool1, 32, 5, 1, 'same', activation=tf.nn.relu) #14x14x32
pool2 = tf.layers.max_pooling2d(conv2, 2, 2) #7x7x32
flat = tf.reshape(pool2, [-1, 7*7*32]) #-> 1d
#layer1
#l1 = add_layer(flat, 7*7*32, 1024, activation_function=tf.nn.relu)
#layer2
# prediction = add_layer(l1, 1024, 10, activation_function=tf.nn.softmax)
l1 = tf.layers.dense(flat, 1024, activation=tf.nn.relu)
prediction = tf.layers.dense(l1, 10, activation=tf.nn.softmax)
#important parameters
loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=prediction)
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

accuracy = tf.metrics.accuracy(
        labels=tf.argmax(tf_y, axis=1),predictions=tf.argmax(prediction,axis=1),)[1]
with tf.Session() as sess:
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # the local var is for accuracy_op
    sess.run(init_op)
    for step in range(50000):
        b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
        _, loss_ = sess.run([train_op, loss], {tf_x: b_x, tf_y: b_y})
        if step % 500 == 0:
            accuracy_, flat_representation = sess.run([accuracy, flat], {tf_x: test_x, tf_y: test_y})
            print('Step:', step, '| train loss: %.4f' % loss_, '| test accuracy: %.4f' % accuracy_)
