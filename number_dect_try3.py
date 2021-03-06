import tensorflow as tf
import numpy as np
#from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
np.random.seed(1234)
#print(mnist.train.images.shape)

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

def next_batch(x, y):
    i = np.random.randint(59900)
    x_bat = np.zeros((100,784),dtype=np.float32)
    y_bat = np.zeros([100,10],dtype=np.float32)
    for k in range(100):
        x_bat[k] = x[(i * 100 + k) % 55000]
        y_bat[k] = y[(i * 100 + k) % 55000]
    # x_bat = x_bat.reshape(100, 784)
    #print('x_bat',x_bat,'y_bat',y_bat)
    return x_bat, y_bat

#define compute accuracy
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs:v_xs})
    #print(y_pre)
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result
#define placeholder
xs = tf.placeholder(tf.float32, [None, 784])     # input x
ys = tf.placeholder(tf.float32, [None, 10])     # input y
#add output layer
l1 = add_layer(xs, 784, 512, activation_function=tf.nn.softmax)
prediction = add_layer(l1, 512, 10, activation_function=tf.nn.softmax)
cross_entropy = -tf.reduce_sum(ys*tf.log(tf.clip_by_value(prediction, 1e-8, 1)),
                                                         reduction_indices=[1])
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(10000):
    batch_xs, batch_ys = next_batch(mnist.train.images, mnist.train.labels)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 500 == 0:
        print('accuracy =',compute_accuracy(
            mnist.test.images,mnist.test.labels))


#regularize shape of x, y
# x_test = x_1.reshape(10000,784)
# y_train = np.zeros([60000,10],dtype=np.float32)
# y_test = np.zeros([10000,10],dtype=np.float32)
# for index in range(60000):
#     y_train[index][y_[index]] = 1
#     if(index < 10000):
#         y_test[index][y_1[index]] = 1
#next batch
# def next_batch(x, y):
#     i = np.random.randint(59900)
#     x_bat = np.zeros((100,28,28),dtype=np.float32)
#     y_bat = np.zeros([100,10],dtype=np.float32)
#     for k in range(100):
#         x_bat[k] = x[(i * 100 + k) % 60000]
#         y_bat[k] = y[(i * 100 + k) % 60000]
#     x_bat = x_bat.reshape(100, 784)
#     #print('x_bat',x_bat,'y_bat',y_bat)
#     return x_bat, y_bat
