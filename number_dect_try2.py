
#This is a negative example. It cannnot learn anything from the data, however, when I put the same code into tf 1.15.0,
#bang, miracle happens! It all works out and the accuracy is 0.93.
#If you got any clue why this piece wouldn't work out in Version 2.1.0, pls contact me at gutentagberry@qq.com. Appreciate a lot!
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
#from tensorflow.examples.tutorials.mnist import input_data
mnist = tf.keras.datasets.mnist
(x_,y_),(x_1,y_1) = mnist.load_data() # x_.shape = 60000 X 28 X 28, x_1.shape = 10000 X 28 X 28

#regularize shape of x, y
x_test = x_1.reshape(10000,784)
y_train = np.zeros([60000,10],dtype=np.float32)
y_test = np.zeros([10000,10],dtype=np.float32)
for index in range(60000):
    y_train[index][y_[index]] = 1
    if(index < 10000):
        y_test[index][y_1[index]] = 1
#next batch
def next_batch(x, y, i):
    x_bat = np.zeros((100,28,28),dtype=np.float32)
    y_bat = np.zeros([100,10],dtype=np.float32)
    for k in range(100):
        x_bat[k] = x[(i * 100 + k) % 60000]
        y_bat[k] = y[(i * 100 + k) % 60000]
    x_bat = x_bat.reshape(100, 784)
    #print('x_bat',x_bat,'y_bat',y_bat)
    return x_bat, y_bat
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

#define compute accuracy
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs:v_xs})
    print(y_pre)
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result
#define placeholder
xs = tf.placeholder(tf.float32, [None, 784])     # input x
ys = tf.placeholder(tf.float32, [None, 10])     # input y
#add output layer
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)
cross_entropy = -tf.reduce_sum(ys*tf.log(tf.clip_by_value(prediction, 1e-8, 1)),
                                                         reduction_indices=[1])
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs, batch_ys = next_batch(x_, y_train, i)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:
        print(compute_accuracy(
            x_test,y_test))

