import csv
from numpy import *

def toInt(array):
    array = mat(array)
    n,m = shape(array)
    newArray = zeros((n,m))
    for i in range(n):
        for j in range(m):
            newArray[i,j] = int(array[i,j])
    return newArray

def dealData(data):
    n,m = shape(data)
    for i in range(n):
        for j in range(m):
            data[i,j] = data[i,j] / 254
    return data

def dealLabel(label):
    n,m = shape(label)
    newArray = zeros((m,10))
    for i in range(m):
        a = int(label[0,i])
        newArray[i,a] = 1
    return newArray

def loadTrainData():
    TrainInput = []
    with open('train.csv') as file:
        lines = csv.reader(file)
        for line in lines:
            TrainInput.append(line)
    TrainInput.remove(TrainInput[0])
    TrainInput = array(TrainInput)
    label = toInt(TrainInput[:,0])
    data = toInt(TrainInput[:,1:])
    return dealData(data), dealLabel(label)

def loadTestData():
    TestInput = []
    with open('test.csv') as file:
        lines = csv.reader(file)
        for line in lines:
            TestInput.append(line)
    TestInput.remove(TestInput[0])
    TestInput = array(TestInput)
    data = toInt(TestInput)
    return dealData(data)

import tensorflow as tf

def output(v_xs):
    global prediction
    with open('result-cnn-dropout.csv','w', newline='') as myFile:
        myWriter=csv.writer(myFile)
        myWriter.writerow(['ImageId', 'Label'])
        for i in range(280):
            # print(i)
            batch_xs = v_xs[i*100 : (i+1)*100]
            v_ys = sess.run(prediction, feed_dict={xs: batch_xs, keep_prob: 1})
            argmax = tf.argmax(v_ys,1)
            result = sess.run(argmax, feed_dict={xs: batch_xs, keep_prob: 1})
            for j in range(100):
                myWriter.writerow([i*100+j+1, result[j]])

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

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

xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

x_image = tf.reshape(xs,[-1,28,28,1])

W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

h_pool2_flat = tf.reshape(h_pool2,[-1, 7*7*64])
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

W_fc2=weight_variable([1024, 10])
b_fc2=bias_variable([10])
# prediction=tf.nn.softmax(tf.matmul(h_fc1,W_fc2) + b_fc2)
prediction=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

train_xs, train_ys = loadTrainData()
# print(train_xs[0])
# print(train_ys[0])
# print(shape(train_xs))
# print(shape(train_ys))
# print(shape(train_xs[0]))
# print(shape(train_ys[0]))

for o in range(3):
    for i in range(420):
        batch_xs = train_xs[i*100 : (i+1)*100]
        batch_ys = train_ys[i*100 : (i+1)*100]
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})

    # print(shape(batch_xs))
    # print(shape(batch_ys))
    print('%.6f%%' % (compute_accuracy(train_xs[:100], train_ys[:100]) * 100))

# output(loadTestData())
