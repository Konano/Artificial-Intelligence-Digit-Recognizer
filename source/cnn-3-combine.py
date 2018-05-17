NAME = "cnn-3-"

import csv
from numpy import *
import tensorflow as tf
import time

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

def dealLabel(label, NUMBER):
    n,m = shape(label)
    newArray = zeros((m))
    for i in range(m):
        if (label[0,i] == NUMBER):
            newArray[i] = 1
        else:
            newArray[i] = 0
    return newArray

def loadTrainData():
    TrainInput = []
    with open('data/train.csv') as file:
        lines = csv.reader(file)
        for line in lines:
            TrainInput.append(line)
    TrainInput.remove(TrainInput[0])
    TrainInput = array(TrainInput)
    label = toInt(TrainInput[:,0])
    data = toInt(TrainInput[:,1:])
    return dealData(data), label

def loadTestData():
    TestInput = []
    with open('data/test.csv') as file:
        lines = csv.reader(file)
        for line in lines:
            TestInput.append(line)
    TestInput.remove(TestInput[0])
    TestInput = array(TestInput)
    data = toInt(TestInput)
    return dealData(data)

def output(result):
    global NAME
    print(len(result))
    n = len(result)
    with open("result/"+NAME+".csv",'w', newline='') as myFile:
        myWriter=csv.writer(myFile)
        myWriter.writerow(['ImageId', 'Label'])
        for i in range(n):
            myWriter.writerow([i+1, result[i]])

right_label = -1
wrong_label = -1

def next_batch(num):
    global train_xs, train_ys, right_label, wrong_label
    n,m = shape(train_xs)
    new_xs = zeros((num*2,m))
    new_ys = zeros((num*2))
    for i in range(num):
        right_label += 1
        while train_ys[right_label % n] != 1:
            right_label += 1
        right_label %= n
        wrong_label += 1
        while train_ys[wrong_label % n] != 0:
            wrong_label += 1
        wrong_label %= n
        new_xs[i*2] = train_xs[right_label]
        new_ys[i*2] = train_ys[right_label]
        new_xs[i*2+1] = train_xs[wrong_label]
        new_ys[i*2+1] = train_ys[wrong_label]
    return new_xs, new_ys

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name='w')

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='b')

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 784], name='x_input')
    ys = tf.placeholder(tf.float32, [None], name='y_input')
    x_image = tf.reshape(xs,[-1,28,28,1], name='image')
    keep_prob = tf.placeholder(tf.float32, name='dropout')
    keep_prob_conv = tf.placeholder(tf.float32, name='dropout_conv')

with tf.name_scope('convolutional_layer1'):
    with tf.name_scope('weights'):
        W_conv1 = weight_variable([4,4,1,32])
    with tf.name_scope('biases'):
        b_conv1 = bias_variable([32])
    with tf.name_scope('conv'):
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    with tf.name_scope('max_pool'):
        h_pool1 = max_pool_2x2(h_conv1)
    with tf.name_scope('dropout'):
        h_pool1_drop = tf.nn.dropout(h_pool1, keep_prob_conv)

with tf.name_scope('convolutional_layer2'):
    with tf.name_scope('weights'):
        W_conv2 = weight_variable([4,4,32,64])
    with tf.name_scope('biases'):
        b_conv2 = bias_variable([64])
    with tf.name_scope('conv'):
        h_conv2 = tf.nn.relu(conv2d(h_pool1_drop, W_conv2) + b_conv2)
    with tf.name_scope('max_pool'):
        h_pool2 = max_pool_2x2(h_conv2)
    with tf.name_scope('dropout'):
        h_pool2_drop = tf.nn.dropout(h_pool2, keep_prob_conv)

with tf.name_scope('fully_connected_layer1'):
    with tf.name_scope('flat'):
        h_pool2_flat = tf.reshape(h_pool2_drop,[-1, 7*7*64])
    with tf.name_scope('weights'):
        W_fc1 = weight_variable([7*7*64, 128])
    with tf.name_scope('biases'):
        b_fc1 = bias_variable([128])
    with tf.name_scope('Wx_plus_b'):
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    with tf.name_scope('dropout'):
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.name_scope('fully_connected_layer2'):
    with tf.name_scope('weights'):
        W_fc2 = weight_variable([128, 1])
    with tf.name_scope('biases'):
        b_fc2 = bias_variable([1])
    with tf.name_scope('Wx_plus_b'):
        h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

with tf.name_scope('prediction'):
    prediction = tf.sigmoid(h_fc2)
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.abs(tf.reshape(prediction, [-1]) - ys))
    tf.summary.scalar('accuracy', accuracy)
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(accuracy)

sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/"+NAME+"/", sess.graph)
saver = tf.train.Saver()
_prediction = tf.reshape(prediction, [-1])

print("START READ TEST DATA")
train_xs = loadTestData()
print("END")

n,m = shape(train_xs)
result = zeros((10,n))

for NUMBER in range(0, 10):
    print(NUMBER)
    saver.restore(sess, "net/"+NAME+str(NUMBER)+".ckpt")
    for i in range(280):
        result[NUMBER][i*100:(i+1)*100] = sess.run(_prediction, feed_dict={xs: train_xs[i*100:(i+1)*100], keep_prob: 1, keep_prob_conv: 1})
    # for i in range(2):
    #     result[NUMBER][i*2:(i+1)*2] = sess.run(_prediction, feed_dict={xs: train_xs[i*2:(i+1)*2], keep_prob: 1, keep_prob_conv: 1})

output(reshape(argmax(result,0), -1))