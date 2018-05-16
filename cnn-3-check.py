NAME = "cnn-3-"

import csv
from numpy import *
import tensorflow as tf
import time
import os

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

def output(v_xs):
    global prediction, NAME
    n,m = shape(v_xs)
    with open("result/"+NAME+".csv",'w', newline='') as myFile:
        myWriter=csv.writer(myFile)
        myWriter.writerow(['ImageId', 'Label'])
        for i in range(280):
            # print(i)
            batch_xs = v_xs[i*100 : (i+1)*100]
            v_ys = sess.run(prediction, feed_dict={xs: batch_xs, keep_prob: 1, keep_prob_conv: 1})
            _result = tf.argmax(v_ys,1)
            result = sess.run(_result, feed_dict={xs: batch_xs, keep_prob: 1, keep_prob_conv: 1})
            for j in range(100):
                myWriter.writerow([i*100+j+1, result[j]])

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
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(accuracy)

sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/"+NAME+"/", sess.graph)
init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=10)

# module_file = tf.train.latest_checkpoint("net")
# print(module_file)
# saver.restore(sess, module_file)

print("START READ TRAINING DATA")
train_xs, train_ys_all = loadTrainData()
print("END")
print(shape(train_xs))
print(shape(train_ys_all))

def paint(pic):
    pic = reshape(pic, [28,28])
    bw = []
    for i in range(28):
        for j in range(28):
            if (pic[i][j] <= 0):
                bw.append(" ")
            elif (pic[i][j] <= 0.2):
                bw.append("*")
            elif (pic[i][j] <= 0.4):
                bw.append("&")
            elif (pic[i][j] <= 0.6):
                bw.append("%")
            elif (pic[i][j] <= 0.8):
                bw.append("$")
            else:
                bw.append("#")
        bw.append("\n")
    print(''.join(bw))

result = tf.abs(tf.reshape(prediction, [-1]) - ys)

for NUMBER in range(0, 10):
    print(NUMBER)
    saver.restore(sess, "net/"+NAME+str(NUMBER)+".ckpt")
    train_ys = dealLabel(train_ys_all, NUMBER)
    total = 0
    for rd in range(42):
        rs = reshape(sess.run(result, feed_dict={xs: train_xs[rd*1000:(rd+1)*1000], ys: train_ys[rd*1000:(rd+1)*1000], keep_prob: 1, keep_prob_conv: 1}), -1)
        for i in range(1000):
            if rs[i] > 0.1:
                total += 1
                # print(rs[i], train_ys_all[0][rd*1000+i])
                # paint(train_xs[rd*1000+i])
                # os.system("pause")
    print(NUMBER, total)
