NAME = "cnn"
mode = "TEST"

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

def dealLabel(label):
    n,m = shape(label)
    newArray = zeros((m,10))
    for i in range(m):
        a = int(label[0,i])
        newArray[i,a] = 1
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
    return dealData(data), dealLabel(label)

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

def output(v_xs):
    global prediction, NAME
    n,m = shape(v_xs)
    with open("result.csv",'w', newline='') as myFile:
        myWriter=csv.writer(myFile)
        myWriter.writerow(['ImageId', 'Label'])
        for i in range(280):
            batch_xs = v_xs[i*100 : (i+1)*100]
            v_ys = sess.run(prediction, feed_dict={xs: batch_xs, keep_prob: 1, keep_prob_conv: 1})
            _result = tf.argmax(v_ys, 1)
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
    ys = tf.placeholder(tf.float32, [None, 10], name='y_input')
    x_image = tf.reshape(xs,[-1,28,28,1], name='image')
    keep_prob = tf.placeholder(tf.float32, name='dropout')
    keep_prob_conv = tf.placeholder(tf.float32, name='dropout_conv')

with tf.name_scope('convolutional_layer1'):
    with tf.name_scope('weights'):
        W_conv1 = weight_variable([4,4,1,64])
    with tf.name_scope('biases'):
        b_conv1 = bias_variable([64])
    with tf.name_scope('conv'):
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    with tf.name_scope('max_pool'):
        h_pool1 = max_pool_2x2(h_conv1)
    with tf.name_scope('dropout'):
        h_pool1_drop = tf.nn.dropout(h_pool1, keep_prob_conv)

with tf.name_scope('convolutional_layer2'):
    with tf.name_scope('weights'):
        W_conv2 = weight_variable([4,4,64,128])
    with tf.name_scope('biases'):
        b_conv2 = bias_variable([128])
    with tf.name_scope('conv'):
        h_conv2 = tf.nn.relu(conv2d(h_pool1_drop, W_conv2) + b_conv2)
        # h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    with tf.name_scope('max_pool'):
        h_pool2 = max_pool_2x2(h_conv2)
    with tf.name_scope('dropout'):
        h_pool2_drop = tf.nn.dropout(h_pool2, keep_prob_conv)

with tf.name_scope('convolutional_layer3'):
    with tf.name_scope('weights'):
        W_conv3 = weight_variable([4,4,128,256])
    with tf.name_scope('biases'):
        b_conv3 = bias_variable([256])
    with tf.name_scope('conv'):
        h_conv3 = tf.nn.relu(conv2d(h_pool2_drop, W_conv3) + b_conv3)
        # h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    with tf.name_scope('max_pool'):
        h_pool3 = max_pool_2x2(h_conv3)
    with tf.name_scope('dropout'):
        h_pool3_drop = tf.nn.dropout(h_pool3, keep_prob_conv)

with tf.name_scope('fully_connected_layer1'):
    with tf.name_scope('flat'):
        h_pool2_flat = tf.reshape(h_pool3_drop,[-1, 4*4*256])
        # h_pool2_flat = tf.reshape(h_pool3,[-1, 4*4*256])
    with tf.name_scope('weights'):
        W_fc1 = weight_variable([4*4*256, 1024])
    with tf.name_scope('biases'):
        b_fc1 = bias_variable([1024])
    with tf.name_scope('Wx_plus_b'):
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    with tf.name_scope('dropout'):
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.name_scope('fully_connected_layer2'):
    with tf.name_scope('weights'):
        W_fc2 = weight_variable([1024, 10])
    with tf.name_scope('biases'):
        b_fc2 = bias_variable([10])
    with tf.name_scope('Wx_plus_b'):
        h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

with tf.name_scope('prediction'):
    prediction = tf.nn.softmax(h_fc2)
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction,1), tf.argmax(ys,1)), tf.float32))
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=h_fc2, labels=ys))
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()

if mode == 'TRAIN':

	init = tf.global_variables_initializer()
	sess.run(init)

	saver = tf.train.Saver(max_to_keep=5)
	min_cross = 1000000.0

	train_xs, train_ys = loadTrainData()
	times = 0
	rs = sess.run(merged, feed_dict={xs: train_xs[41000:42000], ys: train_ys[41000:42000], keep_prob: 1, keep_prob_conv: 1})
	writer.add_summary(rs, times)

	for o in range(100):
	    print(o)
	    for i in range(410):
	        batch_xs = train_xs[i*100 : (i+1)*100]
	        batch_ys = train_ys[i*100 : (i+1)*100]
	        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5, keep_prob_conv: 0.75})
	        times = times + 1;

	        if (times % 50 == 0):
	            rs = sess.run(merged, feed_dict={xs: train_xs[41000:42000], ys: train_ys[41000:42000], keep_prob: 1, keep_prob_conv: 1})
	            writer.add_summary(rs, times)

	    val_cross = sess.run(cross_entropy, feed_dict={xs: train_xs[41000:42000], ys: train_ys[41000:42000], keep_prob: 1, keep_prob_conv: 1})
	    if val_cross < min_cross:
	        min_cross = val_cross
	        saver.save(sess, "net/"+NAME+".ckpt", global_step = o)
	        print('Save! ', min_cross)

	    if ((o+1) % 10 == 0):
	        time.sleep(300)

if mode == 'TEST':

	saver = tf.train.Saver()
	saver.restore(sess, "net/"+NAME+".ckpt")

	output(loadTestData())
