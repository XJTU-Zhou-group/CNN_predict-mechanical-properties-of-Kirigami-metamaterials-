import numpy as np
import pandas as pd
import tensorflow as tf
import random
import os
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
n_train = 700
n_text = 1000-n_train
batch_size = 64
train_step = 100000
step_display = 1000
learning_reat_base = 0.001
learning_reat_step = 100
learning_reat_decay = 0.99
l2reat = 0.04
n1 = 32
n2 = 64
n3 = 32
kp = 1.0
ac_E = 1000.0
ac_v = 100.0
ac_mixtic = np.array([[ac_E, 0], [0, ac_v]])
mc_mixtic = np.array([[1/ac_E, 0], [0, 1/ac_v]])
file_angle = 'D:\\E_v\\3-3\\angle.csv'
file_E_v = 'D:\\E_v\\3-3\\E_v.csv'
file_output_tr = 'D:\\E_v\\3-3\\E_v_output_train_CNN.csv'
file_output_te = 'D:\\E_v\\3-3\\E_v_output_text_CNN.csv'
file_err_E_mean = 'D:\\E_v\\3-3\\err_E_mean_CNN.csv'
file_err_v_mean = 'D:\\E_v\\3-3\\err_v_mean_CNN.csv'
file_save = 'D:\\E_v\\3-3\\model_CNN.ckpt'
angle = pd.read_csv(file_angle, header=None).values
print(angle)
E_v_o = pd.read_csv(file_E_v, header=None).values
E_v = np.matmul(E_v_o, ac_mixtic)
print(E_v)
row, col = angle.shape
row2, col2 = E_v.shape

angle_train = angle[0: n_train]
angle_text = angle[n_train: 1000]
E_v_train = E_v[0: n_train]
E_v_text = E_v[n_train: 1000]

input = tf.placeholder(tf.float32)
output = tf.placeholder(tf.float32)

n_input = int(col)
n_output = int(col2)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):

    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):

    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

keep_prob = tf.placeholder(tf.float32)
input_matrix = tf.reshape(input, [-1, 6, 6, 1])

## conv1 layer ##
W_conv1 = weight_variable([2, 2, 1, 16])
b_conv1 = bias_variable([16])
h_conv1 = tf.nn.relu(conv2d(input_matrix, W_conv1) + b_conv1)
# h_pool1 = max_pool_2x2(h_conv1)

## conv2 layer ##
W_conv2 = weight_variable([2,2, 16, 32])
b_conv2 = bias_variable([32])
h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
# h_pool2 = max_pool_2x2(h_conv2)

## fc1 layer ##
W_fc1 = weight_variable([6*6*32, 512])
b_fc1 = bias_variable([512])
h_conv2_flat = tf.reshape(h_conv2, [-1, 6*6*32])
h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
W_fc2 = weight_variable([512, 64])
b_fc2 = bias_variable([64])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

## fc3 layer ##
W_fc3 = weight_variable([64, 2])
b_fc3 = bias_variable([2])
output_predict = tf.matmul(h_fc2_drop, W_fc3) + b_fc3

loss = tf.reduce_mean(tf.square(output_predict - output))
loss_l2 = tf.contrib.layers.l2_regularizer(l2reat)(W_fc1) + tf.contrib.layers.l2_regularizer(l2reat)(W_fc2) + \
tf.contrib.layers.l2_regularizer(l2reat)(W_fc3)
totalloss = loss + loss_l2
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(learning_reat_base, global_step, learning_reat_step, learning_reat_decay)
train_op = tf.train.AdamOptimizer(learning_rate).minimize(totalloss, global_step)

saver = tf.train.Saver()
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

err_E_mean = np.zeros(shape = (train_step, 1))
err_v_mean = np.zeros(shape = (train_step, 1))

with tf.Session(config=tf_config) as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    for step in range(train_step):
        index = random.sample(range(0, n_train), batch_size)
        sess.run(train_op, feed_dict={input: angle_train[index], output: E_v_train[index], keep_prob: kp})
        E_v_output_text = sess.run(output_predict, feed_dict={input: angle_text, keep_prob: 1.0})
        E_v_output_test = np.matmul(E_v_output_text, mc_mixtic)
        err1 = abs(E_v_output_test - E_v_o[n_train: 1000])
        err_E_mean1 = np.mean(err1[:, 0] / abs(E_v_o[n_train: 1000, 0]))
        err_v_mean1 = np.mean(err1[:, 1] / abs(E_v_o[n_train: 1000, 1]))
        err_E_mean[step] = err_E_mean1*100
        err_v_mean[step] = err_v_mean1*100
        if step % step_display == 0:
            loss_tr = sess.run(loss, feed_dict={input: angle_train, output: E_v_train, keep_prob: 1.0})
            loss_te = sess.run(loss, feed_dict={input: angle_text, output: E_v_text, keep_prob: 1.0})
            lossl2 = sess.run(loss_l2)
            print('step: ' + str(step))
            print('loss_tr: ' + str(loss_tr) + '  loss_te: ' + str(loss_te))
            print('loss_l2: ' + str(lossl2))
    loss_tr = sess.run(loss, feed_dict={input: angle_train, output: E_v_train, keep_prob: 1.0})
    loss_te = sess.run(loss, feed_dict={input: angle_text, output: E_v_text, keep_prob: 1.0})
    print('step: ' + str(step))
    print('loss_tr: ' + str(loss_tr) + '  loss_te: ' + str(loss_te))
    E_v_output_train = sess.run(output_predict, feed_dict={input: angle_train, keep_prob: 1.0})
    E_v_output_train = np.matmul(E_v_output_train, mc_mixtic)
    E_v_output_text = sess.run(output_predict, feed_dict={input: angle_text, keep_prob: 1.0})
    E_v_output_text = np.matmul(E_v_output_text, mc_mixtic)
    np.savetxt(file_output_tr, E_v_output_train)
    np.savetxt(file_output_te, E_v_output_text)
    np.savetxt(file_err_E_mean, err_E_mean)
    np.savetxt(file_err_v_mean, err_v_mean)
    saver.save(sess, file_save)