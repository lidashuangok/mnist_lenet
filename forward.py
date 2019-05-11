# !usr/bin/python
# Author:das
# -*-coding: utf-8 -*-
import tensorflow as tf
import numpy as np
IMAGE_WIDTH = 28
IMAGE_HIGH =28
IMMAGE_CHANNEL =1
CONV1_SIZE =3
CONV1_DEEP =6
CONV2_SIZE =3
CONV2_DEEP =16
FC1_NODES =120
FC2_NODES = 84
OUT_NODES= 10

def get_wight(shape,regularizer):
    w = tf.Variable(tf.random_normal(shape=shape, mean=0.0, stddev=1.0, dtype=tf.float32))
    if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    b = tf.random_normal(shape=shape, mean=0.0, stddev=1.0, dtype=tf.float32)
    return tf.Variable(b)

def conv2d(x,w):
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding="SAME")

def max_pooling_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def forward(x,train,regularizer):
    conv1_w=get_wight([CONV1_SIZE,CONV1_SIZE,IMMAGE_CHANNEL,CONV1_DEEP],regularizer)
    conv1_b=get_bias([CONV1_DEEP])
    conv1=conv2d(x,conv1_w)
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_b))
    max_pool1 = max_pooling_2x2(relu1)
    conv2_w = get_wight([CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP], regularizer)
    conv2_b = get_bias([CONV2_DEEP])
    conv2 = conv2d(max_pool1, conv2_w)
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    max_pool2 = max_pooling_2x2(relu2)

    pool_shape = max_pool2.get_shape().as_list()
    print(pool_shape)
    nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
    reshaped = tf.reshape(max_pool2, [pool_shape[0], nodes])

    fc1_w =get_wight([nodes,FC1_NODES],regularizer)
    fc1_b = get_bias([FC1_NODES])
    fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_w)+fc1_b)
    if train: fc1 = tf.nn.dropout(fc1, 0.5)
    fc2_w = get_wight([FC1_NODES, FC2_NODES], regularizer)
    fc2_b = get_bias([FC2_NODES])
    fc2 = tf.nn.relu(tf.matmul(fc1, fc2_w) + fc2_b)
    fc3_w = get_wight([FC2_NODES, OUT_NODES], regularizer)
    fc3_b = get_bias([OUT_NODES])
    y = tf.matmul(fc2, fc3_w) + fc3_b
    return y