# !usr/bin/python
# Author:das
# -*-coding: utf-8 -*-
import tensorflow as tf
import forward
import os
import tensorflow.examples.tutorials.mnist
from tensorflow.examples.tutorials.mnist import input_data
MODEL_NAME="mnist_model"
import numpy as np
from tensorflow.examples.tutorials.mnist import mnist
LEARNING_RATE_BASE =0.1
LEARNING_RATE_DECAY =0.99
BATCH_SIZE =100
STEPS=50000
MOVING_AVERAGE_DECAY = 0.99
REGULARIZER = 0.0001
MODEL_SAVE_PATH='./model/'
#DECAY_STEP=
def backward(mnist):
    x= tf.placeholder(dtype=tf.float32,shape=[BATCH_SIZE,forward.IMAGE_WIDTH,forward.IMAGE_HIGH,forward.IMMAGE_CHANNEL])
    y_ = tf.placeholder(dtype=tf.float32, shape=[None, 10])
    y =forward.forward(x,True,REGULARIZER)
    global_step = tf.Variable(0,trainable=False)
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step=global_step,decay_steps=60000/BATCH_SIZE,decay_rate=LEARNING_RATE_DECAY,staircase=True)

    # loss = tf.losses.mean_squared_error(labels=y_,predictions=y)
    # loss = loss + tf.add_n(tf.get_collection('losses'))
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))

    train = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)

    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train, ema_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()

    with tf.Session() as sess:
        init= tf.global_variables_initializer()
        sess.run(init)

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(STEPS):
                X,Y_= mnist.train.next_batch(batch_size=BATCH_SIZE)
                X_reshaped = np.reshape(X,(BATCH_SIZE,forward.IMAGE_WIDTH,forward.IMAGE_HIGH,forward.IMMAGE_CHANNEL))
                _,_,loss_v,steps_v=sess.run((train_op,train, loss,global_step), feed_dict={x: X_reshaped , y_: Y_})
                if i%500==0:
                    print("After %d training step(s), loss on training batch is %g." % (steps_v, loss_v))
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
if __name__ == '__main__':
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    backward(mnist)