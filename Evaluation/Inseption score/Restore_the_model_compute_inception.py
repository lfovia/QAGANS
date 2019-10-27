########### for any model, if you want to compute the inception score ######### just load the model 
##### load the model into correct generator and discriminator architecture ###############
##########################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import division, print_function, absolute_import
import cPickle as pickle
import os
import urllib
import numpy as np
import tarfile
import pickle
from tensorflow.python.platform import gfile
import tensorflow as tf
from tensordata.augmentation import random_flip
import adler.tensorflow as atf
import sys
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
slim = tf.contrib.slim
import numpy as np
import tensordata
import functools

# set seeds for reproducibility
np.random.seed(0)
tf.set_random_seed(0)
batch_size = 64
sess = tf.InteractiveSession()
size = 32

def apply_conv(x, filters=32, kernel_size=3, he_init=True):
    if he_init:
        initializer = tf.contrib.layers.variance_scaling_initializer(uniform=True)
    else:
        initializer = tf.contrib.layers.xavier_initializer(uniform=True)

    return tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size,
                            padding='SAME', kernel_initializer=initializer)


def activation(x):
    with tf.name_scope('activation'):
        return tf.nn.relu(x)


def bn(x):
    return tf.contrib.layers.batch_norm(x,
                                    decay=0.9,
                                    center=True,
                                    scale=True,
                                    epsilon=1e-5,
                                    zero_debias_moving_mean=True,
                                    is_training=False)


def stable_norm(x, ord):
    x = tf.contrib.layers.flatten(x)
    alpha = tf.reduce_max(tf.abs(x) + 1e-5, axis=1)
    result = alpha * tf.norm(x / alpha[:, None], ord=ord, axis=1)
    return result


def downsample(x):
    with tf.name_scope('downsample'):
        x = tf.identity(x)
        return tf.add_n([x[:,::2,::2,:], x[:,1::2,::2,:],
                         x[:,::2,1::2,:], x[:,1::2,1::2,:]]) / 4.

def upsample(x):
    with tf.name_scope('upsample'):
        x = tf.identity(x)
        x = tf.concat([x, x, x, x], axis=-1)
        return tf.depth_to_space(x, 2)


def conv_meanpool(x, **kwargs):
    return downsample(apply_conv(x, **kwargs))

def meanpool_conv(x, **kwargs):
    return apply_conv(downsample(x), **kwargs)

def upsample_conv(x, **kwargs):
    return apply_conv(upsample(x), **kwargs)

def resblock(x, filters, resample=None, normalize=False):
    if normalize:
        norm_fn = bn
    else:
        norm_fn = tf.identity

    if resample == 'down':
        conv_1 = functools.partial(apply_conv, filters=filters)
        conv_2 = functools.partial(conv_meanpool, filters=filters)
        conv_shortcut = functools.partial(conv_meanpool, filters=filters,
                                          kernel_size=1, he_init=False)
    elif resample == 'up':
        conv_1 = functools.partial(upsample_conv, filters=filters)
        conv_2 = functools.partial(apply_conv, filters=filters)
        conv_shortcut = functools.partial(upsample_conv, filters=filters,
                                          kernel_size=1, he_init=False)
    elif resample == None:
        conv_1 = functools.partial(apply_conv, filters=filters)
        conv_2 = functools.partial(apply_conv, filters=filters)
        conv_shortcut = tf.identity

    with tf.name_scope('resblock'):
        x = tf.identity(x)
        update = conv_1(activation(norm_fn(x)))
        update = conv_2(activation(norm_fn(update)))

        skip = conv_shortcut(x)
        return skip + update


def resblock_optimized(x, filters):
    with tf.name_scope('resblock'):
        x = tf.identity(x)
        update = apply_conv(x, filters=filters)
        update = conv_meanpool(activation(update), filters=filters)

        skip = meanpool_conv(x, filters=128, kernel_size=1, he_init=False)
        return skip + update

############### the generator architecture ########################
def generator(z, reuse):
    with tf.variable_scope('generator', reuse=reuse):
        with tf.name_scope('pre_process'):
            z = tf.layers.dense(z, 4 * 4 * 128)
            x = tf.reshape(z, [-1, 4, 4, 128])

        with tf.name_scope('x1'):
            x = resblock(x, filters=128, resample='up', normalize=True) # 8
            x = resblock(x, filters=128, resample='up', normalize=True) # 16
            x = resblock(x, filters=128, resample='up', normalize=True) # 32

        with tf.name_scope('post_process'):
            x = activation(bn(x))
            result = apply_conv(x, filters=3, he_init=False)
            return tf.tanh(result)
##################### the discriminator architecture ####################
def discriminator(x, reuse):
    with tf.variable_scope('discriminator', reuse=reuse):
        with tf.name_scope('pre_process'):
            x2 = resblock_optimized(x, filters=128)
        with tf.name_scope('x1'):
            x3 = resblock(x2, filters=128, resample='down') # 8            
            x4 = resblock(x3, filters=128) # 16            
            x5 = resblock(x4, filters=128) # 
        with tf.name_scope('post_process'):
            x6 = activation(x5)
            x7 = tf.reduce_mean(x6, axis=[1, 2])
            flat2 = tf.contrib.layers.flatten(x7)
            flat = tf.layers.dense(flat2, 1)
            return flat

########## generating the images #################################################
with tf.name_scope('gan'):
    z = tf.random_normal([batch_size, 128], name="z")
    x_generated = generator(z, reuse=False)

############# Computing the inception score of the generated images ###############
with tf.name_scope('summaries'):
    # Advanced metrics
    with tf.name_scope('inception'):
        
        def generate_and_classify(z):
            INCEPTION_OUTPUT = 'logits:0'
            x = generator(z, reuse=True)
            x = tf.image.resize_bilinear(x, [299, 299])
            return tf.contrib.gan.eval.run_inception(x, output_tensor=INCEPTION_OUTPUT)

        # Fixed z for fairness between runs
        inception_z = tf.constant(np.random.randn(10000, 128), dtype='float32')
        inception_score = tf.contrib.gan.eval.classifier_score(inception_z,
                                                               classifier_fn=generate_and_classify,
                                                               num_batches=10000 // 100)

###################### intialize the variables ###########################################
sess.run([tf.global_variables_initializer(),
          tf.local_variables_initializer()])
# image files loading and coordinate the reading ##########################################
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)
# Add op to save and restore ###############################################################
saver = tf.train.Saver(max_to_keep=200)
##### restore the optimal model to compute inception score ##################################
####### model path ######
i = 100000 # the model number, which we want to restore. 
name = "./cifar_NIQE_checkpoints"
nn = name + "/model.ckpt-" + str(i)
############## restoring the model from the saved check point ########################
saver.restore(sess,nn)
########## computing the inception score and taking mean and variance
scores = [inception_score.eval(feed_dict={inception_z: np.random.randn(10000, 128)}) for i in range(5)]
print('{}+-{}'.format(np.mean(scores), np.std(scores)))
