from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import division, print_function, absolute_import
import cPickle as pickle
import os
import urllib
from glob import glob
import numpy as np
import tarfile
import pickle
import scipy.misc
from tensorflow.python.platform import gfile
import tensorflow as tf
from tensordata.augmentation import random_flip
import adler.tensorflow as atf
import sys
from SSIM_penalty import *
slim = tf.contrib.slim
import tensorflow as tf
import numpy as np
import tensordata
import functools
"""
This code is mailny to generate the images from the trained model This code can be easily extend to all three architectures 
"""
# User hyper parameters #######################################################################
save_freq = 10
Batch_size = 200
reset = False
##################### interactive session intialization #######################################
np.random.seed(0)
tf.set_random_seed(0)
sess = tf.InteractiveSession()
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
                                    is_training=True)
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
############### resnet architecture ######################################
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
############# resblock optimizised ##############################
def resblock_optimized(x, filters):
    with tf.name_scope('resblock'):
        x = tf.identity(x)
        update = apply_conv(x, filters=filters)
        update = conv_meanpool(activation(update), filters=filters)

        skip = meanpool_conv(x, filters=128, kernel_size=1, he_init=False)
        return skip + update
###### the generator function ##################################
def generator(z, reuse):
    with tf.variable_scope('generator', reuse=reuse):
        with tf.name_scope('pre_process'):
            z = tf.layers.dense(z, 6 * 6 * 128)
            x = tf.reshape(z, [-1, 6, 6, 128])

        with tf.name_scope('x1'):
            x = resblock(x, filters=128, resample='up', normalize=True) # 8
            x = resblock(x, filters=128, resample='up', normalize=True) # 16
            x = resblock(x, filters=128, resample='up', normalize=True) # 32

        with tf.name_scope('post_process'):
            x = activation(bn(x))
            result = apply_conv(x, filters=3, he_init=False)
            return tf.tanh(result)
############### the discrminator function #####################
def discriminator(x, reuse):
    with tf.variable_scope('discriminator', reuse=reuse):
        with tf.name_scope('pre_process'):
            x2 = resblock_optimized(x, filters=128)

        with tf.name_scope('x1'):
            x3 = resblock(x2, filters=128, resample='down') # 8
            x4 = resblock(x3, filters=128) # 16
            x5 = resblock(x4, filters=128) # 32
        with tf.name_scope('post_process'):
            x6 = activation(x5)
            x7 = tf.reduce_mean(x6, axis=[1, 2])
            flat2 = tf.contrib.layers.flatten(x7)
            flat = tf.layers.dense(flat2, 1)
            return flat
############ generating the images from trained generator ####################################
with tf.name_scope('gan'):
    z = tf.random_normal([Batch_size, 128], name="z")
    x_generated = generator(z, reuse=False)
############# intialize the variables in the graph #############################################
sess.run([tf.global_variables_initializer(),
          tf.local_variables_initializer()])
# Coordinate the loading of image files. #######################################################
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)
saver = tf.train.Saver()
#### Enter the path of the model  #######
i =10000 ######### the model number, which you want to use for generating samples #################################
nn = "./STL_SSIM_checkpoints/model.ckpt-" + str(i)
saver.restore(sess,nn)
dirname = './STL_SSIM_generated_images/'
os.makedirs(dirname)
Batch_size = 200
no_times = 1
########## generating the samples and saving generated the samples #####################################
for k in range(no_times):
    z_validate = np.random.randn(Batch_size, 128)  
    xg = sess.run(x_generated,feed_dict={z:z_validate}) 
    for i in range(Batch_size): 
        generated_image=xg[i,:,:,:]
        file_name=dirname+'sample'+str(i)+'_'+str(k)+'_'+'.jpg';
        scipy.misc.imsave(os.path.join(file_name), generated_image)  
