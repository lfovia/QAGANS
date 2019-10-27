from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import division, print_function, absolute_import
import tensorflow as tf
slim = tf.contrib.slim
import numpy as np
import functools
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)
def binary_activation(x):

    cond = tf.less(x, tf.ones(tf.shape(x)))
    out = tf.where(cond, tf.zeros(tf.shape(x)), tf.ones(tf.shape(x)))

    return out
def tf_ssim_modified(img1, img2, cs_map=False, mean_metric=True, size=8, sigma=1.5):
    window = _tf_fspecial_gauss(size, sigma) # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 2  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='SAME')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='SAME')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='SAME') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='SAME') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='SAME') - mu1_mu2
    #####################
    S1 = (2*mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    S2 = (2*sigma12 + C2) / (sigma1_sq + sigma2_sq + C2+0.01)
    binary_S1 = tf.reduce_sum(binary_activation(S1))
    binary_S2 = tf.reduce_sum(binary_activation(S2))
    #print(S1)
    #print(S2)
    value = tf.reduce_mean(tf.sqrt(2-S1-S2),[1,2,3])
    S2_m = tf.reduce_mean(S2,[1,2,3])
    S1_m = tf.reduce_mean(S1,[1,2,3])		
    return value
