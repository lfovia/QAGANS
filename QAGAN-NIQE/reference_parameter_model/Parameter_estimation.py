from __future__ import division, print_function, absolute_import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import division, print_function, absolute_import
import os
import urllib
import tensorflow as tf
import os
from glob import glob
import numpy as np
import math
from niqe_parameter import *
import skimage.transform
from glob import glob
import tarfile
import pickle
from tensordata.augmentation import random_flip
import adler.tensorflow as atf
import sys
import scipy.io as sio
slim = tf.contrib.slim
import tensordata
import functools
MAX_ITERS = 10
SUMMARY_FREQ = 10
BATCH_SIZE = 256
reset = True
np.random.seed(0)
tf.set_random_seed(0)
sess = tf.InteractiveSession()
size = 64
input_fname_pattern = '*.jpg'
data_x = glob(os.path.join("/home/parimala/Basic_tensor_flow_Codes_tensor_board_github/Generative_Models/GANwKeras-master/vamshi/img_align_celeba/", input_fname_pattern))	
def read_input(batch_size):
	with tf.device('/cpu:0'):
		reader = tf.WholeFileReader()
		filename_queue = tf.train.string_input_producer(data_x)
		data_num = len(data_x)
		key, value = reader.read(filename_queue)
		image = tf.image.decode_jpeg(value, channels=3, name="dataset_image")
		image = tf.image.random_flip_left_right(image)
		input_width = 108
		input_height = 108
		image = tf.image.crop_to_bounding_box(image, (218 - input_height) //2, (178 - input_width) // 2, input_height, input_width)
		#if self.crop:
		output_height = 64
		output_width = 64
		image = tf.image.resize_images(image, [output_height,output_width], method=tf.image.ResizeMethod.BICUBIC)
		num_preprocess_threads=4
		num_examples_per_epoch=800
		min_queue_examples = int(0.1 * num_examples_per_epoch)
		img_batch = tf.train.batch([image],
									batch_size=batch_size,
									num_threads=4,
									capacity=min_queue_examples + 2*batch_size)
		img_batch = 2*((tf.cast(img_batch, tf.float32) / 255.) - 0.5)

		return img_batch, data_num

############ reading data from  the input dataset ##############
with tf.name_scope('placeholders'):
	
	x_train_ph, _ = read_input(batch_size=BATCH_SIZE)
	
with tf.name_scope('pre_process'):
	x_train = x_train_ph	
	x_true = x_train
	
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


def generator(z, reuse):
	with tf.variable_scope('generator', reuse=reuse):
		with tf.name_scope('pre_process'):
			z = tf.layers.dense(z, 4 * 4 * 128)
			x = tf.reshape(z, [-1, 4, 4, 128])

		with tf.name_scope('x1'):
			x = resblock(x, filters=128, resample='up', normalize=True) 
			x = resblock(x, filters=128, resample='up', normalize=True) 
			x = resblock(x, filters=128, resample='up', normalize=True) 
			x = resblock(x, filters=128, resample='up', normalize=True) 
		with tf.name_scope('post_process'):
			x = activation(bn(x))
			result = apply_conv(x, filters=3, he_init=False)
			return tf.tanh(result)

def discriminator(x, reuse):
	with tf.variable_scope('discriminator', reuse=reuse):
		with tf.name_scope('pre_process'):
			x2 = resblock_optimized(x, filters=128)

		with tf.name_scope('x1'):
			x3 = resblock(x2, filters=128, resample='down') 
			x4 = resblock(x3, filters=128) 
			x5 = resblock(x4, filters=128) 
		with tf.name_scope('post_process'):
			x6 = activation(x5)
			x7 = tf.reduce_mean(x6, axis=[1, 2])
			flat2 = tf.contrib.layers.flatten(x7)
			flat = tf.layers.dense(flat2, 1)
			return flat


with tf.name_scope('gan'):
	z = tf.random_normal([tf.shape(x_true)[0], 128], name="z")
	x_generated = generator(z, reuse=False)
	d_true = discriminator(x_true, reuse=False)
	
	d_generated = discriminator(x_generated, reuse=True)
	
	z_gen = tf.random_normal([BATCH_SIZE * 2, 128], name="z")
	d_generated_train = discriminator(generator(z_gen, reuse=True), reuse=True)


with tf.name_scope('regularizer'):
	epsilon = tf.random_uniform([tf.shape(x_true)[0], 1, 1, 1], 0.0, 1.0)
	x_hat = epsilon * x_generated + (1 - epsilon) * x_true
	d_hat = discriminator(x_hat, reuse=True)
	gradients = tf.gradients(d_hat, x_hat)[0]
	C_xhat_grad_norm = tf.norm(slim.flatten(gradients), axis=1)  # l2 norm
	d_regularizer1 = tf.reduce_mean(tf.square(C_xhat_grad_norm - 1.))
	gradients_real = tf.gradients(d_true, x_true)[0]
	features_mu_realgrad,features_cov_realgrad = niqe(tf.image.rgb_to_grayscale(gradients_real),64)
	d_regularizer_mean = tf.reduce_mean(tf.square(d_true))
	added_regularizer = d_regularizer1
with tf.name_scope('loss_gan'):
	wasserstein_scaled = (tf.reduce_mean(d_generated) - tf.reduce_mean(d_true))
	wasserstein = wasserstein_scaled 
	g_loss = tf.reduce_mean(d_generated_train) 
	d_loss = (-wasserstein + 1e-5 * d_regularizer_mean + added_regularizer) 
with tf.name_scope('optimizer'):
	ema = atf.EMAHelper(decay=0.99)
	global_step = tf.Variable(0, trainable=False, name='global_step')
	decay = tf.maximum(0., 1.-(tf.cast(global_step, tf.float32)/MAX_ITERS))
	learning_rate = 2e-4 * decay
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0., beta2=0.9)
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='gan/generator')
	g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
	with tf.control_dependencies(update_ops):
		g_train = optimizer.minimize(g_loss, var_list=g_vars,
									 global_step=global_step)
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='gan/discriminator')
	d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
	with tf.control_dependencies(update_ops):
		d_train = optimizer.minimize(d_loss, var_list=d_vars)
# Initialize all TF variables
sess.run([tf.global_variables_initializer(),
		  tf.local_variables_initializer()])

# Coordinate the loading of image files from the image dataset ########
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)

# Train the network for some 1000 iterations ##########################
while True:
	i = sess.run(global_step)
	if i >= MAX_ITERS:
		break
	
	num_d_train = 5
	for j in range(num_d_train):
		_, d_loss_result,features_mu2,features_cov2 = sess.run([d_train, d_loss,features_mu_realgrad,features_cov_realgrad])
		sio.savemat('Model_parmeters_mu_real_grad.mat', {'ref_mu_grad':features_mu2})
		sio.savemat('Model_parmeters_cov_real_grad.mat', {'ref_cov_grad':features_cov2})
	
	_, g_loss_result, _ = sess.run([g_train, g_loss, ema.apply])
	
	   
		
