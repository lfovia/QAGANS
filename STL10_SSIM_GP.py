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
# User selectable parameters
############ Hyper parameters #################################################################
MAX_ITERS = 1000000
SUMMARY_FREQ = 10
save_freq = 1000
BATCH_SIZE = 64
reset = True
lambda_1 = 1.0
lambda_2 = 0.1 
size = 48
stability_regularizer_factor = 1e-5
####### session intialization #########
# set seeds for reproducibility
np.random.seed(0)
tf.set_random_seed(0)
name = './STL_ssim'
log_dir = './STL_ssim_logs'
if not os.path.exists(log_dir):
	os.makedirs(log_dir)
if not os.path.exists(name):
	os.makedirs(name)
sess = tf.InteractiveSession()
######### creating the folders for check points and log files ##############
# Training specific parameters
input_fname_pattern = '*.png'
data_x = glob(os.path.join("/home/parimala/Aa_current/Current_GAN_work/SSIM_Expicit_regularization/WGAN_GP_SSIM_STL_architecture_96_/img_unlabeled_48/", input_fname_pattern))	
def read_input(batch_size):
	with tf.device('/cpu:0'):
		reader = tf.WholeFileReader()
		filename_queue = tf.train.string_input_producer(data_x)
		data_num = len(data_x)
		key, value = reader.read(filename_queue)
		image = tf.image.decode_jpeg(value, channels=3, name="dataset_image")
		image = tf.image.resize_images(image, [48, 48], method=tf.image.ResizeMethod.BICUBIC)
		img_batch = tf.train.batch([image],
									batch_size=batch_size
									)
		img_batch = (tf.cast(img_batch, tf.float32) / 256.0) 

		return img_batch, data_num

with tf.name_scope('placeholders'):
	
	x_train_ph, _ = read_input(batch_size=BATCH_SIZE)
	


with tf.name_scope('pre_process'):
	x_train = (x_train_ph - 0.5) * 2.0

	x_true = x_train
######### the convolution layer ##########################################################

def apply_conv(x, filters=32, kernel_size=3, he_init=True):
	if he_init:
		initializer = tf.contrib.layers.variance_scaling_initializer(uniform=True)
	else:
		initializer = tf.contrib.layers.xavier_initializer(uniform=True)

	return tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size,
							padding='SAME', kernel_initializer=initializer)

############## the activation function #####################################################
def activation(x):
	with tf.name_scope('activation'):
		return tf.nn.relu(x)
############### batch normalization ######################################################

def bn(x):
	return tf.contrib.layers.batch_norm(x,
									decay=0.9,
									center=True,
									scale=True,
									epsilon=1e-5,
									zero_debias_moving_mean=True,
									is_training=True)

##################### All the required building blocks for architecture ##########
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
################### resnet block ##########################################
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

############## resnet block optimized #############################
def resblock_optimized(x, filters):
	with tf.name_scope('resblock'):
		x = tf.identity(x)
		update = apply_conv(x, filters=filters)
		update = conv_meanpool(activation(update), filters=filters)

		skip = meanpool_conv(x, filters=128, kernel_size=1, he_init=False)
		return skip + update

################ the generator architecture ####################
def generator(z, reuse):
	with tf.variable_scope('generator', reuse=reuse):
		with tf.name_scope('pre_process'):
			z = tf.layers.dense(z, 6 * 6 * 128)
			x = tf.reshape(z, [-1, 6, 6, 128])

		with tf.name_scope('x1'):
			x = resblock(x, filters=128, resample='up', normalize=True) 
			x = resblock(x, filters=128, resample='up', normalize=True) 
			x = resblock(x, filters=128, resample='up', normalize=True) 

		with tf.name_scope('post_process'):
			x = activation(bn(x))
			result = apply_conv(x, filters=3, he_init=False)
			return tf.tanh(result)
################### the discriminator architecture #####################
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

################# the gan losses #######################################
with tf.name_scope('gan'):
	z = tf.random_normal([tf.shape(x_true)[0], 128], name="z")
	x_generated = generator(z, reuse=False)
	
	d_true = discriminator(x_true, reuse=False)
########## make sure you keep reuse = True evertime you use the generator or discriminator functions further ###########
	d_generated = discriminator(x_generated, reuse=True)
	
	z_gen = tf.random_normal([BATCH_SIZE * 2, 128], name="z")
	d_generated_train = discriminator(generator(z_gen, reuse=True), reuse=True)

with tf.name_scope('regularizer'):
	epsilon = tf.random_uniform([tf.shape(x_true)[0], 1, 1, 1], 0.0, 1.0)
	############################## this computes the gradient map with respect to X_hat ############
	x_hat = epsilon * x_generated + (1 - epsilon) * x_true
	d_hat = discriminator(x_hat, reuse=True)
	################### This is the auto encoding part ###########
	gradients = tf.gradients(d_hat, x_hat)[0]
	C_xhat_grad_norm = tf.norm(slim.flatten(gradients), axis=1)  # l2 norm
	d_regularizer1 = tf.reduce_mean(tf.square(C_xhat_grad_norm - 1.))
	############### SSIM penalty computation ##########################
	######### ###### SSIM gradient penalty term computation ###############
	ddx_difference = tf.abs(d_true-d_generated)
	SSIM_penalty_numerator = tf.abs(d_true-d_generated)
	
	SSIM_penalty_denominator = tf_ssim_modified(tf.image.rgb_to_grayscale(x_true),tf.image.rgb_to_grayscale(x_generated))
	#SSIM_penalty_denominator_2 = tf.where(tf.is_nan(SSIM_penalty_denominator), 1.414*tf.ones_like(SSIM_penalty_denominator), SSIM_penalty_denominator)
	############ Based on the dynamic range of your input data, constants in the SSIM function  can be modified.
	############## If you face NaN issue for your data. You can comment the following ###################
	coupled_penalty = tf.divide(SSIM_penalty_numerator[:,0],SSIM_penalty_denominator)
	d_regularizer_ssim = tf.reduce_mean((coupled_penalty - 1) ** 2)
	
	d_regularizer_mean_stability = tf.reduce_mean(tf.square(d_true))
	
	added_regularizer = lambda_1*d_regularizer1 + lambda_2*d_regularizer_ssim + stability_regularizer_factor * d_regularizer_mean_stability
	

with tf.name_scope('loss_gan'):
	wasserstein_scaled = (tf.reduce_mean(d_generated) - tf.reduce_mean(d_true))
	wasserstein = wasserstein_scaled 

	g_loss = tf.reduce_mean(d_generated_train) 
######### The proposed QAGAN -SSIM loss function based on SSIM ###############################################
	d_loss = (-wasserstein + added_regularizer) 
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


with tf.name_scope('summaries'):
	tf.summary.scalar('wasserstein_scaled', wasserstein_scaled)
	tf.summary.scalar('wasserstein', wasserstein)

	tf.summary.scalar('g_loss', g_loss)

	tf.summary.scalar('d_loss', d_loss)
	
	tf.summary.scalar('d_regularizer_ssim', d_regularizer_ssim)
	
	tf.summary.scalar('d_regularizer_gp', d_regularizer1)
	
	tf.summary.scalar('learning_rate', learning_rate)
	tf.summary.scalar('added_regularizer', added_regularizer)
	tf.summary.scalar('learning_rate', learning_rate)
	tf.summary.scalar('global_step', global_step)

	atf.image_grid_summary('x_generated', x_generated)
	merged_summary = tf.summary.merge_all()
# Initialize all TF variables
sess.run([tf.global_variables_initializer(),
		  tf.local_variables_initializer()])

# Coordinate the loading of image files.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)
#i = 102819 # Number of the check point to restore
# Add op to save and restore
saver = tf.train.Saver(max_to_keep=2)
######## i = 1000 uncomment and enter the model number for restoring the model #####
sav_freq = 1000
train_summary_writer = tf.summary.FileWriter(log_dir)
################ Standardized validation z ######################################################
while True:
	i = sess.run(global_step)
	if i >= MAX_ITERS:
		break
	num_d_train = 5
################## discriminator training 5 times ##################################################
	for j in range(num_d_train):

		_, d_loss_result = sess.run([d_train, d_loss])
############## generator training ##################################################################
	_, g_loss_result, _ = sess.run([g_train, g_loss, ema.apply])
	
######################### summary writing ###########################################################
	print(' i={}, d_loss={}, g_loss={}'.format(i,d_loss_result,g_loss_result))
																									
	if i % SUMMARY_FREQ == (SUMMARY_FREQ-1):
		ema_dict = ema.average_dict()
		merged_summary_result_train = sess.run(merged_summary)
		train_summary_writer.add_summary(merged_summary_result_train, i)
############### check point writing ###################################################################
	if i % save_freq == save_freq-1:				
		saver.save(sess,name + "/model.ckpt", global_step=i)
	   
