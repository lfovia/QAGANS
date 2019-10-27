from __future__ import division, print_function, absolute_import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import division, print_function, absolute_import
import tensorflow as tf
import os
import math
import numpy as np
import numpy.linalg
from scipy.special import gamma
from scipy.ndimage.filters import gaussian_filter
import scipy.misc
import sys
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
def mscn_function(ref2):
	N = 3
	sd =0.5
	window = _tf_fspecial_gauss(N, sd)
	ref = ref2
	mu1 = tf.nn.conv2d(ref, window, strides=[1, 1, 1, 1], padding='SAME')
	mu1_sq = mu1 * mu1
	sigma1_sq = tf.nn.conv2d(ref*ref, window, strides=[1, 1, 1, 1], padding='SAME') - mu1_sq
	mscn_image = tf.divide((ref-mu1),(sigma1_sq+1))    
	return mscn_image
def generalized_gaussian_ratio(alpha):
	return (gamma(2.0/alpha)**2) / (gamma(1.0/alpha) * gamma(3.0/alpha))

def tf_cov(x):
	x_without_nan = tf.where(tf.is_nan(x), tf.zeros_like(x), x)
	mean_x = tf.reduce_mean(x_without_nan, axis=0, keep_dims=True)
	mx = tf.matmul(tf.transpose(mean_x), mean_x)
	vx = tf.matmul(tf.transpose(x), x)/tf.cast(tf.shape(x)[0], tf.float32)
	cov_xx = vx - mx
	return cov_xx

def generalized_gaussian_ratio_inverse(k):
	a1 = tf.constant(-0.535707356, dtype='float32')
	a2 = tf.constant(1.168939911, dtype='float32')
	a3 = tf.constant(-0.1516189217, dtype='float32')
	b1 = tf.constant(0.9694429, dtype='float32')
	b2 = tf.constant(0.8727534, dtype='float32')
	b3 = tf.constant(0.07350824, dtype='float32')
	c1 = tf.constant(0.3655157, dtype='float32')
	c2 = tf.constant(0.6723532, dtype='float32')
	c3 = tf.constant(0.033834, dtype='float32')
	first01=tf.cast(2.0 ,tf.float32)
	first02= tf.cast(tf.log(tf.divide(27.0,16.0)),tf.float32)
	first1=tf.multiply(first01,first02)
	first001=tf.cast(4.0,tf.float32)
	first002=tf.cast(tf.pow(k,2),tf.float32)
	first0002=tf.multiply(first001,first002)
	first002_2=tf.cast(tf.divide(3.0,first0002),tf.float32)
	first2=tf.log(first002_2)
	#if k < 0.131246:
	final1 = tf.divide(first1,first2)
	second2=-a2+tf.sqrt(tf.pow(a2,2)-tf.multiply(tf.multiply(first001,a1),a3)+tf.multiply(tf.multiply(first001,a1),k))
	one1=tf.cast(1.0 ,tf.float32)
	second002=tf.divide(one1,tf.multiply(first01,a1))
	final2=tf.multiply(second002,second2)
	second002_22=tf.divide(one1,tf.multiply(tf.multiply(first01,b3),k))
	second002_2=(b1-tf.multiply(b2,k)-tf.sqrt(tf.pow((b1-tf.multiply(b2,k)),2)-tf.multiply(tf.multiply(first001,b3),tf.pow(k,2))))
	final3=tf.divide(second002_22,second002_2)
	fourth1=tf.divide(one1,tf.multiply(first01,c3))
	extra=tf.log(tf.divide((3.0-tf.multiply(first001,k)),tf.multiply(first001,c1)))
	fourth2=c2-tf.sqrt(tf.pow(c2,2)+tf.multiply(tf.multiply(first001,c3),extra))
	final4=tf.multiply(fourth1,fourth2)
	xx=(tf.shape(final1))
	def f1(): return tf.constant(final1,dtype=tf.float32)
	def f2(): return tf.constant(final2,dtype=tf.float32)
	def f3(): return tf.constant(final3,dtype=tf.float32)
	def f4(): return tf.constant(final4,dtype=tf.float32)
	def f5(): return tf.constant(0.0001,dtype=tf.float32)
	  
	return final1

	

def estimate_aggd_params(x):
	x_left_mask = tf.less(x, 0)
	x_left = tf.boolean_mask(x, x_left_mask) 
	x_right_mask = tf.greater(x, 0)
	x_right = tf.boolean_mask(x, x_right_mask) 
	size_l=tf.shape(x_left)[0]
	size_r=tf.shape(x_right)[0]
	xx=tf.cast(tf.sqrt(tf.divide(1,(size_l-1))),tf.float32)
	yy=tf.reduce_sum(tf.pow(x_left,2))
	stddev_left_1 = tf.multiply(xx,yy)
	stddev_left_2 = tf.constant(0.001,tf.float32)
	xx2=tf.cast(tf.sqrt(tf.divide(1,(size_r-1))),tf.float32)
	yy2=tf.reduce_sum(tf.pow(x_right,2))
	stddev_right_1 = tf.multiply(xx2,yy2)
	stddev_right_2 = tf.constant(0.001,tf.float32)
	stddev_right = tf.cond(size_r > 0, lambda:stddev_right_1 , lambda:stddev_right_2)
	stddev_left = tf.cond(size_l > 0, lambda:stddev_left_1 , lambda:stddev_left_2)
	if stddev_right == 0:
		return 1, 0, 0 # TODO check this
	size_n = size_l + size_r
	r_hat_1 = tf.pow(tf.reduce_mean(tf.abs(x)),2) / tf.reduce_mean(tf.pow(x,2))
	r_hat_2 = 0.5
	r_hat = tf.cond(tf.math.equal(size_n,0), lambda:r_hat_2 , lambda:r_hat_1)
	
	y_hat = tf.divide(stddev_left,stddev_right)
	vvx=tf.pow((tf.pow(y_hat,2)+1),2)
	vvx2=tf.multiply(tf.multiply(r_hat, (tf.pow(y_hat,3) + 1)) , (y_hat + 1))
	R_hat = tf.divide(vvx2,vvx)  
	alpha = generalized_gaussian_ratio_inverse(R_hat)
	dd_x=tf.divide(tf.exp(tf.lgamma(tf.divide(3.0,alpha))),tf.exp(tf.lgamma(tf.divide(1.0,alpha))))
	beta_left = tf.multiply(stddev_left,dd_x)
	#pdb.set_trace() 
	beta_right = tf.multiply(stddev_right,dd_x)
	return alpha, beta_right, beta_left

def compute_features(img_norm):
	features = []
	############### estimate the aggd parameters from the MSCN coefficeints ###########
	alpha, beta_left, beta_right = estimate_aggd_params(img_norm)
	
	features.extend([ alpha, (beta_left+beta_right)/2 ])
	########## Roating and shifts and multiplyng the patches awith neighbouring patches ###########
	block_image_shifted =img_norm* tf.manip.roll(img_norm, shift=[0,1], axis=[0,1])
	alpha, beta_left, beta_right = estimate_aggd_params(block_image_shifted)
	eta = (beta_right - beta_left) * tf.divide(tf.exp(tf.lgamma(tf.divide(3.0,alpha))) , tf.exp(tf.lgamma(tf.divide(1.0,alpha))))
	features.extend([ alpha,eta,beta_left, beta_right])
	block_image_shifted =img_norm* tf.manip.roll(img_norm, shift=[1,1], axis=[0,1])
	alpha, beta_left, beta_right = estimate_aggd_params(block_image_shifted)
	eta = (beta_right - beta_left) * tf.divide(tf.exp(tf.lgamma(tf.divide(3.0,alpha))) , tf.exp(tf.lgamma(tf.divide(1.0,alpha))))
	features.extend([ alpha,eta,beta_left, beta_right])
	block_image_shifted =img_norm* tf.manip.roll(img_norm, shift=[1,0], axis=[0,1])
	alpha, beta_left, beta_right = estimate_aggd_params(block_image_shifted)
	eta = (beta_right - beta_left) * tf.divide(tf.exp(tf.lgamma(tf.divide(3.0,alpha))) , tf.exp(tf.lgamma(tf.divide(1.0,alpha))))
	features.extend([ alpha,eta,beta_left, beta_right])
	block_image_shifted =img_norm* tf.manip.roll(img_norm, shift=[1,-1], axis=[0,1])
	eta = (beta_right - beta_left) * tf.divide(tf.exp(tf.lgamma(tf.divide(3.0,alpha))) , tf.exp(tf.lgamma(tf.divide(1.0,alpha))))
	alpha, beta_left, beta_right = estimate_aggd_params(block_image_shifted)
	features.extend([ alpha,eta,beta_left, beta_right])
	return features

#####
def pinv(a, rcond=1e-15):
	s, u, v = tf.svd(a)
	# Ignore singular values close to zero to prevent numerical overflow
	limit = rcond * tf.reduce_max(s)
	non_zero = tf.greater(s, limit)

	reciprocal = tf.where(non_zero, tf.reciprocal(s), tf.zeros(s.shape))
	lhs = tf.matmul(v, tf.diag(reciprocal))
	return tf.matmul(lhs, u, transpose_b=True)
###############3 this the implementation of NIQE (no refrence image quality evaluator ###################
def niqe(img,res):
    ############### loading the reference model parameters #############################
	features = tf.placeholder("float", [None,None])
	img_scaled = img
	#################### NIQE computation as follows ###################
	i = 0
	res2 = res
	########### this compuation will be done for two scales ################
	for scale in [1,2]:
		if (scale != 1):
			img_scaled = tf.image.resize_bilinear(img_scaled, [int(res/2),int(res/2)])
			res2 = int(res/2)
		############ the mean subtracted contrast normalization function ###################
		img_norm=mscn_function(img_scaled)
		########### the feature extraction from MSCN coefficients ######################
		scale_features = []
		block_size = res//scale
		for block_col in range(res2//block_size):
			for block_row in range(res2//block_size):
				block_features = compute_features( img_norm[:,block_col*block_size:(block_col+1)*block_size, block_row*block_size:(block_row+1)*block_size,:] )

				scale_features.append(block_features)
		if (i == 0):
			features = tf.stack(scale_features,axis=0)
			i = i+1
		else :
			features = tf.concat([features,scale_features],axis=1) 
	################ after getting the features of length (36)  ##############
	########### MVG will be  fitted on those features ################
	feature_without_nans = tf.where(tf.is_nan(features), tf.zeros_like(features), features)
	features_mu = tf.reduce_mean(feature_without_nans, 0)
	features_cov = tf_cov(features)
	################# Distance between two MVGs ########################
	return features_mu,features_cov
