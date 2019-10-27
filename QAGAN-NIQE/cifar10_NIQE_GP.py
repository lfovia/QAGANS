########### Resnet Architecture of the discriminator and generator ##############
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import division, print_function, absolute_import
import os
import numpy as np
import tarfile
import pickle
from tensorflow.python.platform import gfile
import tensorflow as tf
from NIQE_penalty import *
from tensordata.augmentation import random_flip
import adler.tensorflow as atf
import sys
slim = tf.contrib.slim
import tensorflow as tf
import numpy as np
import functools
from IPython.core.debugger import Pdb
pdb = Pdb()
# User selectable parameters
############ Hyper parameters #################################################################
MAX_ITERS = 1000000
SUMMARY_FREQ = 10
BATCH_SIZE = 64
reset = True
sav_freq = 1000
size = 64
lambda_1 = 1.0
lambda_2 = 0.1 
stability_regularizer_factor = 1e-5
# set seeds for reproducibility
np.random.seed(0)
tf.set_random_seed(0)
####### session intialization #########
sess = tf.InteractiveSession()
# Training specific parameters
######### creating the folders for check points and log files ##############
name = './cifar_NIQE_checkpoints'
log_dir = './cifar_NIQE_logs'
if not os.path.exists(log_dir):
	os.makedirs(log_dir)
if not os.path.exists(name):
	os.makedirs(name)
################# here the main code is only for reading the dataset from cifar files #####################
def unpickle(file):
    with open(file, 'rb') as fo:
        dct = pickle.load(fo)
    return dct
CIFAR10_LABELS = ['airplane',
                  'automobile',
                  'bird',
                  'cat',
                  'deer',
                  'dog',
                  'frog',
                  'horse',
                  'ship',
                  'truck']


class ClassificationDataSet(object):

    """Dataset for classification problems."""

    def __init__(self,
                 images,
                 labels):
        assert images.shape[0] == labels.shape[0]
        assert images.ndim == 4
        assert labels.ndim == 1

        self.num_examples = images.shape[0]

        self.images = images
        self.labels = labels
        self.epochs_completed = 0
        self._index_in_epoch = 0

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        assert batch_size <= self.num_examples

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self.num_examples:
            # Finished epoch
            self.epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch
        return self.images[start:end], self.labels[start:end]
def get_cifar10_dataset(split=None):
    filename = 'cifar-10-python.tar.gz'
    ####### give the path to the dataset #############
    train_dir = '/home/parimala/Aa_current/NIPS_reproducible_stl48/Cifar_SSIM/'
    data = []
    labels = []
    for i in range(1, 7):
        if i < 6:
            path = os.path.join(train_dir, 'cifar-10-batches-py', 'data_batch_{}'.format(i))
        elif i == 6:
            path = os.path.join(train_dir, 'cifar-10-batches-py', 'test_batch')
        dct = unpickle(path)
        data.append(dct[b'data'])
        labels.append(np.array(dct[b'labels'], dtype='int32'))

    data_arr = np.concatenate(data, axis=0)
    raw_float = np.array(data_arr, dtype='float32') / 256.0
    images = raw_float.reshape([-1, 3, 32, 32])
    images = images.transpose([0, 2, 3, 1])

    labels = np.concatenate(labels, axis=0)

    if split is None:
        pass
    elif split == 'train':
        images = images[:-10000]
        labels = labels[:-10000]
    elif split == 'test':
        images = images[-10000:]
        labels = labels[-10000:]
    else:
        raise ValueError('unknown split')

    dataset = ClassificationDataSet(images,
                                    labels)

    return dataset
def get_cifar10_tf(batch_size=32, shape=[32, 32], split=None, augment=True,
                   start_queue_runner=True):
    with tf.name_scope('get_cifar10_tf'):
        dataset = get_cifar10_dataset(split=split)

        images = tf.constant(dataset.images, dtype='float32')
        labels = tf.constant(dataset.labels, dtype='int32')

        image, label = tf.train.slice_input_producer([images, labels],
                                                     shuffle=True)

        images_batch, labels_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=8)

        if augment:
            images_batch = random_flip(images_batch)
            images_batch += tf.random_uniform(tf.shape(images_batch),
                                              0.0, 1.0/256.0)

        if shape != [32, 32]:
            images_batch = tf.image.resize_bilinear(images_batch,
                                                    [shape[0], shape[1]])

        if start_queue_runner:
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(coord=coord)

        return images_batch, labels_batch
######################### till here only reading the dataset ###########################
########## reading the data in batches to x_train ######################################

with tf.name_scope('placeholders'):
    x_train_ph, _ = get_cifar10_tf(batch_size=BATCH_SIZE)
############# normalizing the input data ################################################
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

with tf.name_scope('NIQE_regularizer'):
    epsilon = tf.random_uniform([tf.shape(x_true)[0], 1, 1, 1], 0.0, 1.0)
    x_hat = epsilon * x_generated + (1 - epsilon) * x_true
    d_hat = discriminator(x_hat, reuse=True)
    ############################## this computes the gradient map with respect to X_hat ############
    gradients = tf.gradients(d_hat, x_hat)[0]
    C_xhat_grad_norm = tf.norm(slim.flatten(gradients), axis=1)  # l2 norm of  the gradient 
    ############## gradient penalty term (1-gp) ###################################
    d_regularizer1 = tf.reduce_mean(tf.square(C_xhat_grad_norm - 1.))
    ############### NIQE based penalty computation ##########################
    niqe_score_grad = niqe(tf.image.rgb_to_grayscale(gradients),32)
    niqe_score_mean_grad = tf.reduce_mean(niqe_score_grad)
    d_regularizer_mean_stability = tf.reduce_mean(tf.square(d_true))    
	######### The proposed regularizer based on niqe #############################
    added_regularizer = lambda_1*d_regularizer1 + lambda_2*niqe_score_mean_grad + stability_regularizer_factor * d_regularizer_mean_stability
with tf.name_scope('loss_gan'):
    ############### Wasserstein GAN loss ############################
    wasserstein_scaled = (tf.reduce_mean(d_generated) - tf.reduce_mean(d_true))
    wasserstein = wasserstein_scaled  
    g_loss = tf.reduce_mean(d_generated_train) 
######### The proposed QAGAN -NIQE loss function based on SSIM ###############################################
    d_loss = (-wasserstein + added_regularizer) 
with tf.name_scope('optimizer'):
    ema = atf.EMAHelper(decay=0.99)
    global_step = tf.Variable(0, trainable=False, name='global_step')
    decay = tf.maximum(0., 1.-(tf.cast(global_step, tf.float32)/MAX_ITERS))
    learning_rate = 2e-4 * decay
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0., beta2=0.9)
    ############### training generator with generator loss
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='gan/generator')
    g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    with tf.control_dependencies(update_ops):
        g_train = optimizer.minimize(g_loss, var_list=g_vars,
                                     global_step=global_step)
    ################ training discriminator with discriminator loss ##################
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='gan/discriminator')
    d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
    with tf.control_dependencies(update_ops):
        d_train = optimizer.minimize(d_loss, var_list=d_vars)

############ summary writing ########################################
with tf.name_scope('summaries'):
    tf.summary.scalar('wasserstein_scaled', wasserstein_scaled)
    tf.summary.scalar('wasserstein', wasserstein)
    tf.summary.scalar('g_loss', g_loss)
    tf.summary.scalar('d_loss', d_loss)
    tf.summary.scalar('d_regularizer_niqe', niqe_score_mean_grad)
    tf.summary.scalar('d_regularizer_gp', d_regularizer1)
    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('added_regularizer', added_regularizer)
    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('global_step', global_step)
    atf.image_grid_summary('x_generated', x_generated)
    

    merged_summary = tf.summary.merge_all()
############### intialize the variables ################
sess.run([tf.global_variables_initializer(),
          tf.local_variables_initializer()])

############ The image files and coordinate the loading of image files #########
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)
########### Add op to save and restore #########################################
saver = tf.train.Saver(max_to_keep=10)
######## i = 1000 uncomment and enter the model number for restoring the model #####
if not reset:
    
    nn = name + "/model.ckpt-" + str(i)
    saver.restore(sess,nn)

################ Standardized validation z ######################################################
train_summary_writer = tf.summary.FileWriter(log_dir)
###########################
while True:
    i = sess.run(global_step)
    if i >= MAX_ITERS:
        break
################## discriminator training 5 times ##################################################
    num_d_train = 5
    for j in range(num_d_train):
        _, d_loss_result = sess.run([d_train, d_loss])
############## generator training ##################################################################
    _, g_loss_result, _ = sess.run([g_train, g_loss, ema.apply])
    print('i={}, d_loss={}, g_loss={}'.format(i,d_loss_result,g_loss_result))
######################### summary writing ###########################################################
    if i % SUMMARY_FREQ == SUMMARY_FREQ - 1:
        ema_dict = ema.average_dict()
        merged_summary_result_train = sess.run(merged_summary)
        train_summary_writer.add_summary(merged_summary_result_train, i)
############### check point writing ###################################################################
    if i % sav_freq == sav_freq - 1:
        saver.save(sess,name + "/model.ckpt", global_step=i)
       
