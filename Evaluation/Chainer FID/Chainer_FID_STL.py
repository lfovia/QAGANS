######### This code is mainly to compute the FID scores for STL10 dataset FID scores, We have used these for fair comparison with previous approaches #######
####### Please use more than 50k images for FID compuation ##########
from __future__ import absolute_import, division, print_function
import os
import sys
import math
import glob
from scipy.misc import imread
import numpy as np
#from PIL import Image
import scipy.linalg
from scipy.misc import imresize
import chainer
import chainer.cuda
from chainer import Variable
from chainer import serializers
import chainer.functions as F

sys.path.append(os.path.dirname(__file__))
from inception_score import inception_score, Inception
def load_inception_model():
    infile = "inception_score.model"
    model = Inception()
    serializers.load_hdf5(infile, model)
    model.to_gpu()
    return model
######## Give the path for generated and real images here  #######
datapath = '/home/parimala/Aa_current/Current_GAN_work/SSIM_Expicit_regularization/WGAN_GP_SSIM_STL_architecture_96_/SSIm_based_regularizers_STL_50k_images_96_res_96k_fid'
datapath2 = '/home/parimala/Aa_current/Current_GAN_work/SSIM_Expicit_regularization/WGAN_GP_SSIM_STL_architecture_96_/img_unlabeled_48' # 

def get_mean_cov(model, ims, batch_size=100):
    n, c, w, h = ims.shape
    n_batches = int(math.ceil(float(n) / float(batch_size)))

    xp = model.xp

    print('Batch size:', batch_size)
    print('Total number of images:', n)
    print('Total number of batches:', n_batches)

    ys = xp.empty((n, 2048), dtype=xp.float32)

    for i in range(n_batches):
        print('Running batch', i + 1, '/', n_batches, '...')
        batch_start = (i * batch_size)
        batch_end = min((i + 1) * batch_size, n)

        ims_batch = ims[batch_start:batch_end]
        ims_batch = xp.asarray(ims_batch)  # To GPU if using CuPy
        ims_batch = Variable(ims_batch)

        # Resize image to the shape expected by the inception module
        if (w, h) != (299, 299):
            ims_batch = F.resize_images(ims_batch, (299, 299))  # bilinear

        # Feed images to the inception module to get the features
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            y = model(ims_batch, get_feature=True)
        ys[batch_start:batch_end] = y.data

    mean = chainer.cuda.to_cpu(xp.mean(ys, axis=0))
    # cov = F.cross_covariance(ys, ys, reduce="no").data.get()
    cov = np.cov(chainer.cuda.to_cpu(ys).T)

    return mean, cov

def FID(m0,c0,m1,c1):
    ret = 0
    ret += np.sum((m0-m1)**2)
    ret += np.trace(c0 + c1 - 2.0*scipy.linalg.sqrtm(np.dot(c0, c1)))
    return np.real(ret)

def calc_FID(batchsize=10):
    """Frechet Inception Distance proposed by https://arxiv.org/abs/1706.08500"""
    #@chainer.training.make_extension(
    model = load_inception_model()
    
    xs = []
    n_ims = 50000
    files = glob.glob(os.path.join(datapath, '*.jpg'))
    files2 = files[:50000]
    xs = np.array([imread(str(fn)).astype(np.float32) for fn in files2])
    xs = xs.reshape((n_ims, 3, 96, 96)).astype("f")

    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        mean, cov = get_mean_cov(model, np.asarray(xs).reshape((-1, 3, 96, 96)))
    xs_real = []
    n_ims = 50000
    files = glob.glob(os.path.join(datapath2, '*.png'))
    files3 = files[:50000]
    xs_real = np.array([imread(str(fn)).astype(np.float32) for fn in files3])
    xs_real = xs_real.reshape((n_ims, 3, 96, 96)).astype("f")

    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        mean2, cov2 = get_mean_cov(model, np.asarray(xs_real).reshape((-1, 3, 96, 96)))
    fid = FID(mean2, cov2, mean, cov)
    print(fid)
####### Please mention the batch size ###########    
calc_FID(batchsize=100)    

