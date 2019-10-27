import tensorflow as tf
import os
import glob
from scipy.misc import imresize
from scipy.misc import imread
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
######### images dataset path for generated set ##########
datapath1 = '/home/parimala/Aa_current/Current_GAN_work/SSIM_Expicit_regularization/Models_ssim_regularization_cifar/niqe_91k_4'
############ images dataset path for real set ########
datapath2 = './train'
files = glob.glob(os.path.join(datapath1, '*.jpg'))
images1 = np.array([imread(str(fn),mode='RGB').astype(np.float32) for fn in files])
images222 = tf.convert_to_tensor(images1)
files = glob.glob(os.path.join(datapath2, '*.png'))
images2 = np.array([imread(str(fn),mode='RGB').astype(np.float32) for fn in files])
images222_2 = tf.convert_to_tensor(images2)
print("yes")
def frechet_process(x):
    INCEPTION_FINAL_POOL = 'pool_3:0'
    x = tf.image.resize_bilinear(x, [299, 299])
    return tf.contrib.gan.eval.run_inception(x,output_tensor=INCEPTION_FINAL_POOL)
                                                          
fid = tf.contrib.gan.eval.frechet_classifier_distance(images222,images222_2,classifier_fn =frechet_process,num_batches=200)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(fid))
    
    
