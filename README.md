# QAGANS
Quality Aware Generative Adversarial Networks


Dataset links


cifar : https://www.cs.toronto.edu/~kriz/cifar.html

celebA : http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

stl: https://cs.stanford.edu/~acoates/stl10/

Tensorflow version compatible : 1.10.0

Install : pip install https://github.com/adler-j/adler/archive/master.zip


SSIM references 
1. Zhou Wang et al. “Image quality assessment: from error visibility to structural similarity”.
In: IEEE transactions on image processing 13.4 (2004), pp. 600–612
2. Dominique Brunet, Edward R Vrscay, and Zhou Wang. “On the mathematical properties
of the structural similarity index”. In: IEEE Transactions on Image Processing 21.4
(2011), pp. 1488–1499

########################
Extract the cifar-dataset 

#############

######  Training from scratch #######
1.set reset = true

2. Enter folder path for the real data in data_x or train_dir.

3. run the codes: python cifar10_SSIM_GP.py

#### To Restore the model and training ########

1) set reset = False 
2) comment os.makedirs(log_dir)
		   os.makedirs(name)
3) uncomment the i = 1000 
   enter the i value as the model number, which you want to restore by checking in the checkpoints folder.
######################## Computing inception score while training ################

The example code for computing the inception score while training
 1) can be found in inception_evaluation_while_training.py (The example code is for STL10, can be extended to other datasets as well just by replacing the correct model architecture.)
 
 2) Enter the inception frequency
As inception score computation takes time and this training will take more time

###################################################################################
########################### Final evaluation of the model ########################
1. Generate the 50k images using generate_images by loading the model. (The example code is for STL10, can be extended to other datasets as well just by replacing the correct model architecture.)

   i) Code can be found in generate_more_images.py
   
   ii) Please give the path for the current model.
   
	
2. Evaluation codes can be found in evaluation folder.


We would like to thank Banach WGAN paper author for their architecture code.

https://github.com/adler-j/bwgan

###################
If you are using the code/model/data provided here in a publication, please cite our paper:


@incollection{NIPS2019_8560,
title = {Quality Aware Generative Adversarial Networks},
author = {PARIMALA, KANCHARLA and Channappayya, Sumohana},
booktitle = {Advances in Neural Information Processing Systems 32},
editor = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
pages = {2948--2958},
year = {2019},
publisher = {Curran Associates, Inc.},
url = {http://papers.nips.cc/paper/8560-quality-aware-generative-adversarial-networks.pdf}
}


For any queries Please contact :
ee15m17p100001@iith.ac.in,sumohoan@iith.ac.in





