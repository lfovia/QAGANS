###########
########## We are giving the model parameters from the authors of NIQE 

##################################################################

1. A. Mittal, R. Soundararajan and A. C. Bovik, “ Making a Completely Blind Image Quality Analyzer ”, IEEE Signal processing Letters, pp. 209-212, vol. 22, no. 3, March 2013.

2. NIQE Original implementation : http://live.ece.utexas.edu/research/Quality/index_algorithms.htm
Here The main motivation is to impose statistical constraint. ( So it is valid to take model parameters from images and also pristine grad maps.)

run celebA_NIQE_GP.py

# In our experiments we observed that 
1. Underestimation of pristine parameters will give NANs while training. so we suggest to use sufficient number of images to estimate the pristine images.

#################
We provide the NIQE python implementation and also NIQE tensorflow implementation.


We provide the following parameters in the reference_model_parameters:

1. modelparameters.mat (Natural scene stats from NIQE original implementation)

######## The reference model parameters ##########

2. model_STL_grad.mat	

3. model_cifar_grad.mat


4. model_face_grad.mat

########
Please change the model parameters for various datasets

######################
##parameter estimation #########

# reference_parameter_model folder #######

 You can do in two ways
 
1. Using Matlab
You can save the discriminator gradient maps with respect to real images and use those to estimate the reference parameter model

2. Using python : 
	run Parameter_estimation.py
  

####Please cite our work.
1. Quality Aware Generative Adversarial Networks, Neurips , 2019.

#######################################


