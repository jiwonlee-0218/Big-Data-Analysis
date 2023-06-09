# Big-Data-Analysis


# Dataset
from tslearn.datasets import UCR_UEA_datasets 


# Model Description
+ Temporal AutoEncoder (TAE)
  + Using convolutional layers and LSTM layers to extract spatio-temporal features.
+ Temporal Clustering Layer
  + The temporal clustering layer takes the latent vector z_i (i represents a single data point) generated by the encoder of the TAE model as input.
  + The centroid vector w_j (j represents a single cluster) initialized within the model is also used, along with the Euclidean temporal similarity metric, to compute the similarity between z_i and w_j, which is returned as a probability.
  
