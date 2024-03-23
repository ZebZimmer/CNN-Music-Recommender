# Replication Assignment 2 - Second Deadline

## Team Members
Zeb Zimmer
Hamza Khan

## Paper Citation and Link
### Bibtex
@inproceedings{NIPS2013_b3ba8f1b,
 author = {van den Oord, Aaron and Dieleman, Sander and Schrauwen, Benjamin},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {C.J. Burges and L. Bottou and M. Welling and Z. Ghahramani and K.Q. Weinberger},
 pages = {},
 publisher = {Curran Associates, Inc.},
 title = {Deep content-based music recommendation},
 url = {https://proceedings.neurips.cc/paper_files/paper/2013/file/b3ba8f1bee1238a2f37603d90b58898d-Paper.pdf},
 volume = {26},
 year = {2013}
}

## Goals
### The paper
Deep content-based music recommendation from Ghent University argues that audio music contains latent factors that can be predicted when they cannot be obtained from usage data (cold-start). They use a CNN to try and create recommendations for new and unpopular songs with the predicted latent factors. The weighted matrix facotrization (WMF) algorithm is used to learn the latent factor representations of all users and items in the dataset. These latent factor vectors from WMF were used as the ground truth to train the prediction models. WMF relies on extensive play count data for users and songs.

#### Bag-of-words
The bag-of-words (BoW) technique was considered (in 2013) to be the conventional method where local features are extracted from audio signals and aggregated into a BoW representation. Two models used BoW features to predict latent factors using linear regression and a multilayer perceptron. 

#### Convolutional Neural Networks
The convolutional neural network (CNN) technique involved extracting intermediate time-frequency representations from the audio signals to use as input. MSE was used as the objective function because latent factor vectors are real-valued. 
These predicted latent factor vectors were tested ultimately by creating recommendations by generating latent factor representations from a user's liked songs and then songs not in that list with similar a similar latent factor vector.

## Plan & Status
Use WMF to generate ground truth (latent factor vecotrs for songs) from dataset XXXXX. We may need to create our WMF ourselves (XXXXXXXXX see if this can be found).
Gather audio clips from dataset YYYYYY. Copy the paper's way of preprocessing the audio for CNN training.

Design and adapt a CNN architecture to generate latent factor vectors from audio samples. (The paper does not give code nor the architecture of their CNN.)
Determine if 50-dimensional or 400-dimensional latent factor vectors are best for the given dataset.

Train the CNN to predict latent factors for songs.
Generate latent factor vectors for every song in our dataset.
Generate predictions on song querys by creating a latent factor vector and finding the most similar songs using cosine similarity.  
