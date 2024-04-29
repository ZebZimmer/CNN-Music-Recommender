# Replication Assignment 2 - Second Deadline

## Team Members
Zeb Zimmer
Hamza Khan

## Paper Citation and Link
Find the paper here:
https://proceedings.neurips.cc/paper_files/paper/2013/file/b3ba8f1bee1238a2f37603d90b58898d-Paper.pdf

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
"Deep Content-based Music Recommendation from Ghent University argues that audio music contains latent factors that can be predicted when they cannot be obtained from usage data (cold-start scenarios). They use a CNN to create latent factor representation of new and unpopular songs for recommendation purposes. The weighted matrix facotrization (WMF) algorithm is used to learn the latent factor representations of all users and items in the dataset and these latent factor vectors (LFV) from WMF were used as the ground truth to train the prediction models. WMF relies on extensive play count data for users and songs (non-cold-start scenarios).

#### Bag-of-words
The bag-of-words (BoW) technique was considered (in 2013) to be the conventional method of representing songs where local features are extracted from audio signals and aggregated into a BoW representation. Two models used BoW features to learn predict LFV using linear regression and a multilayer perceptron. These two models are less ideal due to the preprocessing of the audio to get the initial BoW represenation as well as being constrained by the inital BoW input.

#### Convolutional Neural Networks
The convolutional neural network (CNN) technique involved extracting intermediate time-frequency representations from a song's audio signals to use as input. MSE was used as the objective function because LFV are real-valued. 
These predicted LFVs were tested by creating recommendations through finding similar songs to songs already enjoyed by a user. A LFV was created for every song in the database and the liked songs of the user were used as search querys to find similar songs (recommendations).

## Plan & Status
### TODO List
Use WMF to generate ground truth (LFVs for songs) from the Million Song Dataset(1). We plan on using the WMF model from Cornac(2).
Gather audio clips from a dataset(3). Replicate the paper's way of preprocessing the audio for CNN training.

Design and adapt a CNN architecture to generate LFV from audio samples. (The paper does not give code nor the architecture of their CNN.)
Determine if 50-dimensional or 400-dimensional LFV are best for the given dataset.

Train the CNN to predict LFVs for songs.
Generate LFVs for every song in our dataset.
Generate predictions on song querys by creating a LFV and finding the most similar songs using cosine similarity.  

## Links and whatnot
1. http://millionsongdataset.com/
2. https://cornac.readthedocs.io/en/latest/api_ref/models.html#module-cornac.models.wmf.recom_wmf
3. Note: The paper users 7digital.com as a source for ~30sec audio clips from songs. We have reached out to 7digital to get the same treatment and have yet to hear back. Our backup plan is to use the Free Music Archive Dataset: https://github.com/mdeff/fma. 

