# Project Overview
This project is to build a convolutional neural network with Keras to classify dog breeds. 

The model uses the pre-trained Inception model as a fixed feature extractor, where the last convolutional output of Inception is fed as input to the model. Adding a global average polling layer and a fully connected layer to complete the model. After training the model, it can attain at least 80% accuracy on the test set.
