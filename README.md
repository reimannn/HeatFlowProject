# HeatFlowProject

HeatFlowProject is a simple regression Neural Network that solves the problem in the accompanying pdf. The gist of the problem is given boundary data the neural network should be able to deduce the total heat transfer in the given body. This is assuming that the heat conductivity coefficient and the body are the same for any testing or prediction data as it is in the training data.

# KFoldFile

This file has the code for training the data using a k-fold technique of segmenting off a section of the data for validation on each run. This helps debug and validate any techniques without involving the test data too early in the model. 

# FullModel

This file has the code for training the model on the network and validating the results against the test data. 

# Usage

There are many adjustable variables (nodes per hidden layer, optimizer function, etc.) in the beginning of the code. Any adjustments outside of changing these choices will require further development on the user's part. 
Much of this code was written while reading “Deep Learning with Python” 2nd edition by Fancois Chollet. It would be well advised to user’s new to Neural Networks to refer to this work before interacting with the model design.

