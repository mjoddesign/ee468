"""
EE468: Neural Networks and Deep Learning class.
Third assignment-- Neural Networks
===============================================
Instructor notes:
-----------------
This script is meant as a guid for students to learn Python and do
Assignment 3 of the course. It introduces students to python classes
for the purpose of developing complete neural networks. NOTE: Only the
construction of a network is discussed here along with the implementation
of the forward pass. Backward pass and network training is deferred to
another assignment.

Happy coding... ;-)
============================
Author: Muhammad Alrabeiah
Date: Mar. 2022
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from build_net import Sequential
from layers import * # Read about the asterisk sign and its role with import


# Load some data
data_file="./assig3_dataset.mat"
D = sio.loadmat(data_file) #TODO: load dataset from .mat file
x_trn = D['trn_inp'] #TODO: read training inputs
y_trn = D['trn_labels'] #TODO: read training labels
x_val = D['val_inp'] #TODO: read validation inputs
y_val =D['val_labels'] #TODO: read validation labels

#TODO: Plot training and validation data in two separate scatter plots
# Example: plt.scatter(x_trn[:, 0], x_trn[:, 1], c=y_trn, s=40, cmap=plt.cm.Spectral)


plt.figure(1)
plt.scatter(x_trn[:, 0], x_trn[:, 1], c=y_trn, s=40, cmap=plt.cm.Spectral)
plt.grid()
plt.xlabel('x1') 
plt.ylabel('x2') 
plt.title('Training data')
plt.show()

plt.figure(2)
plt.scatter(x_val[:, 0], x_val[:, 1], c=y_val, s=40, cmap=plt.cm.Spectral)
plt.grid()
plt.xlabel('x1') 
plt.ylabel('x2') 
plt.title('Validation data')
plt.show()


# Build a neural network
# TODO: use the classes you coded in script "layers.py" to create objects for
#  each layer of the network
fc_1 = Linear(in_dim=2,out_dim=100) #TODO: first layer of linear combiners
relu_1 = ReLU(fc_1) #TODO: first layer of activation functions
fc_2 = Linear(in_dim=100,out_dim=20) #TODO: second layer of linear combiners
relu_2 = ReLU(fc_2)#TODO: second layer of activation functions
fc_3 = Linear(in_dim=20,out_dim=3) #TODO: third layer of linear combiners
sfm = softmax(fc_3) #TODO: softmax layer

layer_list = [fc_1,relu_1,fc_2,relu_2,fc_3,sfm]#TODO: create an ORDERED list of layers

net = Sequential(layer_list)#TODO: use Sequential class to construct an object representing your network

# Load trained parameters
#TODO: here you should do the following:
# 1) Load the trained parameters from the file "trained_parameters.mat".
# 2) Assign those parameters to their corresponding weights and biases in the network object "net".
# Hint: make sure the dimensions of the weights and biases are correct.

S  =  sio.loadmat('trained_parameters.mat')
w_1=S['w_1'].T
w_2=S['w_2']
w_3=S['w_3']
b_1=S['b_1'].T
b_2=S['b_2']
b_3=S['b_3']

j=0
W=[w_1,w_2,w_3]
B=[b_1,b_2,b_3]
for i in range(len(net.layers)):
        if isinstance(net.layers[i], Linear):
            net.layers[i].weights=W[j] 
            net.layers[i].biases=B[j]
            j=j+1
            
t= net.forward(x_val)
t=np.argmax(t,axis=0).T

sum=0
for i in range(len(t)):
     
      if t[i]== y_val[0,i]:
        sum=sum+1
    
        
acc= sum/300

print('Accuracy is = {}%'.format(acc*100) )


plt.figure(3)
plt.scatter(x_val[:, 0], x_val[:, 1], c=t, s=40, cmap=plt.cm.Spectral)
plt.title('Validation data predicted')
plt.grid()
plt.xlabel('x1') 
plt.ylabel('x2')
plt.show()

# Test network on validation dataset
#TODO: Test the network you have created on the validation data by doing
# the following:
# (1) Compute network prediction accuracy (See assignment for the equation)
# and print it on the screen using "print('Accuracy is {}'.format())". HINT:
# Read about using "format" with python print function
# (2) Record all predictions and plot a scatter plot of x_val where the data
# are colored based on their "predicted" labels not the groundtruth labels,
# i.e., "y_val". HINT: Scatter plot should be similar to that in Line 37
