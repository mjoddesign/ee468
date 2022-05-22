"""
In this script you will learn about Python classes, so do a little bit of reading on the
following topics before you proceed:
1) What is the "constructor" of a class?
2) What is the difference between the "attributes" and "methods" of a class?
3) How to use the "self" argument to store attributes?
----------------------------------------------------------------------------------------
You need to create three important classes each of which could be viewed as a layer
Before completing the script, please carefully read the notes provided with each class
as well as the assignment.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

#########################

class Linear():
    def __init__(self,
                 in_dim,
                 out_dim,
                 use_bias=True,
                 initializer='xavier'):
        """
        A class implementing the linear combiner of a single layer
        NOTE: the weights and biases of this layer are of the shape
        (number of neurons, dimension of input vector) and (number of neurons,),
        respectively
        :param in_dim: size of the input vector (i.e., number of elements in the input vector)
        :param out_dim: number of linear combiners
        :param use_bias: True--> implements bias addition. False--> no bias implemented
        :param initializer: Type of weight initialization
        """
        self.in_dim = in_dim # This is an attribute of Linear
        self.out_dim =  out_dim #TODO
        self.use_bias = use_bias #TODO
        self.initializer = initializer #TODO

        # Define the weights and biases of the linear layer
        if self.initializer == 'xavier': # Read about Xavier initialization
            xavier = -(1/np.sqrt(out_dim)),(1/np.sqrt(out_dim))
            self.weights=np.random.uniform(low=xavier[0], high=xavier[1], size=out_dim) #TODO: What is the equation for Xavier initialization
        elif  self.initializer == 'normal':
            self.weights = np.random.randn(self.in_dim, self.out_dim)

        if self.use_bias:
            self.biases = np.zeros(600)#TODO: initialize biases to zeros

    def forward(self,x):
        """
        forward pass through a linear combiner layer
        :param x: dimensions are number of samples (batch size) x number of elements
        in an input vector (dimension of input vector)
        :return: linear combiner output
        """
     
        try:
            y =np.matmul(self.weights.T,x.T) + self.biases  #TODO: implement the forward pass such that it has the option of applying the biases or not
           
      
        except:
            y =np.matmul(x.T,self.weights.T) 
            y=y+ self.biases
            y=y.T
         
        return y

    def backward(self,dLdy):
        ValueError('Not implemented yet!')

#########################

class ReLU():
    def __init__(self,
                 dim):
        """
        A class implementing element-wise relu function
        :param dim: number of proceeding linear combiners
        """
        self.num_neurons = dim

    def forward(self,x):
        """
        forward pass through a ReLU layer
        :param x: dimensions are number of samples (batch size) x number of linear combiners
        :return: neuron output
        """
        
        y = np.where(np.asarray(x) > 0, x, 0)
       
     
        
        return y

    def backward(self,dLdy):
        ValueError('Not implemented yet!')

#########################

class softmax():
    def __init__(self, dim):
        """
        A class implementing a stable version of softmax
        :param dim:
        """
        self.dim = dim

    def forward(self,x):
        constant = np.max(x,axis=1)
        x=x.T
        x = x - constant
        x=x.T
        nume = np.exp(x)
       
        deno = np.sum(np.exp(x), axis=0)
        y = nume/deno
       
        return y