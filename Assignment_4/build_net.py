'''
Construct neural network
--------------------------
Author: Muhammad Alrabeiah
Date: May, 2022
'''
import torch.nn as nn


class CNN(nn.Module): # Read about class inheritance
    def __init__(self,
                 inp_dim, # Number of color channels
                 out_dim, # Number of classes
                 ):
        super().__init__() # Read about super in Python classes. Useful reference: https://www.educative.io/edpresso/what-is-super-in-python

        # Conv blocks
        self.conv_b_1 = ConvBlock(
            kernel=5,
            inp_ch= inp_dim[0],
            out_ch=6
            #TODO build first conv stack
        )
        
        
         

        
        self.conv_b_2 = ConvBlock(
            kernel=5,
            inp_ch=6*2,
            out_ch=12
            #TODO build second conv stack
        )

        # Fully-Connected blocks
        self.dense_b_1 = DenseBlock(
            inp=1536, out=84, act_type='relu'
            #TODO build first dense stack
        )
        self.dense_b_2 = DenseBlock(
            inp=84, out= out_dim, act_type='relu'
            #TODO build output dense stack
        )

    def forward(self,x):
        x = self.conv_b_1(x)
        
        x = self.conv_b_2(x)
        x =  x.view(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
#TODO convert tensor from shape (batch, channel, height, width) to (batch, channel*heigh*width)
        x = self.dense_b_1(x)
        x = self.dense_b_2(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self,
                 kernel, # kernel size
                 inp_ch, # number of input channels
                 out_ch # number of output channels
                 ):
        super().__init__()

        self.conv_1 = nn.Conv2d(inp_ch, out_ch, kernel,padding=2) #TODO implement 2D convolution with padding (output has same size a input)
        self.relu_1 = nn.ReLU()#TODO implement relu activation
        self.conv_2 = nn.Conv2d(out_ch, out_ch*2, kernel,padding=2)#TODO implement 2D convolution with padding (output has same size a input)
        self.relu_2 = nn.ReLU()#TODO implement relu activation
        self.max_pool =nn.MaxPool2d(2,2) # TODO implement max-pooling with 2 by 2 kernel such that the output has 1/2 input size at each spatial dimension (i.e., 1/2 width, 1/2 height)

    def forward(self,x):
        #TODO define the forward pass
        x= self.conv_1(x)
      
        x=self.relu_1 (x)
        x= self.conv_2(x)
        x=self.relu_2(x)
        x=self.max_pool(x)
       
        return x


class DenseBlock(nn.Module):
    def __init__(self,
                 inp, # Input dimensions (not including batch size)
                 out, # Output dimensions (not including batch size)
                 act_type='relu'):
        super().__init__()

        self.fc = nn.Linear(inp,out)#TODO implement a linear combiner
        if act_type == 'relu': # Think carefully why if statement is needed here!!
            self.relu = nn.ReLU()#TODO implement a relu activation
        else:
            self.relu = None

    def forward(self,x):
        #TODO define forward pass
        x=self.fc(x)
        x=self.relu(x)
     

        return x

