# Assignment 4

It introduces students to PyTorch for the purpose of developing complete CNNs. The assignment revolves around developing an image classification
network.

# Instructions

You are expected to develop a CNN and train it on the popular CIFAR dataset ([website](https://www.cs.toronto.edu/~kriz/cifar.html)) in this assignment.
You should start by doing the following:

1) Make sure you download the Python version of the CIFAR-10 dataset.
2) Extract dataset in a folder named "cifar-10-batches-py"--- see section below for data structure
3) Run the script get_data.py to create the .mat files for training and validation data
4) Run the script utils.py to compute and save image sstatistics.

Now, that the dataset is ready, start developing and training your network by making sure you do the following:
1) Develop a Python script for network construction. To do so you need to:
   1) Understand CNNs.
   2) Understand the functions implement by each layer, i.e., convolution, linear combiner, ReLU,...etc.
   3) Complete the provided script template "build_net.py"
2) Develop a data feeding class that could be used to efficiently fetch data pairs for neural network training and validation and batch
them. To do so, you need to:
   1) Understand mini-batch training.
   2) Complete the provide template script "data_feed.py"
3) Develop a supervised training function. To do so, you need to:
   1) Understand how iterative training works.
   2) Understand the role of gradient descent.
   3) Learn a little bit about the types of gradient descent algorithms like SGD and Adam.
   4) Complete the provided class template "algo_train.py"
4) Develop main project script to combine scriptss and get your network trained. Use the provided script "main.py" to do so.


# CIFAR-10 Raw Data Structure

Your assignment directory after extrcating the data should look like this:
```
Assignment_4/
   main.py
   build_net.py
   algo_train.py
   data_feed.py
   get_data.py
   utils.py
   cifar-10-batches-py/
     |
     |_ data_batch_1
     |_ data_batch_2
     |_ data_batch_3
     .
     .
     . 
     etc
```
Notice how the data files are organized inside a directory named "cifar-10-batches-py" inside the parent directory "Assignment_4." Feel free you call your parent directory any name.
