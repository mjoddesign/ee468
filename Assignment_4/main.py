'''
EE468: Neural Networks and Deep Learning class.
Fourth assignment--CNN and PyTorch
===============================================
Instructor notes:
-----------------
This script is meant as a guid for students to learn Python, PyTorch, and do
Assignment 4 of the course. It introduces students to PyTorch classes and functions
for the purpose of developing a complete CMM. Students are expected to develop
the CNN architecture as well as prepare the dataset and train the network.
Happy coding... ;-)
============================
Author: Muhammad Alrabeiah
Date: May. 2022
'''
import scipy.io as sio
from torch.utils.data import DataLoader
import torchvision.transforms as trfs
import matplotlib.pyplot as plt

from data_feed import DataFeed
from build_net import CNN
from algo_train import sup_trn

# Experment parameters
# NOTE: "opt_dict" is a dictionary containing all the parameters of the experiment.
# It provides the developer/user easy access to all parameters used by all related
# scripts.
opt_dict = {
    'tag': 'cifar_cnn', # A title for the experiment

    # Dataset
    'trn_root': 'training.mat', # Path to training data
    'val_root': 'testing.mat', # Path to validation data
    'stats_file':'image_stats.mat', # Path to image statistics

    # Network architecture
    'inp_dim': (3,32,32),# Dimensions of input images (channels, height, width)
    'out_dim': 10, # Dimension of output == number of class

    # training parameters
    'gpu': True, # Set to "True" if GPU available. Otherwise set to False
    'gpu_idx': 0, # The GPU number on your system (Only needed with multiple-GPU systems)
    'solver': 'Adam', # Type of training optimizer. Either Adam or SGDM
    'tbs': 4096, # Training batch size
    'vbs': 4096, # Validation batch size
    'num_epochs': 3, # Number of epoch
    'lr': 1e-2, # Learning rate
    'wd': 1e-5, # Weight decay factor
    'momentum': 0.9, # Momentum factor for SGDM
    'lr_sch': [70], # Learning rate schedule; at what epoch the learning is reduced
    'lr_factor': 0.1, # The factor by which lr is reduced
    'coll_cycle': 10, # Collection cycle. Collect training information every "X" of iteration
    'display_freq': 10, # Display training progress on the monitor every "X" iterations
    'val_cycle': 30, # Perform validation every "X" iterations
}

# Load stats
stats = sio.loadmat(opt_dict['stats_file'])
opt_dict['stats'] = stats


# Data loaders
trn_feed = DataFeed(root=opt_dict['trn_root'],
                    trf=True,
                    stats=opt_dict['stats'])

trn_loader = DataLoader(trn_feed,
                        batch_size=opt_dict['tbs'],
                        shuffle=True)
print('Number of training data points is {0:}'.format(len(trn_feed)))
val_feed = DataFeed(root=opt_dict['val_root'], #TODO
                    trf=True, #TODO
                    stats=opt_dict['stats'] )
val_loader = DataLoader(val_feed,
                        batch_size=opt_dict['vbs'],
                        shuffle=True)
print('Number of validation data points is {0:}'.format(len(val_feed)))

# Build network
net = CNN(opt_dict['inp_dim'],opt_dict['out_dim'])#TODO
if opt_dict['gpu']:
    net = net.cuda()

# Train network
opt_dict['trn_size'] = len(trn_feed)
opt_dict['val_size'] = len(val_feed)#TODO
net, trn_info = sup_trn(net,trn_loader,val_loader,opt_dict)

plt.figure(1)
plt.plot(trn_info['trn_itr'], trn_info['trn_acc'], '-ob', label='Training accuracy')
plt.plot(trn_info['val_itr'], trn_info['val_acc'], '-or', label='Validation accuracy')
plt.xlabel('Iteration number')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid()
plt.show()
