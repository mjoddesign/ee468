'''
CIFAR-10 training data are stored in separate pickle files, each of which is ~ 30 MB, and CIFAR testing data
are stired in one pickle file. This script must do the following:
(1) Load the different pickle files and save them in one .mat file. More specifically, the script combines
RGB images--with resolution 32x32--from each file in one 50Kx3072 dimensional array, and it also combines
class labels from each file into a 50K dimensional vector.
(2) Save the test images and labels in one .mat file (10Kx3072 dimensional image array and 10K class vector)
-----------
Author: Muhammad Alrabeiah
Date: May, 2022
'''
import os
import pickle
import numpy as np
import scipy.io as sio


src_root = 'cifar-10-batches-py/'
dir_cont = os.listdir(src_root)

# Save training data as mat file
# ------------------------------
files = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']
x = np.zeros((50000,3072),dtype=np.float32)
t = np.zeros((50000,),dtype=np.float32)
counter = 0
for file in dir_cont:
    if file in files:
        st = counter*10000
        ed = st+10000
        src_path = os.path.join(src_root, file)
        with open(src_path,'rb') as fo:
            data_dict = pickle.load(fo, encoding='bytes')
            x[st:ed,:] = data_dict[b'data'].astype(np.float32)
            t[st:ed] = np.array(data_dict[b'labels'],dtype=np.float32)
        counter += 1

sio.savemat('training.mat',{'img':x,'cls':t})

# Save testing data as mat file
# -----------------------------
file = 'test_batch'
src_path = os.path.join(src_root, file)
with open(src_path,'rb') as fo:
    data_dict = pickle.load(fo,encoding='bytes')
    x = data_dict[b'data'].astype(np.float32)
    t = np.array(data_dict[b'labels']).astype(np.float32)

sio.savemat('testing.mat',{'img':x, 'cls':t})

