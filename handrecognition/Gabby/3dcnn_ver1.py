import tensorflow as tf
import tflearn
import numpy as np
from tflearn.layers.conv import conv_3d, max_pool_3d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from random import shuffle

# Defining constants
IMG_SIZE = 54 # 50 x 50 image, for example
TIME_LEN = 10 # 5 time points, for example
LR = 1e-3 # learning rate, for example
CONV_LAYERS = 1
POOL_LAYERS = 1
NO_CHANNELS = 1 # how many input streams i.e. a network with 3 channels could have RGB, depth, and body skeleton streams
VERSION = 1 # to keep track of changes in a particular model
NO_SIGNS = 2 # number of signs to learn - i.e. the alphabet
N_EPOCH = 10 # number of epochs
MODEL_NAME = '3dconv-{}--{}--{}--{}--{}-.model'.format(LR, CONV_LAYERS, POOL_LAYERS, NO_CHANNELS, VERSION)

data = np.load("trainingData.npy")
shuffle(data)
train_data = data[:-10]
test_data = data[-10:]
X = np.array([train_data[i][0] for i in range(0, len(train_data)-1)])
Y = [train_data[i][1] for i in range(0, len(train_data)-1)]

test_X = np.array([test_data[i][0] for i in range(0, len(test_data)-1)])
test_Y = [test_data[i][1] for i in range(0, len(test_data)-1)]

conv_net = input_data(shape = [None, TIME_LEN, IMG_SIZE, IMG_SIZE, NO_CHANNELS], name = 'input')

conv_net = conv_3d(conv_net, 32, 3, strides = 1, padding = 'same', activation = 'relu')

conv_net = max_pool_3d(conv_net, 2, strides = 2)

conv_net = fully_connected(conv_net, 256, activation = 'relu')

conv_net = dropout(conv_net, 0.8)

conv_net = fully_connected(conv_net, NO_SIGNS, activation = 'softmax')

conv_net = regression(conv_net, optimizer = 'adam', learning_rate = LR, loss = 'categorical_crossentropy', name = 'targets')

model = tflearn.DNN(conv_net, tensorboard_dir = 'log')

model.fit({'input':X}, {'targets':Y}, n_epoch = N_EPOCH, validation_set = ({'input':test_X}, {'targets':test_Y}),
          show_metric = True, run_id = MODEL_NAME) 

# to view tensorboard, paste the following into command prompt:
# tensorboard --logdir = foo:C:/.../yourprojectdirectory/log