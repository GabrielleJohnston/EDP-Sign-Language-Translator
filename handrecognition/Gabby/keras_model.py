# 3D CNN in keras

# importing libraries
import tensorflow as tf
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils, generic_utils
from random import shuffle
import keras

# defining parameters
IMG_SIZE = 54 # 54 for a 54 x 54 frame size 
TIME_LEN = 20 # 20 time points, each corresponding to a consecutive frame
LR = 1e-3 # learning rate
CONV_LAYERS = 3
POOL_LAYERS = 2
NO_CHANNELS = 1 # how many input streams i.e. 3 for an RGB image, 1 for a grey image
VERSION = 'motion-keras-1' 
NO_SIGNS = 3 # number of signs to learn
N_EPOCH = 10 # number of epochs
BATCH_SIZE = 64 # number of videos per epoch
MODEL_PATH = '3dconv-{}--{}--{}--{}--{}'.format(LR, CONV_LAYERS, POOL_LAYERS, NO_CHANNELS, VERSION)
MODEL_NAME = '3dconv-{}--{}--{}--{}--{}-.model'.format(LR, CONV_LAYERS, POOL_LAYERS, NO_CHANNELS, VERSION)


# loading in videos
train_data_motion = np.load("trainingData.npy") # videos stored as a numpy array
train_data_motion = list(train_data_motion) # convert numpy array to list 
shuffle(train_data_motion) # rearrange the order of the videos

# reshaping videos to numpy array with dimensions time length x image size x image size x 1
x = np.array([i[0] for i in train_data_motion]).reshape(-1, TIME_LEN, IMG_SIZE, IMG_SIZE, 1)

y = np.array([i[1] for i in train_data_motion]) # convert labels to numpy array

x_train = x[:-98] # all but last 98 videos in training video set
x_test = x[-98:] # last 98 videos in testing video set
y_train = y[:-98] # labels for training video set
y_test = y[-98:] # labels for testing video set

x_train = x_train.astype('float32') # convert pixel values to floats
x_test = x_test.astype('float32')
x_train /= 255 # divide by 255 so all pixel values are between 0 and 1
x_test /= 255


# creating the model
model = Sequential() # model made up of a linear stack of layers

# creating 3D convolution layer, with 32 filters, and a 9 x 11 x 11 filter size (time x height x width)
# filters move across video travelling 1 pixel in each dimension each time they are applied
model.add(Convolution3D(32, (9, 11, 11), padding = 'same', input_shape = (20, 54, 54, 1), strides= (1, 1, 1)))
model.add(Activation('relu')) # performs rectified linear activation 
model.add(Convolution3D(32, (7, 9, 9)))
model.add(Activation('relu'))
model.add(MaxPooling3D(pool_size=(1, 2, 2))) # halves the height and width
model.add(Convolution3D(32, (6, 9, 9)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(512)) # fully connected layer with 512 neurons
model.add(Activation('relu'))
model.add(Dropout(0.5)) # ignores ~ half of the neurons, selected randomly: prevents overfitting
model.add(Dense(3)) # fully connected layer with 3 neurons corresponding to each word
model.add(Activation('softmax'))

# creating callbacks
tbCallBack = keras.callbacks.TensorBoard(log_dir='./log', histogram_freq=1, write_graph=True, write_images=True, 
update_freq = 'batch')
pbCallBack = keras.callbacks.ProgbarLogger(count_mode='samples', stateful_metrics=None)
cpCallBack = keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_loss', verbose=0, save_best_only=False, 
save_weights_only=False, mode='auto', period=1)

# configure model learning 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 


# running the model
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCH, validation_data=(x_test, y_test), shuffle=True, 
callbacks = [tbCallBack, pbCallBack, cpCallBack], verbose = 1)
