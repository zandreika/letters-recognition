from __future__ import print_function
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import RMSprop, Adam
from keras.optimizers import SGD
from keras.utils import np_utils

from emnist import list_datasets
from emnist import extract_training_samples
from emnist import extract_test_samples

import matplotlib.pyplot as plt

np.random.seed(348)  # for reproducibility

# network and training
NB_EPOCH = 20
BATCH_SIZE = 256
VERBOSE = 1
NB_CLASSES = 26   # number of outputs = number of digits
OPTIMIZER = Adam() # optimizer, explainedin this chapter
N_HIDDEN = 64
VALIDATION_SPLIT=0.2 # how much TRAIN is reserved for VALIDATION
DROPOUT = 0.2


print(list_datasets())
X_train, y_train = extract_training_samples('letters')
print("train shape: ", X_train.shape)
print("train labels: ",y_train.shape)
X_test, y_test = extract_test_samples('letters')
print("test shape: ",X_test.shape)
print("test labels: ",y_test.shape)

#for indexing from 0
y_train = y_train-1
y_test = y_test-1

#X_train is 124800 rows of 28x28 values --> reshaped in 124800 x 784
#X_train is 20800 rows of 28x28 values --> reshaped in 20800 x 784
RESHAPED = X_train.shape[1]*X_train.shape[2]

X_train = X_train.reshape(X_train.shape[0], RESHAPED)
X_test = X_test.reshape(X_test.shape[0], RESHAPED)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalize 
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

X_train = X_train.reshape(X_train.shape[0], 1,28,28)
X_test = X_test.reshape(X_test.shape[0], 1,28,28)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)
in_shape = (1,28,28)
# M_HIDDEN hidden layers
# 26 outputs
# final stage is softmax
model = Sequential()
model.add(Convolution2D(20, kernel_size=3, strides=1, activation = 'relu', input_shape=in_shape))
model.add(MaxPooling2D(pool_size=3, strides=1))
model.add(Convolution2D(30, kernel_size=3, strides=1, activation = 'relu'))
model.add(MaxPooling2D(pool_size=3, strides=1))
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(52, activation = 'relu'))
model.add(Dense(52, activation = 'linear'))
model.add(Dense(NB_CLASSES, activation = 'softmax'))
model.summary()


model.compile(loss='categorical_crossentropy',
              optimizer=OPTIMIZER,
              metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    batch_size=BATCH_SIZE, epochs=NB_EPOCH,
                    verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
print("\nTest score:", score[0])
print('Test accuracy:', score[1])


# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()