#!/usr/bin/env python
# coding: utf-8

# # Training simple neural networks on the MNIST dataset using keras

# You can create a suitable conda environment to run this notebook using:
#
#     conda create -n keras_demo python=3 tensorflow matplotlib ipykernel scikit-learn pandas
#
# If you haven't got the latest cuda drivers, you may also need to specify `cudatoolkit=9.0` or similar.

# ### First some common imports...

# In[1]:


#get_ipython().magic(u'matplotlib inline')
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


# ### Set up TensorFlow in a friendly manner

# In[2]:


import tensorflow as tf

# if multiple GPUs, only use one of them
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# avoid hogging all the GPU memory
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)


# In[3]:


# check that we have the devices we expect available
from tensorflow.python.client import device_lib
device_lib.list_local_devices()

plt.close("all")
# ### Import keras bits and pieces

# In[4]:


# if you have a recent version of tensorflow, keras is included
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import RMSprop


# ### We will use a standard demonstration data set: MNIST handwritten digits

# In[5]:


# the data, split between train and test sets
(x_train, y_train), (x_valtest, y_valtest) = mnist.load_data()
print np.shape(x_train)

# select
#y_train = y_train[(np.where(y_train==0)) or (np.where(y_train==1))]
#y_train = y_train[(y_train == 0) or (y_train ==1)]

x_zeros = x_train[(y_train == 0)]
x_ones = x_train[(y_train == 1)]
y_zeros = y_train[y_train == 0]
y_ones = y_train[(y_train == 1)]

x_train = np.concatenate((x_zeros,x_ones),axis=0)
y_train = np.concatenate((y_zeros,y_ones),axis=0)

# We will combine x data with y data so that we shuffle them in the same manners

# This requires x data to be reshaped to 1D first
def process_data(x,nxbins,nybins):
    x = x.reshape(-1, nxbins*nybins)
    x = x.astype('float32')
    #x /= 255
    return x


x_train = process_data(x_train,28,28)

# Now we can shuffle the data, keeping indicies correct
data = np.column_stack((x_train,y_train))
datafinal = np.take(data,np.random.permutation(data.shape[0]),axis=0,out=data)

x_train = data[:,:-1]
y_train = data[:,784]

# A check that the data is sorted correctly:
print y_train[0:5]

# Reshaping only so we can view the image

# Now do the same for the validation data

x_valtest_zeros = x_valtest[(y_valtest == 0)]
x_valtest_ones = x_valtest[(y_valtest == 1)]
y_valtest_zeros = y_valtest[y_valtest == 0]
y_valtest_ones = y_valtest[(y_valtest == 1)]

x_valtest = np.concatenate((x_valtest_zeros,x_valtest_ones),axis=0)
y_valtest = np.concatenate((y_valtest_zeros,y_valtest_ones),axis=0)


x_valtest = process_data(x_valtest,28,28)

val_data = np.column_stack((x_valtest,y_valtest))
val_datafinal = np.take(val_data,np.random.permutation(val_data.shape[0]),axis=0,out=val_data)

x_valtest = val_data[:,:-1]
y_valtest = val_data[:,784]






# Reshape x data back to 1D array
#x_train = process_data(x_train)
#x_valtest = process_data(x_valtest)

# Scale to max pixel value
x_train /= 255
x_valtest /=255

# In[6]:


# demo the effect of noisy data
#x_train = (x_train/5 + np.random.poisson(200, size=x_train.shape)).clip(0, 255)
#x_valtest = (x_valtest/5 + np.random.poisson(200, size=x_valtest.shape)).clip(0, 255)


# In[7]:


# if we wanted to test our model on limited data,
# we could reduce the amount of training data like this...
'''
idx = np.random.choice(len(x_train), size=len(x_train)//100)
x_train = x_train[idx]
y_train = y_train[idx]
'''

# If your data isn't split, or want a validation set, need to do split manually

# In[ ]:


from sklearn.model_selection import train_test_split
x_test, x_val, y_test, y_val = train_test_split(x_valtest, y_valtest, test_size=0.5)

# I want to rebin the data, averaging over neaighboring or neaby pixels to find values for the new bbox_inches

# A function which will do this (requires array to be 2D)
def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
x_val = x_val.reshape(-1, 28, 28, 1)

new_nxbins = 28
new_nybins = 28

reshaped_x_train = np.zeros((x_train.shape[0],new_nxbins,new_nybins,x_train.shape[3]))
for i in range(x_train.shape[0]):
    reshaped_x_train[i,:,:,0] = rebin(x_train[i,:,:,0],(new_nxbins,new_nybins))

reshaped_x_test = np.zeros((x_test.shape[0],new_nxbins,new_nybins,x_test.shape[3]))
for i in range(x_test.shape[0]):
    reshaped_x_test[i,:,:,0] = rebin(x_test[i,:,:,0],(new_nxbins,new_nybins))

reshaped_x_val = np.zeros((x_val.shape[0],new_nxbins,new_nybins,x_val.shape[3]))
for i in range(x_test.shape[0]):
    reshaped_x_val[i,:,:,0] = rebin(x_val[i,:,:,0],(new_nxbins,new_nybins))

Nplots = 4
for j in range(1,Nplots+1):
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    Nsubplots = 10
    for i in range(1,Nsubplots+1):
        ax1 = fig.add_subplot(2, Nsubplots, i)
        ax2 = fig.add_subplot(2, Nsubplots, i+Nsubplots)
        ax1.imshow(x_train.reshape(-1,28,28,1)[j*10 + i-1,:,:,0])
        ax2.imshow(reshaped_x_train.reshape(-1,new_nxbins,new_nybins,1)[j*10 + i-1,:,:,0])

x_train = reshaped_x_train
x_test = reshaped_x_test
x_val = reshaped_x_val




# convert class vectors to "one-hot" binary class matrices
num_classes = 2
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)

x_train = process_data(x_train,new_nxbins,new_nybins)
x_test = process_data(x_test,new_nxbins,new_nybins)
x_val = process_data(x_val,new_nxbins,new_nybins)

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print(x_val.shape[0], 'validation samples')
plt.show(block=False)


# Construct the networt

# a fairly small network for speed
fcmodel = Sequential()
fcmodel.add(Dense(20, activation='relu', input_shape=(new_nxbins*new_nybins,)))
fcmodel.add(Dense(20, activation='relu'))
# could include dropout, regularisation, ...
fcmodel.add(Dense(num_classes, activation='softmax'))


# In[ ]:


fcmodel.summary()


# In[ ]:


fcmodel.compile(loss='categorical_crossentropy',
                optimizer=RMSprop(),
                metrics=['mae'])


# ### Train the network

# In[ ]:


batch_size = 32
epochs = 10


# In[ ]:


history = fcmodel.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=2,
                      validation_data=(x_test, y_test))


# In[ ]:


score = fcmodel.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:


# really we should not look at the score for this set of data
# until we have finished tuning our model
score = fcmodel.evaluate(x_val, y_val, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:


def histplot(history):
    hist = pd.DataFrame(history.history)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    hist.plot(y=['loss', 'val_loss'], ax=ax1)
    min_loss = hist['val_loss'].min()
    ax1.hlines(min_loss, 0, len(hist), linestyle='dotted',
               label='min(val_loss) = {:.3f}'.format(min_loss))
    ax1.legend(loc='upper right')
    hist.plot(y=['acc', 'val_acc'], ax=ax2)
    max_acc = hist['val_acc'].max()
    ax2.hlines(max_acc, 0, len(hist), linestyle='dotted',
               label='max(val_acc) = {:.3f}'.format(max_acc))
    ax2.legend(loc='lower right', fontsize='large')


# In[ ]:


#histplot(history)

plt.show(block=False)
# ## Convolutional Neural Network (CNN)

# ### Reshape the data

# In[ ]:

'''
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)


# ### Construct the network

# In[ ]:


# a fairly small network for speed
cnnmodel = Sequential()
cnnmodel.add(Conv2D(8, (3, 3), activation='relu', input_shape=(28, 28, 1)))
cnnmodel.add(MaxPooling2D((3, 3)))
cnnmodel.add(Conv2D(16, (3, 3), activation='relu'))
cnnmodel.add(MaxPooling2D((2, 2)))
cnnmodel.add(Flatten())
cnnmodel.add(Dense(num_classes, activation='softmax'))
# could include dropout, regularisation, ...


# In[ ]:


cnnmodel.compile(loss='categorical_crossentropy',
                optimizer=RMSprop(),
                metrics=['accuracy'])


# In[ ]:


cnnmodel.summary()


# In[ ]:


# save weights for reinitialising below
cnnmodel.save_weights('/tmp/cnnmodel_init_weights.tf')


# ### Train the network

# In[ ]:


history = cnnmodel.fit(x_train, y_train,
                       batch_size=batch_size,
                       epochs=3*epochs,
                       verbose=2,
                       validation_data=(x_test, y_test))


# In[ ]:


histplot(history)


# ## Online data augmentation

# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=20.0,
    zoom_range=0.05)

#datagen.fit(x_train)  # only required if normalizing


# In[ ]:


gen = datagen.flow(x_train, y_train, batch_size=1)


# In[ ]:


# run this several times to see more augmented examples
i = 3
fig, axarr = plt.subplots(1, 5)
for ax in axarr:
    img = gen[i][0][0, : , :, 0]
    ax.imshow(img, cmap='gray');
    ax.axis('off')
print('label =', gen[i][1][0].argmax())


# In[ ]:


# Reinitialise model
cnnmodel.load_weights('/tmp/cnnmodel_init_weights.tf')


# In[ ]:


cnnmodel.compile(loss='categorical_crossentropy',
                 optimizer=RMSprop(),
                 metrics=['accuracy'])


# In[ ]:


# fits the model on batches with real-time data augmentation:
# the accuracy continues to (slowly) rise, due to the augmentation
history = cnnmodel.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                 epochs=10*epochs,
                                 verbose=2,
                                 validation_data=(x_test, y_test))


# In[ ]:


histplot(history)


# Data augmentation reduced number of misclassifications by half.
'''
