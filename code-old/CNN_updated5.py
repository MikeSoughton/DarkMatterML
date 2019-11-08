#matplotlib inline
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pylab as plab
from scipy import *
from numpy import *
from numpy import ma
from pylab import *
import pylab as plab
import matplotlib.mlab as mlab


import tensorflow as tf

# if multiple GPUs, only use one of them
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# avoid hogging all the GPU memory
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)


# check that we have the devices we expect available
from tensorflow.python.client import device_lib
device_lib.list_local_devices()



# if you have a recent version of tensorflow, keras is included
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split

plt.close("all")

def split(arr, count):
     # syntax of a[x::y] means get every yth element starting at position x
     return [arr[i::count] for i in range(count)]

n_xbins = 20
n_ybins = 40

xedges =np.linspace(-4.0, 4.0, num=n_xbins)
#print len(xedges)
yedges =np.linspace(100.0,2600.0, num=n_ybins) #changed this from 2000 to 2600 as there are still events up there and they are porbably the more important ones for distinguihsing signals
#print len(yedges)

mono_ptj,mono_etaj,mono_phij,mono_signal=loadtxt("monoj.csv", unpack=True,skiprows=1)

DM_ptj,DM_etaj,DM_phij,DM_signal=loadtxt("spin1med.csv", unpack=True,skiprows=1)

#data splitting into (,N) parts
N = 200
Nsmall = 200 # here small refers to number of events, not number of batches

#data processing for ALP process
batched_mono_ptj=split(mono_ptj,N)
batched_mono_etaj=split(mono_etaj,N)

small_batched_mono_ptj=split(mono_ptj,Nsmall)
small_batched_mono_etaj=split(mono_etaj,Nsmall)

# Produce ALP monojet density arrays for training data with large number of events
for i in range(1,N+1):
      H_mono, xedges, yedges = np.histogram2d(batched_mono_etaj[i-1],batched_mono_ptj[i-1],bins=(xedges, yedges))
      H_mono = H_mono.reshape((n_xbins-1)*(n_ybins-1)) #1-D array from a matrix # could this be the line which messes things up?
      if i==1:
         H_mono_save = H_mono
      else:
         H_mono_save = np.vstack([H_mono_save,H_mono])

mono_signal=np.ones((N,1))
H_mono_savef=np.column_stack((H_mono_save, mono_signal))
print  H_mono_savef.shape

# Produce ALP monojet density arrays for test data with small number of events
for i in range(1,Nsmall+1):
      H_small_mono, xedges, yedges = np.histogram2d(small_batched_mono_etaj[i-1],small_batched_mono_ptj[i-1],bins=(xedges, yedges))
      H_small_mono = H_small_mono.reshape((n_xbins-1)*(n_ybins-1)) #1-D array from a matrix # could this be the line which messes things up?
      if i==1:
         H_small_mono_save = H_small_mono
      else:
         H_small_mono_save = np.vstack([H_small_mono_save,H_small_mono])

small_mono_signal=np.ones((Nsmall,1))
H_small_mono_savef=np.column_stack((H_small_mono_save, small_mono_signal))
print  H_small_mono_savef.shape

#data processing for DM process
batched_DM_ptj=split(DM_ptj,N)
batched_DM_etaj=split(DM_etaj,N)

small_batched_DM_ptj=split(DM_ptj,Nsmall)
small_batched_DM_etaj=split(DM_etaj,Nsmall)

# Produce DM density arrays for training data with large number of events
for i in range(1,N+1):
      H_DM, xedges, yedges = np.histogram2d(batched_DM_etaj[i-1],batched_DM_ptj[i-1],bins=(xedges, yedges))
      H_DM = H_DM.reshape((n_xbins-1)*(n_ybins-1)) #1-D array from a matrix # could this be the line which messes things up?
      if i==1:
         H_DM_save = H_DM
      else:
         H_DM_save = np.vstack([H_DM_save,H_DM])

DM_signal=np.zeros((N,1))
H_DM_savef=np.column_stack((H_DM_save, DM_signal))
print  H_DM_savef.shape

# Produce DM density arrays for test data with small number of events
for i in range(1,Nsmall+1):
      H_small_DM, xedges, yedges = np.histogram2d(small_batched_DM_etaj[i-1],small_batched_DM_ptj[i-1],bins=(xedges, yedges))
      H_small_DM = H_small_DM.reshape((n_xbins-1)*(n_ybins-1)) #1-D array from a matrix # could this be the line which messes things up?
      if i==1:
         H_small_DM_save = H_small_DM
      else:
         H_small_DM_save = np.vstack([H_small_DM_save,H_small_DM])

small_DM_signal=np.ones((Nsmall,1))
H_small_DM_savef=np.column_stack((H_small_DM_save, small_DM_signal))
print  H_small_DM_savef.shape



data=np.vstack([H_mono_savef,H_DM_savef])
#data /= np.max(data)
print  data.shape

datafinal = np.take(data,np.random.permutation(data.shape[0]),axis=0,out=data)

small_data = np.vstack([H_small_mono_savef,H_small_DM_savef])

small_datafinal = np.take(small_data,np.random.permutation(small_data.shape[0]),axis=0,out=small_data)



#Separating out the last coulmn : which is Y_label(1 or 0)
x_all=data[:,:-1]
x_all /= np.max(x_all)

small_x_all=small_data[:,:-1]
small_x_all /= np.max(small_x_all)

y_all=data[:,(n_xbins-1)*(n_ybins-1)]
small_y_all=small_data[:,(n_xbins-1)*(n_ybins-1)]

#plt.figure(2)
#plt.imshow(x_all.reshape(-1, n_xbins-1, n_ybins-1)[0,:,:])
#plt.imshow(x_all.reshape(-1, n_xbins-1, n_ybins-1,1)[0,:,:][:,:,0])


x_train, x_valtest, y_train, y_valtest = train_test_split(x_all, y_all, test_size=0.8)
x_test, x_val, y_test, y_val = train_test_split(x_valtest, y_valtest, test_size=0.5)

# this line is sort of redundant as we might as we don't need training for this batch of less events
small_x_train, small_x_valtest, small_y_train, small_y_valtest = train_test_split(small_x_all, small_y_all, test_size=0.8)
small_x_test, small_x_val, small_y_test, small_y_val = train_test_split(small_x_valtest, small_y_valtest, test_size=0.5)

#x_test_small =

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print(x_val.shape[0], 'validation samples')

print x_train

f=open('test.txt','ab')
np.savetxt(f,x_train[0])
f.close()

# convert class vectors to "one-hot" binary class matrices
num_classes = 2
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
small_y_test = keras.utils.to_categorical(small_y_test, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)

# a fairly small network for speed
fcmodel = Sequential()
fcmodel.add(Dense(20, activation='relu', input_shape=((n_xbins-1)*(n_ybins-1),)))
fcmodel.add(Dense(20, activation='relu'))
# could include dropout, regularisation, ...
fcmodel.add(Dense(num_classes, activation='softmax'))

fcmodel.summary()


fcmodel.compile(loss='categorical_crossentropy',
                optimizer=RMSprop(),
                metrics=['accuracy'])



batch_size = 5
epochs = 5


history1 = fcmodel.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=2,
                      validation_data=(x_test, y_test))

history2 = fcmodel.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=2,
                      validation_data=(small_x_test, small_y_test))


score = fcmodel.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

score = fcmodel.evaluate(small_x_test, small_y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



# really we should not look at the score for this set of data
# until we have finished tuning our model
score = fcmodel.evaluate(x_val, y_val, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



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
    #plab.savefig('new1.pdf', bbox_inches=0,dpi=100)
#plt.figure(1)
histplot(history1)

#plt.figure(2)
histplot(history2)
plt.show(block=False)



'''

#Convolutional Neural Network (CNN)

x_train2D = x_train.reshape(-1, n_xbins-1, n_ybins-1, 1)
x_test2D = x_test.reshape(-1, n_xbins-1, n_ybins-1, 1)


# a fairly small network for speed
cnnmodel = Sequential()
cnnmodel.add(Conv2D(8, (3, 3), activation='relu', input_shape=(n_xbins-1,n_ybins-1, 1)))
cnnmodel.add(MaxPooling2D((3, 3)))
cnnmodel.add(Conv2D(16, (3, 3), activation='relu'))
cnnmodel.add(MaxPooling2D((2, 2)))
cnnmodel.add(Flatten())
cnnmodel.add(Dense(num_classes, activation='softmax'))
# could include dropout, regularisation, ...


cnnmodel.compile(loss='categorical_crossentropy',
                optimizer=RMSprop(),
                metrics=['accuracy'])


cnnmodel.summary()


# save weights for reinitialising below
cnnmodel.save_weights('/tmp/cnnmodel_init_weights.tf')


history3 = cnnmodel.fit(x_train2D, y_train,
                       batch_size=batch_size,
                       epochs=1*epochs,
                       verbose=2,
                       validation_data=(x_test2D, y_test))

#histplot(history3)
plt.show(block=False)

'''
'''
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=20.0,
    zoom_range=0.05)

#datagen.fit(x_train)  # only required if normalizing


gen = datagen.flow(x_train, y_train, batch_size=1)


# run this several times to see more augmented examples
i = 3
fig, axarr = plt.subplots(1, 5)
for ax in axarr:
    img = gen[i][0][0, : , :, 0]
    ax.imshow(img, cmap='gray');
    ax.axis('off')
print('label =', gen[i][1][0].argmax())


# Reinitialise model
cnnmodel.load_weights('/tmp/cnnmodel_init_weights.tf')

cnnmodel.compile(loss='categorical_crossentropy',
                 optimizer=RMSprop(),
                 metrics=['accuracy'])


# fits the model on batches with real-time data augmentation:
# the accuracy continues to (slowly) rise, due to the augmentation
history = cnnmodel.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                 epochs=1*epochs,
                                 verbose=2,
                                 validation_data=(x_test, y_test))


histplot(history)
'''
