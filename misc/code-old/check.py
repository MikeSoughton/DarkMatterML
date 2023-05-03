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


def split(arr, count):
     return [arr[i::count] for i in range(count)]

n_xbins =29
n_ybins =29


yedges =np.linspace(-4.0, 4.0, num=n_xbins)
#print len(xedges)
xedges =np.linspace(100.0,2000.0, num=n_ybins)
#print len(yedges)

ap,bp,cp,dp=loadtxt("../axion/monojet/monoj.csv", unpack=True)

a0,b0,c0,d0=loadtxt("../spin1/spin1med.csv", unpack=True)

#data processing for process A
#data splitting into (,N) parts

N =10000

app=split(ap,N)
bpp=split(bp,N)

for i in range(1,N+1):
      H, xedges, yedges = np.histogram2d(app[i-1],bpp[i-1],bins=(xedges, yedges))
#      H = H.T #density matrix
      if i==1:
         H1=H
      if i==2:
         H2=H
      if i==3:
         H3=H
      if i==4:
         H4=H
      H = H.reshape((n_xbins-1)*(n_ybins-1)) #1-D array from a matrix
      
#      H /= max(H)
      if i==1:
         Hsave = H
      else:
         Hsave = np.vstack([Hsave,H])
#      print sum(H)
#      print Hsave.shape
B=np.ones((N,1))
Hsavef=np.column_stack((Hsave, B))

print  Hsavef.shape


#Htot1, xedges, yedges = np.histogram2d(app,bpp,bins=(xedges, yedges))

#fig = plt.figure()
#ax = fig.add_subplot(111)
#plt.imshow(Htot1, interpolation='nearest', origin='low')
#plab.savefig('tot1.png',origin='(left, bottom)')
#plt.show()


fig = plt.figure()
ax = fig.add_subplot(111)
plt.imshow(H1, interpolation='nearest', origin='low')
plab.savefig('m40pro1s1.png',origin='(left, bottom)')
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111)
plt.imshow(H3, interpolation='nearest', origin='low')
plab.savefig('m40pro1s2.png',origin='(left, bottom)')
plt.show()


#fig = plt.figure()
#ax = fig.add_subplot(111)
#plt.imshow(H4, interpolation='nearest', origin='low')
#plab.savefig('proa4.png',origin='(left, bottom)')
#plt.show()


#data processing for process B
ap0=split(a0,N)
#print app[0]
bp0=split(b0,N)

for i in range(1,N+1):
      H0, xedges, yedges = np.histogram2d(ap0[i-1],bp0[i-1],bins=(xedges, yedges))
#      H0 = H0.T
      H1=H0
      
      if i==1:
         H1b=H0
      if i==2:
         H2b=H0
      if i==3:
         H3b=H0
      if i==4:
         H4b=H0
      
      H0 = H0.reshape((n_xbins-1)*(n_ybins-1))
      
#      H0 /= max(H0)
      if i==1:
         H0save = H0
      else:
         H0save = np.vstack([H0save,H0])
#      print sum(H)
#      print Hsave.shape
B0=np.zeros((N,1))
H0savef=np.column_stack((H0save, B0))
print  H0savef.shape

#Htot2, xedges, yedges = np.histogram2d(ap0,bp0,bins=(xedges, yedges))

#fig = plt.figure()
#ax = fig.add_subplot(111)
#plt.imshow(htot2, interpolation='nearest', origin='low')
#plab.savefig('tot2.png',origin='(left, bottom)')
#plt.show()


fig = plt.figure()
ax = fig.add_subplot(111)
plt.imshow(H2b, interpolation='nearest', origin='low')
plab.savefig('m40pro2s1.png',origin='(left, bottom)')
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111)
plt.imshow(H3b, interpolation='nearest', origin='low')
plab.savefig('m40pro2s2.png',origin='(left, bottom)')
plt.show()


#fig = plt.figure()
#ax = fig.add_subplot(111)
#plt.imshow(H4b, interpolation='nearest', origin='low')
#plab.savefig('prob4.png',origin='(left, bottom)')
#plt.show()



data=np.vstack([Hsavef,H0savef])
#data /= np.max(data)
print  data.shape



datafinal =np.take(data,np.random.permutation(data.shape[0]),axis=0,out=data)
print datafinal



#Separating out the last coulmn : which is Y_label(1 or 0) 
x_all=data[:,:-1]
x_all /= np.max(x_all)
#x_all /= np.()
y_all=data[:,(n_xbins-1)*(n_ybins-1)]



x_train, x_valtest, y_train, y_valtest = train_test_split(x_all, y_all, test_size=0.8)

#x_train=x_valtest
#y_train=y_valtest
#print y_train



x_test, x_val, y_test, y_val = train_test_split(x_valtest, y_valtest, test_size=0.5)


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
y_val = keras.utils.to_categorical(y_val, num_classes)

# a fairly small network for speed
fcmodel = Sequential()
fcmodel.add(Dense(20, activation='relu', input_shape=((n_xbins-1)*(n_ybins-1),)))
fcmodel.add(Dense(20, activation='relu'))
#fcmodel.add(Dense(20, activation='relu'))
#fcmodel.add(Dense(20, activation='relu'))
#fcmodel.add(Dropout(0.1))
# could include dropout, regularisation, ...
fcmodel.add(Dense(num_classes, activation='softmax'))



fcmodel.summary()


fcmodel.compile(loss='categorical_crossentropy',
                optimizer=RMSprop(),
                metrics=['accuracy'])
                
                
                
batch_size =4000
epochs = 400               
                
#model.fit_generator                
history = fcmodel.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=2,
                      validation_data=(x_test, y_test))                
                
                
score = fcmodel.evaluate(x_test, y_test, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])                
                
                
                
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
    plab.savefig('new1.pdf', bbox_inches=0,dpi=100)
histplot(history)


#Convolutional Neural Network (CNN)

x_train = x_train.reshape(-1,(n_xbins-1), (n_ybins-1), 1)
x_test = x_test.reshape(-1, (n_xbins-1), (n_ybins-1), 1)


xre = x_train.reshape(-1, n_xbins-1, n_ybins-1)


print x_train.shape

hnew1=xre[0,:,:]
hnew2=xre[1,:,:]

fig = plt.figure()
ax = fig.add_subplot(111)
plt.imshow(hnew1, interpolation='nearest', origin='low')
plab.savefig('hnew1.png',origin='(left, bottom)')
plt.show()

 
fig = plt.figure()
ax = fig.add_subplot(111)
plt.imshow(hnew2, interpolation='nearest', origin='low')
plab.savefig('hnew2.png',origin='(left, bottom)')
plt.show()

 
#import sys
#clear = lambda: os.system('cls')
#clear()
 
for name in dir():
    if not name.startswith('_'):
        del globals()[name]

for name in dir():
    if not name.startswith('_'):
        del locals()[name] 
 
#xre = x_train.reshape(-1,(n_xbins-1), (n_ybins-1),1)
#plt.imshow(x_train[0], interpolation='nearest')
#fig = plt.figure()
#ax = fig.add_subplot(111)
#plt.imshow(xre[0], interpolation='nearest', origin='low')
#plab.savefig('afterprob.png',origin='(left, bottom)')
#plt.show()




