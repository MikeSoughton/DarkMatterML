#matplotlib inline
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
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

n_xbins = 29
n_ybins = 29

yedges =np.linspace(-4.0, 4.0, num=n_xbins)
#print len(xedges)
xedges =np.linspace(130.0,2000.0, num=n_ybins) #changed this from 2000 to 2600 as there are still events up there and they are porbably the more important ones for distinguihsing signals
#print len(yedges)

BG_ptj1,BG_ptj2,BG_etaj1,BG_etaj2,BG_phij,BG_MET,BG_metphij1,BG_metphij2,BG_signal=np.loadtxt("smnp1dijet.csv", unpack=True)#,skiprows=1
ALP_ptj1,ALP_ptj2,ALP_etaj1,ALP_etaj2,ALP_phij,ALP_MET,ALP_metphij1,ALP_metphij2,ALP_signal=np.loadtxt("alpnp1dijet.csv", unpack=True)#,skiprows=1
EFT_ptj1,EFT_ptj2,EFT_etaj1,EFT_etaj2,EFT_phij,EFT_MET,EFT_metphij1,EFT_metphij2,EFT_signal=np.loadtxt("spin1meddijet.csv", unpack=True)#,skiprows=1
SUSY1_ptj1,SUSY1_ptj2,SUSY1_etaj1,SUSY1_etaj2,SUSY1_phij,SUSY1_MET,SUSY1_metphij1,SUSY1_metphij2,SUSY1_signal=np.loadtxt("susycor1dijet.csv", unpack=True)#,skiprows=1
SUSY2_ptj1,SUSY2_ptj2,SUSY2_etaj1,SUSY2_etaj2,SUSY2_phij,SUSY2_MET,SUSY2_metphij1,SUSY2_metphij2,SUSY2_signal=np.loadtxt("susycor2dijet.csv", unpack=True)#,skiprows=1
SUSY3_ptj1,SUSY3_ptj2,SUSY3_etaj1,SUSY3_etaj2,SUSY3_phij,SUSY3_MET,SUSY3_metphij1,SUSY3_metphij2,SUSY3_signal=np.loadtxt("susycor3dijet.csv", unpack=True)#,skiprows=1

# Split data into N images - this is redundant for the future since we now fix N_images through N_events _per_image instead
# but fixing it here and commenting the line where it is set below will shorten the size of the dataset and save time running code
#N_images = 125

# Calculation to find number of events per image for the total number of events:
N_events = len(BG_ptj1)
#N_events_per_image_total = N_events/N_images

# Now manually define the number of events per image and shrink the size of the data (number of events) accordingly
N_events_per_image = 20

# Set N_images to the max possible value if desired - then the next part of code will not shorten the dataset
N_images = N_events/N_events_per_image

# Reduce the size of the datasets such that N_events is fixed by N_images and N_events_per_image
N_events = N_images*N_events_per_image
'''
idx = np.random.choice(len(BG_ptj1), size=N_events)
BG_ptj = BG_ptj[idx]
BG_etaj = BG_etaj[idx]
ALP_ptj = ALP_ptj[idx]
ALP_etaj = ALP_etaj[idx]
EFT_ptj = EFT_ptj[idx]
EFT_etaj = EFT_etaj[idx]
SUSY1_ptj = SUSY1_ptj[idx]
SUSY1_etaj = SUSY1_etaj[idx]
SUSY2_ptj = SUSY2_ptj[idx]
SUSY2_etaj = SUSY2_etaj[idx]
SUSY3_ptj = SUSY3_ptj[idx]
SUSY3_etaj = SUSY3_etaj[idx]
'''
#data processing for BG process
batched_BG_ptj1=split(BG_ptj1,N_images)
batched_BG_metphij1=split(BG_metphij1,N_images)

# Produce BG BGjet density arrays for training data with large number of events
for i in range(1,N_images+1):
      H_BG, xedges, yedges = np.histogram2d(batched_BG_ptj1[i-1],batched_BG_metphij1[i-1],bins=(xedges, yedges))
      H2 = H_BG
      H_BG = H_BG.reshape((n_xbins-1)*(n_ybins-1)) #1-D array from a matrix # could this be the line which messes things up?
      if i==1:
         H_BG_save = H_BG
      else:
         H_BG_save = np.vstack([H_BG_save,H_BG])

BG_signal=np.zeros((N_images,1))
H_BG_savef=np.column_stack((H_BG_save, BG_signal))
print  H_BG_savef.shape

#data processing for ALP process
batched_ALP_ptj1=split(ALP_ptj1,N_images)
batched_ALP_metphij1=split(ALP_metphij1,N_images)

# Produce ALP ALPjet density arrays for training data with large number of events
for i in range(1,N_images+1):
      H_ALP, xedges, yedges = np.histogram2d(batched_ALP_ptj1[i-1],batched_ALP_metphij1[i-1],bins=(xedges, yedges))
      H2 = H_ALP
      H_ALP = H_ALP.reshape((n_xbins-1)*(n_ybins-1)) #1-D array from a matrix # could this be the line which messes things up?
      if i==1:
         H_ALP_save = H_ALP
      else:
         H_ALP_save = np.vstack([H_ALP_save,H_ALP])

ALP_signal=np.ones((N_images,1))
H_ALP_savef=np.column_stack((H_ALP_save, ALP_signal))
print  H_ALP_savef.shape

#data processing for EFT process
batched_EFT_ptj1=split(EFT_ptj1,N_images)
batched_EFT_metphij1=split(EFT_metphij1,N_images)

# Produce EFT density arrays for training data with large number of events
for i in range(1,N_images+1):
      H_EFT, xedges, yedges = np.histogram2d(batched_EFT_ptj1[i-1],batched_EFT_metphij1[i-1],bins=(xedges, yedges))
      H_EFT = H_EFT.reshape((n_xbins-1)*(n_ybins-1)) #1-D array from a matrix # could this be the line which messes things up?
      if i==1:
         H_EFT_save = H_EFT
      else:
         H_EFT_save = np.vstack([H_EFT_save,H_EFT])

EFT_signal=np.ones((N_images,1))
H_EFT_savef=np.column_stack((H_EFT_save, EFT_signal))

#data processing for SUSY1 process
batched_SUSY1_ptj1=split(SUSY1_ptj1,N_images)
batched_SUSY1_metphij1=split(SUSY1_metphij1,N_images)

# Produce SUSY1 density arrays for training data with large number of events
for i in range(1,N_images+1):
      H_SUSY1, xedges, yedges = np.histogram2d(batched_SUSY1_ptj1[i-1],batched_SUSY1_metphij1[i-1],bins=(xedges, yedges))
      H_SUSY1 = H_SUSY1.reshape((n_xbins-1)*(n_ybins-1)) #1-D array from a matrix # could this be the line which messes things up?
      if i==1:
         H_SUSY1_save = H_SUSY1
      else:
         H_SUSY1_save = np.vstack([H_SUSY1_save,H_SUSY1])

SUSY1_signal=np.ones((N_images,1))
H_SUSY1_savef=np.column_stack((H_SUSY1_save, SUSY1_signal))

#data processing for SUSY2 process
batched_SUSY2_ptj1=split(SUSY2_ptj1,N_images)
batched_SUSY2_metphij1=split(SUSY2_metphij1,N_images)

# Produce SUSY2 density arrays for training data with large number of events
for i in range(1,N_images+1):
      H_SUSY2, xedges, yedges = np.histogram2d(batched_SUSY2_ptj1[i-1],batched_SUSY2_metphij1[i-1],bins=(xedges, yedges))
      H_SUSY2 = H_SUSY2.reshape((n_xbins-1)*(n_ybins-1)) #1-D array from a matrix # could this be the line which messes things up?
      if i==1:
         H_SUSY2_save = H_SUSY2
      else:
         H_SUSY2_save = np.vstack([H_SUSY2_save,H_SUSY2])

SUSY2_signal=np.ones((N_images,1))
H_SUSY2_savef=np.column_stack((H_SUSY2_save, SUSY2_signal))

#data processing for SUSY3 process
batched_SUSY3_ptj1=split(SUSY3_ptj1,N_images)
batched_SUSY3_metphij1=split(SUSY3_metphij1,N_images)

# Produce SUSY3 density arrays for training data with large number of events
for i in range(1,N_images+1):
      H_SUSY3, xedges, yedges = np.histogram2d(batched_SUSY3_ptj1[i-1],batched_SUSY3_metphij1[i-1],bins=(xedges, yedges))
      H_SUSY3 = H_SUSY3.reshape((n_xbins-1)*(n_ybins-1)) #1-D array from a matrix # could this be the line which messes things up?
      if i==1:
         H_SUSY3_save = H_SUSY3
      else:
         H_SUSY3_save = np.vstack([H_SUSY3_save,H_SUSY3])

SUSY3_signal=np.ones((N_images,1))
H_SUSY3_savef=np.column_stack((H_SUSY3_save, SUSY3_signal))

# ========================== WRITE DATA INTO PROPER FORMAT FOR NN ========
#========================================= EFT ===========================

EFTdata=np.vstack([H_BG_savef,H_EFT_savef])
EFTdatafinal = np.take(EFTdata,np.random.permutation(EFTdata.shape[0]),axis=0,out=EFTdata)

#Separating out the last coulmn : which is Y_label(1 or 0)
EFTx_all=EFTdata[:,:-1]
EFTx_all /= np.max(EFTx_all)

EFTy_all=EFTdata[:,(n_xbins-1)*(n_ybins-1)]

EFTx_train, EFTx_valtest, EFTy_train, EFTy_valtest = train_test_split(EFTx_all, EFTy_all, test_size=0.4)
EFTx_test, EFTx_val, EFTy_test, EFTy_val = train_test_split(EFTx_valtest, EFTy_valtest, test_size=0.5)

print(EFTx_train.shape[0], 'EFT train samples')
print(EFTx_test.shape[0], 'EFT test samples')
print(EFTx_val.shape[0], 'EFT validation samples')

# convert class vectors to "one-hot" binary class matrices
num_classes = 2
EFTy_train = keras.utils.to_categorical(EFTy_train, num_classes)
EFTy_test = keras.utils.to_categorical(EFTy_test, num_classes)
EFTy_val = keras.utils.to_categorical(EFTy_val, num_classes)

#========================================= ALP ===========================

ALPdata=np.vstack([H_BG_savef,H_ALP_savef])
ALPdatafinal = np.take(ALPdata,np.random.permutation(ALPdata.shape[0]),axis=0,out=ALPdata)

#Separating out the last coulmn : which is Y_label(1 or 0)
ALPx_all=ALPdata[:,:-1]
ALPx_all /= np.max(ALPx_all)

ALPy_all=ALPdata[:,(n_xbins-1)*(n_ybins-1)]

ALPx_train, ALPx_valtest, ALPy_train, ALPy_valtest = train_test_split(ALPx_all, ALPy_all, test_size=0.4)
ALPx_test, ALPx_val, ALPy_test, ALPy_val = train_test_split(ALPx_valtest, ALPy_valtest, test_size=0.5)

print(ALPx_train.shape[0], 'ALP train samples')
print(ALPx_test.shape[0], 'ALP test samples')
print(ALPx_val.shape[0], 'ALP validation samples')

# convert class vectors to "one-hot" binary class matrices
num_classes = 2
ALPy_train = keras.utils.to_categorical(ALPy_train, num_classes)
ALPy_test = keras.utils.to_categorical(ALPy_test, num_classes)
ALPy_val = keras.utils.to_categorical(ALPy_val, num_classes)

#========================================= SUSY1 ===========================

SUSY1data=np.vstack([H_BG_savef,H_SUSY1_savef])
SUSY1datafinal = np.take(SUSY1data,np.random.permutation(SUSY1data.shape[0]),axis=0,out=SUSY1data)

#Separating out the last coulmn : which is Y_label(1 or 0)
SUSY1x_all=SUSY1data[:,:-1]
SUSY1x_all /= np.max(SUSY1x_all)

SUSY1y_all=SUSY1data[:,(n_xbins-1)*(n_ybins-1)]

SUSY1x_train, SUSY1x_valtest, SUSY1y_train, SUSY1y_valtest = train_test_split(SUSY1x_all, SUSY1y_all, test_size=0.4)
SUSY1x_test, SUSY1x_val, SUSY1y_test, SUSY1y_val = train_test_split(SUSY1x_valtest, SUSY1y_valtest, test_size=0.5)

print(SUSY1x_train.shape[0], 'SUSY1 train samples')
print(SUSY1x_test.shape[0], 'SUSY1 test samples')
print(SUSY1x_val.shape[0], 'SUSY1 validation samples')

# convert class vectors to "one-hot" binary class matrices
num_classes = 2
SUSY1y_train = keras.utils.to_categorical(SUSY1y_train, num_classes)
SUSY1y_test = keras.utils.to_categorical(SUSY1y_test, num_classes)
SUSY1y_val = keras.utils.to_categorical(SUSY1y_val, num_classes)

#========================================= SUSY2 ===========================

SUSY2data=np.vstack([H_BG_savef,H_SUSY2_savef])
SUSY2datafinal = np.take(SUSY2data,np.random.permutation(SUSY2data.shape[0]),axis=0,out=SUSY2data)

#Separating out the last coulmn : which is Y_label(1 or 0)
SUSY2x_all=SUSY2data[:,:-1]
SUSY2x_all /= np.max(SUSY2x_all)

SUSY2y_all=SUSY2data[:,(n_xbins-1)*(n_ybins-1)]

SUSY2x_train, SUSY2x_valtest, SUSY2y_train, SUSY2y_valtest = train_test_split(SUSY2x_all, SUSY2y_all, test_size=0.4)
SUSY2x_test, SUSY2x_val, SUSY2y_test, SUSY2y_val = train_test_split(SUSY2x_valtest, SUSY2y_valtest, test_size=0.5)

print(SUSY2x_train.shape[0], 'SUSY2 train samples')
print(SUSY2x_test.shape[0], 'SUSY2 test samples')
print(SUSY2x_val.shape[0], 'SUSY2 validation samples')

# convert class vectors to "one-hot" binary class matrices
num_classes = 2
SUSY2y_train = keras.utils.to_categorical(SUSY2y_train, num_classes)
SUSY2y_test = keras.utils.to_categorical(SUSY2y_test, num_classes)
SUSY2y_val = keras.utils.to_categorical(SUSY2y_val, num_classes)

#========================================= SUSY3 ===========================

SUSY3data=np.vstack([H_BG_savef,H_SUSY3_savef])
SUSY3datafinal = np.take(SUSY3data,np.random.permutation(SUSY3data.shape[0]),axis=0,out=SUSY3data)

#Separating out the last coulmn : which is Y_label(1 or 0)
SUSY3x_all=SUSY3data[:,:-1]
SUSY3x_all /= np.max(SUSY3x_all)

SUSY3y_all=SUSY3data[:,(n_xbins-1)*(n_ybins-1)]

SUSY3x_train, SUSY3x_valtest, SUSY3y_train, SUSY3y_valtest = train_test_split(SUSY3x_all, SUSY3y_all, test_size=0.4)
SUSY3x_test, SUSY3x_val, SUSY3y_test, SUSY3y_val = train_test_split(SUSY3x_valtest, SUSY3y_valtest, test_size=0.5)

print(SUSY3x_train.shape[0], 'SUSY3 train samples')
print(SUSY3x_test.shape[0], 'SUSY3 test samples')
print(SUSY3x_val.shape[0], 'SUSY3 validation samples')

# convert class vectors to "one-hot" binary class matrices
num_classes = 2
SUSY3y_train = keras.utils.to_categorical(SUSY3y_train, num_classes)
SUSY3y_test = keras.utils.to_categorical(SUSY3y_test, num_classes)
SUSY3y_val = keras.utils.to_categorical(SUSY3y_val, num_classes)



# a fairly small network for speed
EFTfcmodel = Sequential()
EFTfcmodel.add(Dense(20, activation='relu', input_shape=((n_xbins-1)*(n_ybins-1),)))
EFTfcmodel.add(Dense(20, activation='relu'))
# could include dropout, regularisation, ...
EFTfcmodel.add(Dense(num_classes, activation='softmax'))
EFTfcmodel.summary()
EFTfcmodel.compile(loss='binary_crossentropy', # does changing to binary crossentropy change this?
                optimizer=RMSprop(),
                metrics=['accuracy'])

'''
# a fairly small network for speed
fcmodel = Sequential()
fcmodel.add(Dense(20, activation='relu', input_shape=((n_xbins-1)*(n_ybins-1),)))
fcmodel.add(Dropout(0.5))
fcmodel.add(Dense(20, activation='relu'))
fcmodel.add(Dropout(0.5))
fcmodel.add(Dense(20, activation='relu'))
fcmodel.add(Dense(num_classes, activation='softmax'))
fcmodel.summary()
fcmodel.compile(loss='binary_crossentropy', # does changing to binary crossentropy change this?
                optimizer=RMSprop(),
                metrics=['accuracy'])
'''

# a fairly small network for speed
ALPfcmodel = Sequential()
ALPfcmodel.add(Dense(20, activation='relu', input_shape=((n_xbins-1)*(n_ybins-1),)))
ALPfcmodel.add(Dense(20, activation='relu'))
# could include dropout, regularisation, ...
ALPfcmodel.add(Dense(num_classes, activation='softmax'))
ALPfcmodel.summary()
ALPfcmodel.compile(loss='binary_crossentropy', # does changing to binary crossentropy change this?
                optimizer=RMSprop(),
                metrics=['accuracy'])

# a fairly small network for speed
SUSY1fcmodel = Sequential()
SUSY1fcmodel.add(Dense(20, activation='relu', input_shape=((n_xbins-1)*(n_ybins-1),)))
SUSY1fcmodel.add(Dense(20, activation='relu'))
# could include dropout, regularisation, ...
SUSY1fcmodel.add(Dense(num_classes, activation='softmax'))
SUSY1fcmodel.summary()
SUSY1fcmodel.compile(loss='binary_crossentropy', # does changing to binary crossentropy change this?
                optimizer=RMSprop(),
                metrics=['accuracy'])

# a fairly small network for speed
SUSY2fcmodel = Sequential()
SUSY2fcmodel.add(Dense(20, activation='relu', input_shape=((n_xbins-1)*(n_ybins-1),)))
SUSY2fcmodel.add(Dense(20, activation='relu'))
# could include dropout, regularisation, ...
SUSY2fcmodel.add(Dense(num_classes, activation='softmax'))
SUSY2fcmodel.summary()
SUSY2fcmodel.compile(loss='binary_crossentropy', # does changing to binary crossentropy change this?
                optimizer=RMSprop(),
                metrics=['accuracy'])

# a fairly small network for speed
SUSY3fcmodel = Sequential()
SUSY3fcmodel.add(Dense(20, activation='relu', input_shape=((n_xbins-1)*(n_ybins-1),)))
SUSY3fcmodel.add(Dense(20, activation='relu'))
# could include dropout, regularisation, ...
SUSY3fcmodel.add(Dense(num_classes, activation='softmax'))
SUSY3fcmodel.summary()
SUSY3fcmodel.compile(loss='binary_crossentropy', # does changing to binary crossentropy change this?
                optimizer=RMSprop(),
                metrics=['accuracy'])

batch_size = 500
epochs = 300

EFThistory = EFTfcmodel.fit(EFTx_train, EFTy_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=2,
                      validation_data=(EFTx_test, EFTy_test))

EFTscore = EFTfcmodel.evaluate(EFTx_test, EFTy_test, verbose=0)
print('EFT Test loss:', EFTscore[0])
print('EFT Test accuracy:', EFTscore[1]) #now mae

ALPhistory = ALPfcmodel.fit(ALPx_train, ALPy_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=2,
                      validation_data=(ALPx_test, ALPy_test))

ALPscore = ALPfcmodel.evaluate(ALPx_test, ALPy_test, verbose=0)
print('ALP Test loss:', ALPscore[0])
print('ALP Test accuracy:', ALPscore[1]) #now mae

SUSY1history = SUSY1fcmodel.fit(SUSY1x_train, SUSY1y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=2,
                      validation_data=(SUSY1x_test, SUSY1y_test))

SUSY1score = SUSY1fcmodel.evaluate(SUSY1x_test, SUSY1y_test, verbose=0)
print('SUSY1 Test loss:', SUSY1score[0])
print('SUSY1 Test accuracy:', SUSY1score[1]) #now mae

SUSY2history = SUSY2fcmodel.fit(SUSY2x_train, SUSY2y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=2,
                      validation_data=(SUSY2x_test, SUSY2y_test))

SUSY2score = SUSY2fcmodel.evaluate(SUSY2x_test, SUSY2y_test, verbose=0)
print('SUSY2 Test loss:', SUSY2score[0])
print('SUSY2 Test accuracy:', SUSY2score[1]) #now mae

SUSY3history = SUSY3fcmodel.fit(SUSY3x_train, SUSY3y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=2,
                      validation_data=(SUSY3x_test, SUSY3y_test))

SUSY3score = SUSY3fcmodel.evaluate(SUSY3x_test, SUSY3y_test, verbose=0)
print('SUSY3 Test loss:', SUSY3score[0])
print('SUSY3 Test accuracy:', SUSY3score[1]) #now mae

EFTprobs = EFTfcmodel.predict(EFTx_test)
EFTfpr, EFTtpr, EFTthresholds = roc_curve(EFTy_test[:,1], EFTprobs[:,1])
EFTsig_rate = EFTtpr
EFTbg_rate = EFTfpr
EFTroc_auc = auc(EFTbg_rate, EFTsig_rate)

ALPprobs = ALPfcmodel.predict(ALPx_test)
ALPfpr, ALPtpr, ALPthresholds = roc_curve(ALPy_test[:,1], ALPprobs[:,1])
ALPsig_rate = ALPtpr
ALPbg_rate = ALPfpr
ALProc_auc = auc(ALPbg_rate, ALPsig_rate)

SUSY1probs = SUSY1fcmodel.predict(SUSY1x_test)
SUSY1fpr, SUSY1tpr, SUSY1thresholds = roc_curve(SUSY1y_test[:,1], SUSY1probs[:,1])
SUSY1sig_rate = SUSY1tpr
SUSY1bg_rate = SUSY1fpr
SUSY1roc_auc = auc(SUSY1bg_rate, SUSY1sig_rate)

SUSY2probs = SUSY2fcmodel.predict(SUSY2x_test)
SUSY2fpr, SUSY2tpr, SUSY2thresholds = roc_curve(SUSY2y_test[:,1], SUSY2probs[:,1])
SUSY2sig_rate = SUSY2tpr
SUSY2bg_rate = SUSY2fpr
SUSY2roc_auc = auc(SUSY2bg_rate, SUSY2sig_rate)

SUSY3probs = SUSY3fcmodel.predict(SUSY3x_test)
SUSY3fpr, SUSY3tpr, SUSY3thresholds = roc_curve(SUSY3y_test[:,1], SUSY3probs[:,1])
SUSY3sig_rate = SUSY3tpr
SUSY3bg_rate = SUSY3fpr
SUSY3roc_auc = auc(SUSY3bg_rate, SUSY3sig_rate)


plt.figure()
plt.plot(ALPbg_rate, ALPsig_rate,color='red', lw=2, label='ALPs, AUC = %0.2f'%(ALProc_auc))
plt.plot(EFTbg_rate, EFTsig_rate,color='green', lw=2, label='EFT, AUC = %0.2F'%(EFTroc_auc))
plt.plot(SUSY1bg_rate, SUSY1sig_rate,color='orange', lw=2, label='SUSY1, AUC = %0.2f'%(SUSY1roc_auc))
plt.plot(SUSY2bg_rate, SUSY2sig_rate,color='blue', lw=2, label='SUSY2, AUC = %0.2f'%(SUSY2roc_auc))
plt.plot(SUSY3bg_rate, SUSY3sig_rate,color='magenta', lw=2, label='SUSY3, AUC = %0.2f'%(SUSY3roc_auc))
plt.xlabel(r'$\epsilon_{B}$',fontsize=16)
plt.ylabel(r'$\epsilon_{S}$',fontsize=16)
plt.title(r'DNN')
legend1=plt.legend(loc='lower right')#,prop={'size':10}
plt.savefig('2D_DNN_sig_vs_bg_delphes_pt-metphij1_ROC.png')


# really we should not look at the score for this set of data
# until we have finished tuning our model
EFTvalscore = EFTfcmodel.evaluate(EFTx_val, EFTy_val, verbose=0)
print('EFT val Test loss:', EFTvalscore[0])
print('EFT val Test accuracy:', EFTvalscore[1])

# OKAY SO THIS CODE HAS TEST AND VAL THE WRONG WAY ROUND
# I HAVE ONLY CHANGED THIS SO FAR IN THE LEGEND OF THE plots
# BUT IT SHOULD BE CHANGED THROUGHOUT THE CODE
# I AM JUST TOO LAZY TO DO IT NOW

#Convolutional Neural Network (CNN)

EFTx_train = EFTx_train.reshape(-1, n_xbins-1, n_ybins-1, 1)
EFTx_test = EFTx_test.reshape(-1, n_xbins-1, n_ybins-1, 1)

ALPx_train = ALPx_train.reshape(-1, n_xbins-1, n_ybins-1, 1)
ALPx_test = ALPx_test.reshape(-1, n_xbins-1, n_ybins-1, 1)

SUSY1x_train = SUSY1x_train.reshape(-1, n_xbins-1, n_ybins-1, 1)
SUSY1x_test = SUSY1x_test.reshape(-1, n_xbins-1, n_ybins-1, 1)

SUSY2x_train = SUSY2x_train.reshape(-1, n_xbins-1, n_ybins-1, 1)
SUSY2x_test = SUSY2x_test.reshape(-1, n_xbins-1, n_ybins-1, 1)

SUSY3x_train = SUSY3x_train.reshape(-1, n_xbins-1, n_ybins-1, 1)
SUSY3x_test = SUSY3x_test.reshape(-1, n_xbins-1, n_ybins-1, 1)

# a fairly small network for speed
EFTcnnmodel = Sequential()
EFTcnnmodel.add(Conv2D(8, (3, 3), activation='relu', input_shape=(n_xbins-1,n_ybins-1, 1)))
EFTcnnmodel.add(MaxPooling2D((3, 3)))
EFTcnnmodel.add(Conv2D(16, (3, 3), activation='relu'))
EFTcnnmodel.add(MaxPooling2D((2, 2)))
EFTcnnmodel.add(Flatten())
EFTcnnmodel.add(Dense(num_classes, activation='softmax'))
# could include dropout, regularisation, ...


EFTcnnmodel.compile(loss='categorical_crossentropy',
                optimizer=RMSprop(),
                metrics=['accuracy'])
EFTcnnmodel.summary()

CNN_EFThistory = EFTcnnmodel.fit(EFTx_train, EFTy_train,
                       batch_size=batch_size,
                       epochs=1*epochs,
                       verbose=2,
                       validation_data=(EFTx_test, EFTy_test))

EFTscore = EFTcnnmodel.evaluate(EFTx_test, EFTy_test, verbose=0)
print('CNN EFT Test loss:', EFTscore[0])
print('CNN EFT Test accuracy:', EFTscore[1]) #now mae

# a fairly small network for speed
ALPcnnmodel = Sequential()
ALPcnnmodel.add(Conv2D(8, (3, 3), activation='relu', input_shape=(n_xbins-1,n_ybins-1, 1)))
ALPcnnmodel.add(MaxPooling2D((3, 3)))
ALPcnnmodel.add(Conv2D(16, (3, 3), activation='relu'))
ALPcnnmodel.add(MaxPooling2D((2, 2)))
ALPcnnmodel.add(Flatten())
ALPcnnmodel.add(Dense(num_classes, activation='softmax'))
# could include dropout, regularisation, ...


ALPcnnmodel.compile(loss='categorical_crossentropy',
                optimizer=RMSprop(),
                metrics=['accuracy'])
ALPcnnmodel.summary()

CNN_ALPhistory = ALPcnnmodel.fit(ALPx_train, ALPy_train,
                       batch_size=batch_size,
                       epochs=1*epochs,
                       verbose=2,
                       validation_data=(ALPx_test, ALPy_test))

ALPscore = ALPcnnmodel.evaluate(ALPx_test, ALPy_test, verbose=0)
print('CNN ALP Test loss:', ALPscore[0])
print('CNN ALP Test accuracy:', ALPscore[1]) #now mae

# a fairly small network for speed
SUSY1cnnmodel = Sequential()
SUSY1cnnmodel.add(Conv2D(8, (3, 3), activation='relu', input_shape=(n_xbins-1,n_ybins-1, 1)))
SUSY1cnnmodel.add(MaxPooling2D((3, 3)))
SUSY1cnnmodel.add(Conv2D(16, (3, 3), activation='relu'))
SUSY1cnnmodel.add(MaxPooling2D((2, 2)))
SUSY1cnnmodel.add(Flatten())
SUSY1cnnmodel.add(Dense(num_classes, activation='softmax'))
# could include dropout, regularisation, ...


SUSY1cnnmodel.compile(loss='categorical_crossentropy',
                optimizer=RMSprop(),
                metrics=['accuracy'])
SUSY1cnnmodel.summary()

CNN_SUSY1history = SUSY1cnnmodel.fit(SUSY1x_train, SUSY1y_train,
                       batch_size=batch_size,
                       epochs=1*epochs,
                       verbose=2,
                       validation_data=(SUSY1x_test, SUSY1y_test))

SUSY1score = SUSY1cnnmodel.evaluate(SUSY1x_test, SUSY1y_test, verbose=0)
print('CNN SUSY1 Test loss:', SUSY1score[0])
print('CNN SUSY1 Test accuracy:', SUSY1score[1]) #now mae

# a fairly small network for speed
SUSY2cnnmodel = Sequential()
SUSY2cnnmodel.add(Conv2D(8, (3, 3), activation='relu', input_shape=(n_xbins-1,n_ybins-1, 1)))
SUSY2cnnmodel.add(MaxPooling2D((3, 3)))
SUSY2cnnmodel.add(Conv2D(16, (3, 3), activation='relu'))
SUSY2cnnmodel.add(MaxPooling2D((2, 2)))
SUSY2cnnmodel.add(Flatten())
SUSY2cnnmodel.add(Dense(num_classes, activation='softmax'))
# could include dropout, regularisation, ...


SUSY2cnnmodel.compile(loss='categorical_crossentropy',
                optimizer=RMSprop(),
                metrics=['accuracy'])
SUSY2cnnmodel.summary()

CNN_SUSY2history = SUSY2cnnmodel.fit(SUSY2x_train, SUSY2y_train,
                       batch_size=batch_size,
                       epochs=1*epochs,
                       verbose=2,
                       validation_data=(SUSY2x_test, SUSY2y_test))

SUSY2score = SUSY2cnnmodel.evaluate(SUSY2x_test, SUSY2y_test, verbose=0)
print('CNN SUSY2 Test loss:', SUSY2score[0])
print('CNN SUSY2 Test accuracy:', SUSY2score[1]) #now mae

# a fairly small network for speed
SUSY3cnnmodel = Sequential()
SUSY3cnnmodel.add(Conv2D(8, (3, 3), activation='relu', input_shape=(n_xbins-1,n_ybins-1, 1)))
SUSY3cnnmodel.add(MaxPooling2D((3, 3)))
SUSY3cnnmodel.add(Conv2D(16, (3, 3), activation='relu'))
SUSY3cnnmodel.add(MaxPooling2D((2, 2)))
SUSY3cnnmodel.add(Flatten())
SUSY3cnnmodel.add(Dense(num_classes, activation='softmax'))
# could include dropout, regularisation, ...


SUSY3cnnmodel.compile(loss='categorical_crossentropy',
                optimizer=RMSprop(),
                metrics=['accuracy'])
SUSY3cnnmodel.summary()

CNN_SUSY3history = SUSY3cnnmodel.fit(SUSY3x_train, SUSY3y_train,
                       batch_size=batch_size,
                       epochs=1*epochs,
                       verbose=2,
                       validation_data=(SUSY3x_test, SUSY3y_test))

SUSY3score = SUSY3cnnmodel.evaluate(SUSY3x_test, SUSY3y_test, verbose=0)
print('CNN SUSY3 Test loss:', SUSY3score[0])
print('CNN SUSY3 Test accuracy:', SUSY3score[1]) #now mae


EFTprobs = EFTcnnmodel.predict(EFTx_test)
EFTfpr, EFTtpr, EFTthresholds = roc_curve(EFTy_test[:,1], EFTprobs[:,1])
EFTsig_rate = EFTtpr
EFTbg_rate = EFTfpr
EFTroc_auc = auc(EFTbg_rate, EFTsig_rate)

ALPprobs = ALPcnnmodel.predict(ALPx_test)
ALPfpr, ALPtpr, ALPthresholds = roc_curve(ALPy_test[:,1], ALPprobs[:,1])
ALPsig_rate = ALPtpr
ALPbg_rate = ALPfpr
ALProc_auc = auc(ALPbg_rate, ALPsig_rate)

SUSY1probs = SUSY1cnnmodel.predict(SUSY1x_test)
SUSY1fpr, SUSY1tpr, SUSY1thresholds = roc_curve(SUSY1y_test[:,1], SUSY1probs[:,1])
SUSY1sig_rate = SUSY1tpr
SUSY1bg_rate = SUSY1fpr
SUSY1roc_auc = auc(SUSY1bg_rate, SUSY1sig_rate)

SUSY2probs = SUSY2cnnmodel.predict(SUSY2x_test)
SUSY2fpr, SUSY2tpr, SUSY2thresholds = roc_curve(SUSY2y_test[:,1], SUSY2probs[:,1])
SUSY2sig_rate = SUSY2tpr
SUSY2bg_rate = SUSY2fpr
SUSY2roc_auc = auc(SUSY2bg_rate, SUSY2sig_rate)

SUSY3probs = SUSY3cnnmodel.predict(SUSY3x_test)
SUSY3fpr, SUSY3tpr, SUSY3thresholds = roc_curve(SUSY3y_test[:,1], SUSY3probs[:,1])
SUSY3sig_rate = SUSY3tpr
SUSY3bg_rate = SUSY3fpr
SUSY3roc_auc = auc(SUSY3bg_rate, SUSY3sig_rate)


plt.figure()
plt.plot(ALPbg_rate, ALPsig_rate,color='red' ,lw=2, label='ALPs, AUC = %0.2f'%(ALProc_auc))
plt.plot(EFTbg_rate, EFTsig_rate,color='green' , lw=2, label='EFT, AUC = %0.2F'%(EFTroc_auc))
plt.plot(SUSY1bg_rate, SUSY1sig_rate, color='orange' ,lw=2, label='SUSY1, AUC = %0.2f'%(SUSY1roc_auc))
plt.plot(SUSY2bg_rate, SUSY2sig_rate,color='blue' , lw=2, label='SUSY2, AUC = %0.2f'%(SUSY2roc_auc))
plt.plot(SUSY3bg_rate, SUSY3sig_rate,color='magenta' , lw=2, label='SUSY3, AUC = %0.2f'%(SUSY3roc_auc))
plt.xlabel(r'$\epsilon_{B}$',fontsize=16)
plt.ylabel(r'$\epsilon_{S}$',fontsize=16)
plt.title(r'CNN')
legend1=plt.legend(loc='lower right')#,prop={'size':16}
plt.savefig('2D_CNN_sig_vs_bg_delphes_pt-metphij1_ROC.png')


plt.show(block=False)
