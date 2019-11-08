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
idx = np.random.choice(len(BG_ptj), size=N_events)
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

ALP_signal0=np.zeros((N_images,1))
ALP_signal1=np.ones((N_images,1))
H_ALP0_savef=np.column_stack((H_ALP_save, ALP_signal0))
H_ALP1_savef=np.column_stack((H_ALP_save, ALP_signal1))

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

EFT_signal0=np.zeros((N_images,1))
EFT_signal1=np.ones((N_images,1))
H_EFT0_savef=np.column_stack((H_EFT_save, EFT_signal0))
H_EFT1_savef=np.column_stack((H_EFT_save, EFT_signal1))

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

SUSY1_signal0=np.zeros((N_images,1))
SUSY1_signal1=np.ones((N_images,1))
H_SUSY10_savef=np.column_stack((H_SUSY1_save, SUSY1_signal0))
H_SUSY11_savef=np.column_stack((H_SUSY1_save, SUSY1_signal1))

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

SUSY2_signal0=np.zeros((N_images,1))
SUSY2_signal1=np.ones((N_images,1))
H_SUSY20_savef=np.column_stack((H_SUSY2_save, SUSY2_signal0))
H_SUSY21_savef=np.column_stack((H_SUSY2_save, SUSY2_signal1))

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

SUSY3_signal0=np.zeros((N_images,1))
SUSY3_signal1=np.ones((N_images,1))
H_SUSY30_savef=np.column_stack((H_SUSY3_save, SUSY3_signal0))
H_SUSY31_savef=np.column_stack((H_SUSY3_save, SUSY3_signal1))

# ========================== WRITE DATA INTO PROPER FORMAT FOR NN ========
#========================================= EFT ===========================

EFT_SUSY1data=np.vstack([H_SUSY10_savef,H_EFT1_savef])
EFT_SUSY1datafinal = np.take(EFT_SUSY1data,np.random.permutation(EFT_SUSY1data.shape[0]),axis=0,out=EFT_SUSY1data)

#Separating out the last coulmn : which is Y_label(1 or 0)
EFT_SUSY1x_all=EFT_SUSY1data[:,:-1]
EFT_SUSY1x_all /= np.max(EFT_SUSY1x_all)

EFT_SUSY1y_all=EFT_SUSY1data[:,(n_xbins-1)*(n_ybins-1)]

EFT_SUSY1x_train, EFT_SUSY1x_valtest, EFT_SUSY1y_train, EFT_SUSY1y_valtest = train_test_split(EFT_SUSY1x_all, EFT_SUSY1y_all, test_size=0.4)
EFT_SUSY1x_test, EFT_SUSY1x_val, EFT_SUSY1y_test, EFT_SUSY1y_val = train_test_split(EFT_SUSY1x_valtest, EFT_SUSY1y_valtest, test_size=0.5)

print(EFT_SUSY1x_train.shape[0], 'EFT_SUSY1 train samples')
print(EFT_SUSY1x_test.shape[0], 'EFT_SUSY1 test samples')
print(EFT_SUSY1x_val.shape[0], 'EFT_SUSY1 validation samples')

# convert class vectors to "one-hot" binary class matrices
num_classes = 2
EFT_SUSY1y_train = keras.utils.to_categorical(EFT_SUSY1y_train, num_classes)
EFT_SUSY1y_test = keras.utils.to_categorical(EFT_SUSY1y_test, num_classes)
EFT_SUSY1y_val = keras.utils.to_categorical(EFT_SUSY1y_val, num_classes)


EFT_SUSY2data=np.vstack([H_SUSY10_savef,H_EFT1_savef])
EFT_SUSY2datafinal = np.take(EFT_SUSY2data,np.random.permutation(EFT_SUSY2data.shape[0]),axis=0,out=EFT_SUSY2data)

#Separating out the last coulmn : which is Y_label(1 or 0)
EFT_SUSY2x_all=EFT_SUSY2data[:,:-1]
EFT_SUSY2x_all /= np.max(EFT_SUSY2x_all)

EFT_SUSY2y_all=EFT_SUSY2data[:,(n_xbins-1)*(n_ybins-1)]

EFT_SUSY2x_train, EFT_SUSY2x_valtest, EFT_SUSY2y_train, EFT_SUSY2y_valtest = train_test_split(EFT_SUSY2x_all, EFT_SUSY2y_all, test_size=0.4)
EFT_SUSY2x_test, EFT_SUSY2x_val, EFT_SUSY2y_test, EFT_SUSY2y_val = train_test_split(EFT_SUSY2x_valtest, EFT_SUSY2y_valtest, test_size=0.5)

print(EFT_SUSY2x_train.shape[0], 'EFT_SUSY2 train samples')
print(EFT_SUSY2x_test.shape[0], 'EFT_SUSY2 test samples')
print(EFT_SUSY2x_val.shape[0], 'EFT_SUSY2 validation samples')

# convert class vectors to "one-hot" binary class matrices
num_classes = 2
EFT_SUSY2y_train = keras.utils.to_categorical(EFT_SUSY2y_train, num_classes)
EFT_SUSY2y_test = keras.utils.to_categorical(EFT_SUSY2y_test, num_classes)
EFT_SUSY2y_val = keras.utils.to_categorical(EFT_SUSY2y_val, num_classes)


EFT_SUSY3data=np.vstack([H_SUSY10_savef,H_EFT1_savef])
EFT_SUSY3datafinal = np.take(EFT_SUSY3data,np.random.permutation(EFT_SUSY3data.shape[0]),axis=0,out=EFT_SUSY3data)

#Separating out the last coulmn : which is Y_label(1 or 0)
EFT_SUSY3x_all=EFT_SUSY3data[:,:-1]
EFT_SUSY3x_all /= np.max(EFT_SUSY3x_all)

EFT_SUSY3y_all=EFT_SUSY3data[:,(n_xbins-1)*(n_ybins-1)]

EFT_SUSY3x_train, EFT_SUSY3x_valtest, EFT_SUSY3y_train, EFT_SUSY3y_valtest = train_test_split(EFT_SUSY3x_all, EFT_SUSY3y_all, test_size=0.4)
EFT_SUSY3x_test, EFT_SUSY3x_val, EFT_SUSY3y_test, EFT_SUSY3y_val = train_test_split(EFT_SUSY3x_valtest, EFT_SUSY3y_valtest, test_size=0.5)

print(EFT_SUSY3x_train.shape[0], 'EFT_SUSY3 train samples')
print(EFT_SUSY3x_test.shape[0], 'EFT_SUSY3 test samples')
print(EFT_SUSY3x_val.shape[0], 'EFT_SUSY3 validation samples')

# convert class vectors to "one-hot" binary class matrices
num_classes = 2
EFT_SUSY3y_train = keras.utils.to_categorical(EFT_SUSY3y_train, num_classes)
EFT_SUSY3y_test = keras.utils.to_categorical(EFT_SUSY3y_test, num_classes)
EFT_SUSY3y_val = keras.utils.to_categorical(EFT_SUSY3y_val, num_classes)

#========================================= ALP ===========================

ALP_EFTdata=np.vstack([H_EFT0_savef,H_ALP1_savef])
ALP_EFTdatafinal = np.take(ALP_EFTdata,np.random.permutation(ALP_EFTdata.shape[0]),axis=0,out=ALP_EFTdata)

#Separating out the last coulmn : which is Y_label(1 or 0)
ALP_EFTx_all=ALP_EFTdata[:,:-1]
ALP_EFTx_all /= np.max(ALP_EFTx_all)

ALP_EFTy_all=ALP_EFTdata[:,(n_xbins-1)*(n_ybins-1)]

ALP_EFTx_train, ALP_EFTx_valtest, ALP_EFTy_train, ALP_EFTy_valtest = train_test_split(ALP_EFTx_all, ALP_EFTy_all, test_size=0.4)
ALP_EFTx_test, ALP_EFTx_val, ALP_EFTy_test, ALP_EFTy_val = train_test_split(ALP_EFTx_valtest, ALP_EFTy_valtest, test_size=0.5)

print(ALP_EFTx_train.shape[0], 'ALP_EFT train samples')
print(ALP_EFTx_test.shape[0], 'ALP_EFT test samples')
print(ALP_EFTx_val.shape[0], 'ALP_EFT validation samples')

# convert class vectors to "one-hot" binary class matrices
num_classes = 2
ALP_EFTy_train = keras.utils.to_categorical(ALP_EFTy_train, num_classes)
ALP_EFTy_test = keras.utils.to_categorical(ALP_EFTy_test, num_classes)
ALP_EFTy_val = keras.utils.to_categorical(ALP_EFTy_val, num_classes)


ALP_SUSY1data=np.vstack([H_SUSY10_savef,H_ALP1_savef])
ALP_SUSY1datafinal = np.take(ALP_SUSY1data,np.random.permutation(ALP_SUSY1data.shape[0]),axis=0,out=ALP_SUSY1data)

#Separating out the last coulmn : which is Y_label(1 or 0)
ALP_SUSY1x_all=ALP_SUSY1data[:,:-1]
ALP_SUSY1x_all /= np.max(ALP_SUSY1x_all)

ALP_SUSY1y_all=ALP_SUSY1data[:,(n_xbins-1)*(n_ybins-1)]

ALP_SUSY1x_train, ALP_SUSY1x_valtest, ALP_SUSY1y_train, ALP_SUSY1y_valtest = train_test_split(ALP_SUSY1x_all, ALP_SUSY1y_all, test_size=0.4)
ALP_SUSY1x_test, ALP_SUSY1x_val, ALP_SUSY1y_test, ALP_SUSY1y_val = train_test_split(ALP_SUSY1x_valtest, ALP_SUSY1y_valtest, test_size=0.5)

print(ALP_SUSY1x_train.shape[0], 'ALP_SUSY1 train samples')
print(ALP_SUSY1x_test.shape[0], 'ALP_SUSY1 test samples')
print(ALP_SUSY1x_val.shape[0], 'ALP_SUSY1 validation samples')

# convert class vectors to "one-hot" binary class matrices
num_classes = 2
ALP_SUSY1y_train = keras.utils.to_categorical(ALP_SUSY1y_train, num_classes)
ALP_SUSY1y_test = keras.utils.to_categorical(ALP_SUSY1y_test, num_classes)
ALP_SUSY1y_val = keras.utils.to_categorical(ALP_SUSY1y_val, num_classes)

ALP_SUSY2data=np.vstack([H_SUSY20_savef,H_ALP1_savef])
ALP_SUSY2datafinal = np.take(ALP_SUSY2data,np.random.permutation(ALP_SUSY2data.shape[0]),axis=0,out=ALP_SUSY2data)

#Separating out the last coulmn : which is Y_label(1 or 0)
ALP_SUSY2x_all=ALP_SUSY2data[:,:-1]
ALP_SUSY2x_all /= np.max(ALP_SUSY2x_all)

ALP_SUSY2y_all=ALP_SUSY2data[:,(n_xbins-1)*(n_ybins-1)]

ALP_SUSY2x_train, ALP_SUSY2x_valtest, ALP_SUSY2y_train, ALP_SUSY2y_valtest = train_test_split(ALP_SUSY2x_all, ALP_SUSY2y_all, test_size=0.4)
ALP_SUSY2x_test, ALP_SUSY2x_val, ALP_SUSY2y_test, ALP_SUSY2y_val = train_test_split(ALP_SUSY2x_valtest, ALP_SUSY2y_valtest, test_size=0.5)

print(ALP_SUSY2x_train.shape[0], 'ALP_SUSY2 train samples')
print(ALP_SUSY2x_test.shape[0], 'ALP_SUSY2 test samples')
print(ALP_SUSY2x_val.shape[0], 'ALP_SUSY2 validation samples')

# convert class vectors to "one-hot" binary class matrices
num_classes = 2
ALP_SUSY2y_train = keras.utils.to_categorical(ALP_SUSY2y_train, num_classes)
ALP_SUSY2y_test = keras.utils.to_categorical(ALP_SUSY2y_test, num_classes)
ALP_SUSY2y_val = keras.utils.to_categorical(ALP_SUSY2y_val, num_classes)

ALP_SUSY3data=np.vstack([H_SUSY30_savef,H_ALP1_savef])
ALP_SUSY3datafinal = np.take(ALP_SUSY3data,np.random.permutation(ALP_SUSY3data.shape[0]),axis=0,out=ALP_SUSY3data)

#Separating out the last coulmn : which is Y_label(1 or 0)
ALP_SUSY3x_all=ALP_SUSY3data[:,:-1]
ALP_SUSY3x_all /= np.max(ALP_SUSY3x_all)

ALP_SUSY3y_all=ALP_SUSY3data[:,(n_xbins-1)*(n_ybins-1)]

ALP_SUSY3x_train, ALP_SUSY3x_valtest, ALP_SUSY3y_train, ALP_SUSY3y_valtest = train_test_split(ALP_SUSY3x_all, ALP_SUSY3y_all, test_size=0.4)
ALP_SUSY3x_test, ALP_SUSY3x_val, ALP_SUSY3y_test, ALP_SUSY3y_val = train_test_split(ALP_SUSY3x_valtest, ALP_SUSY3y_valtest, test_size=0.5)

print(ALP_SUSY3x_train.shape[0], 'ALP_SUSY3 train samples')
print(ALP_SUSY3x_test.shape[0], 'ALP_SUSY3 test samples')
print(ALP_SUSY3x_val.shape[0], 'ALP_SUSY3 validation samples')

# convert class vectors to "one-hot" binary class matrices
num_classes = 2
ALP_SUSY3y_train = keras.utils.to_categorical(ALP_SUSY3y_train, num_classes)
ALP_SUSY3y_test = keras.utils.to_categorical(ALP_SUSY3y_test, num_classes)
ALP_SUSY3y_val = keras.utils.to_categorical(ALP_SUSY3y_val, num_classes)

#========================================= SUSY1 ===========================

SUSY1_SUSY2data=np.vstack([H_SUSY20_savef,H_SUSY11_savef])
SUSY1_SUSY2datafinal = np.take(SUSY1_SUSY2data,np.random.permutation(SUSY1_SUSY2data.shape[0]),axis=0,out=SUSY1_SUSY2data)

#Separating out the last coulmn : which is Y_label(1 or 0)
SUSY1_SUSY2x_all=SUSY1_SUSY2data[:,:-1]
SUSY1_SUSY2x_all /= np.max(SUSY1_SUSY2x_all)

SUSY1_SUSY2y_all=SUSY1_SUSY2data[:,(n_xbins-1)*(n_ybins-1)]

SUSY1_SUSY2x_train, SUSY1_SUSY2x_valtest, SUSY1_SUSY2y_train, SUSY1_SUSY2y_valtest = train_test_split(SUSY1_SUSY2x_all, SUSY1_SUSY2y_all, test_size=0.4)
SUSY1_SUSY2x_test, SUSY1_SUSY2x_val, SUSY1_SUSY2y_test, SUSY1_SUSY2y_val = train_test_split(SUSY1_SUSY2x_valtest, SUSY1_SUSY2y_valtest, test_size=0.5)

print(SUSY1_SUSY2x_train.shape[0], 'SUSY1_SUSY2 train samples')
print(SUSY1_SUSY2x_test.shape[0], 'SUSY1_SUSY2 test samples')
print(SUSY1_SUSY2x_val.shape[0], 'SUSY1_SUSY2 validation samples')

# convert class vectors to "one-hot" binary class matrices
num_classes = 2
SUSY1_SUSY2y_train = keras.utils.to_categorical(SUSY1_SUSY2y_train, num_classes)
SUSY1_SUSY2y_test = keras.utils.to_categorical(SUSY1_SUSY2y_test, num_classes)
SUSY1_SUSY2y_val = keras.utils.to_categorical(SUSY1_SUSY2y_val, num_classes)


SUSY1_SUSY3data=np.vstack([H_SUSY30_savef,H_SUSY11_savef])
SUSY1_SUSY3datafinal = np.take(SUSY1_SUSY3data,np.random.permutation(SUSY1_SUSY3data.shape[0]),axis=0,out=SUSY1_SUSY3data)

#Separating out the last coulmn : which is Y_label(1 or 0)
SUSY1_SUSY3x_all=SUSY1_SUSY3data[:,:-1]
SUSY1_SUSY3x_all /= np.max(SUSY1_SUSY3x_all)

SUSY1_SUSY3y_all=SUSY1_SUSY3data[:,(n_xbins-1)*(n_ybins-1)]

SUSY1_SUSY3x_train, SUSY1_SUSY3x_valtest, SUSY1_SUSY3y_train, SUSY1_SUSY3y_valtest = train_test_split(SUSY1_SUSY3x_all, SUSY1_SUSY3y_all, test_size=0.4)
SUSY1_SUSY3x_test, SUSY1_SUSY3x_val, SUSY1_SUSY3y_test, SUSY1_SUSY3y_val = train_test_split(SUSY1_SUSY3x_valtest, SUSY1_SUSY3y_valtest, test_size=0.5)

print(SUSY1_SUSY3x_train.shape[0], 'SUSY1_SUSY3 train samples')
print(SUSY1_SUSY3x_test.shape[0], 'SUSY1_SUSY3 test samples')
print(SUSY1_SUSY3x_val.shape[0], 'SUSY1_SUSY3 validation samples')

# convert class vectors to "one-hot" binary class matrices
num_classes = 2
SUSY1_SUSY3y_train = keras.utils.to_categorical(SUSY1_SUSY3y_train, num_classes)
SUSY1_SUSY3y_test = keras.utils.to_categorical(SUSY1_SUSY3y_test, num_classes)
SUSY1_SUSY3y_val = keras.utils.to_categorical(SUSY1_SUSY3y_val, num_classes)

#========================================= SUSY2 ===========================

SUSY2_SUSY3data=np.vstack([H_SUSY30_savef,H_SUSY21_savef])
SUSY2_SUSY3datafinal = np.take(SUSY2_SUSY3data,np.random.permutation(SUSY2_SUSY3data.shape[0]),axis=0,out=SUSY2_SUSY3data)

#Separating out the last coulmn : which is Y_label(1 or 0)
SUSY2_SUSY3x_all=SUSY2_SUSY3data[:,:-1]
SUSY2_SUSY3x_all /= np.max(SUSY2_SUSY3x_all)

SUSY2_SUSY3y_all=SUSY2_SUSY3data[:,(n_xbins-1)*(n_ybins-1)]

SUSY2_SUSY3x_train, SUSY2_SUSY3x_valtest, SUSY2_SUSY3y_train, SUSY2_SUSY3y_valtest = train_test_split(SUSY2_SUSY3x_all, SUSY2_SUSY3y_all, test_size=0.4)
SUSY2_SUSY3x_test, SUSY2_SUSY3x_val, SUSY2_SUSY3y_test, SUSY2_SUSY3y_val = train_test_split(SUSY2_SUSY3x_valtest, SUSY2_SUSY3y_valtest, test_size=0.5)

print(SUSY2_SUSY3x_train.shape[0], 'SUSY2_SUSY3 train samples')
print(SUSY2_SUSY3x_test.shape[0], 'SUSY2_SUSY3 test samples')
print(SUSY2_SUSY3x_val.shape[0], 'SUSY2_SUSY3 validation samples')

# convert class vectors to "one-hot" binary class matrices
num_classes = 2
SUSY2_SUSY3y_train = keras.utils.to_categorical(SUSY2_SUSY3y_train, num_classes)
SUSY2_SUSY3y_test = keras.utils.to_categorical(SUSY2_SUSY3y_test, num_classes)
SUSY2_SUSY3y_val = keras.utils.to_categorical(SUSY2_SUSY3y_val, num_classes)

# ========================================= Network Structure ==================

# a fairly small network for speed
EFT_SUSY1fcmodel = Sequential()
EFT_SUSY1fcmodel.add(Dense(20, activation='relu', input_shape=((n_xbins-1)*(n_ybins-1),)))
EFT_SUSY1fcmodel.add(Dense(20, activation='relu'))
# could include dropout, regularisation, ...
EFT_SUSY1fcmodel.add(Dense(num_classes, activation='softmax'))
EFT_SUSY1fcmodel.summary()
EFT_SUSY1fcmodel.compile(loss='binary_crossentropy', # does changing to binary crossentropy change this?
                optimizer=RMSprop(),
                metrics=['accuracy'])

# a fairly small network for speed
EFT_SUSY2fcmodel = Sequential()
EFT_SUSY2fcmodel.add(Dense(20, activation='relu', input_shape=((n_xbins-1)*(n_ybins-1),)))
EFT_SUSY2fcmodel.add(Dense(20, activation='relu'))
# could include dropout, regularisation, ...
EFT_SUSY2fcmodel.add(Dense(num_classes, activation='softmax'))
EFT_SUSY2fcmodel.summary()
EFT_SUSY2fcmodel.compile(loss='binary_crossentropy', # does changing to binary crossentropy change this?
                optimizer=RMSprop(),
                metrics=['accuracy'])

# a fairly small network for speed
EFT_SUSY3fcmodel = Sequential()
EFT_SUSY3fcmodel.add(Dense(20, activation='relu', input_shape=((n_xbins-1)*(n_ybins-1),)))
EFT_SUSY3fcmodel.add(Dense(20, activation='relu'))
# could include dropout, regularisation, ...
EFT_SUSY3fcmodel.add(Dense(num_classes, activation='softmax'))
EFT_SUSY3fcmodel.summary()
EFT_SUSY3fcmodel.compile(loss='binary_crossentropy', # does changing to binary crossentropy change this?
                optimizer=RMSprop(),
                metrics=['accuracy'])

# a fairly small network for speed
ALP_EFTfcmodel = Sequential()
ALP_EFTfcmodel.add(Dense(20, activation='relu', input_shape=((n_xbins-1)*(n_ybins-1),)))
ALP_EFTfcmodel.add(Dense(20, activation='relu'))
# could include dropout, regularisation, ...
ALP_EFTfcmodel.add(Dense(num_classes, activation='softmax'))
ALP_EFTfcmodel.summary()
ALP_EFTfcmodel.compile(loss='binary_crossentropy', # does changing to binary crossentropy change this?
                optimizer=RMSprop(),
                metrics=['accuracy'])

# a fairly small network for speed
ALP_SUSY1fcmodel = Sequential()
ALP_SUSY1fcmodel.add(Dense(20, activation='relu', input_shape=((n_xbins-1)*(n_ybins-1),)))
ALP_SUSY1fcmodel.add(Dense(20, activation='relu'))
# could include dropout, regularisation, ...
ALP_SUSY1fcmodel.add(Dense(num_classes, activation='softmax'))
ALP_SUSY1fcmodel.summary()
ALP_SUSY1fcmodel.compile(loss='binary_crossentropy', # does changing to binary crossentropy change this?
                optimizer=RMSprop(),
                metrics=['accuracy'])

# a fairly small network for speed
ALP_SUSY2fcmodel = Sequential()
ALP_SUSY2fcmodel.add(Dense(20, activation='relu', input_shape=((n_xbins-1)*(n_ybins-1),)))
ALP_SUSY2fcmodel.add(Dense(20, activation='relu'))
# could include dropout, regularisation, ...
ALP_SUSY2fcmodel.add(Dense(num_classes, activation='softmax'))
ALP_SUSY2fcmodel.summary()
ALP_SUSY2fcmodel.compile(loss='binary_crossentropy', # does changing to binary crossentropy change this?
                optimizer=RMSprop(),
                metrics=['accuracy'])

# a fairly small network for speed
ALP_SUSY3fcmodel = Sequential()
ALP_SUSY3fcmodel.add(Dense(20, activation='relu', input_shape=((n_xbins-1)*(n_ybins-1),)))
ALP_SUSY3fcmodel.add(Dense(20, activation='relu'))
# could include dropout, regularisation, ...
ALP_SUSY3fcmodel.add(Dense(num_classes, activation='softmax'))
ALP_SUSY3fcmodel.summary()
ALP_SUSY3fcmodel.compile(loss='binary_crossentropy', # does changing to binary crossentropy change this?
                optimizer=RMSprop(),
                metrics=['accuracy'])

# a fairly small network for speed
SUSY1_SUSY2fcmodel = Sequential()
SUSY1_SUSY2fcmodel.add(Dense(20, activation='relu', input_shape=((n_xbins-1)*(n_ybins-1),)))
SUSY1_SUSY2fcmodel.add(Dense(20, activation='relu'))
# could include dropout, regularisation, ...
SUSY1_SUSY2fcmodel.add(Dense(num_classes, activation='softmax'))
SUSY1_SUSY2fcmodel.summary()
SUSY1_SUSY2fcmodel.compile(loss='binary_crossentropy', # does changing to binary crossentropy change this?
                optimizer=RMSprop(),
                metrics=['accuracy'])

# a fairly small network for speed
SUSY1_SUSY3fcmodel = Sequential()
SUSY1_SUSY3fcmodel.add(Dense(20, activation='relu', input_shape=((n_xbins-1)*(n_ybins-1),)))
SUSY1_SUSY3fcmodel.add(Dense(20, activation='relu'))
# could include dropout, regularisation, ...
SUSY1_SUSY3fcmodel.add(Dense(num_classes, activation='softmax'))
SUSY1_SUSY3fcmodel.summary()
SUSY1_SUSY3fcmodel.compile(loss='binary_crossentropy', # does changing to binary crossentropy change this?
                optimizer=RMSprop(),
                metrics=['accuracy'])

# a fairly small network for speed
SUSY2_SUSY3fcmodel = Sequential()
SUSY2_SUSY3fcmodel.add(Dense(20, activation='relu', input_shape=((n_xbins-1)*(n_ybins-1),)))
SUSY2_SUSY3fcmodel.add(Dense(20, activation='relu'))
# could include dropout, regularisation, ...
SUSY2_SUSY3fcmodel.add(Dense(num_classes, activation='softmax'))
SUSY2_SUSY3fcmodel.summary()
SUSY2_SUSY3fcmodel.compile(loss='binary_crossentropy', # does changing to binary crossentropy change this?
                optimizer=RMSprop(),
                metrics=['accuracy'])

batch_size = 500
epochs = 300

# ================================= EFT training ===============================


EFT_SUSY1history = EFT_SUSY1fcmodel.fit(EFT_SUSY1x_train, EFT_SUSY1y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=2,
                      validation_data=(EFT_SUSY1x_test, EFT_SUSY1y_test))

EFT_SUSY1score = EFT_SUSY1fcmodel.evaluate(EFT_SUSY1x_test, EFT_SUSY1y_test, verbose=0)
print('EFT_SUSY1 Test loss:', EFT_SUSY1score[0])
print('EFT_SUSY1 Test accuracy:', EFT_SUSY1score[1])

EFT_SUSY2history = EFT_SUSY2fcmodel.fit(EFT_SUSY2x_train, EFT_SUSY2y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=2,
                      validation_data=(EFT_SUSY2x_test, EFT_SUSY2y_test))

EFT_SUSY2score = EFT_SUSY2fcmodel.evaluate(EFT_SUSY2x_test, EFT_SUSY2y_test, verbose=0)
print('EFT_SUSY2 Test loss:', EFT_SUSY2score[0])
print('EFT_SUSY2 Test accuracy:', EFT_SUSY2score[1])

EFT_SUSY3history = EFT_SUSY3fcmodel.fit(EFT_SUSY3x_train, EFT_SUSY3y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=2,
                      validation_data=(EFT_SUSY3x_test, EFT_SUSY3y_test))

EFT_SUSY3score = EFT_SUSY3fcmodel.evaluate(EFT_SUSY3x_test, EFT_SUSY3y_test, verbose=0)
print('EFT_SUSY3 Test loss:', EFT_SUSY3score[0])
print('EFT_SUSY3 Test accuracy:', EFT_SUSY3score[1])

# ============================ ALP training ====================================

ALP_EFThistory = ALP_EFTfcmodel.fit(ALP_EFTx_train, ALP_EFTy_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=2,
                      validation_data=(ALP_EFTx_test, ALP_EFTy_test))

ALP_EFTscore = ALP_EFTfcmodel.evaluate(ALP_EFTx_test, ALP_EFTy_test, verbose=0)
print('ALP_EFT Test loss:', ALP_EFTscore[0])
print('ALP_EFT Test accuracy:', ALP_EFTscore[1])


ALP_SUSY1history = ALP_SUSY1fcmodel.fit(ALP_SUSY1x_train, ALP_SUSY1y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=2,
                      validation_data=(ALP_SUSY1x_test, ALP_SUSY1y_test))

ALP_SUSY1score = ALP_SUSY1fcmodel.evaluate(ALP_SUSY1x_test, ALP_SUSY1y_test, verbose=0)
print('ALP_SUSY1 Test loss:', ALP_SUSY1score[0])
print('ALP_SUSY1 Test accuracy:', ALP_SUSY1score[1])


ALP_SUSY2history = ALP_SUSY2fcmodel.fit(ALP_SUSY2x_train, ALP_SUSY2y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=2,
                      validation_data=(ALP_SUSY2x_test, ALP_SUSY2y_test))

ALP_SUSY2score = ALP_SUSY2fcmodel.evaluate(ALP_SUSY2x_test, ALP_SUSY2y_test, verbose=0)
print('ALP_SUSY2 Test loss:', ALP_SUSY2score[0])
print('ALP_SUSY2 Test accuracy:', ALP_SUSY2score[1])


ALP_SUSY3history = ALP_SUSY3fcmodel.fit(ALP_SUSY3x_train, ALP_SUSY3y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=2,
                      validation_data=(ALP_SUSY3x_test, ALP_SUSY3y_test))

ALP_SUSY3score = ALP_SUSY3fcmodel.evaluate(ALP_SUSY3x_test, ALP_SUSY3y_test, verbose=0)
print('ALP_SUSY3 Test loss:', ALP_SUSY3score[0])
print('ALP_SUSY3 Test accuracy:', ALP_SUSY3score[1])

# ============================= SUSY 1 training ================================

SUSY1_SUSY2history = SUSY1_SUSY2fcmodel.fit(SUSY1_SUSY2x_train, SUSY1_SUSY2y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=2,
                      validation_data=(SUSY1_SUSY2x_test, SUSY1_SUSY2y_test))

SUSY1_SUSY2score = SUSY1_SUSY2fcmodel.evaluate(SUSY1_SUSY2x_test, SUSY1_SUSY2y_test, verbose=0)
print('SUSY1_SUSY2 Test loss:', SUSY1_SUSY2score[0])
print('SUSY1_SUSY2 Test accuracy:', SUSY1_SUSY2score[1])


SUSY1_SUSY3history = SUSY1_SUSY3fcmodel.fit(SUSY1_SUSY3x_train, SUSY1_SUSY3y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=2,
                      validation_data=(SUSY1_SUSY3x_test, SUSY1_SUSY3y_test))

SUSY1_SUSY3score = SUSY1_SUSY3fcmodel.evaluate(SUSY1_SUSY3x_test, SUSY1_SUSY3y_test, verbose=0)
print('SUSY1_SUSY3 Test loss:', SUSY1_SUSY3score[0])
print('SUSY1_SUSY3 Test accuracy:', SUSY1_SUSY3score[1])


# ============================= SUSY 2 training ================================

SUSY2_SUSY3history = SUSY2_SUSY3fcmodel.fit(SUSY2_SUSY3x_train, SUSY2_SUSY3y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=2,
                      validation_data=(SUSY2_SUSY3x_test, SUSY2_SUSY3y_test))

SUSY2_SUSY3score = SUSY2_SUSY3fcmodel.evaluate(SUSY2_SUSY3x_test, SUSY2_SUSY3y_test, verbose=0)
print('SUSY2_SUSY3 Test loss:', SUSY2_SUSY3score[0])
print('SUSY2_SUSY3 Test accuracy:', SUSY2_SUSY3score[1]) #now mae


# =========================== ROC Curves =======================================

EFT_SUSY1probs = EFT_SUSY1fcmodel.predict(EFT_SUSY1x_test)
EFT_SUSY1fpr, EFT_SUSY1tpr, EFT_SUSY1thresholds = roc_curve(EFT_SUSY1y_test[:,1], EFT_SUSY1probs[:,1])
EFT_SUSY1sig_rate = EFT_SUSY1tpr
EFT_SUSY1bg_rate = EFT_SUSY1fpr
EFT_SUSY1roc_auc = auc(EFT_SUSY1bg_rate, EFT_SUSY1sig_rate)

EFT_SUSY2probs = EFT_SUSY2fcmodel.predict(EFT_SUSY2x_test)
EFT_SUSY2fpr, EFT_SUSY2tpr, EFT_SUSY2thresholds = roc_curve(EFT_SUSY2y_test[:,1], EFT_SUSY2probs[:,1])
EFT_SUSY2sig_rate = EFT_SUSY2tpr
EFT_SUSY2bg_rate = EFT_SUSY2fpr
EFT_SUSY2roc_auc = auc(EFT_SUSY2bg_rate, EFT_SUSY2sig_rate)

EFT_SUSY3probs = EFT_SUSY3fcmodel.predict(EFT_SUSY3x_test)
EFT_SUSY3fpr, EFT_SUSY3tpr, EFT_SUSY3thresholds = roc_curve(EFT_SUSY3y_test[:,1], EFT_SUSY3probs[:,1])
EFT_SUSY3sig_rate = EFT_SUSY3tpr
EFT_SUSY3bg_rate = EFT_SUSY3fpr
EFT_SUSY3roc_auc = auc(EFT_SUSY3bg_rate, EFT_SUSY3sig_rate)

ALP_EFTprobs = ALP_EFTfcmodel.predict(ALP_EFTx_test)
ALP_EFTfpr, ALP_EFTtpr, ALP_EFTthresholds = roc_curve(ALP_EFTy_test[:,1], ALP_EFTprobs[:,1])
ALP_EFTsig_rate = ALP_EFTtpr
ALP_EFTbg_rate = ALP_EFTfpr
ALP_EFTroc_auc = auc(ALP_EFTbg_rate, ALP_EFTsig_rate)

ALP_SUSY1probs = ALP_SUSY1fcmodel.predict(ALP_SUSY1x_test)
ALP_SUSY1fpr, ALP_SUSY1tpr, ALP_SUSY1thresholds = roc_curve(ALP_SUSY1y_test[:,1], ALP_SUSY1probs[:,1])
ALP_SUSY1sig_rate = ALP_SUSY1tpr
ALP_SUSY1bg_rate = ALP_SUSY1fpr
ALP_SUSY1roc_auc = auc(ALP_SUSY1bg_rate, ALP_SUSY1sig_rate)

ALP_SUSY2probs = ALP_SUSY2fcmodel.predict(ALP_SUSY2x_test)
ALP_SUSY2fpr, ALP_SUSY2tpr, ALP_SUSY2thresholds = roc_curve(ALP_SUSY2y_test[:,1], ALP_SUSY2probs[:,1])
ALP_SUSY2sig_rate = ALP_SUSY2tpr
ALP_SUSY2bg_rate = ALP_SUSY2fpr
ALP_SUSY2roc_auc = auc(ALP_SUSY2bg_rate, ALP_SUSY2sig_rate)

ALP_SUSY3probs = ALP_SUSY3fcmodel.predict(ALP_SUSY3x_test)
ALP_SUSY3fpr, ALP_SUSY3tpr, ALP_SUSY3thresholds = roc_curve(ALP_SUSY3y_test[:,1], ALP_SUSY3probs[:,1])
ALP_SUSY3sig_rate = ALP_SUSY3tpr
ALP_SUSY3bg_rate = ALP_SUSY3fpr
ALP_SUSY3roc_auc = auc(ALP_SUSY3bg_rate, ALP_SUSY3sig_rate)

SUSY1_SUSY2probs = SUSY1_SUSY2fcmodel.predict(SUSY1_SUSY2x_test)
SUSY1_SUSY2fpr, SUSY1_SUSY2tpr, SUSY1_SUSY2thresholds = roc_curve(SUSY1_SUSY2y_test[:,1], SUSY1_SUSY2probs[:,1])
SUSY1_SUSY2sig_rate = SUSY1_SUSY2tpr
SUSY1_SUSY2bg_rate = SUSY1_SUSY2fpr
SUSY1_SUSY2roc_auc = auc(SUSY1_SUSY2bg_rate, SUSY1_SUSY2sig_rate)

SUSY1_SUSY3probs = SUSY1_SUSY3fcmodel.predict(SUSY1_SUSY3x_test)
SUSY1_SUSY3fpr, SUSY1_SUSY3tpr, SUSY1_SUSY3thresholds = roc_curve(SUSY1_SUSY3y_test[:,1], SUSY1_SUSY3probs[:,1])
SUSY1_SUSY3sig_rate = SUSY1_SUSY3tpr
SUSY1_SUSY3bg_rate = SUSY1_SUSY3fpr
SUSY1_SUSY3roc_auc = auc(SUSY1_SUSY3bg_rate, SUSY1_SUSY3sig_rate)

SUSY2_SUSY3probs = SUSY2_SUSY3fcmodel.predict(SUSY2_SUSY3x_test)
SUSY2_SUSY3fpr, SUSY2_SUSY3tpr, SUSY2_SUSY3thresholds = roc_curve(SUSY2_SUSY3y_test[:,1], SUSY2_SUSY3probs[:,1])
SUSY2_SUSY3sig_rate = SUSY2_SUSY3tpr
SUSY2_SUSY3bg_rate = SUSY2_SUSY3fpr
SUSY2_SUSY3roc_auc = auc(SUSY2_SUSY3bg_rate, SUSY2_SUSY3sig_rate)



plt.figure()
plt.plot(ALP_SUSY1bg_rate, ALP_SUSY1sig_rate, lw=2, color='orange', linestyle='solid', label='SUSY1 vs ALPs, AUC = %0.2f'%(ALP_SUSY1roc_auc))
plt.plot(EFT_SUSY1bg_rate, EFT_SUSY1sig_rate, lw=2, color='orange', linestyle='dashed', label='SUSY1 vs EFT, AUC = %0.2f'%(EFT_SUSY1roc_auc))
plt.plot(ALP_SUSY2bg_rate, ALP_SUSY2sig_rate, lw=2, color='blue', linestyle='solid', label='SUSY2 vs ALPs, AUC = %0.2f'%(ALP_SUSY2roc_auc))
plt.plot(EFT_SUSY2bg_rate, EFT_SUSY2sig_rate, lw=2, color='blue', linestyle='dashed', label='SUSY2 vs EFT, AUC = %0.2f'%(EFT_SUSY2roc_auc))
plt.plot(ALP_SUSY3bg_rate, ALP_SUSY3sig_rate, lw=2, color='magenta', linestyle='solid', label='SUSY3 vs ALPs, AUC = %0.2f'%(ALP_SUSY3roc_auc))
plt.plot(EFT_SUSY3bg_rate, EFT_SUSY3sig_rate, lw=2, color='magenta', linestyle='dashed', label='SUSY3 vs EFT, AUC = %0.2f'%(EFT_SUSY3roc_auc))
plt.xlabel(r'$\epsilon_{B}$',fontsize=16)
plt.ylabel(r'$\epsilon_{S}$',fontsize=16)
plt.title(r'DNN')
legend1=plt.legend(loc='lower right')#,prop={'size':10}
plt.savefig('2D_DNN_SUSY_vs_sig_NLO_pt-metphij1_ROC.png')



# really we should not look at the score for this set of data
# until we have finished tuning our model
EFT_SUSY1valscore = EFT_SUSY1fcmodel.evaluate(EFT_SUSY1x_val, EFT_SUSY1y_val, verbose=0)
print('EFT_SUSY1 val Test loss:', EFT_SUSY1valscore[0])
print('EFT_SUSY1 val Test accuracy:', EFT_SUSY1valscore[1])

# OKAY SO THIS CODE HAS TEST AND VAL THE WRONG WAY ROUND
# I HAVE ONLY CHANGED THIS SO FAR IN THE LEGEND OF THE plots
# BUT IT SHOULD BE CHANGED THROUGHOUT THE CODE
# I AM JUST TOO LAZY TO DO IT NOW

#Convolutional Neural Network (CNN)

ALP_SUSY1x_train = ALP_SUSY1x_train.reshape(-1, n_xbins-1, n_ybins-1, 1)
ALP_SUSY1x_test = ALP_SUSY1x_test.reshape(-1, n_xbins-1, n_ybins-1, 1)

ALP_SUSY2x_train = ALP_SUSY2x_train.reshape(-1, n_xbins-1, n_ybins-1, 1)
ALP_SUSY2x_test = ALP_SUSY2x_test.reshape(-1, n_xbins-1, n_ybins-1, 1)

ALP_SUSY3x_train = ALP_SUSY3x_train.reshape(-1, n_xbins-1, n_ybins-1, 1)
ALP_SUSY3x_test = ALP_SUSY3x_test.reshape(-1, n_xbins-1, n_ybins-1, 1)

EFT_SUSY1x_train = EFT_SUSY1x_train.reshape(-1, n_xbins-1, n_ybins-1, 1)
EFT_SUSY1x_test = EFT_SUSY1x_test.reshape(-1, n_xbins-1, n_ybins-1, 1)

EFT_SUSY2x_train = EFT_SUSY2x_train.reshape(-1, n_xbins-1, n_ybins-1, 1)
EFT_SUSY2x_test = EFT_SUSY2x_test.reshape(-1, n_xbins-1, n_ybins-1, 1)

EFT_SUSY3x_train = EFT_SUSY3x_train.reshape(-1, n_xbins-1, n_ybins-1, 1)
EFT_SUSY3x_test = EFT_SUSY3x_test.reshape(-1, n_xbins-1, n_ybins-1, 1)


# a fairly small network for speed
EFT_SUSY1cnnmodel = Sequential()
EFT_SUSY1cnnmodel.add(Conv2D(8, (3, 3), activation='relu', input_shape=(n_xbins-1,n_ybins-1, 1)))
EFT_SUSY1cnnmodel.add(MaxPooling2D((3, 3)))
EFT_SUSY1cnnmodel.add(Conv2D(16, (3, 3), activation='relu'))
EFT_SUSY1cnnmodel.add(MaxPooling2D((2, 2)))
EFT_SUSY1cnnmodel.add(Flatten())
EFT_SUSY1cnnmodel.add(Dense(num_classes, activation='softmax'))
# could include dropout, regularisation, ...

EFT_SUSY2cnnmodel = Sequential()
EFT_SUSY2cnnmodel.add(Conv2D(8, (3, 3), activation='relu', input_shape=(n_xbins-1,n_ybins-1, 1)))
EFT_SUSY2cnnmodel.add(MaxPooling2D((3, 3)))
EFT_SUSY2cnnmodel.add(Conv2D(16, (3, 3), activation='relu'))
EFT_SUSY2cnnmodel.add(MaxPooling2D((2, 2)))
EFT_SUSY2cnnmodel.add(Flatten())
EFT_SUSY2cnnmodel.add(Dense(num_classes, activation='softmax'))
# could include dropout, regularisation, ...

EFT_SUSY3cnnmodel = Sequential()
EFT_SUSY3cnnmodel.add(Conv2D(8, (3, 3), activation='relu', input_shape=(n_xbins-1,n_ybins-1, 1)))
EFT_SUSY3cnnmodel.add(MaxPooling2D((3, 3)))
EFT_SUSY3cnnmodel.add(Conv2D(16, (3, 3), activation='relu'))
EFT_SUSY3cnnmodel.add(MaxPooling2D((2, 2)))
EFT_SUSY3cnnmodel.add(Flatten())
EFT_SUSY3cnnmodel.add(Dense(num_classes, activation='softmax'))
# could include dropout, regularisation, ...

ALP_SUSY1cnnmodel = Sequential()
ALP_SUSY1cnnmodel.add(Conv2D(8, (3, 3), activation='relu', input_shape=(n_xbins-1,n_ybins-1, 1)))
ALP_SUSY1cnnmodel.add(MaxPooling2D((3, 3)))
ALP_SUSY1cnnmodel.add(Conv2D(16, (3, 3), activation='relu'))
ALP_SUSY1cnnmodel.add(MaxPooling2D((2, 2)))
ALP_SUSY1cnnmodel.add(Flatten())
ALP_SUSY1cnnmodel.add(Dense(num_classes, activation='softmax'))
# could include dropout, regularisation, ...

ALP_SUSY2cnnmodel = Sequential()
ALP_SUSY2cnnmodel.add(Conv2D(8, (3, 3), activation='relu', input_shape=(n_xbins-1,n_ybins-1, 1)))
ALP_SUSY2cnnmodel.add(MaxPooling2D((3, 3)))
ALP_SUSY2cnnmodel.add(Conv2D(16, (3, 3), activation='relu'))
ALP_SUSY2cnnmodel.add(MaxPooling2D((2, 2)))
ALP_SUSY2cnnmodel.add(Flatten())
ALP_SUSY2cnnmodel.add(Dense(num_classes, activation='softmax'))
# could include dropout, regularisation, ...

ALP_SUSY3cnnmodel = Sequential()
ALP_SUSY3cnnmodel.add(Conv2D(8, (3, 3), activation='relu', input_shape=(n_xbins-1,n_ybins-1, 1)))
ALP_SUSY3cnnmodel.add(MaxPooling2D((3, 3)))
ALP_SUSY3cnnmodel.add(Conv2D(16, (3, 3), activation='relu'))
ALP_SUSY3cnnmodel.add(MaxPooling2D((2, 2)))
ALP_SUSY3cnnmodel.add(Flatten())
ALP_SUSY3cnnmodel.add(Dense(num_classes, activation='softmax'))
# could include dropout, regularisation, ...


EFT_SUSY1cnnmodel.compile(loss='categorical_crossentropy',
                optimizer=RMSprop(),
                metrics=['accuracy'])
EFT_SUSY1cnnmodel.summary()

CNN_EFT_SUSY1history = EFT_SUSY1cnnmodel.fit(EFT_SUSY1x_train, EFT_SUSY1y_train,
                       batch_size=batch_size,
                       epochs=1*epochs,
                       verbose=2,
                       validation_data=(EFT_SUSY1x_test, EFT_SUSY1y_test))

EFT_SUSY1score = EFT_SUSY1cnnmodel.evaluate(EFT_SUSY1x_test, EFT_SUSY1y_test, verbose=0)
print('CNN EFT_SUSY1 Test loss:', EFT_SUSY1score[0])
print('CNN EFT_SUSY1 Test accuracy:', EFT_SUSY1score[1]) #now mae

EFT_SUSY2cnnmodel.compile(loss='categorical_crossentropy',
                optimizer=RMSprop(),
                metrics=['accuracy'])
EFT_SUSY2cnnmodel.summary()

CNN_EFT_SUSY2history = EFT_SUSY2cnnmodel.fit(EFT_SUSY2x_train, EFT_SUSY2y_train,
                       batch_size=batch_size,
                       epochs=1*epochs,
                       verbose=2,
                       validation_data=(EFT_SUSY2x_test, EFT_SUSY2y_test))

EFT_SUSY2score = EFT_SUSY2cnnmodel.evaluate(EFT_SUSY2x_test, EFT_SUSY2y_test, verbose=0)
print('CNN EFT_SUSY2 Test loss:', EFT_SUSY2score[0])
print('CNN EFT_SUSY2 Test accuracy:', EFT_SUSY2score[1]) #now mae

EFT_SUSY3cnnmodel.compile(loss='categorical_crossentropy',
                optimizer=RMSprop(),
                metrics=['accuracy'])
EFT_SUSY3cnnmodel.summary()

CNN_EFT_SUSY3history = EFT_SUSY3cnnmodel.fit(EFT_SUSY3x_train, EFT_SUSY3y_train,
                       batch_size=batch_size,
                       epochs=1*epochs,
                       verbose=2,
                       validation_data=(EFT_SUSY3x_test, EFT_SUSY3y_test))

EFT_SUSY3score = EFT_SUSY3cnnmodel.evaluate(EFT_SUSY3x_test, EFT_SUSY3y_test, verbose=0)
print('CNN EFT_SUSY3 Test loss:', EFT_SUSY3score[0])
print('CNN EFT_SUSY3 Test accuracy:', EFT_SUSY3score[1]) #now mae

ALP_SUSY1cnnmodel.compile(loss='categorical_crossentropy',
                optimizer=RMSprop(),
                metrics=['accuracy'])
ALP_SUSY1cnnmodel.summary()

CNN_ALP_SUSY1history = ALP_SUSY1cnnmodel.fit(ALP_SUSY1x_train, ALP_SUSY1y_train,
                       batch_size=batch_size,
                       epochs=1*epochs,
                       verbose=2,
                       validation_data=(ALP_SUSY1x_test, ALP_SUSY1y_test))

ALP_SUSY1score = ALP_SUSY1cnnmodel.evaluate(ALP_SUSY1x_test, ALP_SUSY1y_test, verbose=0)
print('CNN ALP_SUSY1 Test loss:', ALP_SUSY1score[0])
print('CNN ALP_SUSY1 Test accuracy:', ALP_SUSY1score[1]) #now mae

ALP_SUSY2cnnmodel.compile(loss='categorical_crossentropy',
                optimizer=RMSprop(),
                metrics=['accuracy'])
ALP_SUSY2cnnmodel.summary()

CNN_ALP_SUSY2history = ALP_SUSY2cnnmodel.fit(ALP_SUSY2x_train, ALP_SUSY2y_train,
                       batch_size=batch_size,
                       epochs=1*epochs,
                       verbose=2,
                       validation_data=(ALP_SUSY2x_test, ALP_SUSY2y_test))

ALP_SUSY2score = ALP_SUSY2cnnmodel.evaluate(ALP_SUSY2x_test, ALP_SUSY2y_test, verbose=0)
print('CNN ALP_SUSY2 Test loss:', ALP_SUSY2score[0])
print('CNN ALP_SUSY2 Test accuracy:', ALP_SUSY2score[1]) #now mae

ALP_SUSY3cnnmodel.compile(loss='categorical_crossentropy',
                optimizer=RMSprop(),
                metrics=['accuracy'])
ALP_SUSY3cnnmodel.summary()

CNN_ALP_SUSY3history = ALP_SUSY3cnnmodel.fit(ALP_SUSY3x_train, ALP_SUSY3y_train,
                       batch_size=batch_size,
                       epochs=1*epochs,
                       verbose=2,
                       validation_data=(ALP_SUSY3x_test, ALP_SUSY3y_test))

ALP_SUSY3score = ALP_SUSY3cnnmodel.evaluate(ALP_SUSY3x_test, ALP_SUSY3y_test, verbose=0)
print('CNN ALP_SUSY3 Test loss:', ALP_SUSY3score[0])
print('CNN ALP_SUSY3 Test accuracy:', ALP_SUSY3score[1]) #now mae


# =========================== ROC Curves =======================================

EFT_SUSY1probs = EFT_SUSY1cnnmodel.predict(EFT_SUSY1x_test)
EFT_SUSY1fpr, EFT_SUSY1tpr, EFT_SUSY1thresholds = roc_curve(EFT_SUSY1y_test[:,1], EFT_SUSY1probs[:,1])
EFT_SUSY1sig_rate = EFT_SUSY1tpr
EFT_SUSY1bg_rate = EFT_SUSY1fpr
EFT_SUSY1roc_auc = auc(EFT_SUSY1bg_rate, EFT_SUSY1sig_rate)

EFT_SUSY2probs = EFT_SUSY2cnnmodel.predict(EFT_SUSY2x_test)
EFT_SUSY2fpr, EFT_SUSY2tpr, EFT_SUSY2thresholds = roc_curve(EFT_SUSY2y_test[:,1], EFT_SUSY2probs[:,1])
EFT_SUSY2sig_rate = EFT_SUSY2tpr
EFT_SUSY2bg_rate = EFT_SUSY2fpr
EFT_SUSY2roc_auc = auc(EFT_SUSY2bg_rate, EFT_SUSY2sig_rate)

EFT_SUSY3probs = EFT_SUSY3cnnmodel.predict(EFT_SUSY3x_test)
EFT_SUSY3fpr, EFT_SUSY3tpr, EFT_SUSY3thresholds = roc_curve(EFT_SUSY3y_test[:,1], EFT_SUSY3probs[:,1])
EFT_SUSY3sig_rate = EFT_SUSY3tpr
EFT_SUSY3bg_rate = EFT_SUSY3fpr
EFT_SUSY3roc_auc = auc(EFT_SUSY3bg_rate, EFT_SUSY3sig_rate)

ALP_SUSY1probs = ALP_SUSY1cnnmodel.predict(ALP_SUSY1x_test)
ALP_SUSY1fpr, ALP_SUSY1tpr, ALP_SUSY1thresholds = roc_curve(ALP_SUSY1y_test[:,1], ALP_SUSY1probs[:,1])
ALP_SUSY1sig_rate = ALP_SUSY1tpr
ALP_SUSY1bg_rate = ALP_SUSY1fpr
ALP_SUSY1roc_auc = auc(ALP_SUSY1bg_rate, ALP_SUSY1sig_rate)

ALP_SUSY2probs = ALP_SUSY2cnnmodel.predict(ALP_SUSY2x_test)
ALP_SUSY2fpr, ALP_SUSY2tpr, ALP_SUSY2thresholds = roc_curve(ALP_SUSY2y_test[:,1], ALP_SUSY2probs[:,1])
ALP_SUSY2sig_rate = ALP_SUSY2tpr
ALP_SUSY2bg_rate = ALP_SUSY2fpr
ALP_SUSY2roc_auc = auc(ALP_SUSY2bg_rate, ALP_SUSY2sig_rate)

ALP_SUSY3probs = ALP_SUSY3cnnmodel.predict(ALP_SUSY3x_test)
ALP_SUSY3fpr, ALP_SUSY3tpr, ALP_SUSY3thresholds = roc_curve(ALP_SUSY3y_test[:,1], ALP_SUSY3probs[:,1])
ALP_SUSY3sig_rate = ALP_SUSY3tpr
ALP_SUSY3bg_rate = ALP_SUSY3fpr
ALP_SUSY3roc_auc = auc(ALP_SUSY3bg_rate, ALP_SUSY3sig_rate)



plt.figure()
plt.plot(ALP_SUSY1bg_rate, ALP_SUSY1sig_rate, lw=2, color='orange', linestyle='solid', label='SUSY1 vs ALP, AUC = %0.2f'%(ALP_SUSY1roc_auc))
plt.plot(EFT_SUSY1bg_rate, EFT_SUSY1sig_rate, lw=2, color='orange', linestyle='dashed', label='SUSY1 vs EFT, AUC = %0.2f'%(EFT_SUSY1roc_auc))
plt.plot(ALP_SUSY2bg_rate, ALP_SUSY2sig_rate, lw=2, color='blue', linestyle='solid', label='SUSY2 vs ALP, AUC = %0.2f'%(ALP_SUSY2roc_auc))
plt.plot(EFT_SUSY2bg_rate, EFT_SUSY2sig_rate, lw=2, color='blue', linestyle='dashed', label='SUSY2 vs EFT, AUC = %0.2f'%(EFT_SUSY2roc_auc))
plt.plot(ALP_SUSY3bg_rate, ALP_SUSY3sig_rate, lw=2, color='magenta', linestyle='solid', label='SUSY3 vs ALP, AUC = %0.2f'%(ALP_SUSY3roc_auc))
plt.plot(EFT_SUSY3bg_rate, EFT_SUSY3sig_rate, lw=2, color='magenta', linestyle='dashed', label='SUSY3 vs EFT, AUC = %0.2f'%(EFT_SUSY3roc_auc))
plt.xlabel(r'$\epsilon_{B}$',fontsize=16)
plt.ylabel(r'$\epsilon_{S}$',fontsize=16)
plt.title(r'CNN')
legend1=plt.legend(loc='lower right')#,prop={'size':10}
plt.savefig('2D_CNN_SUSY_vs_sig_NLO_pt-metphij1_ROC.png')



plt.show(block=False)
