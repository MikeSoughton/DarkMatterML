#!/usr/bin/env python
# coding: utf-8

# # Notebook 19: Variational Autoencoders with Keras and MNIST

# ## Learning Goals
# The goals of this notebook is to learn how to code a variational autoencoder in Keras. We will discuss hyperparameters, training, and loss-functions. In addition, we will familiarize ourselves with the Keras sequential GUI as well as how to visualize results and make predictions using a VAE with a small number of latent dimensions.
#
# ## Overview
#
# This notebook teaches the reader how to build a Variational Autoencoder (VAE) with Keras. The code is a minimally modified, stripped-down version of the code from Lous Tiao in his wonderful [blog post](http://tiao.io/posts/implementing-variational-autoencoders-in-keras-beyond-the-quickstart-tutorial/) which the reader is strongly encouraged to also read.
#
# Our VAE will have Gaussian Latent variables and a Gaussian Posterior distribution $q_\phi({\mathbf z}|{\mathbf x})$ with a diagonal covariance matrix.
#
# Recall, that a VAE consists of four essential elements:
#
# * A latent variable ${\mathbf z}$ drawn from a distribution $p({\mathbf z})$ which in our case will be a Gaussian with mean zero and standard
# deviation $\epsilon$.
# * A decoder $p(\mathbf{x}|\mathbf{z})$ that maps latent variables ${\mathbf z}$ to visible variables ${\mathbf x}$. In our case, this is just a Multi-Layer Perceptron (MLP) - a neural network with one hidden layer.
# * An encoder $q_\phi(\mathbf{z}|\mathbf{x})$ that maps examples to the latent space. In our case, this map is just a Gaussian with means and variances that depend on the input: $q_\phi({\bf z}|{\bf x})= \mathcal{N}({\bf z}, \boldsymbol{\mu}({\bf x}), \mathrm{diag}(\boldsymbol{\sigma}^2({\bf x})))$
# * A cost function consisting of two terms: the reconstruction error and an additional regularization term that minimizes the KL-divergence between the variational and true encoders. Mathematically, the reconstruction error is just the cross-entropy between the samples and their reconstructions. The KL-divergence term can be calculated analytically for this term and can be written as
#
# $$-D_{KL}(q_\phi({\bf z}|{\bf x})|p({\bf z}))={1 \over 2} \sum_{j=1}^J \left (1+\log{\sigma_j^2({\bf x})}-\mu_j^2({\bf x}) -\sigma_j^2({\bf x})\right).
# $$
#

# ## Importing Data and specifying hyperparameters
#
# In the next section of code, we import the data and specify hyperparameters. The MNIST data are gray scale ranging in values from 0 to 255 for each pixel. We normalize this range to lie between 0 and 1.
#
# The hyperparameters we need to specify the architecture and train the VAE are:
#
# * The dimension of the hidden layers for encoders and decoders (`intermediate_dim`)
# * The dimension of the latent space (`latent_dim`)
# * The standard deviation of latent variables (`epsilon_std`)
# * Optimization hyper-parameters: `batch_size`, `epochs`
#


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras import backend as K

from keras.layers import (Input, InputLayer, Dense, Lambda, Layer,
                          Add, Multiply)
from keras.models import Model, Sequential
from keras.datasets import mnist
import pandas as pd

#============================= LOADING DATA AS BEFORE ===================================

# if you have a recent version of tensorflow, keras is included
from sklearn.model_selection import train_test_split

plt.close("all")

def split(arr, count):
     # syntax of a[x::y] means get every yth element starting at position x
     return [arr[i::count] for i in range(count)]

n_xbins = 28
n_ybins = 28

yedges =np.linspace(-4.0, 4.0, num=n_xbins)
#print len(xedges)
xedges =np.linspace(100.0,2000.0, num=n_ybins) #changed this from 2000 to 2600 as there are still events up there and they are porbably the more important ones for distinguihsing signals
#print len(yedges)

mono_ptj,mono_etaj,mono_phij,mono_signal=np.loadtxt("monoj.csv", unpack=True,skiprows=1)

DM_ptj,DM_etaj,DM_phij,DM_signal=np.loadtxt("spin1med.csv", unpack=True,skiprows=1)

#data splitting into (,N) parts
N = 1000

#data processing for ALP process
batched_mono_ptj=split(mono_ptj,N)
batched_mono_etaj=split(mono_etaj,N)

# Produce ALP monojet density arrays for training data with large number of events
for i in range(1,N+1):
      H_mono, xedges, yedges = np.histogram2d(batched_mono_ptj[i-1],batched_mono_etaj[i-1],bins=(xedges, yedges))
      H2 = H_mono
      H_mono = H_mono.reshape((n_xbins-1)*(n_ybins-1)) #1-D array from a matrix # could this be the line which messes things up?
      if i==1:
         H_mono_save = H_mono
      else:
         H_mono_save = np.vstack([H_mono_save,H_mono])

mono_signal=np.ones((N,1))
H_mono_savef=np.column_stack((H_mono_save, mono_signal))
print  H_mono_savef.shape

#data processing for DM process
batched_DM_ptj=split(DM_ptj,N)
batched_DM_etaj=split(DM_etaj,N)

# Produce DM density arrays for training data with large number of events
for i in range(1,N+1):
      H_DM, xedges, yedges = np.histogram2d(batched_DM_ptj[i-1],batched_DM_etaj[i-1],bins=(xedges, yedges))
      H_DM = H_DM.reshape((n_xbins-1)*(n_ybins-1)) #1-D array from a matrix # could this be the line which messes things up?
      if i==1:
         H_DM_save = H_DM
      else:
         H_DM_save = np.vstack([H_DM_save,H_DM])

DM_signal=np.zeros((N,1))
H_DM_savef=np.column_stack((H_DM_save, DM_signal))
print  H_DM_savef.shape

data=np.vstack([H_mono_savef,H_DM_savef])
#data /= np.max(data)
print  data.shape

datafinal = np.take(data,np.random.permutation(data.shape[0]),axis=0,out=data)

#Separating out the last coulmn : which is Y_label(1 or 0)
x_all=data[:,:-1]
x_all /= np.max(x_all)

y_all=data[:,(n_xbins-1)*(n_ybins-1)]

x_train, x_valtest, y_train, y_valtest = train_test_split(x_all, y_all, test_size=0.2)
x_test, x_val, y_test, y_val = train_test_split(x_valtest, y_valtest, test_size=0.5)

x_train_reshaped = x_train.reshape(-1, n_xbins-1, n_ybins-1, 1)
x_test_reshaped = x_test.reshape(-1, n_xbins-1, n_ybins-1, 1)

#==========================END OF LOADING DATA===============================================

#Load Data and map gray scale 256 to number between zero and 1
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#x_train = np.expand_dims(x_train, axis=-1) / 255.
#x_test = np.expand_dims(x_test, axis=-1) / 255.

#x_train = np.expand_dims(x_train, axis=-1)
#x_test = np.expand_dims(x_test, axis=-1)

print(x_train.shape)

# Find dimensions of input images
img_rows, img_cols, img_chns = x_train_reshaped.shape[1:]

# Specify hyperparameters
original_dim = img_rows * img_cols
intermediate_dim = 20000 # number of possible values which th pixels can take => was 256 before but will be something else now
latent_dim = 2
batch_size = 100
epochs = 3
epsilon_std = 0.05


# ## Specifying the loss function
#
# Here we specify the loss function. The first block of code is just the reconstruction error which is given by the cross-entropy. The second block of code calculates the KL-divergence analytically and adds it to the loss function with the line `self.add_loss`. It represents the KL-divergence as just another layer in the neural network with the inputs equal to the outputs: the means and variances for the variational encoder (i.e. $\boldsymbol{\mu}({\bf x})$ and $\boldsymbol{\sigma}^2({\bf x})$).

def nll(y_true, y_pred):
    """ Negative log likelihood (Bernoulli). """

    # keras.losses.binary_crossentropy gives the mean
    # over the last axis. we require the sum
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)

class KLDivergenceLayer(Layer):

    """ Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_var = inputs

        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs


# # Encoder and Decoder
#
# The following specifies both the encoder and decoder. The encoder is a MLP with three layers that maps ${\bf x}$ to $\boldsymbol{\mu}({\bf x})$ and $\boldsymbol{\sigma}^2({\bf x})$, followed by the generation of a latent variable using the reparametrization trick (see main text). The decoder is specified as a single sequential Keras layer.
#

# Encoder

x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)

z_mu = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])

# Reparametrization trick
z_sigma = Lambda(lambda t: K.exp(.5*t))(z_log_var)

eps = Input(tensor=K.random_normal(shape=(K.shape(x)[0],
                                          latent_dim)))
z_eps = Multiply()([z_sigma, eps])
z = Add()([z_mu, z_eps])

# This defines the Encoder which takes noise and input and outputs
# the latent variable z
encoder = Model(inputs=[x, eps], outputs=z)

# Decoder is MLP specified as single Keras Sequential Layer
decoder = Sequential([
    Dense(intermediate_dim, input_dim=latent_dim, activation='relu'),
    Dense(original_dim, activation='sigmoid')
])

x_pred = decoder(z)


# ## Training the model
#
# We now train the model. Even though the loss function is the negative log likelihood (cross-entropy), recall that the KL-layer adds the analytic form of the loss function as well. We also have to reshape the data to make it a vector, and specify an optimizer.

vae = Model(inputs=[x, eps], outputs=x_pred, name='vae')
vae.compile(optimizer='rmsprop', loss=nll)

#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#x_train = x_train.reshape(-1, original_dim) / 255.
#x_test = x_test.reshape(-1, original_dim) / 255.

hist = vae.fit(
    x_train,
    x_train,
    shuffle=True,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(x_test, x_test)
)


# ## Visualizing the loss function
#
# We can automatically visualize the loss function as a function of the epoch using the standard Keras interface for fitting.

#get_ipython().magic(u'matplotlib inline')

#for pretty plots
golden_size = lambda width: (width, 2. * width / (1 + np.sqrt(5)))

fig, ax = plt.subplots(figsize=golden_size(6))

hist_df = pd.DataFrame(hist.history)
hist_df.plot(ax=ax)

ax.set_ylabel('NELBO')
ax.set_xlabel('# epochs')

ax.set_ylim(.99*hist_df[1:].values.min(),
            1.1*hist_df[1:].values.max())



# ## Visualizing embedding in latent space
#
# Since our latent space is two dimensional, we can think of our encoder as defining a dimensional reduction of the original 784 dimensional space to just two dimensions! We can visualize the structure of this mapping by plotting the MNIST dataset in the latent space, with each point colored by which number it is $[0,1,\ldots,9]$.

x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=golden_size(6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test, cmap='nipy_spectral')
plt.colorbar()
#plt.savefig('VAE_MNIST_latent.pdf')



# ## Generating new examples
#
# One of the nice things about VAEs is that they are generative models. Thus, we can generate new examples or fantasy particles much like we did for RBMs and DBMs. We will generate the particles in two different ways
#
# * Sampling uniformally in the latent space
# * Sampling accounting for the fact that the latent space is Gaussian so that we expect most of the data points to be centered around (0,0) and fall off exponentially in all directions. This is done by transforming the uniform grid using the inverse Cumulative Distribution Function (CDF) for the Gaussian.
#
#


# display a 2D manifold of the images
n = 5  # figure with 15x15 images
quantile_min = 0.01
quantile_max = 0.99

# Linear Sampling
# we will sample n points within [-15, 15] standard deviations
z1_u = np.linspace(5, -5, n)
z2_u = np.linspace(5, -5, n)
z_grid = np.dstack(np.meshgrid(z1_u, z2_u))

x_pred_grid = decoder.predict(z_grid.reshape(n*n, latent_dim))                      .reshape(n, n, img_rows, img_cols)

# Plot figure
fig, ax = plt.subplots(figsize=golden_size(10))

ax.imshow(np.block(list(map(list, x_pred_grid))), cmap='gray')

ax.set_xticks(np.arange(0, n*img_rows, img_rows) + .5 * img_rows)
ax.set_xticklabels(map('{:.2f}'.format, z1_u), rotation=90)

ax.set_yticks(np.arange(0, n*img_cols, img_cols) + .5 * img_cols)
ax.set_yticklabels(map('{:.2f}'.format, z2_u))

ax.set_xlabel('$z_1$')
ax.set_ylabel('$z_2$')
ax.set_title('Uniform')
ax.grid(False)
#plt.savefig('VAE_MNIST_fantasy_uniform.pdf')


# Inverse CDF sampling
z1 = norm.ppf(np.linspace(quantile_min, quantile_max, n))
z2 = norm.ppf(np.linspace(quantile_max, quantile_min, n))
z_grid2 = np.dstack(np.meshgrid(z1, z2))

x_pred_grid2 = decoder.predict(z_grid2.reshape(n*n, latent_dim))                      .reshape(n, n, img_rows, img_cols)

# Plot figure Inverse CDF sampling
fig, ax = plt.subplots(figsize=golden_size(10))

ax.imshow(np.block(list(map(list, x_pred_grid2))), cmap='gray')

ax.set_xticks(np.arange(0, n*img_rows, img_rows) + .5 * img_rows)
ax.set_xticklabels(map('{:.2f}'.format, z1), rotation=90)

ax.set_yticks(np.arange(0, n*img_cols, img_cols) + .5 * img_cols)
ax.set_yticklabels(map('{:.2f}'.format, z2))

ax.set_xlabel('$z_1$')
ax.set_ylabel('$z_2$')
ax.set_title('Inverse CDF')
ax.grid(False)
#plt.savefig('VAE_MNIST_fantasy_invCDF.pdf')

plt.show(block=False)


# ## Exercises
#
# * Play with the standard deviation of the latent variables $\epsilon$. How does this effect your results?
# * Generate samples as you increase the number of latent dimensions. Do your generated samples look better? Visualize the latent variables using a dimensional reduction technique such as PCA or t-SNE. How does it compare to the case with two latent dimensions showed above?
# * Repeat this analysis with the supersymmetry dataset? Are the supersymmetric and non-supersymmetric examples separated in the latent dimensions?
