import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA #import the PCA class from scikit-learn

BG_data=np.loadtxt("smnp1dijet.csv", unpack=True)#,skiprows=1
ALP_data=np.loadtxt("alpnp1dijet.csv", unpack=True)#,skiprows=1
EFT_data=np.loadtxt("spin1meddijet.csv", unpack=True)#,skiprows=1
SUSY1_data=np.loadtxt("susycor1dijet.csv", unpack=True)#,skiprows=1
SUSY2_data=np.loadtxt("susycor2dijet.csv", unpack=True)#,skiprows=1
SUSY3_data=np.loadtxt("susycor3dijet.csv", unpack=True)#,skiprows=1

# Drop signal (y data)
BG_xdata = BG_data[0:BG_data.shape[0]-1]
ALP_xdata = ALP_data[0:ALP_data.shape[0]-1]
EFT_xdata = EFT_data[0:EFT_data.shape[0]-1]
SUSY1_xdata = SUSY1_data[0:SUSY1_data.shape[0]-1]
SUSY2_xdata = SUSY2_data[0:SUSY2_data.shape[0]-1]
SUSY3_xdata = SUSY3_data[0:SUSY3_data.shape[0]-1]


# Drop metPhi
#new_BG_xdata = np.delete(BG_xdata,7,axis=1)
#new_ALP_xdata = np.delete(ALP_xdata,7,axis=1)
#new_EFT_xdata = np.delete(EFT_xdata,7,axis=1)
#new_SUSY1_xdata = np.delete(SUSY1_xdata,7,axis=1)
#new_SUSY2_xdata = np.delete(SUSY2_xdata,7,axis=1)
#new_SUSY3_xdata = np.delete(SUSY3_xdata,7,axis=1)

# Check the shape of the data
print np.shape(BG_xdata)

# Choose the number of components
num_components = BG_xdata.shape[0]# - 1

# Set the stage needed to perform linear dimensionality reduction using Singular Valuar Decomposition
# The first line only sets the number of dimensions to use
# The second line is performing the transformation to the pricipal component basis
# After that we can then choose to eliminate any dimensions

def pca_func(data,num_components):
    #data = data[0:num_components]
    pca = PCA(n_components=num_components)
    pca.fit(data)
    principal_components = pca.fit_transform(data)
    explained_variance_ratio = pca.explained_variance_ratio_
    return principal_components,explained_variance_ratio

#skeleton script for bar chart of the principal components
plt.close("all")

plt.figure()
plt.bar(np.arange(num_components),pca_func(BG_xdata,num_components)[1],tick_label=np.arange(num_components)+1)
plt.title("BG principal components")
plt.yscale("log")
plt.ylabel("Variance encompassed by each principal component")
plt.xlabel("Principal component no.");
plt.savefig("BG_PCA")

plt.figure()
plt.bar(np.arange(num_components),pca_func(ALP_xdata,num_components)[1],tick_label=np.arange(num_components)+1)
plt.title("ALP principal components")
plt.yscale("log")
plt.ylabel("Variance encompassed by each principal component")
plt.xlabel("Principal component no.");
plt.savefig("ALP_PCA")

plt.figure()
plt.bar(np.arange(num_components),pca_func(EFT_xdata,num_components)[1],tick_label=np.arange(num_components)+1)
plt.title("EFT principal components")
plt.yscale("log")
plt.ylabel("Variance encompassed by each principal component")
plt.xlabel("Principal component no.");
plt.savefig("EFT_PCA")

plt.figure()
plt.bar(np.arange(num_components),pca_func(SUSY1_xdata,num_components)[1],tick_label=np.arange(num_components)+1)
plt.title("SUSY1 principal components")
plt.yscale("log")
plt.ylabel("Variance encompassed by each principal component")
plt.xlabel("Principal component no.");
plt.savefig("SUSY1_PCA")

plt.figure()
plt.bar(np.arange(num_components),pca_func(SUSY2_xdata,num_components)[1],tick_label=np.arange(num_components)+1)
plt.title("SUSY2 principal components")
plt.yscale("log")
plt.ylabel("Variance encompassed by each principal component")
plt.xlabel("Principal component no.");
plt.savefig("SUSY2_PCA")

plt.figure()
plt.bar(np.arange(num_components),pca_func(SUSY3_xdata,num_components)[1],tick_label=np.arange(num_components)+1)
plt.title("SUSY3 principal components")
plt.yscale("log")
plt.ylabel("Variance encompassed by each principal component")
plt.xlabel("Principal component no.");
plt.savefig("SUSY3_PCA")

plt.show(block=False)
