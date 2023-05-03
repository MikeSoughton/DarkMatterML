import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA #import the PCA class from scikit-learn

#BG_data=np.loadtxt("dijetax.csv", unpack=True,skiprows=0)
ALP_data=np.loadtxt("dijetax.csv", unpack=True,skiprows=0)
EFT_data=np.loadtxt("dijetspin1md10.csv", unpack=True,skiprows=0)

# Check the shape of the data
print np.shape(ALP_data)

# Choose the number of components
num_components = 7

# Set the stage needed to perform linear dimensionality reduction using Singular Valuar Decomposition
# The first line only sets the number of dimensions to use
# The second line is performing the transformation to the pricipal component basis
# After that we can then choose to eliminate any dimensions

def pca(data,num_components):
    #data = data[0:num_components]
    pca = PCA(n_components=num_components)
    pca.fit(data)
    principal_components = pca.fit_transform(data)
    explained_variance_ratio = pca.explained_variance_ratio_
    return principal_components,explained_variance_ratio

#print principal_components

#skeleton script for bar chart of the principal components
plt.close("all")

plt.figure()
plt.bar(np.arange(num_components),pca(ALP_data,num_components)[1],tick_label=np.arange(num_components)+1)
plt.title("ALP principal_components")
plt.yscale("log")
plt.ylabel("Variance encompassed by each principal component")
plt.xlabel("Principal component no.");
plt.savefig("ALP_PCA")

plt.figure()
plt.bar(np.arange(num_components),pca(EFT_data,num_components)[1],tick_label=np.arange(num_components)+1)
plt.title("EFT principal_components")
plt.yscale("log")
plt.ylabel("Variance encompassed by each principal component")
plt.xlabel("Principal component no.");
plt.savefig("EFT_PCA")

plt.show(block=False)
