import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA #import the PCA class from scikit-learn

BG_data=np.loadtxt("smnp1dijet.csv", unpack=True)#,skiprows=1
#ALP_data=np.loadtxt("dijeta.csv", unpack=True,skiprows=0)
#EFT_data=np.loadtxt("spin1meddelphes.csv", unpack=True,skiprows=1)
#SUSY1_data=np.loadtxt("susyfulldelphes1.csv", unpack=True,skiprows=1)
#SUSY2_data=np.loadtxt("susyfulldelphes2.csv", unpack=True,skiprows=1)
#SUSY3_data=np.loadtxt("susyfulldelphes3.csv", unpack=True,skiprows=1)

# Check the shape of the data
print np.shape(BG_data)

BG_xdata = BG_data[0:BG_data.shape[0]-1]




# Choose the number of components
##num_components = BG_xdata.shape[0] -1

num_components = BG_xdata.shape[0] 


# load dataset into Pandas DataFrame
df_BG = pd.read_csv('modified_smnp1dijet.csv')
df_ALP = pd.read_csv('modified_alpnp1dijet.csv')
df_EFT = pd.read_csv('modified_spin1meddijet.csv')
df_SUSY1 = pd.read_csv('modified_susycor1dijet.csv')
df_SUSY2 = pd.read_csv('modified_susycor2dijet.csv')
df_SUSY3 = pd.read_csv('modified_susycor3dijet.csv')

#df_BG = df_BG.drop(columns=['metPhi','signal'])
#df_ALP = df_ALP.drop(columns=['metPhi','signal'])
#df_EFT = df_EFT.drop(columns=['metPhi','signal'])
#df_SUSY1 = df_SUSY1.drop(columns=['metPhi','signal'])
#df_SUSY2 = df_SUSY2.drop(columns=['metPhi','signal'])
#df_SUSY3 = df_SUSY3.drop(columns=['metPhi','signal'])

df_BG = df_BG.drop(columns=['signal'])
df_ALP = df_ALP.drop(columns=['signal'])
df_EFT = df_EFT.drop(columns=['signal'])
df_SUSY1 = df_SUSY1.drop(columns=['signal'])
df_SUSY2 = df_SUSY2.drop(columns=['signal'])
df_SUSY3 = df_SUSY3.drop(columns=['signal'])



# normalize data
from sklearn import preprocessing
BG_data_scaled = pd.DataFrame(preprocessing.scale(df_BG),columns = df_BG.columns)
ALP_data_scaled = pd.DataFrame(preprocessing.scale(df_ALP),columns = df_ALP.columns)
EFT_data_scaled = pd.DataFrame(preprocessing.scale(df_EFT),columns = df_EFT.columns)
SUSY1_data_scaled = pd.DataFrame(preprocessing.scale(df_SUSY1),columns = df_SUSY1.columns)
SUSY2_data_scaled = pd.DataFrame(preprocessing.scale(df_SUSY2),columns = df_SUSY2.columns)
SUSY3_data_scaled = pd.DataFrame(preprocessing.scale(df_SUSY3),columns = df_SUSY3.columns)

# PCA
pca_BG = PCA(n_components=num_components)
pca_ALP = PCA(n_components=num_components)
pca_EFT = PCA(n_components=num_components)
pca_SUSY1 = PCA(n_components=num_components)
pca_SUSY2 = PCA(n_components=num_components)
pca_SUSY3 = PCA(n_components=num_components)

pca_BG.fit_transform(BG_data_scaled)
pca_ALP.fit_transform(ALP_data_scaled)
pca_EFT.fit_transform(EFT_data_scaled)
pca_SUSY1.fit_transform(SUSY1_data_scaled)
pca_SUSY2.fit_transform(SUSY2_data_scaled)
pca_SUSY3.fit_transform(SUSY3_data_scaled)

# Dump components relations with features:
print "Background: \n", pd.DataFrame(pca_BG.components_,columns=BG_data_scaled.columns,index = ['PC-1','PC-2','PC-3','PC-4','PC-5','PC-6','PC-7','PC-8'])
print "\nALP: \n", pd.DataFrame(pca_ALP.components_,columns=ALP_data_scaled.columns,index = ['PC-1','PC-2','PC-3','PC-4','PC-5','PC-6','PC-7','PC-8'])
print "\nEFT: \n", pd.DataFrame(pca_EFT.components_,columns=EFT_data_scaled.columns,index = ['PC-1','PC-2','PC-3','PC-4','PC-5','PC-6','PC-7','PC-8'])
print "\nSUSY1: \n", pd.DataFrame(pca_SUSY1.components_,columns=SUSY1_data_scaled.columns,index = ['PC-1','PC-2','PC-3','PC-4','PC-5','PC-6','PC-7','PC-8'])
print "\nSUSY2: \n", pd.DataFrame(pca_SUSY2.components_,columns=SUSY2_data_scaled.columns,index = ['PC-1','PC-2','PC-3','PC-4','PC-5','PC-6','PC-7','PC-8'])
print "\nSUSY3: \n", pd.DataFrame(pca_SUSY3.components_,columns=SUSY3_data_scaled.columns,index = ['PC-1','PC-2','PC-3','PC-4','PC-5','PC-6','PC-7','PC-8'])

# Function that will print horizontal lines when exporting to latex format
def latex_with_lines(df, *args, **kwargs):
    kwargs['column_format'] = '|'.join([''] + ['l'] * df.index.nlevels
                                            + ['r'] * df.shape[1] + [''])
    res = df.to_latex(*args, **kwargs)
    return res.replace('\\\\\n', '\\\\ \\hline\n')

# Export to latex format
print latex_with_lines(pd.DataFrame(pca_BG.components_,columns=BG_data_scaled.columns,index = ['PC-1','PC-2','PC-3','PC-4','PC-5','PC-6','PC-7','PC-8']).round(2))
print latex_with_lines(pd.DataFrame(pca_ALP.components_,columns=ALP_data_scaled.columns,index = ['PC-1','PC-2','PC-3','PC-4','PC-5','PC-6','PC-7','PC-8']).round(2))
print latex_with_lines(pd.DataFrame(pca_EFT.components_,columns=EFT_data_scaled.columns,index = ['PC-1','PC-2','PC-3','PC-4','PC-5','PC-6','PC-7','PC-8']).round(2))
print latex_with_lines(pd.DataFrame(pca_SUSY1.components_,columns=SUSY1_data_scaled.columns,index = ['PC-1','PC-2','PC-3','PC-4','PC-5','PC-6','PC-7','PC-8']).round(2))
print latex_with_lines(pd.DataFrame(pca_SUSY2.components_,columns=SUSY2_data_scaled.columns,index = ['PC-1','PC-2','PC-3','PC-4','PC-5','PC-6','PC-7','PC-8']).round(2))
print latex_with_lines(pd.DataFrame(pca_SUSY3.components_,columns=SUSY3_data_scaled.columns,index = ['PC-1','PC-2','PC-3','PC-4','PC-5','PC-6','PC-7','PC-8']).round(2))
