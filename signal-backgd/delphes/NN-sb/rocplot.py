import numpy as np
import warnings
import matplotlib.pyplot as plt
import pylab as plab



tpr1=np.loadtxt("axion-sm/tpr.txt", unpack=True)
fpr1=np.loadtxt("axion-sm/fpr.txt", unpack=True)

tpr2=np.loadtxt("eft-sm/tpr.txt", unpack=True)
fpr2=np.loadtxt("eft-sm/fpr.txt", unpack=True)


tpr3=np.loadtxt("susy1-sm/tpr.txt", unpack=True)
fpr3=np.loadtxt("susy1-sm/fpr.txt", unpack=True)

tpr4=np.loadtxt("susy2-sm/tpr.txt", unpack=True)
fpr4=np.loadtxt("susy2-sm/fpr.txt", unpack=True)

tpr5=np.loadtxt("susy3-sm/tpr.txt", unpack=True)
fpr5=np.loadtxt("susy3-sm/fpr.txt", unpack=True)



fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(fpr1,tpr1,lw=2,color='red', label=r'ALPs, AUC=0.58')
ax.plot(fpr2,tpr2,lw=2,color='green', label=r'EFT, AUC=0.72')
ax.plot(fpr3,tpr3,lw=2,color='blue', label=r'SUSY1, AUC=0.62')#WIMP BP1
ax.plot(fpr4,tpr4,lw=2,color='blue',linestyle='--', label=r'SUSY2, AUC=0.68')#WIMP BP2
ax.plot(fpr5,tpr5,lw=2,color='blue',linestyle=':', label=r'SUSY3, AUC=0.71')#WIMP BP3



legend1= plt.legend(loc='upper left')


legend2=plt.legend([r"Neural Network (Detector simulation)"],loc=5,prop={'size':10},handlelength=0, handletextpad=0,frameon=False)
ax.add_artist(legend1)

legend3=plt.legend([r"Signal vs Background"],loc=4,prop={'size':10},handlelength=0, handletextpad=0,frameon=False)
ax.add_artist(legend2)


plt.xlabel(r'$\epsilon_{B}$',fontsize=16)
plt.ylabel(r'$\epsilon_{S}$',fontsize=20)
plab.savefig('ROCsbNNdelphes.png', bbox_inches=0,dpi=100)
plt.show()





