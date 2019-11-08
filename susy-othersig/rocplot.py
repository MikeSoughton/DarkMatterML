import numpy as np
import warnings
import matplotlib.pyplot as plt
import pylab as plab
import math
tpr1=np.loadtxt("susy1-axion/tpr.txt", unpack=True)
fpr1=np.loadtxt("susy1-axion/fpr.txt", unpack=True)

#fp1=np.arange(0.452384174,0)
#tp1=np.arange(0.570439339,0)



tpr2=np.loadtxt("susy1-eft/tpr.txt", unpack=True)
fpr2=np.loadtxt("susy1-eft/fpr.txt", unpack=True)

tpr3=np.loadtxt("susy2-axion/tpr.txt", unpack=True)
fpr3=np.loadtxt("susy2-axion/fpr.txt", unpack=True)


tpr4=np.loadtxt("susy2-eft/tpr.txt", unpack=True)
fpr4=np.loadtxt("susy2-eft/fpr.txt", unpack=True)

tpr5=np.loadtxt("susy3-axion/tpr.txt", unpack=True)
fpr5=np.loadtxt("susy3-axion/fpr.txt", unpack=True)

tpr6=np.loadtxt("susy3-eft/tpr.txt", unpack=True)
fpr6=np.loadtxt("susy3-eft/fpr.txt", unpack=True)

fig = plt.figure()
ax = fig.add_subplot(111)
#plt.plot([0.452384174,0],[0.570439339,0],lw=4,color='black',linestyle='--', label=r'AUC=0.58')
ax.plot(fpr1,tpr1,lw=2,color='Orange', label=r'SUSY1 vs ALPs, AUC=0.58')
ax.plot(fpr2,tpr2,lw=2,color='Orange',linestyle='--',label=r'SUSY1 vs EFT, AUC=0.60')
ax.plot(fpr3,tpr3,lw=2,color='blue', label=r'SUSY2 vs ALPs, AUC=0.64')
ax.plot(fpr4,tpr4,lw=2,color='blue',linestyle='--', label=r'SUSY2 vs EFT, AUC=0.53')
ax.plot(fpr5,tpr5,lw=2,color='magenta', label=r'SUSY3 vs ALPs, AUC=0.67')
ax.plot(fpr6,tpr6,lw=2,color='magenta',linestyle='--', label=r'SUSY3 vs EFT, AUC=0.50')


legend1= plt.legend(loc='upper left')

legend2=plt.legend([r"Parton Level"],loc=5,prop={'size':10},handlelength=0, handletextpad=0,frameon=False)
ax.add_artist(legend1)

#legend2=plt.legend([r"Neural Network, Solid Curves: Axion(B), Dashed Curves: EFT(B)"],loc=5,prop={'size':10},handlelength=0, handletextpad=0,frameon=False)
#ax.add_artist(legend1)

#legend3=plt.legend([r"Signal: Red-WIMP BP1, Blue-WIMP BP2, Green-WIMP BP3"],loc=4,prop={'size':10},handlelength=0, handletextpad=0,frameon=False)
#ax.add_artist(legend2)

plt.title(r'DNN')
plt.xlabel(r'$\epsilon_{B}$',fontsize=16)
plt.ylabel(r'$\epsilon_{S}$',fontsize=20)
plab.savefig('ROCssNN.png', bbox_inches=0,dpi=100)
plt.show()





