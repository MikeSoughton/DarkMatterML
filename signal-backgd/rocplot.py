import numpy as np
import warnings
import matplotlib.pyplot as plt
import pylab as plab



#tpr1=np.loadtxt("axion-sm/tpr.txt", unpack=True)
#fpr1=np.loadtxt("axion-sm/fpr.txt", unpack=True)

#tpr2=np.loadtxt("eft-sm/tpr.txt", unpack=True)
#fpr2=np.loadtxt("eft-sm/fpr.txt", unpack=True)


#tpr3=np.loadtxt("susy1-sm/tpr.txt", unpack=True)
#fpr3=np.loadtxt("susy1-sm/fpr.txt", unpack=True)

#tpr4=np.loadtxt("susy2-sm/tpr.txt", unpack=True)
#fpr4=np.loadtxt("susy2-sm/fpr.txt", unpack=True)

tpr5par=np.loadtxt("partonlevel/NN-sb/susy3-sm/tpr.txt", unpack=True)
fpr5par=np.loadtxt("partonlevel/NN-sb/susy3-sm/fpr.txt", unpack=True)


tpr5del=np.loadtxt("delphes-revised/NN-sb/susy3-sm/tpr.txt", unpack=True)
fpr5del=np.loadtxt("delphes-revised/NN-sb/susy3-sm/fpr.txt", unpack=True)


fig = plt.figure()
ax = fig.add_subplot(111)

#ax.plot(fpr1,tpr1,lw=2,color='red', label=r'ALPs, AUC=0.59')
#ax.plot(fpr2,tpr2,lw=2,color='green', label=r'EFT, AUC=0.72')
#ax.plot(fpr3,tpr3,lw=2,color='blue', label=r'WIMP BP1, AUC=0.64')
#ax.plot(fpr4,tpr4,lw=2,color='blue',linestyle='--', label=r'WIMP BP2, AUC=0.70')
ax.plot(fpr5par,tpr5par,lw=2,color='magenta', label=r'SUSY3 - Parton Level, AUC=0.72')#linestyle=':'
ax.plot(fpr5del,tpr5del,lw=2,color='magenta', linestyle='--', label=r'SUSY3 - Detector Level, AUC=0.69')



legend1= plt.legend(loc='upper left')


#legend2=plt.legend([r"Neural Network"],loc=5,prop={'size':10},handlelength=0, handletextpad=0,frameon=False)
#ax.add_artist(legend1)

legend2=plt.legend([r"Signal vs Background"],loc=4,prop={'size':10},handlelength=0, handletextpad=0,frameon=False)
ax.add_artist(legend1)

plt.title(r'DNN')
plt.xlabel(r'$\epsilon_{B}$',fontsize=16)
plt.ylabel(r'$\epsilon_{S}$',fontsize=20)
plab.savefig('ROCsbNNcomparePLandDelphes.png', bbox_inches=0,dpi=100)
plt.show()





