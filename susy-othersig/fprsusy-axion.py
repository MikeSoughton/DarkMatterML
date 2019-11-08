from scipy import * 
from numpy import *
from numpy import ma
from pylab import *
import numpy as np
import pylab as plab
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import scipy.stats as ss
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.tri import *
import math
from matplotlib.tri import Triangulation, TriAnalyzer, UniformTriRefiner
import scipy.interpolate
import matplotlib.tri as mtri
from matplotlib import colors, ticker, cm
from scipy.ndimage import gaussian_filter1d



a2,b2= loadtxt("wimp-axion.dat", unpack=True)
a3,b3= loadtxt("wimp-eft.dat", unpack=True)


fig = plt.figure()
ax = fig.add_subplot(111)
#plt.plot([0.452384174,0],[0.570439339,0],lw=4,color='black',linestyle='--', label=r'AUC=0.58')
#ax.plot(fpr1,tpr1,lw=2,color='red', label=r'AUC=0.58')
#ax.plot(fpr2,tpr2,lw=2,color='red',linestyle='--',label=r'AUC=0.60')


################
a=np.column_stack((a2,b2))

x, y = a.T
t = np.linspace(0, 2, len(x))
t2 = np.linspace(0, 2, 200)

x2 = np.interp(t2, t, x)
y2 = np.interp(t2, t, y)
sigma = 5
x3 = gaussian_filter1d(x2, sigma)
y3 = gaussian_filter1d(y2, sigma)

x4 = np.interp(t, t2, x3)
y4 = np.interp(t, t2, y3)

#plot(x, y, "o-", lw=2)
plot(x3, y3, "r", lw=1.8, linestyle="-", label="ALPs as SUSY WIMP")
#plot(x4, y4, "o", lw=2)


###############
###############


a125=np.column_stack((a3,b3))

xpp, ypp = a125.T
tpp = np.linspace(0, 2, len(xpp))
t2pp = np.linspace(0, 2, 200)

x2pp = np.interp(t2pp, tpp, xpp)
y2pp = np.interp(t2pp, tpp, ypp)
sigma = 5
x3pp = gaussian_filter1d(x2pp, sigma)
y3pp = gaussian_filter1d(y2pp, sigma)

x4pp = np.interp(tpp, t2pp, x3pp)
y4pp = np.interp(tpp, t2pp, y3pp)

#plot(x, y, "o-", lw=2)
plot(x3pp, y3pp, "g", lw=2, linewidth=1.8, linestyle="-",label="EFT as SUSY WIMP")
#plot(x4, y4, "o", lw=2)
################








legend1= plt.legend(loc='upper left')



#legend2=plt.legend([r"Neural Network, Solid Curves: Axion(B), Dashed Curves: EFT(B)"],loc=5,prop={'size':10},handlelength=0, handletextpad=0,frameon=False)
#ax.add_artist(legend1)

#legend3=plt.legend([r"Signal: Red-WIMP BP1, Blue-WIMP BP2, Green-WIMP BP3"],loc=4,prop={'size':10},handlelength=0, handletextpad=0,frameon=False)
#ax.add_artist(legend2)

plt.ylim(0.3,0.6)
plt.ylabel(r'$P_{ALPs/EFT}$',fontsize=14)
plt.xlabel(r'$M_{\tilde \chi_0}$ (GeV)',fontsize=14)
plab.savefig('fprsusy-ALPeft.png', bbox_inches=0,dpi=100)

plt.show()





