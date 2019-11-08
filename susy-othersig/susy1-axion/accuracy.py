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



a1,b1= loadtxt("list200k.dat", unpack=True)
a2,b2= loadtxt("list400k.dat", unpack=True)
a3,b3= loadtxt("list600k.dat", unpack=True)
a4,b4= loadtxt("list800k.dat", unpack=True)

fig = plt.figure()
ax = fig.add_subplot(111)
#plt.plot([0.452384174,0],[0.570439339,0],lw=4,color='black',linestyle='--', label=r'AUC=0.58')
#ax.plot(fpr1,tpr1,lw=2,color='red', label=r'AUC=0.58')
#ax.plot(fpr2,tpr2,lw=2,color='red',linestyle='--',label=r'AUC=0.60')


################
a=np.column_stack((a1,b1))

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
plot(x3, y3, "r", lw=1.8, linestyle="-", label="200K")
#plot(x4, y4, "o", lw=2)


###############
###############


a125=np.column_stack((a2,b2))

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
plot(x3pp, y3pp,color="blue", linewidth=1.8, linestyle="-",label="400K")
#plot(x4, y4, "o", lw=2)
################
a126=np.column_stack((a3,b3))

xp, yp = a126.T
tp = np.linspace(0, 2, len(xp))
t2p = np.linspace(0, 2, 200)

x2p = np.interp(t2p, tp, xp)
y2p = np.interp(t2p, tp, yp)
sigma = 5
x3p = gaussian_filter1d(x2p, sigma)
y3p = gaussian_filter1d(y2p, sigma)

x4p = np.interp(tp, t2p, x3p)
y4p = np.interp(tp, t2p, y3p)

#plot(x, y, "o-", lw=2)
plot(x3p, y3p,color="green", linewidth=1.8, linestyle="-", label="600K")
#plot(x4, y4, "o", lw=2)

###############


ann=np.column_stack((a4,b4))

xpn, ypn = ann.T
tpn = np.linspace(0, 2, len(xpn))
t2pn = np.linspace(0, 2, 200)

x2pn = np.interp(t2pn, tpn, xpn)
y2pn = np.interp(t2pn, tpn, ypn)
sigma = 5
x3pn = gaussian_filter1d(x2pn, sigma)
y3pn = gaussian_filter1d(y2pn, sigma)

x4pn = np.interp(tpn, t2pn, x3pn)
y4pn = np.interp(tpn, t2pn, y3pn)

#plot(x, y, "o-", lw=2)
plot(x3pn, y3pn,color="black", linewidth=1.8, linestyle="-", label="800K")
#plot(x4, y4, "o", lw=2)

###############




legend1= plt.legend(loc='upper left')



legend2=plt.legend([r"SUSY1 vs ALPs"],loc=4,prop={'size':10},handlelength=0, handletextpad=0,frameon=False)
ax.add_artist(legend1)

#legend3=plt.legend([r"Signal: Red-WIMP BP1, Blue-WIMP BP2, Green-WIMP BP3"],loc=4,prop={'size':10},handlelength=0, handletextpad=0,frameon=False)
#ax.add_artist(legend2)

plt.ylim(0.6,1.0)
plt.ylabel(r'Accuracy',fontsize=14)
plt.xlabel(r'Events/image (r)',fontsize=14)
plab.savefig('accuracyplot2dSUSY1vsAxion.png', bbox_inches=0,dpi=100)

plt.show()





