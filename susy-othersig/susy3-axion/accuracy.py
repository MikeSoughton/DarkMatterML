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

a10,b10,error10= loadtxt("list10k.dat", unpack=True)

a20,b20,error20= loadtxt("list20k.dat", unpack=True)

a1,b1,error50= loadtxt("list50k.dat", unpack=True)
a2,b2= loadtxt("list100k.dat", unpack=True)
a3,b3= loadtxt("list200k.dat", unpack=True)
a4,b4= loadtxt("list800k.dat", unpack=True)

fig = plt.figure()
ax = fig.add_subplot(111)
#plt.plot([0.452384174,0],[0.570439339,0],lw=4,color='black',linestyle='--', label=r'AUC=0.58')
#ax.plot(fpr1,tpr1,lw=2,color='red', label=r'AUC=0.58')
#ax.plot(fpr2,tpr2,lw=2,color='red',linestyle='--',label=r'AUC=0.60')



################
ap10=np.column_stack((a10,b10))

a10p = np.asarray([10, 25, 50, 100, 200])
b10p = np.asarray([0.72, 0.875, 0.885, 0.95, 0.95])
error10p = np.asarray([0.06, 0.065, 0.085, 0.050, 0.050])

#plt.errorbar(a10p,b10p,error10p)
#plt.fill_between(a10p, b10p-error10p, b10p+error10p)



x10, y10 = ap10.T
t10 = np.linspace(0, 3, len(x10))
t210 = np.linspace(0, 4, 120)

x210 = np.interp(t210, t10, x10)
y210 = np.interp(t210, t10, y10)
sigma = 12
x310 = gaussian_filter1d(x210, sigma)
y310 = gaussian_filter1d(y210, sigma)

x410 = np.interp(t10, t210, x310)
y410 = np.interp(t10, t210, y310)

#plot(x, y, "o-", lw=2)
plot(x310, y310, "orange", lw=1.8, linestyle="-", label="10K")
#plot(x4, y4, "o", lw=2)










################
ap20=np.column_stack((a20,b20))

x20, y20 = ap20.T
t20 = np.linspace(0, 3, len(x20))
t220 = np.linspace(0, 4, 150)

x220 = np.interp(t220, t20, x20)
y220 = np.interp(t220, t20, y20)
sigma = 5
x320 = gaussian_filter1d(x220, sigma)
y320 = gaussian_filter1d(y220, sigma)

x420 = np.interp(t20, t220, x320)
y420 = np.interp(t20, t220, y320)

#plot(x, y, "o-", lw=2)
plot(x320, y320, "magenta", lw=1.8, linestyle="-", label="20K")
#plot(x4, y4, "o", lw=2)



################
a=np.column_stack((a1,b1))

x, y = a.T
t = np.linspace(0, 2, len(x))
t2 = np.linspace(0, 2, 150)

x2 = np.interp(t2, t, x)
y2 = np.interp(t2, t, y)
sigma = 5
x3 = gaussian_filter1d(x2, sigma)
y3 = gaussian_filter1d(y2, sigma)

x4 = np.interp(t, t2, x3)
y4 = np.interp(t, t2, y3)

#plot(x, y, "o-", lw=2)
plot(x3, y3, "r", lw=1.8, linestyle="-", label="50K")
#plot(x4, y4, "o", lw=2)


###############
###############


a125=np.column_stack((a2,b2))

xpp, ypp = a125.T
tpp = np.linspace(0, 2, len(xpp))
t2pp = np.linspace(0, 2, 150)

x2pp = np.interp(t2pp, tpp, xpp)
y2pp = np.interp(t2pp, tpp, ypp)
sigma = 5
x3pp = gaussian_filter1d(x2pp, sigma)
y3pp = gaussian_filter1d(y2pp, sigma)

x4pp = np.interp(tpp, t2pp, x3pp)
y4pp = np.interp(tpp, t2pp, y3pp)

#plot(x, y, "o-", lw=2)
plot(x3pp, y3pp,color="blue", linewidth=1.8, linestyle="-",label="100K")
#plot(x4, y4, "o", lw=2)
################
a126=np.column_stack((a3,b3))

xp, yp = a126.T
tp = np.linspace(0, 2, len(xp))
t2p = np.linspace(0, 2, 150)

x2p = np.interp(t2p, tp, xp)
y2p = np.interp(t2p, tp, yp)
sigma = 5
x3p = gaussian_filter1d(x2p, sigma)
y3p = gaussian_filter1d(y2p, sigma)

x4p = np.interp(tp, t2p, x3p)
y4p = np.interp(tp, t2p, y3p)

#plot(x, y, "o-", lw=2)
plot(x3p, y3p,color="green", linewidth=1.8, linestyle="-", label="200K")
#plot(x4, y4, "o", lw=2)

###############


ann=np.column_stack((a4,b4))

xpn, ypn = ann.T
tpn = np.linspace(0, 2, len(xpn))
t2pn = np.linspace(0, 2, 150)

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




legend1= plt.legend(loc='lower right')



legend2=plt.legend([r"SUSY3 vs ALPs"],loc=5,prop={'size':10},handlelength=0, handletextpad=0,frameon=False)
ax.add_artist(legend1)

#legend3=plt.legend([r"Signal: Red-WIMP BP1, Blue-WIMP BP2, Green-WIMP BP3"],loc=4,prop={'size':10},handlelength=0, handletextpad=0,frameon=False)
#ax.add_artist(legend2)

plt.ylim(0.70,1.0)
plt.ylabel(r'Accuracy',fontsize=14)
plt.xlabel(r'Events/image (r)',fontsize=14)
plab.savefig('accuracyplot2dSUSY3vsAxion.png', bbox_inches=0,dpi=100)

plt.show()





