# Importing the EFT Data set
import numpy as np
import warnings
import matplotlib.pyplot as plt
import pylab as plab

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




a1new,b1new,c1new,d1new=loadtxt("susycor1.csv", unpack=True,skiprows=1)
a2new,b2new,c2new,d2new=loadtxt("susycor2.csv", unpack=True,skiprows=1)
a3new,b3new,c3new,d3new=loadtxt("susycor3.csv", unpack=True,skiprows=1)
asmnew,bsmnew,csmnew,dsmnew=loadtxt("smnp1.csv", unpack=True,skiprows=1)

axnew,bxnew,cxnew,dxnew=loadtxt("alpnp1.csv", unpack=True,skiprows=1)

aspnew,bspnew,cspnew,dspnew=loadtxt("spin1med.csv", unpack=True,skiprows=1)

w1 = np.full(200000,0.000005)



plt.hist(axnew,color="red",bins=50,normed=False,histtype='step',weights = w1,range=(0,1000), linewidth=1.8,label=r"ALPs")
plt.hist(aspnew,color="green",bins=50,normed=False,histtype='step',weights = w1,range=(0,1000), linewidth=1.8,label=r"$M_Y$=1 TeV, $M_{\chi}=10$ GeV")
plt.hist(a1new,color="orange",bins=50,normed=False,histtype='step',weights = w1,range=(0,1000), linewidth=1.8,label=r"$M_{\tilde\chi^0_1}=100$ GeV")
plt.hist(a2new,color="blue",bins=50,normed=False,histtype='step',weights = w1,range=(0,1000), linewidth=1.8,label=r"$M_{\tilde\chi^0_1}=200$ GeV")
plt.hist(a3new,color="magenta",bins=50,normed=False,histtype='step',weights = w1,range=(0,1000), linewidth=1.8,label=r"$M_{\tilde\chi^0_1}=300$ GeV")
plt.hist(asmnew,color="black",bins=50,normed=False,histtype='step',weights = w1,range=(0,1000),linestyle="--", linewidth=1.8,label=r"Background")
ax = gca()
leg = plt.legend(loc='lower left')
plt.xlabel(r'$p_T$',fontsize=16)
ax.set_yscale('log')
plt.ylabel(r'Fraction of events',fontsize=18)
plab.savefig('ptjallsbdelphes.png', bbox_inches=0,dpi=100)
plt.show()



plt.hist(bxnew,color="red",bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5), linewidth=1.8,label=r"ALPs")
plt.hist(bspnew,color="green",bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5), linewidth=1.8,label=r"$M_Y$=1 TeV, $M_{\chi}=10$ GeV")
plt.hist(b1new,color="orange",bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5), linewidth=1.8,label=r"$M_{\tilde\chi^0_1}=100$ GeV")
plt.hist(b2new,color="blue",bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5), linewidth=1.8,label=r"$M_{\tilde\chi^0_1}=200$ GeV")
plt.hist(b3new,color="magenta",bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5), linewidth=1.8,label=r"$M_{\tilde\chi^0_1}=300$ GeV")
plt.hist(bsmnew,color="black",bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5),linestyle="--", linewidth=1.8,label=r"Background")
leg = plt.legend(loc='lower center')
plt.xlabel(r'$\eta$',fontsize=16)
plt.ylabel(r'Fraction of events',fontsize=18)
plab.savefig('etajallsbdelphes.png', bbox_inches=0,dpi=100)
plt.show()



plt.hist(cxnew,color="red", bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5), linewidth=1.8,label=r"ALPs")
plt.hist(cspnew,color="green",bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5), linewidth=1.8,label=r"$M_Y$=1 TeV, $M_{\chi}=10$ GeV")
plt.hist(c1new,color="orange",bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5), linewidth=1.8,label=r"$M_{\tilde\chi^0_1}=100$ GeV")
plt.hist(c2new,color="blue",bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5), linewidth=1.8,label=r"$M_{\tilde\chi^0_1}=200$ GeV")
plt.hist(c3new,color="magenta",bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5), linewidth=1.8,label=r"$M_{\tilde\chi^0_1}=300$ GeV")
plt.hist(csmnew,color="black",bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5),linestyle="--", linewidth=1.8,label=r"Background")
leg = plt.legend(loc='lower center')
plt.xlabel(r'$\phi$',fontsize=16)
plt.ylabel(r'Fraction of events',fontsize=18)
plab.savefig('phijallsbdelphes.png', bbox_inches=0,dpi=100)
plt.show()






# ptj     etaj    phij   missinget
#  a      b       c      d          




