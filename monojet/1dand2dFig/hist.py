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

ap,bp,cp,dp=loadtxt("../../../../axion/monojet/monoj.csv", unpack=True)

a1,b1,c1,d1=loadtxt("../../../../spin1/monojet/spin1med10.csv", unpack=True)

a1p,b1p,c1p,d1p=loadtxt("../../../../susy/monojet/susy1.csv", unpack=True)
a2,b2,c2,d2,e2=loadtxt("../../../../susy/monojet/susy2.csv", unpack=True)
a3,b3,c3,d3=loadtxt("../../../../susy/monojet/susy3.csv", unpack=True)


asm,bsm,csm,dsm=loadtxt("../../../../smbackground/smbckgd.csv", unpack=True)

w1 = np.full(400000,0.0000025)

plt.hist(ap,color="red",bins=50,normed=False,histtype='step',weights = w1,range=(0,1000), linewidth=1.8,label=r"ALPs")
plt.hist(a1,color="green",bins=50,normed=False,histtype='step',weights = w1,range=(0,1000), linewidth=1.8,label=r"$M_Y$=1 TeV, $M_{\chi}=10$ GeV")
plt.hist(a1p,color="orange",bins=50,normed=False,histtype='step',weights = w1,range=(0,1000), linewidth=1.8,label=r"$M_{\tilde\chi^0_1}=100$ GeV")
plt.hist(a2,color="blue",bins=50,normed=False,histtype='step',weights = w1,range=(0,1000), linewidth=1.8,label=r"$M_{\tilde\chi^0_1}=200$ GeV")
plt.hist(a3,color="magenta",bins=50,normed=False,histtype='step',weights = w1,range=(0,1000), linewidth=1.8,label=r"$M_{\tilde\chi^0_1}=300$ GeV")
plt.hist(asm,color="black",bins=50,normed=False,histtype='step',weights = w1,range=(0,1000),linestyle="--", linewidth=1.8,label=r"Background")
ax = gca()
leg = plt.legend(loc='lower left')
plt.xlabel(r'$p_T$',fontsize=16)
ax.set_yscale('log')
plt.ylabel(r'Fraction of events',fontsize=18)
plab.savefig('ptjallsb.png', bbox_inches=0,dpi=100)
plt.show()



plt.hist(bp,color="red",bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5), linewidth=1.8,label=r"ALPs")
plt.hist(b1,color="green",bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5), linewidth=1.8,label=r"$M_Y$=1 TeV, $M_{\chi}=10$ GeV")
plt.hist(b1p,color="orange",bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5), linewidth=1.8,label=r"$M_{\tilde\chi^0_1}=100$ GeV")
plt.hist(b2,color="blue",bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5), linewidth=1.8,label=r"$M_{\tilde\chi^0_1}=200$ GeV")
plt.hist(b3,color="magenta",bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5), linewidth=1.8,label=r"$M_{\tilde\chi^0_1}=300$ GeV")
plt.hist(bsm,color="black",bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5),linestyle="--", linewidth=1.8,label=r"Background")
leg = plt.legend(loc='lower center')
plt.xlabel(r'$\eta$',fontsize=16)
plt.ylabel(r'Fraction of events',fontsize=18)
plab.savefig('etajallsb.png', bbox_inches=0,dpi=100)
plt.show()



plt.hist(cp,color="red", bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5), linewidth=1.8,label=r"ALPs")
plt.hist(c1,color="green",bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5), linewidth=1.8,label=r"$M_Y$=1 TeV, $M_{\chi}=10$ GeV")
plt.hist(c1p,color="orange",bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5), linewidth=1.8,label=r"$M_{\tilde\chi^0_1}=100$ GeV")
plt.hist(c2,color="blue",bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5), linewidth=1.8,label=r"$M_{\tilde\chi^0_1}=200$ GeV")
plt.hist(c3,color="magenta",bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5), linewidth=1.8,label=r"$M_{\tilde\chi^0_1}=300$ GeV")
plt.hist(csm,color="black",bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5),linestyle="--", linewidth=1.8,label=r"Background")
leg = plt.legend(loc='lower center')
plt.xlabel(r'$\phi$',fontsize=16)
plt.ylabel(r'Fraction of events',fontsize=18)
plab.savefig('phijallsb.png', bbox_inches=0,dpi=100)
plt.show()


#plt.hist(dp,color="red", bins=32,normed=False,histtype='step',range=(0,800), linewidth=1.8,label=r"ALPs")
#plt.hist(d1,color="green",bins=32,normed=False,histtype='step',range=(0,800), linewidth=1.8,label=r"$M_Y$=1 TeV, $M_{\chi}=10$ GeV")
##plt.hist(d1,color="blue",bins=50,normed=True,histtype='step',range=(0,1000), linewidth=1.8,label=r"Spin1-med-100")
##plt.hist(d2,color="black",bins=50,normed=True,histtype='step',range=(0,1000), linewidth=1.8,label=r"Spin1-med-500")
#leg = plt.legend(loc='lower right')
#plt.xlabel(r'MET',fontsize=16)
#plt.ylabel(r'Fraction of events',fontsize=18)
#plab.savefig('metaxspin1.png', bbox_inches=0,dpi=100)
#plt.show()


plt.scatter(b1,a1,color="green",facecolors='none',s=60,label=r"$M_Y$=1 TeV, $M_{\chi}=10$ GeV")
plt.scatter(bp,ap,color="red",facecolors='none',s=60,label=r"ALPs")
legend1=plt.legend(loc='upper left',prop={'size':16}, fontsize=20)
plt.ylabel(r'$p_T$ (GeV)',fontsize=14)
plt.xlabel(r'$\eta$',fontsize=16)
#plt.ylim([0, max(d1)])
#plt.xlim([0, max(c1)])
plab.savefig('ptjetajaxspin1.png',origin='(left, bottom)')
plt.show()



plt.scatter(b1p,a1p,color="green",facecolors='none',s=60,label=r"$M_{\tilde\chi^0_1}=100$ GeV")
plt.scatter(bsm,asm,color="red",facecolors='none',s=60,label=r"Background")
legend1=plt.legend(loc='upper left',prop={'size':16}, fontsize=20)
plt.ylabel(r'$p_T$ (GeV)',fontsize=14)
plt.xlabel(r'$\eta$',fontsize=16)
#plt.ylim([0, max(d1)])
#plt.xlim([0, max(c1)])
plab.savefig('ptjetajsmsusy1.png',origin='(left, bottom)')
plt.show()


plt.scatter(b3,a3,color="green",facecolors='none',s=60,label=r"$M_{\tilde\chi^0_1}=300$ GeV")
plt.scatter(bsm,asm,color="red",facecolors='none',s=60,label=r"Background")
legend1=plt.legend(loc='upper left',prop={'size':16}, fontsize=20)
plt.ylabel(r'$p_T$ (GeV)',fontsize=14)
plt.xlabel(r'$\eta$',fontsize=16)
#plt.ylim([0, max(d1)])
#plt.xlim([0, max(c1)])
plab.savefig('ptjetajsmsusy3.png',origin='(left, bottom)')
plt.show()





plt.scatter(b1,a1,color="green",facecolors='none',s=60,label=r"$M_Y$=1 TeV, $M_{\chi}=10$ GeV")
plt.scatter(bsm,asm,color="red",facecolors='none',s=60,label=r"Background")
legend1=plt.legend(loc='upper left',prop={'size':16}, fontsize=20)
plt.ylabel(r'$p_T$ (GeV)',fontsize=14)
plt.xlabel(r'$\eta$',fontsize=16)
#plt.ylim([0, max(d1)])
#plt.xlim([0, max(c1)])
plab.savefig('ptjetajsmeft.png',origin='(left, bottom)')
plt.show()


plt.scatter(bp,ap,color="green",facecolors='none',s=60,label=r"ALPs")
plt.scatter(bsm,asm,color="red",facecolors='none',s=60,label=r"Background")
legend1=plt.legend(loc='upper left',prop={'size':16}, fontsize=20)
plt.ylabel(r'$p_T$ (GeV)',fontsize=14)
plt.xlabel(r'$\eta$',fontsize=16)
#plt.ylim([0, max(d1)])
#plt.xlim([0, max(c1)])
plab.savefig('ptjetajsmaxion.png',origin='(left, bottom)')
plt.show()

plt.scatter(b1p,a1p,color="green",facecolors='none',s=60,label=r"$M_{\tilde\chi^0_1}=100$ GeV")
plt.scatter(bp,ap,color="red",facecolors='none',s=60,label=r"ALPs")
legend1=plt.legend(loc='upper left',prop={'size':16}, fontsize=20)
plt.ylabel(r'$p_T$ (GeV)',fontsize=14)
plt.xlabel(r'$\eta$',fontsize=16)
#plt.ylim([0, max(d1)])
#plt.xlim([0, max(c1)])
plab.savefig('ptjetajaxsusy1.png',origin='(left, bottom)')
plt.show()



# ptj     etaj    phij   missinget
#  a      b       c      d          




