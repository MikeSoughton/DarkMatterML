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

ap,bp,cp,dp,ep,fp,gp,hp,ip=loadtxt("alpnp1dijet.csv", unpack=True,skiprows=1)
a1,b1,c1,d1,e1,f1,g1,h1,i1=loadtxt("spin1meddijet.csv", unpack=True,skiprows=1)
a1p,b1p,c1p,d1p,e1p,f1p,g1p,h1p,i1p=loadtxt("susycor1dijet.csv", unpack=True,skiprows=1)
a2,b2,c2,d2,e2,f2,g2,h2,i2=loadtxt("susycor2dijet.csv", unpack=True,skiprows=1)
a3,b3,c3,d3,e3,f3,g3,h3,i3=loadtxt("susycor3dijet.csv", unpack=True,skiprows=1)


asm,bsm,csm,dsm,esm,fsm,gsm,hsm,ism=loadtxt("smnp1dijet.csv", unpack=True,skiprows=1)

w1 = np.full(50000,0.00002)


fig = plt.figure()  # create a figure object
ax = fig.add_axes([0.12, 0.13, 0.8, 0.8]) 

ax.hist(ap,color="red",bins=50,normed=False,histtype='step',weights = w1,range=(0,1000), linewidth=1.8,label=r"ALPs")
ax.hist(a1,color="green",bins=50,normed=False,histtype='step',weights = w1,range=(0,1000), linewidth=1.8,label=r"$M_Y$=1 TeV, $M_{\chi}=10$ GeV")
ax.hist(a1p,color="orange",bins=50,normed=False,histtype='step',weights = w1,range=(0,1000), linewidth=1.8,label=r"$M_{\tilde\chi^0_1}=100$ GeV")
ax.hist(a2,color="blue",bins=50,normed=False,histtype='step',weights = w1,range=(0,1000), linewidth=1.8,label=r"$M_{\tilde\chi^0_1}=200$ GeV")
ax.hist(a3,color="magenta",bins=50,normed=False,histtype='step',weights = w1,range=(0,1000), linewidth=1.8,label=r"$M_{\tilde\chi^0_1}=300$ GeV")
ax.hist(asm,color="black",bins=50,normed=False,histtype='step',weights = w1,range=(0,1000),linestyle="--", linewidth=1.8,label=r"Background")
#ax = gca()
leg = plt.legend(loc='lower left')
plt.xlabel(r'$p_T^{j_1}$',fontsize=16)
ax.set_yscale('log')
plt.ylabel(r'Fraction of events',fontsize=18)
plab.savefig('ptj1allsbdelphesdijet.png', bbox_inches=0,dpi=100)
plt.show()

#PTj1,ptj2,Eta1,Eta2,(jet1->Phi-jet2->Phi),<met->MET,<<(met->Phi-jet1->Phi),(met->Phi-jet2->Phi)
#A,    B,   c,   d,       e,                f,                  g,              h,               
#ptj1  ptj2 eta1 eta2	phijj	           met		    phimetj1	    phimetj2           


fig = plt.figure()  # create a figure object
ax = fig.add_axes([0.12, 0.13, 0.8, 0.8]) 
ax.hist(bp,color="red",bins=25,normed=False,histtype='step',weights = w1,range=(0,500), linewidth=1.8,label=r"ALPs")
ax.hist(b1,color="green",bins=25,normed=False,histtype='step',weights = w1,range=(0,500), linewidth=1.8,label=r"$M_Y$=1 TeV, $M_{\chi}=10$ GeV")
ax.hist(b1p,color="orange",bins=25,normed=False,histtype='step',weights = w1,range=(0,500), linewidth=1.8,label=r"$M_{\tilde\chi^0_1}=100$ GeV")
ax.hist(b2,color="blue",bins=25,normed=False,histtype='step',weights = w1,range=(0,500), linewidth=1.8,label=r"$M_{\tilde\chi^0_1}=200$ GeV")
ax.hist(b3,color="magenta",bins=25,normed=False,histtype='step',weights = w1,range=(0,500), linewidth=1.8,label=r"$M_{\tilde\chi^0_1}=300$ GeV")
ax.hist(bsm,color="black",bins=25,normed=False,histtype='step',weights = w1,range=(0,500),linestyle="--", linewidth=1.8,label=r"Background")
leg = plt.legend(loc='upper right')
plt.xlabel(r'$p_T^{j_2}$',fontsize=16)
ax.set_yscale('log')
plt.ylabel(r'Fraction of events',fontsize=18)
plab.savefig('ptj2allsbdelphesdijet.png', bbox_inches=0,dpi=100)
plt.show()


fig = plt.figure()  # create a figure object
ax = fig.add_axes([0.12, 0.13, 0.8, 0.8])
ax.hist(cp,color="red", bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5), linewidth=1.8,label=r"ALPs")
ax.hist(c1,color="green",bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5), linewidth=1.8,label=r"$M_Y$=1 TeV, $M_{\chi}=10$ GeV")
ax.hist(c1p,color="orange",bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5), linewidth=1.8,label=r"$M_{\tilde\chi^0_1}=100$ GeV")
ax.hist(c2,color="blue",bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5), linewidth=1.8,label=r"$M_{\tilde\chi^0_1}=200$ GeV")
ax.hist(c3,color="magenta",bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5), linewidth=1.8,label=r"$M_{\tilde\chi^0_1}=300$ GeV")
ax.hist(csm,color="black",bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5),linestyle="--", linewidth=1.8,label=r"Background")
leg = plt.legend(loc='lower center')
plt.xlabel(r'$\eta_{j_1}$',fontsize=16)
plt.ylabel(r'Fraction of events',fontsize=18)
plab.savefig('etaj1allsbdelphesdijet.png', bbox_inches=0,dpi=100)
plt.show()


fig = plt.figure()  # create a figure object
ax = fig.add_axes([0.12, 0.13, 0.8, 0.8])
ax.hist(dp,color="red", bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5), linewidth=1.8,label=r"ALPs")
ax.hist(d1,color="green",bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5), linewidth=1.8,label=r"$M_Y$=1 TeV, $M_{\chi}=10$ GeV")
ax.hist(d1p,color="orange",bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5), linewidth=1.8,label=r"$M_{\tilde\chi^0_1}=100$ GeV")
ax.hist(d2,color="blue",bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5), linewidth=1.8,label=r"$M_{\tilde\chi^0_1}=200$ GeV")
ax.hist(d3,color="magenta",bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5), linewidth=1.8,label=r"$M_{\tilde\chi^0_1}=300$ GeV")
ax.hist(dsm,color="black",bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5),linestyle="--", linewidth=1.8,label=r"Background")
leg = plt.legend(loc='lower center')
plt.xlabel(r'$\eta_{j_1}$',fontsize=16)
plt.ylabel(r'Fraction of events',fontsize=18)
plab.savefig('etaj2allsbdelphesdijet.png', bbox_inches=0,dpi=100)
plt.show()

fig = plt.figure()  # create a figure object
ax = fig.add_axes([0.12, 0.13, 0.8, 0.8]) 
ax.hist(ep,color="red", bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5), linewidth=1.8,label=r"ALPs")
ax.hist(e1,color="green",bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5), linewidth=1.8,label=r"$M_Y$=1 TeV, $M_{\chi}=10$ GeV")
ax.hist(e1p,color="orange",bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5), linewidth=1.8,label=r"$M_{\tilde\chi^0_1}=100$ GeV")
ax.hist(e2,color="blue",bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5), linewidth=1.8,label=r"$M_{\tilde\chi^0_1}=200$ GeV")
ax.hist(e3,color="magenta",bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5), linewidth=1.8,label=r"$M_{\tilde\chi^0_1}=300$ GeV")
ax.hist(esm,color="black",bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5),linestyle="--", linewidth=1.8,label=r"Background")
leg = plt.legend(loc='lower center')
plt.xlabel(r'$\Delta\phi_{j_1j_2}$',fontsize=16)
plt.ylabel(r'Fraction of events',fontsize=18)
plab.savefig('deltaphij1j2allsbdelphesdijet.png', bbox_inches=0,dpi=100)
plt.show()




fig = plt.figure()  # create a figure object
ax = fig.add_axes([0.12, 0.13, 0.8, 0.8])
ax.hist(fp,color="red",bins=40,normed=False,histtype='step',weights = w1,range=(0,800), linewidth=1.8,label=r"ALPs")
ax.hist(f1,color="green",bins=40,normed=False,histtype='step',weights = w1,range=(0,800), linewidth=1.8,label=r"$M_Y$=1 TeV, $M_{\chi}=10$ GeV")
ax.hist(f1p,color="orange",bins=40,normed=False,histtype='step',weights = w1,range=(0,800), linewidth=1.8,label=r"$M_{\tilde\chi^0_1}=100$ GeV")
ax.hist(f2,color="blue",bins=40,normed=False,histtype='step',weights = w1,range=(0,800), linewidth=1.8,label=r"$M_{\tilde\chi^0_1}=200$ GeV")
ax.hist(f3,color="magenta",bins=40,normed=False,histtype='step',weights = w1,range=(0,800), linewidth=1.8,label=r"$M_{\tilde\chi^0_1}=300$ GeV")
ax.hist(fsm,color="black",bins=40,normed=False,histtype='step',weights = w1,range=(0,800),linestyle="--", linewidth=1.8,label=r"Background")
leg = plt.legend(loc='upper right')
plt.xlabel(r'MET',fontsize=16)
plt.ylabel(r'Fraction of events',fontsize=18)
plab.savefig('METallsbdelphesdijet.png', bbox_inches=0,dpi=100)
plt.show()       



fig = plt.figure()  # create a figure object
ax = fig.add_axes([0.12, 0.13, 0.8, 0.8]) 
ax.hist(gp,color="red", bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5), linewidth=1.8,label=r"ALPs")
ax.hist(g1,color="green",bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5), linewidth=1.8,label=r"$M_Y$=1 TeV, $M_{\chi}=10$ GeV")
ax.hist(g1p,color="orange",bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5), linewidth=1.8,label=r"$M_{\tilde\chi^0_1}=100$ GeV")
ax.hist(g2,color="blue",bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5), linewidth=1.8,label=r"$M_{\tilde\chi^0_1}=200$ GeV")
ax.hist(g3,color="magenta",bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5), linewidth=1.8,label=r"$M_{\tilde\chi^0_1}=300$ GeV")
ax.hist(gsm,color="black",bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5),linestyle="--", linewidth=1.8,label=r"Background")
leg = plt.legend(loc='lower center')
plt.xlabel(r'$\Delta\phi_{{MET}}^{j_1}$',fontsize=16)
plt.ylabel(r'Fraction of events',fontsize=18)
plab.savefig('deltaphiMETj1allsbdelphesdijet.png', bbox_inches=0,dpi=100)
plt.show()



fig = plt.figure()  # create a figure object
ax = fig.add_axes([0.12, 0.13, 0.8, 0.8]) 
ax.hist(hp,color="red", bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5), linewidth=1.8,label=r"ALPs")
ax.hist(h1,color="green",bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5), linewidth=1.8,label=r"$M_Y$=1 TeV, $M_{\chi}=10$ GeV")
ax.hist(h1p,color="orange",bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5), linewidth=1.8,label=r"$M_{\tilde\chi^0_1}=100$ GeV")
ax.hist(h2,color="blue",bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5), linewidth=1.8,label=r"$M_{\tilde\chi^0_1}=200$ GeV")
ax.hist(h3,color="magenta",bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5), linewidth=1.8,label=r"$M_{\tilde\chi^0_1}=300$ GeV")
ax.hist(hsm,color="black",bins=28,normed=False,histtype='step',weights = w1,range=(-3.5,3.5),linestyle="--", linewidth=1.8,label=r"Background")
leg = plt.legend(loc='lower right')
plt.xlabel(r'$\Delta\phi_{{MET}}^{j_2}$',fontsize=16)
plt.ylabel(r'Fraction of events',fontsize=18)
plab.savefig('deltaphiMETj2allsbdelphesdijet.png', bbox_inches=0,dpi=100)
plt.show()




