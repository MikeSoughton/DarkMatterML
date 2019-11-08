import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def hist1dError(data,weights=[],**kwargs):

    binCentres,hist,err = hist1dErrorInputs(data,weights)

    plt.bar(binCentres,hist,yerr=err,width=binCentres[1]-binCentres[0],**kwargs)
    plt.errorbar(binCentres,hist,yerr=err,drawstyle='steps-mid-')
    pass

def hist1dErrorInputs(data,weights=None,**kwargs):

    #if there aren't any weights just set them to one
    if not weights: weights = np.ones(len(data))
    
    #n,bins = np.histogram(data, **kwargs)
    hist,bins = np.histogram(data, weights=weights, **kwargs)
    err2,bins = np.histogram(data, weights=weights*weights, **kwargs)
    err=np.sqrt(err2)


    return (bins[1:]+bins[:-1])/2,hist,err

