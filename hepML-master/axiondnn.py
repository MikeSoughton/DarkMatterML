#!/usr/bin/env python
# coding: utf-8
# # First load os and sys so I can update the sys.path with new functions
# change the luminosity to 80 /fb
# generate the 3 plots as in aewol paper.

import os
import sys
#take the paths to the functions we nedd
#module_path = os.path.abspath(os.path.join('./pandasPlotting/'))
#module2_path = os.path.abspath(os.path.join('./MlClasses/'))
#module3_path = os.path.abspath(os.path.join('./MlFunctions/'))
# this part will include in the sys.path variables the paths for our new functions
#if [module_path, module2_path, module3_path] not in sys.path:
#    sys.path.append(module_path)


# here we are going to load what we will need, keras + tensorflow, plot functions, etc..
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import math
import time

from keras import callbacks

from pandasPlotting.Plotter import Plotter
from pandasPlotting.dfFunctions import expandArrays
from pandasPlotting.dtFunctions import featureImportance

from MlClasses.MlData import MlData
from MlClasses.Bdt import Bdt
from MlClasses.Dnn import Dnn
from MlClasses.ComparePerformances import ComparePerformances

from MlFunctions.DnnFunctions import significanceLoss,significanceLossInvert,significanceLoss2Invert,significanceLossInvertSqrt,significanceFull,asimovSignificanceLoss,asimovSignificanceLossInvert,asimovSignificanceFull,truePositive,falsePositive

from linearAlgebraFunctions import gram,addGramToFlatDF

# I don't have patience to training 200 epochs ¯\_(ツ)_/¯
earlyStopping = callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=5)

#load our data files
signal=pd.read_csv("../analysis/data/mon-delphes-updated/susycor3.csv",sep='\s+',engine='python')
bkgd=pd.read_csv("../analysis/data/mon-delphes-updated/smnp1.csv",sep='\s+',engine='python')


#combine them into one dataset
combined = pd.concat([signal,bkgd]).sample(frac=1)

print(combined.keys())

# change thes vars depend on which dataset you are loading, I will implement a better solution.
chosenVars = {
            # #A vanilla analysis with HL variables and lead 3 jets
            '0L':['ptj',     'etaj', 'phij',   'signal'],
            # , 'phij'
}
#'PTJ1', 'PTj2', 'Etaj1', 'Etaj2', 'phijj', 'MET', 'metPhi', 'metphij1', 'metphij2', 'signal'
trainedModels={}
#needed to plot asimov significane
asimovSigLossSysts=[0.01,0.05,0.1,0.2,0.3,0.4,0.5]



# here I have included one archtecture I got from my ES scan, pls comment my entry and use dnn_batch4096 instead.
dnnConfigs={
    'dnn_ZH_0L_cHW_0d03_batch_1024':{'epochs':100,'batch_size':500,'dropOut':0.2,'l2Regularization':None,'hiddenLayers':[5.0,64,64,64],
                 'optimizer':'adadelta', 'activation':'tanh'}
    }


lumi=80. #luminosity in /fb
expectedSignal=14*lumi 
expectedBkgd=2*lumi #cross section of ttbar sample in fb times efficiency measured by Marco
systematic=0.1 #systematic for the asimov signficance



for varSetName,varSet in chosenVars.items():
    #Pick out the expanded arrays
    columnsInDataFrame = []
    for k in combined.keys():
        for v in varSet:
            #Little trick to ensure only the start of the string is checked
            if varSetName is '0L':
                if ' '+v+' ' in ' '+k+' ': columnsInDataFrame.append(k)
            elif ' '+v in ' '+k: columnsInDataFrame.append(k)


    #Select just the features we're interested in
    #For now setting NaNs to 0 for compatibility
    combinedToRun = combined[columnsInDataFrame].copy()
    combinedToRun.fillna(0,inplace=True)
    
    combinedToRun.index = np.arange(0,400000)
    mlData = MlData(combinedToRun,'signal')

    mlData.prepare(evalSize=0.0,testSize=0.3,limitSize=None)

    for name,config in dnnConfigs.items():
        dnn = Dnn(mlData,'testPlots/'+varSetName+'/'+name)
        dnn.setup(hiddenLayers=config['hiddenLayers'], dropOut=config['dropOut'],
                  l2Regularization=config['l2Regularization'],
                  loss='binary_crossentropy',
                extraMetrics=[])
        dnn.fit(epochs=config['epochs'],batch_size=config['batch_size'])
        
        dnn.explainPredictions()
        dnn.diagnostics(batchSize=config['batch_size'])
        dnn.makeHepPlots(expectedSignal,expectedBkgd,asimovSigLossSysts,makeHistograms=False)

        trainedModels[varSetName+'_'+name]=dnn

