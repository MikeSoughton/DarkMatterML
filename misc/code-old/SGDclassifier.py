import sys, os
import pandas as pd
import seaborn as sns
import numpy as np
import warnings
#Commnet this to turn on warnings
import pylab as plab
#import ml_style as style
#import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
#from sklearn import standardscalars 
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import roc_curve, auc, accuracy_score, precision_recall_curve

scaler = StandardScaler()
SM_df=pd.read_csv("spin1med.csv",sep='\s+',engine='python')
cHW_df=pd.read_csv("monoj.csv",sep='\s+',engine='python')

min_max_scaler = preprocessing.MinMaxScaler()
def scaleColumns(df, cols_to_scale):
    for col in cols_to_scale:
        df[col] = pd.DataFrame(min_max_scaler.fit_transform(pd.DataFrame(df[col])),columns=[col])
    return df


def scaleColumns_new(df_train, df_test, cols_to_scale):
    for col in cols_to_scale:
        df_train[col] = pd.DataFrame(min_max_scaler.fit_transform(pd.DataFrame(df_train[col])),columns=[col])
        df_test[col] = pd.DataFrame(min_max_scaler.transform(pd.DataFrame(df_test[col])), columns=[col])
    return df_train, df_test




#ignore_index=True
data = pd.concat([cHW_df,SM_df]).sample(frac=1)
#data1a.index = np.arange(0,20000)
print data
df_train, df_test = train_test_split(data,test_size = 0.3)
df_train.index = np.arange(0,280000)
df_test.index = np.arange(0,120000)

df_train= scaleColumns(df_train,["ptj", "etaj", "phij"])


df_test = scaleColumns(df_test,["ptj", "etaj", "phij"])


df_train =df_train[["signal","ptj", "etaj", "phij"]]


df_test = df_test[["signal","ptj", "etaj", "phij"]]


 


def getTrainData(nVar):
    ExamplesTrain = df_train.iloc[:,1:nVar+1].as_matrix()
    #now the signal
    ResultsTrain = df_train.iloc[:,0:1].as_matrix()
    return (ExamplesTrain,ResultsTrain)

def getTestData(nVar):
    ExamplesTest = df_test.iloc[:,1:nVar+1].as_matrix()
    #now the signal
    ResultsTest = df_test.iloc[:,0:1].as_matrix()
    return (ExamplesTest,ResultsTest)





def runSciKitRegressionL2(nVar,alpha):
    X_train, y_train = getTrainData(nVar)
    X_test, y_test = getTestData(nVar)
    clf = SGDClassifier(loss="log", penalty="l2",alpha=alpha,max_iter=5,tol=None)
    clf.fit(X_train,y_train.ravel())
    predictions = clf.predict(X_test)
    print('Accuracy on test data with alpha %.2E : %.3f' %(alpha,clf.score(X_test,y_test)) )
#    print('Accuracy on train data with alpha %.2E : %.3f' %(alpha,clf.score(X_train,y_train)) )
    probs = clf.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, probs[:, 1])
    roc_auc = auc(fpr, tpr)
    return (probs,tpr,fpr)


def runSciKitRegressionL1(nVar,alpha):
    X_train, y_train = getTrainData(nVar)
    X_test, y_test = getTestData(nVar)
    clf = SGDClassifier(loss="log", penalty="l1",alpha=alpha,max_iter=5,tol=None)
    clf.fit(X_train,y_train.ravel())
    predictions = clf.predict(X_test)
    print('Accuracy on test data with alpha %.2E : %.3f' %(alpha,clf.score(X_test,y_test)) )
    probs = clf.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, probs[:, 1])
    roc_auc = auc(fpr, tpr)
    return (probs,tpr,fpr,roc_auc)

alphas = np.logspace(-9,4,14)
#alphas = np.arange(0.1,0.4,0.01)
fig = plt.figure()
ax = fig.add_subplot(111)
it=0
#for alpha in alphas:

probs,tpr,fpr,roc_auc = runSciKitRegressionL1(3,0.0001)
#    ax.scatter(accept,rej,c=[c1,c2,c3],label='Alpha: %.1E' %alpha)
ax.plot(fpr,tpr, lw=2, label='ROC(area = %0.2f)'%(roc_auc))
#    it+=1
legend1=plt.legend(loc='upper left',prop={'size':16});
legend2=plt.legend([r"SGDClassifier"],loc=4,prop={'size':16},handlelength=0, handletextpad=0,frameon=False)
ax.add_artist(legend1)
#legend3=plt.legend([r"Max accuracy 82 $\%$"],loc=2,prop={'size':16},handlelength=0, handletextpad=0,frameon=False)
#ax.add_artist(legend2)
ax.set_xlabel(r'$\epsilon_{s_1}$',fontsize=18)
ax.set_ylabel(r'$\epsilon_{s_2}$',fontsize=18)
plt.grid()
legend1.get_frame().set_alpha(0.5)
for item in legend2.legendHandles:
    item.set_visible(False)
#for item in legend3.legendHandles:
#    item.set_visible(False)    
    
    plab.savefig('ROCsgd2.pdf', bbox_inches=0,dpi=100)
plt.show()


probsSimple,accep,rej = runSciKitRegressionL1(3,0.00001)
Signal = df_test.iloc[:,0:1]
Signal.index = np.arange(0,30000)
df_test_acc = pd.DataFrame({'PROB':probsSimple[:,1]})
df_test_acc['SIG']=Signal
df_test_acc_sig = df_test_acc.query('SIG==1')
df_test_acc_bkg = df_test_acc.query('SIG==0')

df_test_acc_sig.plot(kind='hist',y='PROB',color='blue',alpha=0.5,bins=np.linspace(0,1,10),label=r'Signal,$\alpha=1.0 \times 10^{-5}$')
plab.savefig('SIGskL1chwp1WH.png', bbox_inches=0,dpi=100)

df_test_acc_bkg.plot(kind='hist',y='PROB',color='red',alpha=0.5,bins=np.linspace(0,1,10),label=r'Background,$\alpha=1.0 \times 10^{-5}$')
plab.savefig('BCKskL1chwp1WH.png', bbox_inches=0,dpi=100)
plt.show()





