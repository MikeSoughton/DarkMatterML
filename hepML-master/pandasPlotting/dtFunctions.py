from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.metrics import cohen_kappa_score
import numpy as np
import matplotlib.pyplot as plt
import os

def classifyScore(score,classification=0):
    '''Returns 0 or 1 depending whether the score is above or below classification'''
    if score > classification: return 1
    else: return 0

def featureImportance(df,classifier,output,exceptions=[],error=False):
    '''Classify the variable importance for a given set of trees code is taken
    from:
    http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html\
            #sphx-glr-auto-examples-ensemble-plot-forest-importances-py
    '''

    # Build a forest and compute the feature importances
    forest = ExtraTreesClassifier(n_estimators=1000,
                                  random_state=0)

    # Make the array of features, X, and the classifier, y

    y = df[classifier]
    X = df.drop(classifier,axis=1)

    #Remove duplicates
    X = X.loc[:,~X.columns.duplicated()]
    for e in exceptions:
        X.drop(e,axis=1,inplace=True)        

    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    if not os.path.exists(output): os.makedirs(output)
    outTxtFile = open(os.path.join(output,'featureList.txt'),'w')
    outTxtFile.write("Feature ranking:\n")

    for f in range(X.shape[1]):
        outTxtFile.write("%d. feature %s (%f)" % (f + 1, X.columns.values.tolist()[indices[f]], importances[indices[f]])+'\n')

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    if error:
        plt.bar(range(X.shape[1]), importances[indices],
               color="r", yerr=std[indices], align="center")
    else:
        plt.bar(range(X.shape[1]), importances[indices],
               color="r", align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.savefig(os.path.join(output,'featureImportances.pdf'))
    plt.close()

    pass

def decisionTree(df,classifier,profit,output,subset=None,split=None,drawTree=False,append='',**kwargs):

    #Define and make the output
    out = os.path.join(output,'decisionTree',append)
    if not os.path.exists(out): os.makedirs(out)

    #Build the tree and partition the data
    dt =  DecisionTreeClassifier(**kwargs)

    y = df[classifier]
    if subset!=None:
        X = df[subset]
    else:
        X = df.drop(classifier,axis=1)

    #Now split into testing and training
    if isinstance(split,list):

        # if it's a list split based on the list
        assert len(split)==2
        assert split[0]+split[1]==1
        trainLen=int(len(X)*split[0])

        X_test=X[trainLen:]
        X_train=X[:trainLen]
        y_test=y[trainLen:]
        y_train=y[:trainLen]
        df_test=df[trainLen:]
        df_train=df[:trainLen]

    elif isinstance(split,str):
        if split=='kfold': 
            print 'kfold not implemented'
            return False
        else:
            print split,'not implemented, not doing decision tree'
            return False
    else:
        # train and test on the same thing
        X_test=X
        X_train=X
        y_test=y
        y_train=y
        df_test=df
        df_train=df

    # Now carry out the fit on the training set
    dt.fit(X_train,y_train)

    #Make the outputs of the tree

    if drawTree:
        #Print out the tree
        outTree=os.path.join(out,'tree.dot')
        export_graphviz(dt,out_file=outTree,feature_names=X_train.columns.values.tolist(),\
                        filled=True, rounded=True,special_characters=True)
        os.system('dot -Tpng '+outTree+' -o '+os.path.join(out,'tree.png'))

    #feature importances
    featuresOut = open(os.path.join(out,'featureImportances.txt'),'w')

    featuresOut.write("Feature ranking:\n")

    featureIndices = np.argsort(dt.feature_importances_)[::-1]
    for f in range(X_train.shape[1]):
        featuresOut.write("%d. feature %s (%f)" % (f + 1, X_train.columns.values.tolist()[featureIndices[f]], dt.feature_importances_[featureIndices[f]])+'\n')

 
    #Assess how it did for train and test

    resultOut = {
            'Train':{'X':X_train,'y':y_train,'df':df_train},
            'Test':{'X':X_test,'y':y_test,'df':df_test},
            }

    textOut = open(os.path.join(out,'dtResults.txt'),'w')
    #write the config
    textOut.write('training split: '+str(split)+'\n')
    textOut.write('DT inputs '+str(kwargs)+'\n\n')

    def myPrint(phrase):
        phrase =' '.join([str(x) for x in phrase])
        print ' >> ',phrase
        textOut.write(phrase+'\n') 

    
    # different assessment of test and training sets
    for name,data in resultOut.iteritems():

        predicted = dt.predict(data['X'])

        #Basic profit info
        myPrint(('>>>>>>>',))
        myPrint((name,'set'))
        myPrint(('Pre tree profit:',data['df'][profit].sum()))
        myPrint(('Post tree 1 profit:',data['df'][predicted==1][profit].sum()))
        myPrint(('Post tree 0 profit:',data['df'][predicted==0][profit].sum()))
        myPrint(('Godlike 1 profit:',data['df'][data['df'][classifier]==1][profit].sum()))
        myPrint(('Godlike 0 profit:',data['df'][data['df'][classifier]==0][profit].sum(),'\n'))

        #output tree performance
        correct = (predicted==data['df'][classifier]).sum()
        myPrint(('Correct/Total',round(correct*100.0/len(predicted),1),'%'))
        myPrint(('Cohens kappa (% better than random)',round(cohen_kappa_score(data['df'][classifier],predicted)*100,1),'%\n'))

        #make a ROC curve
        #...



    return True

