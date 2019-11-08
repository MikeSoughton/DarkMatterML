import pandas as pd

def expandArrays(df):

    for var in df.keys():
        #Check if it's a list
        if df[var].dtype.kind is 'O':

            ############ Failed attempt to generalise it to 2d #######
            #If it's a nested list 
            # if hasattr(df[var].iloc[0][0], "__len__"):
            #     print 'calling nested',var
            #     new= df[var].apply(pd.Series)
            #     new.columns = [var+str(n)+'_' for n in new.keys()]
            #     df = df.drop([var],axis=1)
            #     df = pd.concat([new,df],axis=1)
            #     #call recursively (hopefully i'm not being too clever)
            #     df = expandArrays(df)
            #     return df
            # else:
            ##########################################################

            #Assuming it's a row, so expand it out
            new= df[var].apply(pd.Series)
            new.columns = [var+str(n) for n in new.keys()]
            #drop the old one
            df = df.drop([var],axis=1)
            #include the expanded frame
            df = pd.concat([df,new],axis=1)


    return df

