from Plotter import Plotter
import seaborn as sns

class PlotterSns(Plotter):
    def plotAllHists2DSns(self,varWrt,extraExceptions=[],pairPlot=False,**kwargs):

    if pairPlot: append='PP'
    else: append='JP'
    out = os.path.join(self.outputDir,'hists2dSns'+append+varWrt)
    if not os.path.exists(out): os.makedirs(out)

    for var in self.df.keys():
        if var in extraExceptions: continue
        if pairPlot:
            sns.pairplot(self.df,vars=[varWrt,var],**kwargs)
        else:
            sns.jointplot(varWrt,var,data=self.df)
        plt.savefig(os.path.join(out,var+'.pdf'))
        plt.close()

    return True
    
