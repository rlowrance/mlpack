import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import math
from sklearn.metrics import roc_curve, auc
import pickle

def evenSplit(dat,fld):
    '''
    Evenly splits the data on a given binary field, returns a shuffled dataframe
    '''
    pos=dat[(dat[fld]==1)]
    neg=dat[(dat[fld]==0)]
    neg_shuf=neg.reindex(np.random.permutation(neg.index))
    fin_temp=pos.append(neg_shuf[:pos.shape[0]],ignore_index=True)
    fin_temp=fin_temp.reindex(np.random.permutation(fin_temp.index))
    return fin_temp


def trainTest(dat, pct):
    '''
    Randomly splits data into train and test
    '''
    dat_shuf = dat.reindex(np.random.permutation(dat.index))
    trn = dat_shuf[:int(np.floor(dat_shuf.shape[0]*pct))]
    tst = dat_shuf[int(np.floor(dat_shuf.shape[0]*pct)):]
    return [trn, tst]

def downSample(dat,fld,mult):
    '''
    Evenly splits the data on a given binary field, returns a shuffled dataframe
    '''
    pos=dat[(dat[fld]==1)]
    neg=dat[(dat[fld]==0)]
    neg_shuf=neg.reindex(np.random.permutation(neg.index))
    tot=min(pos.shape[0]*mult,neg.shape[0])
    fin_temp=pos.append(neg_shuf[:tot],ignore_index=True)
    fin_temp=fin_temp.reindex(np.random.permutation(fin_temp.index))
    return fin_temp


def scaleData(d):
    '''
    This function takes data and normalizes it to have the same scale (num-min)/(max-min)
    '''
    #Note, by creating df_scale like this we preserve the index
    df_scale=pd.DataFrame(d.iloc[:,1],columns=['temp'])
    for c in d.columns.values:
        df_scale[c]=(d[c]-d[c].min())/(d[c].max()-d[c].min())
    return df_scale.drop('temp',1)


def plot_dec_line(mn,mx,b0,b1,a,col,lab):
    '''
    This function plots a line in a 2 dim space
    '''
    x = np.random.uniform(mn,mx,100)
    dec_line = map(lambda x_i: -1*(x_i*b0/b1+a/b1),x)
    plt.plot(x,dec_line,col,label=lab)



def plotSVM(X, Y, my_svm):
    '''
    Plots the separating line along with SV's and margin lines
    Code here derived or taken from this example http://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane.html
    '''
    # get the separating hyperplane
    w = my_svm.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(X.iloc[:,0].min(), X.iloc[:,1].max())
    yy = a * xx - (my_svm.intercept_[0]) / w[1]
    # plot the parallels to the separating hyperplane that pass through the
    # support vectors
    b = my_svm.support_vectors_[0]
    yy_down = a * xx + (b[1] - a * b[0])
    b = my_svm.support_vectors_[-1]
    yy_up = a * xx + (b[1] - a * b[0])
    # plot the line, the points, and the nearest vectors to the plane
    plt.plot(xx, yy, 'k-')
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')
    plt.scatter(my_svm.support_vectors_[:, 0], my_svm.support_vectors_[:, 1], s=80, facecolors='none')
    plt.plot(X[(Y==-1)].iloc[:,0], X[(Y==-1)].iloc[:,1],'r.')
    plt.plot(X[(Y==1)].iloc[:,0], X[(Y==1)].iloc[:,1],'b+')
    #plt.axis('tight')
    #plt.show()


def getP(val):
    '''
    Get f(x) where f is the logistic function
    '''
    return (1+math.exp(-1*val))**-1

def getY(val):
    '''
    Return a binary indicator based on a binomial draw with prob=f(val). f the logistic function.
    '''
    return (int(getP(val)>np.random.uniform(0,1,1)[0]))

def gen_logistic_dataframe(n,alpha,betas):
    '''
    Aa function that generates a random logistic dataset
    n is the number of samples
    alpha, betas are the logistic truth
    '''
    X = np.random.random([n,len(betas)])
    Y = map(getY,X.dot(betas)+alpha)
    d = pd.DataFrame(X,columns=['f'+str(j) for j in range(X.shape[1])])
    d['Y'] = Y
    return d


def plotAUC(truth, pred, lab):
    fpr, tpr, thresholds = roc_curve(truth, pred)
    roc_auc = auc(fpr, tpr)
    c = (np.random.rand(), np.random.rand(), np.random.rand())
    plt.plot(fpr, tpr, color=c, label= lab+' (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC')
    plt.legend(loc="lower right")



def LogLoss(dat, beta, alpha):
    X = dat.drop('Y',1)
    Y = dat['Y']
    XB=X.dot(np.array(beta))+alpha*np.ones(len(Y))
    P=(1+np.exp(-1*XB))**-1
    return ((Y==1)*np.log(P)+(Y==0)*np.log(1-P)).mean()


def plotSVD(sig):
    norm = math.sqrt(sum(sig*sig))
    energy_k = [math.sqrt(k)/norm for k in np.cumsum(sig*sig)]

    plt.figure()
    ax1 = plt.subplot(211)
    ax1.bar(range(len(sig+1)), [0]+sig, 0.35)
    plt.title('Kth Singular Value')
    plt.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')

    ax2 = plt.subplot(212)
    plt.plot(range(len(sig)+1), [0]+energy_k)
    plt.title('Normalized Sum-of-Squares of Kth Singular Value')

    ax2.set_xlabel('Kth Singular Value')
    ax2.set_ylim([0, 1])


def genY(x, err, betas):
    '''
    Goal: generate a Y variable as Y=XB+e
    Input
    1. an np array x of length n
    2. a random noise vector r of length n
    3. a (d+1) x 1 vector of coefficients b - each represents ith degree of x
    '''
    d = pd.DataFrame(x, columns=['x'])
    y = err
    for i,b in enumerate(betas):
        y = y + b*x**i
    d['y'] = y
    return d


def makePolyFeat(d, deg):
    '''
    Goal: Generate features up to X**deg
    1. a data frame with two features X and Y
    4. a degree 'deg' (from which we make polynomial features

    '''
    #Generate Polynomial terms
    for i in range(2, deg+1):
        d['x'+str(i)] = d['x']**i
    return d


def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'r') as f:
        return pickle.load(f)
