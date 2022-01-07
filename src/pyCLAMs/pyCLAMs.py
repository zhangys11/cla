import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import uuid
import scipy
import pylab, matplotlib
from .plt2base64 import *
from .plotComponents2D import *

# generate and plot 2D multivariate gaussian data set
def mvg(
    nobs = 20, # number of observations / samples
    md = 2, # distance between means, respect to std, i.e. (mu2 - mu1) / std, or how many stds is the difference.
    dims = 2, # 1 or 2
    ):
 
    N = nobs

    if (dims == 2):    
    
        s = 1 # reference std = 1
        mu1 = [- md * s/2,  0]
        mu2 = [md * s/2 , 0]
        s1 = [s, s]  # use diagonal covariance. Naive assuption: features are uncorrelated
        s2 = [s, s]

        cov1 = np.array([[s1[0], 0],[0,s1[1]]])
        cov2 = np.array([[s2[0], 0],[0,s2[1]]])

        xc1 = np.random.multivariate_normal(mu1, cov1, N)
        xc2 = np.random.multivariate_normal(mu2, cov2, N)

        y = np.concatenate((np.zeros(N), np.ones(N))).astype(int)
        X = np.vstack((xc1,xc2))

    elif (dims == 1):
        
        s = 1
        xc1 = np.random.randn(N) *s - md * s/ 2
        xc2 = np.random.randn(N) *s + md * s/ 2
        
        X = np.concatenate((xc1, xc2)).reshape(-1,1)
        y = np.concatenate((np.zeros(N), np.ones(N))).astype(int)
        
    else:
        
        raise Exception('only support 1 or 2 dims')

    return X, y

def save_file(X, y, fn):

    # fn = str(uuid.uuid1()) + '.csv'
    M = np.hstack((X,y.reshape(-1,1)))
    np.savetxt(fn, M, delimiter=',', fmt='%.3f,'*X.shape[1] + '%i', header='X1,X2,...,Y') # fmt='%f'
    return fn

def load_file(fn):
    M = np.loadtxt(fn, delimiter=',', skiprows=1)
    X = M[:,:-1]
    y = M[:,-1].astype(int)
    return X,y

#
# plot contour of multivariate Gaussian distributions

def plot_gaussian_contour (X, y, mu1, s1, mu2, s2, alpha = 0.4, ax = None):

    plt.figure() # figsize = (9, 6)

    X1 = X[:,0]
    if (X.shape[1]>1):
        X2 = X[:,1]
    else:
        X2 = X1.copy()

    dx = X1.max()-X1.min()
    dy = X2.max()-X2.max()

    Xg1, Xg2 = np.mgrid[X1.min() - dx * 0.2 : X1.max() + dx * 0.2: 0.01, 
                        X2.min() - dy * 0.2 : X2.max() + dy * 0.2: 0.01]
    pos = np.empty(Xg1.shape + (2,))
    pos[:, :, 0] = Xg1
    pos[:, :, 1] = Xg2

    rv1 = scipy.stats.multivariate_normal(mu1, s1)
    c1 = plt.contour(Xg1, Xg2, rv1.pdf(pos),alpha = alpha, cmap='Reds')
    plt.clabel(c1, inline=True, fontsize=10)

    rv2 = scipy.stats.multivariate_normal(mu2, s2)
    c2 = plt.contour(Xg1, Xg2, rv2.pdf(pos),alpha = alpha, cmap='Reds')
    plt.clabel(c2, inline=True, fontsize=10)
   
    # print('C1: X~N(', mu1, ',', s1, ')')
    # print('C2: X~N(', mu2, ',', s2, ')')
    
    return plt.gca()


from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import GaussianNB
from scipy.integrate import quad


def BER(X ,y, M = 10000, NSigma = 10, show = False, save_fig = ''):
    """
    We draw random samples from the bayes distribution models to calculate BER
    
    M - sample count
    NSgima - the sampling range
    """    

    nb = GaussianNB(priors  = [0.5, 0.5]) # we have no strong prior assumption.
    nb.fit(X, y)
    
    labels = list(set(y))
    
    # For multi-class classification, use one vs rest strategy
    assert len(labels) == 2
    
    Xc1 = X[y == labels[0]]
    Xc2 = X[y == labels[1]]
    
    n1 = len(Xc1)
    n2 = len(Xc2)
    
    mu1 = np.mean(Xc1, axis = 0)
    mu2 = np.mean(Xc2, axis = 0)
    
    s1 = np.std(Xc1, axis = 0, ddof=1)
    s2 = np.std(Xc2, axis = 0, ddof=1)
    
    #print(mu1, mu2)
    #print(s1, s2)
    #print(nb.sigma_)
    
    lb = np.minimum(mu1 - NSigma*s1, mu2 - NSigma*s2)
    ub = np.maximum(mu1 + NSigma*s1, mu2 + NSigma*s2)
    
    # we use M random samples to calculate BER 
    XM = np.zeros((M, X.shape[1]))
    for i in range(M):
        rnd = np.random.random(X.shape[1])
        XM[i, :] = lb + (ub - lb) * rnd
    
    # quad(lambda x: guassian, -3std, 3std) ...        
    
    y_pred = nb.predict_proba(XM)

    sum_of_max_prob = 0.0
    sum_of_min_prob = 0.0

    for p in y_pred:
        sum_of_max_prob += p.max()
        sum_of_min_prob += p.min()

    BER = 1 - sum_of_max_prob/len(y_pred)
    BER2 = sum_of_min_prob/len(y_pred)
    IMG = ''
        
    if X.shape[1] == 2:
        # for 2-dimensional data, plot the contours        
        ax = plot_gaussian_contour (X, y, nb.theta_[0], nb.sigma_[0], nb.theta_[1], nb.sigma_[1], alpha = 0.3)
        plotComponents2D(X, y, set(y), use_markers = False, ax = ax)
        plt.legend()
        title = ' $ \mu $ = ' + str(np.round(nb.theta_, 3) ) + ', $\sigma$ = ' + str( np.round(nb.sigma_,3) ).replace('\n','')
        plt.title(title)
        
        if (save_fig != '' and save_fig != None):   
            if save_fig.endswith('.jpg') == False:
                save_fig += '.jpg'
            plt.savefig(save_fig)
            print('figure saved to ' + save_fig)
        
        IMG = plt2html(plt)
        
        if show:
            plt.show()
        else:
            plt.close()        

    return BER, IMG #, BER2

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.metrics import *

CLF_METRICS = ['classification.ACC',
               'classification.Kappa',
               'classification.F1_Score',
               'classification.Jaccard', # The Jaccard index, or Jaccard similarity coefficient, defined as the size of the intersection divided by the size of the union of two label sets
               'classification.Precision',
               'classification.Recall',
               ## The following requires a model that outputs probability
               'classification.CrossEntropy', # cross-entropy loss / log loss
               'classification.AP',
               'classification.Brier',               
               'classification.ROC_AUC',
               'classification.PR_AUC']


def grid_search_svm_hyperparams(X, y, test_size = 0.2, tuned_parameters = [
                                    {'kernel': ['rbf'], 'gamma': [10, 1, 1e-1, 1e-2], 'C': [0.01, 0.1, 1, 10, 100, 1000]},         
                                    {'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10, 100, 1000,10000,100000]}], cv = 5, verbose = True):
    '''
    Find the optimal SVM model by grid search.
    Returns the best model and ACCs on training / testing / all data set
    '''

    X_train, X_test, y_train, y_test = train_test_split(
        X, np.array(y), test_size=test_size) # , random_state=0
    
    log = ''
    # log += "X_train: " + str(X_train.shape) + ", y_train: " + str(y_train.shape)

    gs = GridSearchCV(SVC(), tuned_parameters, cv=cv) #iid = True. accept an estimator object that implements the scikit-learn estimator interface. Explicitly set iid to avoid DeprecationWarning.
    gs.fit(X_train, y_train)

    log += "\nBest parameters set found by GridSearchCV: \n"
    log += str(gs.best_params_)
    log += "\nGrid scores on cv set:\n"
        
    means = gs.cv_results_['mean_test_score']
    stds = gs.cv_results_['std_test_score']


    for mean, std, params in zip(means, stds, gs.cv_results_['params']):
        log += "{} (+/- {}) for {} \n".format(round( mean,5), round(std * 2,5), params)    

    log += ("\nDetailed classification report:\n")


    log += ('\n#### Training Set ####\n')
    y_true, y_pred = y_train, gs.predict(X_train)
    log += classification_report(y_true, y_pred)

    log +=('\n\n#### Test Set ####\n')
    y_true, y_pred = y_test, gs.predict(X_test)
    log += classification_report(y_true, y_pred)
    
    log += ('\n\n#### All Set ####\n')
    y_true, y_pred = y, gs.predict(X)
    log += classification_report(y_true, y_pred)    
    
    if verbose: 
        print(log)
    
    return gs.best_params_, gs.best_estimator_, log # , gs.score(X_train, y_train), gs.score(X_test, y_test), gs.score(X,y), log

from sklearn.linear_model import LogisticRegressionCV

def CLF(X, y, verbose = False, show = False, save_fig = ''):
    '''
    X,y - features and labels
    '''
    
    dct = {}
    clf_metrics = []
    LOG = ''

    '''
    ###
    # Use grid search to train the best SVM model
    C = np.logspace(-20,1,7) # np.logspace(-20,2,10)
    tuned_parameters = [ #{'kernel': ['rbf'], 'gamma': [10, 1e-1, 1e-3],'C': C},
                    {'kernel': ['linear'], 'C': C}]

    ###
    # find the best SVC model
    best_params, clf, LOG = grid_search_svm_hyperparams(X, y, 0.2, 
                                                   tuned_parameters,
                                                   cv = 3,
                                                  verbose = verbose
                                                  )
    '''

    ###
    # train a logistic regression model
    # clf = LogisticRegressionCV(cv=10, solver = 'saga', penalty = 'elasticnet', max_iter = 5000, l1_ratios = [0,0.2,0.4,0.6,0.8,1]).fit(X, y) # with Elasticnet regularization, but it is too time consuming. We don't require sparse solution, so ridge suffices.
    # LOG += "regularization strength\t" + str(clf.C_) + "\nL1 reg ratio" + str(clf.l1_ratio_) + "\n\n"

    clf = LogisticRegressionCV(cv=10, max_iter = 1000).fit(X, y) # ridge(L2) regularization
    LOG += "regularization strength\t" + str(clf.C_) + "\n\n"
    # l1_ratio: while 1 is equivalent to using penalty='l1'. For 0 < l1_ratio <1, the penalty is a combination of L1 and L2.

    IMG = ''
    
    # visualize the decision boundary in a 2D plane if X has two features
    if (X.shape[1] == 2): 
        
        plt.figure()
        
        # plt.scatter(data['X1'], data['X2'], s=50, c=clf.predict_proba(data[['X1', 'X2']])[:,0], cmap='seismic')
        plt.scatter(X[:,0], X[:,1], s=50, c=clf.decision_function(X), cmap='seismic')

        # plot the decision function
        ax = plt.gca() # get current axes
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # create grid to evaluate model
        xx = np.linspace(xlim[0], xlim[1], 30)
        yy = np.linspace(ylim[0], ylim[1], 30)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = clf.decision_function(xy).reshape(XX.shape)

        # plot decision boundary and margins
        out = ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
                   linestyles=['--', '-', '--'])
        
        # in some cases, there can be multiple unconnected decision boundaries
        
        #for i in range(len(out.collections[1].get_paths())):
        #    vertice_set.append(out.collections[1].get_paths()[i].vertices)
        
        # plot support vectors if using SVM
        if (hasattr(clf, "support_vectors_")):
            ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
                       linewidth=1, facecolors='none', edgecolors='k')
            title = 'kernel = ' + best_params['kernel'] + ", C = " + str(best_params['C']) + " , acc = " + str(round(clf.score(X, y),3)) 
        
            if 'gamma' in best_params:
                title += ", $\gamma$ = " + str(best_params['gamma'])
            ax.set_title(title)
        
        if (save_fig != '' and save_fig != None):   
            if save_fig.endswith('.jpg') == False:
                save_fig += '.jpg'
            plt.savefig(save_fig)
            print('figure saved to ' + save_fig)        
        
        IMG = plt2html(plt)
        
        if show:
            plt.show()
        else:
            plt.close()
            
    y_pred = clf.predict(X)
    clf_metrics.append( globals()["accuracy_score"](y, y_pred) )
    clf_metrics.append( globals()["cohen_kappa_score"](y, y_pred) ) # use ground truth and prediction as 1st and 2nd raters
    clf_metrics.append( globals()["f1_score"](y, y_pred) )
    clf_metrics.append( globals()["jaccard_score"](y, y_pred) )
    clf_metrics.append( globals()["precision_score"](y, y_pred) )
    clf_metrics.append( globals()["recall_score"](y, y_pred) )
    
    ###        
    # train a logistic regression model to compute Brier score, PR AUC, etc.
    # clf = LogisticRegressionCV(cv=5, random_state=0).fit(X, y)    
    y_prob = clf.predict_proba(X)[:,1] # for binary classification, use the second proba as P(Y=1|X)
    y_pred = clf.predict(X)
        
    clf_metrics.append( globals()["log_loss"](y, y_prob) )
    clf_metrics.append( globals()["average_precision_score"](y, y_prob) )
    clf_metrics.append( globals()["brier_score_loss"](y, y_prob) )
    clf_metrics.append( globals()["roc_auc_score"](y, y_prob) )
    precisions, recalls, _ = precision_recall_curve(y, y_prob)
    clf_metrics.append( globals()["auc"](recalls, precisions) )
    
    rpt = ''    
    for v in zip(CLF_METRICS, clf_metrics):
        rpt += v[0] + "\t" + str(v[1]) + "\n"
        dct[v[0]] = v[1]
        
    LOG += "\n\n" + rpt
        
    return dct, IMG, LOG # acc_train, acc_test, acc_all # , vertice_set # vertice_set[0] is the decision boundary

from sklearn.feature_selection import mutual_info_classif
    
def IG(X, y, show = False, save_fig = ''):
    """
    Return the feature-wise information gains
    """
    mi = mutual_info_classif(X, y, discrete_features=False)
    mi_sorted = np.sort(mi)[::-1] # sort in desceding order
    mi_sorted_idx = np.argsort(mi)[::-1]

    plt.figure()

    xlabels = []
    for i, v in enumerate(mi_sorted):
        xlabels.append("X"+str(mi_sorted_idx[i] + 1))
        #if (len(mi_sorted) < 20): # don't show text anno if feature number is large
        plt.text(i-0.001, v+0.001,  str(round(v, 1)))

    plt.bar(xlabels, mi_sorted, facecolor="none", edgecolor = "black", width = 0.3, hatch='/')

    plt.title('Info Gain of all features in descending order')
    # plt.xticks ([]) 
            
    if (save_fig != '' and save_fig != None):   
        if save_fig.endswith('.jpg') == False:
            save_fig += '.jpg'
        plt.savefig(save_fig)
        print('figure saved to ' + save_fig)   
    
    IMG = plt2html(plt)
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return mi, IMG


import statsmodels
from statsmodels.base.model import Model
from statsmodels.multivariate import manova
import re
import math

def ANOVA(X,y, verbose = False, show = False):
    """
    Performa feature-wise ANOVA test. Returns an array of p-values on all the features and its minimum.

    y - support up to 5 classes
    """       

    if (len(set(y)) < 2):
        raise Exception('The dataset must have at least two classes.')
    
    ps = []
    IMG = ''
    Fs = []
    
    for i in range(X.shape[1]):
        Xi = X[:,i]
        Xcis = []

        labels = []
        for c in set(y):    
            Xc = Xi[y == c]
            Xcis.append(Xc) 
            labels.append("$ X_"+str(i+1)+"^{( y_"+str(c)+" )} $")

        Xcis = np.array(Xcis)         
        
        f,p= scipy.stats.f_oneway(Xcis[0], Xcis[1]) # equal to ttest_ind() in case of 2 groups 
        
        if (len(set(y)) == 3):
            f,p= scipy.stats.f_oneway(Xcis[0], Xcis[1], Xcis[2])
        elif (len(set(y)) == 4):
            f,p= scipy.stats.f_oneway(Xcis[0], Xcis[1], Xcis[2], Xcis[3])
        elif (len(set(y)) >= 5): # if there are five or more classes
            f,p= scipy.stats.f_oneway(Xcis[0], Xcis[1], Xcis[2], Xcis[3], Xcis[4])
        
        """
        Alternative implementation using sm.stats.anova_lm
        
        import statsmodels.api as sm
        from statsmodels.formula.api import ols

        df = pd.DataFrame( np.vstack( (X[:,0],y) ).T) 

        df.columns = ['X0', 'y']
        df

        # Ordinary Least Squares (OLS) model
        model = ols('y ~ C(X0)', data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=1)
        anova_table
        """
        
        ps.append(p)
        Fs.append(f)

        plt.figure()
        plt.boxplot(Xcis.T, notch=False, labels=labels) # plot ith feature of different classes   
        test_result = "ANOVA on X{}: f={},p={}".format(i+1, f, round(p,3))
        # plt.legend(labels)
        plt.title(test_result)
        IMG += plt2html(plt)+ '<br/>'
    
        if show:
            plt.show()
        else:
            plt.close()

        if verbose:
            print(test_result)

    return ps, Fs, IMG

import sys

def MANOVA(X,y, verbose = False):
    """
    MANOVA test of the first two features.
    """    
    
    if (X.shape[1] <= 1):
        txt = 'There must be more than one dependent variable to fit MANOVA! Use ANOVA to substitute MANOVA.'
        anova_p, anova_p_log, _, _ = ANOVA_p_value(X,y)
        return anova_p, anova_p_log, txt
        
    X1 = X[:,0]
    X2 = X[:,1]
    df = pd.DataFrame({'X1': X1,'X2': X2,'y':y})
    mv = manova.MANOVA.from_formula('X1 + X2 ~ y', data=df) # Intercept is included by default.
    r = mv.mv_test()       
    
    LOG = ''
    LOG += 'endog: ' + str(mv.endog_names) + '\n'
    LOG += 'exog: ' + str(mv.exog_names) + '\n\n'
    LOG += str(r)
    
    delimiters = "Wilks' lambda", "Pillai's trace"
    regexPattern = '|'.join(map(re.escape, delimiters))
    ss = re.split(regexPattern, str(r.results['y']['stat']['Pr > F']) )
    manova_p = float(ss[1].strip())
    
    ss = re.split(regexPattern, str(r.results['y']['stat']['F Value']) )
    manova_F = float(ss[1].strip())
    
    # print(manova_F, manova_p) # use one of the four tests (their results are the same for almost all the time)
    
    if (manova_p == 0): # add a very small amount to make log legal
        manova_p = sys.float_info.epsilon 
    manova_p_log = math.log(manova_p,10) # round( math.log(manova_p, 10) , 3)

    if (verbose):
        print(LOG)
        
    return manova_p, manova_F, LOG

def MWW(X,y, verbose = False, show = False):
    """
    Performa feature-wise MWW test. Returns an array of p-values on all the features and its minimum.

    y - support 2 classes
    """       

    if (len(set(y)) != 2):
        raise Exception('The dataset must have 2 classes.')
    
    ps = []
    Us = []
    IMG = ''

    for i in range(X.shape[1]):
        Xi = X[:,i]
        Xcis = []

        for c in set(y):    
            Xc = Xi[y == c]
            Xcis.append(Xc) 

        Xcis = np.array(Xcis)         
        
        U,p = scipy.stats.mannwhitneyu(Xcis[0], Xcis[1])    
        
        ps.append(p)
        Us.append(U)

        plt.figure()
        plt.hist(Xcis.T, bins = min(12, int(len(y)/3)), alpha=0.4, edgecolor='black', label = ["$ X_"+str(i+1)+"^{( y_"+str(0)+")} $", "$ X_"+str(i+1)+"^{( y_"+str(1)+")} $"]) # plot ith feature of different classes   
        test_result = "MWW test on X{}: U={},p={}".format(i+1, U, round(p,3))
        plt.title('Feature X{} histogram on different classes\n'.format(i+1) + test_result)
        plt.legend()
        IMG += plt2html(plt) + '<br/>'
        
        if show:               
            plt.show() 
        else:
            plt.close()

        if verbose:
            print(test_result)

    return ps, Us, IMG


import numpy as np

def cohen_d(X, y, show = False, save_fig = ''):
    
    labels = list(set(y))
    
    # only support binary classifiction. For multi-class classification, use one vs rest strategy
    assert len(labels) == 2
    
    Xc1 = X[y == labels[0]]
    Xc2 = X[y == labels[1]]
    
    n1 = len(Xc1)
    n2 = len(Xc2)
    dof = n1 + n2 - 2
    
    # replace 0 stds with the medium value
    pooled_std = np.sqrt(((n1-1)*np.std(Xc1, axis = 0, ddof=1) ** 2 
                          + (n2-1)*np.std(Xc2, axis = 0, ddof=1) ** 2)/ dof)
    pooled_std_median = np.median(pooled_std[pooled_std > 0])
    pooled_std[pooled_std == 0] = pooled_std_median
    
    d = np.abs(np.mean(Xc1, axis = 0) - np.mean(Xc2, axis = 0)) / pooled_std

    plt.figure()
    
    d_sorted = np.sort(d)[::-1] # sort in desceding order
    d_sorted_idx = np.argsort(d)[::-1] 
    xlabels = []
    for i, v in enumerate(d_sorted):
        xlabels.append("X"+str(d_sorted_idx[i] + 1))
        plt.text(i-0.01, v+0.01,  str(round(v, 1)))

    plt.bar(xlabels, d_sorted, facecolor="none", edgecolor = "black", width = 0.3, hatch="\\")
    plt.title("Effect Size (Cohen's d) for all features in descending order")
    # plt.xticks ([])

    if (save_fig != '' and save_fig != None):   
        if save_fig.endswith('.jpg') == False:
            save_fig += '.jpg'
        plt.savefig(save_fig)
        print('figure saved to ' + save_fig)   
    
    IMG = plt2html(plt)
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return d, IMG # d is a 1xn array. n is feature num

def correlate(X,y, verbose = False):
    """
    Performa correlation tests between each feature Xi and y.
    
    """       

    dct = {}
    
    rs = []
    rhos = []
    taus = []
    
    prs = []
    prhos = []
    ptaus = []
    
    LOG = ''

    for i in range(X.shape[1]):
        
        Xi = X[:,i]
        
        LOG += '\n\n#### Correlation between X{} and y ####\n'.format(i+1)

        r, p = scipy.stats.pearsonr(Xi, y)
        LOG += '\nPearson r: {}, p-value: {}'.format(round(r,3), round(p,3))
        rs.append(r)
        prs.append(p)

        rho, p = scipy.stats.spearmanr(Xi, y)
        LOG += '\nSpearman rho: {}, p-value: {}'.format(round(rho,3), round(p,3))
        rhos.append(rho)
        prhos.append(p)

        tau, p = scipy.stats.kendalltau(Xi, y)
        LOG += "\nKendall's tau: {}, p-value: {}".format(round(tau,3), round(p,3))
        taus.append(tau)   
        ptaus.append(p)
        
    if verbose:
        print(LOG)

    dct['correlation.r'] = rs
    dct['correlation.r.p'] = prs
    dct['correlation.r.max'] = np.abs(rs).max() # abs max
    dct['correlation.r.p.min'] = np.min(prs)
    
    dct['correlation.rho'] = rhos
    dct['correlation.rho.p'] = prhos
    dct['correlation.rho.max'] = np.abs(rhos).max() # abs max
    dct['correlation.rho.p.min'] = np.min(prhos)
    
    dct['correlation.tau'] = taus
    dct['correlation.tau.p'] = ptaus
    dct['correlation.tau.max'] = np.abs(taus).max() # abs max
    dct['correlation.tau.p.min'] = np.min(ptaus)
    
    return dct, LOG

def KS(X,y, show = False):
    """
    Performa feature-wise KS test.

    y - Because it is two-sample KS test, only support 2 classes
    """       

    if (len(set(y)) != 2):
        raise Exception('The dataset must have two classes. If you have more than 2 classes, use OVR (one-vs-rest) strategy.')
    
    ps = []
    Ds = []
    IMG = ''

    for i in range(X.shape[1]):
        Xi = X[:,i]
        Xcis = []

        for c in set(y):    
            Xc = Xi[y == c]
            Xcis.append(Xc) 

        Xcis = np.array(Xcis)         
        
        D,p= scipy.stats.ks_2samp(Xcis[0], Xcis[1])
        
        ps.append(p)
        Ds.append(D)

        plt.figure()
        plt.hist(Xcis.T, cumulative=True, histtype=u'step', bins = min(12, int(len(y)/3)), label = ["$ CDF( X_"+str(i+1)+"^{(y_"+str(0)+")} ) $", "$ CDF( X_"+str(i+1)+"^{(y_"+str(1)+")} ) $"]) # plot ith feature of different classes   
        test_result = "KS test on X{}: D={},p={}".format(i+1, D, round(p,3))
        plt.title('Feature X{} CDF on the two classes\n'.format(i+1) + test_result)
        plt.legend(loc='upper left')
        IMG += plt2html(plt) + '<br/>'
    
        if show:
            plt.show()
        else:
            plt.close()

    return ps, Ds, IMG


import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri# Defining the R script and loading the instance in Python
import rpy2
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector,FloatVector
import rpy2.robjects.numpy2ri
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

ECoL_METRICS = ['overlapping.F1.mean',
 'overlapping.F1.sd',
 'overlapping.F1v.mean',
 'overlapping.F1v.sd',
 'overlapping.F2.mean',
 'overlapping.F2.sd',
 'overlapping.F3.mean',
 'overlapping.F3.sd',
 'overlapping.F4.mean',
 'overlapping.F4.sd',
 'neighborhood.N1',
 'neighborhood.N2.mean',
 'neighborhood.N2.sd',
 'neighborhood.N3.mean',
 'neighborhood.N3.sd',
 'neighborhood.N4.mean',
 'neighborhood.N4.sd',
 'neighborhood.T1.mean',
 'neighborhood.T1.sd',
 'neighborhood.LSC',
 'linearity.L1.mean',
 'linearity.L1.sd',
 'linearity.L2.mean',
 'linearity.L2.sd',
 'linearity.L3.mean',
 'linearity.L3.sd',
 'dimensionality.T2',
 'dimensionality.T3',
 'dimensionality.T4',
 'balance.C1',
 'balance.C2',
 'network.Density',
 'network.ClsCoef',
 'network.Hubs.mean',
 'network.Hubs.sd']


def setup_ECoL():
    '''
    Need to call this function only once.
    Install the ECoL R package.
    '''

    # import R's utility package
    utils = rpackages.importr('utils')

    # select a mirror for R packages
    utils.chooseCRANmirror(ind=1) 

    # R package names
    packnames = ('ECoL')

    # Selectively install what needs to be install.
    # We are fancy, just because we can.
    names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
    if len(names_to_install) > 0:
        utils.install_packages(StrVector(names_to_install))

def ECoL_metrics(X,y):
    '''
    Use rpy2 to call ECoL R package. ECoL has implemented many metrics. 
    Returns a text report and a dict
    '''    
    
    ### ECoL requires df as input
    # robjects.numpy2ri.activate()
    # rX = robjects.r.matrix(X, nrow=X.shape[0], ncol=X.shape[1])
    # robjects.r.assign("rX", rX)
    # ry = FloatVector(y.tolist())
    # robjects.globalenv['rX'] = rX
    # robjects.globalenv['ry'] = ry
    
    # ys = map(lambda x : 'Class ' + str(x), y)
    # M = np.hstack((X,np.array(list(ys)).reshape(-1,1)))
    M = np.hstack((X,y.reshape(-1,1)))
    df = pd.DataFrame(M)
    # rdf = com.convert_to_r_dataframe(df)
    # rdf = pandas2ri.py2rpy_pandasdataframe(df)
    with localconverter(robjects.default_converter + pandas2ri.converter):
        rdf = robjects.conversion.py2rpy(df)
    robjects.globalenv['rdf'] = rdf

    metrics = robjects.r('''
            # install.packages("ECoL")
            library("ECoL") # , lib.loc = "ECoL" to use the local lib
            complexity(rdf[,1:ncol(rdf)-1], rdf[,ncol(rdf)])
            ''')
    
    rpt = ''
    dct = {}
    for v in zip(ECoL_METRICS, metrics):
        rpt += v[0] + "\t" + str(v[1]) + "\n"
        dct[v[0]] = v[1]
        
    return dct, rpt

import os

def analyze_file(fn):
    if os.path.isfile(fn) == False:
        return 'File ' + fn + ' does not exist.'
        
    X,y = load_file(fn)
    return get_html(X,y)
    
def get_metrics(X,y):
    
    dct,_,_ = CLF(X,y)
    ber, _ = BER(X,y)
    dct['classification.BER'] = ber
    ig, _ = IG(X,y)
    dct['correlation.IG'] = ig
    dct['correlation.IG.max'] = ig.max()

    dct_cor,_ = correlate(X,y)
    dct.update(dct_cor)

    es, _ = cohen_d(X, y)
    dct['test.ES'] = es
    dct['test.ES.max'] = es.max()

    p, F, _ = ANOVA(X,y)
    dct['test.ANOVA'] = p
    dct['test.ANOVA.min'] = np.min(p)
    dct['test.ANOVA.min.log10'] = np.log10 (np.min(p) )
    dct['test.ANOVA.F'] = F
    dct['test.ANOVA.F.max'] = np.max(F)

    p, F, _ = MANOVA(X,y)
    dct['test.MANOVA'] = p
    dct['test.MANOVA.log10'] = np.log10 (p)
    dct['test.MANOVA.F'] = F

    p, U, _ = MWW(X,y)
    dct['test.MWW'] = p
    dct['test.MWW.min'] = np.min(p)
    dct['test.MWW.min.log10'] = np.log10 (np.min(p) )
    dct['test.MWW.U'] = U
    dct['test.MWW.U.min'] = np.min(U) 

    p, D, _ = KS(X,y)
    dct['test.KS'] = p
    dct['test.KS.min'] = np.min(p)
    dct['test.KS.min.log10'] = np.log10 (np.min(p) )
    dct['test.KS.D'] = D
    dct['test.KS.D.max'] = np.max(D) 

    dct_ecol,_ = ECoL_metrics(X,y)
    dct.update(dct_ecol)
    
    dct_s = {}
    
    for k, v in dct.items():
        if hasattr(v, "__len__"): # this is an np array or list
            dct[k] = list(v)
        else: # this only contains single-value metrics
            dct_s[k] = v

    return dct, dct_s

def metrics_keys():
    X, y = mvg(md = 2, nobs = 10)
    dct = get_metrics(X,y)
    return list(dct[0].keys())

import json
def get_json(X,y):    
    return json.dumps(get_metrics(X,y))

    
def get_html(X,y):
    '''
    Generate a summary report in HTML format
    '''
    html = '<table class="table table-striped">'

    tr = '<tr><th> Metric/Statistic </th><tr>' # <th> Value </th><th> Details </th>
    html += tr
    
    ber, ber_img = BER(X,y,show = False)

    # tr = '<tr><td> BER </td><td>' + str(ber) + '</td><td>' + ber_img + '</td><tr>'
    tr = '<tr><td> BER = ' + str(ber) + '<br/>' + ber_img + '</td><tr>'
    html += tr
    
    clf, clf_img, clf_log = CLF(X,y,show = False)

    # tr = '<tr><td> ACC </td><td>' + str(acc) + '</td><td>' + acc_img + '<br/><pre>' + acc_log + '</pre></td><tr>'
    tr = '<tr><td>' + str(clf) + '<br/>' + clf_img + '<br/><pre>' + clf_log + '</pre></td><tr>'
    html += tr
    
    ig, ig_img = IG(X,y,show = False)

    tr = '<tr><td> IG = ' + str(ig) + '<br/>' + ig_img + '</td><tr>'
    html += tr
    
    _, corr_log = correlate(X,y, verbose = False)
    tr = '<tr><td><pre>' + corr_log + '</pre></td><tr>'
    html += tr

    anova_p, _, anova_img = ANOVA(X,y)
    
    tr = '<tr><td> ANOVA p' + str(anova_p) + '<br/>' + anova_img + '</td><tr>'
    html += tr    
    
    manova_p, _, manova_log = MANOVA(X,y)
    
    tr = '<tr><td> MANOVA p = ' + str(manova_p) + '<br/><pre>' + manova_log + '</pre></td><tr>'
    html += tr  
    
    mww_p, _, mww_img = MWW(X,y)
    
    tr = '<tr><td> MWW p = ' + str(mww_p) + '<br/>' + mww_img + '</td><tr>'
    html += tr
    
    ks_p,_, ks_img = KS(X,y)
    
    tr = '<tr><td> K-S p = ' + str(ks_p) + '<br/>' + ks_img + '</td><tr>'
    html += tr    
    
    
    es, es_img = cohen_d(X,y)

    tr = '<tr><td> ES = ' + str(es) + '<br/>' + es_img + '</td><tr>'
    html += tr
    
    _, ecol = ECoL_metrics(X,y)

    tr = '<tr><td> ECoL metrics' + '<br/><br/><pre>' + ecol + '</pre></td><tr>'
    html += tr

    # dataset summary
    tr = '<tr><th> Dataset Summary </th><tr>' 
    html += tr

    tr = '<tr><td>' + str(len(y)) + ' samples, ' + str(X.shape[1]) +  ' features, ' + str(len(set(y)))  + ' classes. <br/> X shape: ' + str(X.shape) + ', y shape: ' + str(y.shape) + '</td><tr>'
    html += tr

    html += "</table>"
    # html += '<style> td { text-align:center; vertical-align:middle } </style>'

    return html


# try different sample sizes (nobs)

from tqdm import tqdm, tqdm_notebook
from sklearn.decomposition import PCA
import seaborn as sns

def simulate(mds, repeat = 1, nobs = 100, dims = 2):

    dcts = {}
    

    ## splits (divide 1 std into how many sections) * repeat         
    pbar = tqdm(total = len(mds) * repeat, position=0)
    
    ## only do visualization when repeat == 1 and draw == True 
    #detailed = (repeat == 1 and draw == True)

    for md in mds:
        
        dct = {}
        
        for i in range(repeat):
            X,y = mvg(
                # mu = mu, # mean, row vector
                # s = s, # std, row vector
                nobs = nobs, # number of observations / samples
                md = md, # distance between means, respect to std, i.e. (mu2 - mu1) / std, or how many stds is the difference.
                dims = dims
            )

            ## if detailed:
            #    print('d = ', round(md,3))
            
            _, dct1 = get_metrics(X,y)
            for k, v in dct1.items():
                if k in dct:
                    dct[k].append(v) # dct[k] = dct[k] + v
                else:
                    dct[k] = [v] # v
            
            pbar.update()
            ## End of inner iteration ##

        for k, v in dct.items():

            trim_size = int(repeat / 10) 

            if (repeat > 10): # remove the max and min
                dct[k] = np.mean(sorted(v)[trim_size:-trim_size])
            else:
                dct[k] = np.mean(v)
            
        ## End of outer iteration ##
    
        for k, v in dct.items():
            if k in dcts:
                dcts[k] = np.append(dcts[k], v)
            else:
                dcts[k] = np.array([v])
    
    dcts['d'] = np.array(mds)
    
    return dcts

import matplotlib.ticker as mticker
def visualize_dcts(dcts):
    
    N = len(dcts) - 1
    
    fig = plt.figure(figsize=(48, 10*(N/10+1)))
    plt.rcParams.update({'font.size': 18})
    
    i = 0
    for k, v in dcts.items():
        if k == 'd':
            pass
        else:
            ax = fig.add_subplot(N/6+1, 6, i+1)
            ax.scatter(dcts['d'], v, label = k)
            ax.plot(dcts['d'], v)
            ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
            ax.legend()
            ax.axis('tight')
        i += 1
        
    plt.axis('tight')
    plt.show()

    plt.rcParams.update({'font.size': 10})
    
def visualize_corr_matrix(dcts, cmap = 'coolwarm', exclude_d = True):
    
    M = dcts['d']
    names = ['d']
    
    for k, v in dcts.items():
        if k == 'd' or len(v) != len(dcts['d']):
            pass
        else:
            M = np.vstack((M, v))
            names.append(k)
            
    # exclude the d column
    if (exclude_d):
        M = M[1:]
        names = names[1:]

    plt.figure(figsize = (36,30))
    dfM = pd.DataFrame(M.T, columns = names) #, columns = ['d', 'BER', 'ACC', 'TOR', 'IG', 'MAN'])

    # plot the heatmap
    # sns.heatmap(np.abs(dfM.corr()), cmap="YlGnBu", annot=True)
    sns.heatmap(dfM.corr(), cmap=cmap, annot=True) # use diverging colormap, e.g., seismic, coolwarm, bwr

    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.tick_params(axis='both', which='minor', labelsize=10)
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    # plt.rcParams.update({'font.size': 10})   # restore fontsize
    plt.show()
    
    plt.figure(figsize = (20, 7))
    plt.bar(names[1:], np.abs(dfM.corr().values[0,1:]), facecolor="none", edgecolor = "black", width = 0.8, label = 'correlation coefficient (abs)') # ,width = 0.6, hatch='x'
    plt.plot(names[1:], [0.9] * len(names[1:]), '--', label = 'threshold at 0.9')
    plt.title('correlation with between-class distance')
    plt.xticks(rotation = 90)
    plt.legend(loc = 'lower right')
    plt.show()
    
    # print(np.where( np.abs(dfM.corr().values[0,1:])>0.9 ))
    print('Metrics above the threshold: ', np.array(names[1:]) [np.where( np.abs(dfM.corr().values[0,1:])>0.9 )])

def extract_PC(dcts):
    
    M = dcts['d']
    names = []
    
    for k, v in dcts.items():
        if k == 'd' or len(v) != len(dcts['d']) or np.isnan(v).any():
            pass
        else:
            M = np.vstack((M, v))
            names.append(k)
            
    dfM = pd.DataFrame(M[1:].T, columns = names)
    pca = PCA()
    PCs = pca.fit_transform(dfM.values)
    
    plt.figure(figsize = (3*len(pca.explained_variance_ratio_), 6))
    plt.bar(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_, facecolor="none", edgecolor = "black", width = 0.4, hatch='/')
    plt.title('explained variance ratio')
    plt.xticks ([]) 
    plt.show()
    
    
    plt.figure(figsize = (20, 7))
    plt.bar(names, pca.components_[0], facecolor="none", edgecolor = "black", hatch='/')
    plt.title('loadings / coefficients')
    plt.xticks(rotation = 90)
    plt.show()
    
    print("1st PC: ", PCs[:,0])
    
    return pca