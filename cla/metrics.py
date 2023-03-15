import sys
import os
import math
import re
import json
import scipy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from sklearn.metrics import *  # we use global() to access the imported functions
from sklearn.preprocessing import MinMaxScaler
# from scipy.integrate import quad
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import OneHotEncoder
from statsmodels.multivariate import manova
from statsmodels.stats.contingency_tables import mcnemar, cochrans_q

import rpy2

if __package__:
    from .vis.plt2base64 import plt2html
    from .vis.plotComponents2D import plotComponents2D
    from .vis.feature_importance import plot_feature_importance
    from .vis.unsupervised_dimension_reductions import unsupervised_dimension_reductions
else:
    VIS_DIR = os.path.dirname(__file__) + '/vis'
    if VIS_DIR not in sys.path:
        sys.path.append(VIS_DIR)

    from plt2base64 import plt2html
    from plotComponents2D import plotComponents2D
    from feature_importance import plot_feature_importance
    from unsupervised_dimension_reductions import unsupervised_dimension_reductions

ENABLE_R = True
if sys.platform == "win32" and rpy2.__version__ >= '3.0.0':
    print('rpy2 3.X may not support Windows. ECoL metrics may not be available.')
    # ENABLE_R = False

if ENABLE_R:
    try:
        import rpy2.robjects as robjects
        # Defining the R script and loading the instance in Python
        from rpy2.robjects import pandas2ri
        import rpy2.robjects.packages as rpackages
        from rpy2.robjects.vectors import StrVector
        # import rpy2.robjects.numpy2ri
        from rpy2.robjects.conversion import localconverter
    except Exception as e:
        print(e)

# generate and plot 2D multivariate gaussian data set


def mvg(
    nobs=20,  # number of observations / samples per class
    # distance between means, respect to std, i.e. (mu2 - mu1) / std, or how many stds is the difference.
    md=2,
    dims=2,  # 1 or 2
):
    '''
    Draw samples from two multivarite Gaussian distributions with different 𝜇 and Σ values.
    The simulated data can be used to analyze the properties of different metrics.

    mvg has two important parameters, md and nobs.
    Theoretically, cohen's d effect size is related to md (distance between means, divided by std) and nobs
    '''

    N = nobs

    if (dims == 2):

        s = 1  # reference std = 1
        mu1 = [- md * s/2,  0]
        mu2 = [md * s/2, 0]
        # use diagonal covariance. Naive assuption: features are uncorrelated
        s1 = [s, s]
        s2 = [s, s]

        cov1 = np.array([[s1[0], 0], [0, s1[1]]])
        cov2 = np.array([[s2[0], 0], [0, s2[1]]])

        xc1 = np.random.multivariate_normal(mu1, cov1, N)
        xc2 = np.random.multivariate_normal(mu2, cov2, N)

        y = np.concatenate((np.zeros(N), np.ones(N))).astype(int)
        X = np.vstack((xc1, xc2))

    elif (dims == 1):

        s = 1
        xc1 = np.random.randn(N) * s - md * s / 2
        xc2 = np.random.randn(N) * s + md * s / 2

        X = np.concatenate((xc1, xc2)).reshape(-1, 1)
        y = np.concatenate((np.zeros(N), np.ones(N))).astype(int)

    else:

        xc1 = np.random.randn(N, dims) - md / 2
        xc2 = np.random.randn(N, dims) + md / 2

        X = np.vstack((xc1, xc2))
        y = np.concatenate((np.zeros(N), np.ones(N))).astype(int)

        # raise Exception('only support 1 or 2 dims')
        print(
            'You specified more than 2 dims. Some metric visualizations will be disabled.')

    return X, y


def mvgx(
    mu,  # mean, row vector
    s,  # std, row vector
    md=2,
    nobs=15
):
    '''
    Generate simulated high-dim (e.g., spectroscopic profiling) data
    This is an extended version of mvg() that accepts specified mu and s vectors.

    Parameters
    ----------
    mu : the mean vector of the target domain dataset. must be row vector
    s : the standard deviation vector. must be row vector
    md : distance between means, respect to std, i.e. (mu2 - mu1) / std, or how many stds is the difference.
    nobs : number of observations / samples per class

    Example
    -------
    X,y = mvgx(
    [-1,0,-1], # mean, row vector
    [1,2,0.5], # variance, row vector
    md = 1, # distance between means, respect to std, i.e. (mu2 - mu1) / std, or how many stds is the difference.
    nobs = 5, # number of observations / samples    
    )
    '''

    mu = np.array(mu)
    s = np.array(s)

    N = nobs
    cov = np.diag(s**2)

    mu1 = mu - s * md / 2
    mu2 = mu + s * md / 2

    xc1 = np.random.multivariate_normal(mu1, cov, N)
    xc2 = np.random.multivariate_normal(mu2, cov, N)

    y = np.concatenate((np.zeros(N), np.ones(N))).astype(int)
    X = np.vstack((xc1, xc2))

    return X, y


def save_file(X, y, pathname):
    '''
    Save data to a csv file    
    '''

    # fn = str(uuid.uuid1()) + '.csv'
    M = np.hstack((X, y.reshape(-1, 1)))
    np.savetxt(pathname, M, delimiter=',', fmt='%.3f,' *
               X.shape[1] + '%i', header='X1,X2,...,Y')  # fmt='%f'
    return pathname


def load_file(pathname):
    '''
    Load data from a csv file    
    '''
    M = np.loadtxt(pathname, delimiter=',', skiprows=1)
    X = M[:, :-1]
    y = M[:, -1].astype(int)
    return X, y

#
# plot contour of multivariate Gaussian distributions


def plot_gaussian_contour(X, y, mu1, s1, mu2, s2, alpha=0.4):
    '''
    Plot contour of multivariate Gaussian distributions   
    '''

    plt.figure()  # figsize = (9, 6)

    X1 = X[:, 0]
    if (X.shape[1] > 1):
        X2 = X[:, 1]
    else:
        X2 = X1.copy()

    dx = X1.max()-X1.min()
    dy = X2.max()-X2.max()

    Xg1, Xg2 = np.mgrid[X1.min() - dx * 0.2: X1.max() + dx * 0.2: 0.01,
                        X2.min() - dy * 0.2: X2.max() + dy * 0.2: 0.01]
    pos = np.empty(Xg1.shape + (2,))
    pos[:, :, 0] = Xg1
    pos[:, :, 1] = Xg2

    rv1 = scipy.stats.multivariate_normal(mu1, s1)
    c1 = plt.contour(Xg1, Xg2, rv1.pdf(pos), alpha=alpha, cmap='Reds')
    plt.clabel(c1, inline=True, fontsize=10)

    rv2 = scipy.stats.multivariate_normal(mu2, s2)
    c2 = plt.contour(Xg1, Xg2, rv2.pdf(pos), alpha=alpha, cmap='Reds')
    plt.clabel(c2, inline=True, fontsize=10)

    # print('C1: X~N(', mu1, ',', s1, ')')
    # print('C2: X~N(', mu2, ',', s2, ')')

    return plt.gca()


def select_features(X, y, metric, metric_name='', N=30, feature_names=None):
    '''
    Perform feature selection via a specified metric

    Parameters
    ----------
    metric : an array of feature-wise metric, e.g., IG, correlation.r2, etc. Usually should be non-negative
    N : number of feature to select
    '''
    N = min(N, X.shape[1])

    plot_feature_importance(np.array(metric), metric_name, row_size=300)
    # idx = np.where(F > 30)[0] # np.where(pval < 0.00001)
    idx = np.argsort(metric)[::-1][:N]

    if feature_names is not None:
        print('Top ' + str(N) + ' important feature indices: ', idx)
        print('Top ' + str(N) + ' important feature names: ',
              np.array(feature_names)[np.array(idx)])

    unsupervised_dimension_reductions(X[:, idx], y, set(y))

    return idx


def BER(X, y, nobs=10000, NSigma=10, show=False, save_fig=''):
    """
    We draw random samples from the bayes distribution models to calculate BER

    nobs - number of observations, i.e., sample size
    NSgima - the sampling range
    """

    nb = GaussianNB(priors=[0.5, 0.5])  # we have no strong prior assumption.
    nb.fit(X, y)

    labels = list(set(y))

    # For multi-class classification, use one vs rest strategy
    assert len(labels) == 2

    Xc1 = X[y == labels[0]]
    Xc2 = X[y == labels[1]]

    n1 = len(Xc1)
    n2 = len(Xc2)

    mu1 = np.mean(Xc1, axis=0)
    mu2 = np.mean(Xc2, axis=0)

    s1 = np.std(Xc1, axis=0, ddof=1)
    s2 = np.std(Xc2, axis=0, ddof=1)

    #print(mu1, mu2)
    #print(s1, s2)
    # print(nb.var_)

    lb = np.minimum(mu1 - NSigma*s1, mu2 - NSigma*s2)
    ub = np.maximum(mu1 + NSigma*s1, mu2 + NSigma*s2)

    # we use M random samples to calculate BER
    XM = np.zeros((nobs, X.shape[1]))
    for i in range(nobs):
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
    IMG = ''

    if X.shape[1] == 2:
        # for 2-dimensional data, plot the contours
        ax = plot_gaussian_contour(X, y, nb.theta_[0], np.sqrt(
            nb.var_[0]), nb.theta_[1], np.sqrt(nb.var_[1]), alpha=0.3)
        plotComponents2D(X, y, set(y), use_markers=False, ax=ax)
        plt.legend()
        title = ' $ \mu $ = ' + str(np.round(nb.theta_, 3)) + \
            ', $\sigma^2$ = ' + str(np.round(nb.var_, 3)).replace('\n', '')
        plt.title(title)

        if (save_fig != '' and save_fig is not None):
            if not save_fig.endswith('.jpg'):
                save_fig += '.jpg'
            plt.savefig(save_fig)
            print('figure saved to ' + save_fig)

        IMG = plt2html(plt)

        if show:
            plt.show()
        else:
            plt.close()

    return BER, IMG  # , BER2


def Mean_KLD(P, Q):
    '''
    Calculate the mean KL divergence between ground truth and predicted one-hot encodings for an entire data set.
    P and Q must be both m x K numpy arrays. m = sample number, K = class number
    '''
    klds = []
    for idx, ground_truth in enumerate(P):
        prediction = Q[idx]
        # If 2nd param is not None, then compute the Kullback-Leibler divergence. S = sum(pk * log(pk / qk), axis=axis).
        kld = scipy.stats.entropy(ground_truth, prediction)

        # from scipy.special import rel_entr
        # #calculate (Q || P)
        # sum(rel_entr(Q, P))

        klds.append(kld)
    return np.mean(klds), klds


CLF_METRICS = ['classification.ACC',
               'classification.Kappa',
               'classification.F1_Score',
               'classification.Jaccard',  # The Jaccard index, or Jaccard similarity coefficient, defined as the size of the intersection divided by the size of the union of two label sets
               'classification.Precision',
               'classification.Recall',
               'classification.McNemar',
               'classification.McNemar.CHI2',
               'classification.CochranQ',
               'classification.CochranQ.T',
               # The following requires a model that outputs probability
               'classification.CrossEntropy',  # cross-entropy loss / log loss
               'classification.Mean_KLD',
               # 'classification.AP',
               'classification.Brier',
               'classification.ROC_AUC',
               'classification.PR_AUC']

########## Section: SVM / LR ###########


def grid_search_svm_hyperparams(X, y, test_size=0.2, tuned_parameters=[
    {'kernel': ['rbf'], 'gamma': [10, 1, 1e-1, 1e-2],
        'C': [0.01, 0.1, 1, 10, 100, 1000]},
        {'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}], cv=5, verbose=True):
    '''
    Find the optimal SVM model by grid search.
    Returns the best model and ACCs on training / testing / all data set
    '''

    X_train, X_test, y_train, y_test = train_test_split(
        X, np.array(y), test_size=test_size)  # , random_state=0

    log = ''
    # log += "X_train: " + str(X_train.shape) + ", y_train: " + str(y_train.shape)

    # iid = True. accept an estimator object that implements the scikit-learn estimator interface. Explicitly set iid to avoid DeprecationWarning.
    gs = GridSearchCV(SVC(), tuned_parameters, cv=cv)
    gs.fit(X_train, y_train)

    log += "\nBest parameters set found by GridSearchCV: \n"
    log += str(gs.best_params_)
    log += "\nGrid scores on cv set:\n"

    means = gs.cv_results_['mean_test_score']
    stds = gs.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, gs.cv_results_['params']):
        log += "{} (+/- {}) for {} \n".format(round(mean, 5),
                                              round(std * 2, 5), params)

    log += ("\nDetailed classification report:\n")

    log += ('\n#### Training Set ####\n')
    y_true, y_pred = y_train, gs.predict(X_train)
    log += classification_report(y_true, y_pred)

    log += ('\n\n#### Test Set ####\n')
    y_true, y_pred = y_test, gs.predict(X_test)
    log += classification_report(y_true, y_pred)

    log += ('\n\n#### All Set ####\n')
    y_true, y_pred = y, gs.predict(X)
    log += classification_report(y_true, y_pred)

    if verbose:
        print(log)

    # , gs.score(X_train, y_train), gs.score(X_test, y_test), gs.score(X,y), log
    return gs.best_params_, gs.best_estimator_, log


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]
                    )  # requires clf to accept 2D data input
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def plot_svm_boundary(X, y, clf, Xn=None):
    '''
    Xn : new data samples to be tested. Will be shown in strong color.
    '''
    return plot_clf_boundary(X, y, clf, Xn=None, clf_type='svm')


def plot_lr_boundary(X, y, clf, Xn=None):
    '''
    Xn : new data samples to be tested. Will be shown in strong color.    
    '''
    return plot_clf_boundary(X, y, clf, Xn=None, clf_type='lr')


def plot_clf_boundary(X, y, clf, Xn=None, clf_type='svm'):
    '''
    clf_type : svm or lr (logistic regression)
    Xn : data samples to be tested. Will be shown in strong color.
    '''
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    cmap = matplotlib.colors.ListedColormap(
        ['0.8', '0.1', 'red', 'blue', 'black', 'orange', 'green', 'cyan', 'purple', 'gray'])

    plt.figure()
    plot_contours(plt, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.1)
    plt.scatter(X0, X1, c=y, s=70, facecolors=cmap,
                edgecolors='gray', alpha=.4)  # cmap='gray'
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.xticks(())
    plt.yticks(())

    if clf_type == 'lr':
        # Plot K one-against-all classifiers
        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()
        coef = clf.coef_
        intercept = clf.intercept_

        def plot_hyperplane(c, color):
            def line(x0):
                return (-(x0 * coef[c, 0]) - intercept[c]) / coef[c, 1]
            plt.plot([xmin, xmax], [line(xmin), line(xmax)],
                     ls=":", color=color, label='clf'+str(c))

        if (len(clf.classes_) > 2):
            for i, color in zip(clf.classes_, 'bgr'):
                plot_hyperplane(i, color)

    if Xn is not None:
        Xn0, Xn1 = Xn[:, 0], Xn[:, 1]
        plt.scatter(Xn0, Xn1, c='r', s=120, facecolors=cmap,
                    edgecolors='k', alpha=1, label='Sample')  # cmap='gray'

    plt.legend()
    plt.show()
    print("SVC({})".format(clf.get_params()))


def classify_with_svm(X, y):
    """Train SVM classifiers

    Parameters
    ----------
    X: feature matrix of with 2 columns 
    y: label    
    """

    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    C = 1.0  # SVM regularization parameter
    models = (SVC(kernel='linear', C=C),
              LinearSVC(C=C),
              SVC(kernel='rbf', gamma=0.7, C=C),
              SVC(kernel='poly', degree=3, C=C))
    models = (clf.fit(X, y) for clf in models)

    # title for the plots
    titles = ('SVC - linear kernel',
              'LinearSVC',
              'SVC RBF kernel',
              'SVC polynomial kernel (degree=3)')

    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(2, 2, figsize=(15, 15))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    for clf, title, ax in zip(models, titles, sub.flatten()):
        plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.1)
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title + '\n(score: {0:.2})'.format(clf.score(X, y)))

    plt.show()

########### End of SVM / LR Section ##########


def CLF(X, y, verbose=False, show=False, save_fig=''):
    '''
    X,y - features and labels
    '''

    dic = {}
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

    grp_samples = []
    for yv in set(y):
        grp_samples.append((y == yv).sum())

    # min(grp_samples) is the minimum sample size among all categories.
    # CV requires to be not greater than this value.

    try:
        clf = LogisticRegressionCV(cv=min(3, min(grp_samples)), max_iter=1000).fit(
            X, y)  # ridge(L2) regularization
    except:
        print('Exception in LogisticRegressionCV().')
        return None, None, None

    LOG += "regularization strength\t" + str(clf.C_) + "\n\n"
    # l1_ratio: while 1 is equivalent to using penalty='l1'. For 0 < l1_ratio <1, the penalty is a combination of L1 and L2.

    IMG = ''

    # visualize the decision boundary in a 2D plane if X has two features
    if (X.shape[1] == 2):

        plt.figure()

        # plt.scatter(data['X1'], data['X2'], s=50, c=clf.predict_proba(data[['X1', 'X2']])[:,0], cmap='seismic')
        plt.scatter(X[:, 0], X[:, 1], s=50,
                    c=clf.decision_function(X), cmap='seismic')

        # plot the decision function
        ax = plt.gca()  # get current axes
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # create grid to evaluate model
        xx = np.linspace(xlim[0], xlim[1], 30)
        yy = np.linspace(ylim[0], ylim[1], 30)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = clf.decision_function(xy).reshape(XX.shape)

        # plot decision boundary and margins
        _ = ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
                         linestyles=['--', '-', '--'])

        # in some cases, there can be multiple unconnected decision boundaries

        # for i in range(len(out.collections[1].get_paths())):
        #    vertice_set.append(out.collections[1].get_paths()[i].vertices)

        # plot support vectors if using SVM
        if (hasattr(clf, "support_vectors_")):
            ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
                       linewidth=1, facecolors='none', edgecolors='k')
            # title = 'kernel = ' + best_params['kernel'] + ", C = " + str(best_params['C']) + " , acc = " + str(round(clf.score(X, y),3))
            #
            # if 'gamma' in best_params:
            #    title += ", $\gamma$ = " + str(best_params['gamma'])
            # ax.set_title(title)

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
    clf_metrics.append(globals()["accuracy_score"](y, y_pred))
    # use ground truth and prediction as 1st and 2nd raters. 0 - no agreement, 1 - perfect agreement.
    clf_metrics.append(globals()["cohen_kappa_score"](y, y_pred))
    clf_metrics.append(globals()["f1_score"](y, y_pred))
    clf_metrics.append(globals()["jaccard_score"](y, y_pred))
    clf_metrics.append(globals()["precision_score"](y, y_pred))
    clf_metrics.append(globals()["recall_score"](y, y_pred))

    res = mcnemar(confusion_matrix(y, y_pred), exact=False, correction=True)
    clf_metrics.append(res.pvalue)
    clf_metrics.append(res.statistic)

    data = np.hstack((np.array(y).reshape(-1, 1),
                     np.array(y_pred).reshape(-1, 1)))
    res = cochrans_q(data)
    clf_metrics.append(res.pvalue)
    clf_metrics.append(res.statistic)

    ###
    # train a logistic regression model to compute Brier score, PR AUC, etc.
    # clf = LogisticRegressionCV(cv=5, random_state=0).fit(X, y)
    y_prob_ohe = clf.predict_proba(X)
    # for binary classification, use the second proba as P(Y=1|X)
    y_prob = y_prob_ohe[:, 1]

    enc = OneHotEncoder(handle_unknown='ignore')
    y_ohe = enc.fit_transform(np.array(y).reshape(-1, 1)).toarray()
    # print(y_ohe)
    # print(y_prob_ohe)

    clf_metrics.append(globals()["log_loss"](y, y_prob))

    mkld, _ = Mean_KLD(y_ohe, y_prob_ohe)
    clf_metrics.append(mkld)

    # clf_metrics.append(globals()["average_precision_score"](y, y_prob)) # According to the sklearn doc, this should equal to the P-R curve AUC. However, the actual result diifers. Implementation may have problems.
    # The Brier score measures the mean squared difference between the predicted probability and the actual outcome.
    clf_metrics.append(globals()["brier_score_loss"](y, y_prob))
    clf_metrics.append(globals()["roc_auc_score"](y, y_prob))

    # set pos_label for cases when y is not {0,1} or {-1,1}
    precisions, recalls, _ = precision_recall_curve(
        y, y_prob, pos_label=max(y))
    clf_metrics.append(globals()["auc"](recalls, precisions))

    rpt = ''
    for v in zip(CLF_METRICS, clf_metrics):
        rpt += v[0] + "\t" + str(v[1]) + "\n"
        dic[v[0]] = v[1]

    LOG += "\n\n" + rpt

    # acc_train, acc_test, acc_all # , vertice_set # vertice_set[0] is the decision boundary
    return dic, IMG, LOG


def SVM_Margin_Width(X, y, scale=True, show=False):
    '''
    SVM hyperplane margin width

    Note
    ----
    When the between-class distance is small (< 3std), there are many overlaps, 
    the margin width is big. 
    This metric is only linear after the distance is big enough.
    '''

    if scale:
        X = MinMaxScaler().fit_transform(X)

    svc_model = SVC(kernel='linear')  # C = 10
    svc_model = svc_model.fit(X, y)

    support_vectors = svc_model.support_vectors_
    w = svc_model.coef_[0]
    p = np.linalg.norm(w, ord=2)
    width = 2/p

    IMG = ''

    if len(support_vectors[1]) == 2 and len(set(y)) == 2:
        df = pd.DataFrame(X)

        x_min = np.min(df.iloc[:, 0]) - 0.5
        x_max = np.max(df.iloc[:,  0]) + 0.5
        y_min = np.min(df.iloc[:, 1]) - 0.5
        y_max = np.max(df.iloc[:, 1]) + 0.5

        x = np.arange(x_min, x_max, 0.1)
        y0 = -svc_model.intercept_/w[1]-w[0]/w[1]*x
        y_up = (1-svc_model.intercept_)/w[1]-w[0]/w[1]*x
        y_down = (-1 - svc_model.intercept_) / w[1] - w[0] / w[1] * x

        plt.plot(x, y0)
        plt.plot(x, y_up, linestyle='--')
        plt.plot(x, y_down, linestyle='--')
        plt.ylim(y_min, y_max)
        plt.xlim(x_min, x_max)

        labels = set(y)

        for label in labels:
            cluster = X[np.where(y == label)]
            plt.scatter(cluster[:, 0], cluster[:, 1])

        IMG = plt2html(plt)

        if show:
            plt.show()
        else:
            plt.close()

    return width, IMG


def IG(X, y, show=False, save_fig=''):
    """
    Return the feature-wise information gains.
    It can be proven that Info Gain = Mutual information
    We can use mutual_info_classif() to calculate info gain.

    Usually we prefer Info Gain over Correlation, because:
    Correlation only measures the linear relationship (Pearson's correlation) or monotonic relationship (Spearman's correlation) between two variables.
    Mutual information is more general and measures the reduction of uncertainty in Y after observing X. It is the KL distance between the joint density and the product of the individual densities. So MI can measure non-monotonic relationships and other more complicated relationships.

    Note
    ----
    When X is multi-dimensional (i.e., X = X1, X2, …, Xn), IG(Y|X) = IG(Y| X1, X2, …Xn).
    If we cannot assume the infogain from X1, X2, ..., Xn are not overlapped, we cannot simply add up IG(Y|X1), IG(Y|X2), ..., IG(Y|Xn).

    Implementation Details
    ----------------------
    Be cautious about the discrete_features parameter : {‘auto’, bool, array_like}, default ‘auto’ in funciton mutual_info_classif().
    If bool, then determines whether to consider all features discrete or continuous. If array, then it should be either a boolean mask with shape (n_features,) or array with indices of discrete features. If ‘auto’, it is assigned to False for dense X and to True for sparse X.  
    For continous feature, use discrete_features = False.

    The term “discrete features” is used instead of naming them “categorical”, because it describes the essence more accurately. For example, pixel intensities of an image are discrete features (but hardly categorical) and you will get better results if mark them as such. Also note, that treating a continuous variable as discrete and vice versa will usually give incorrect results, so be attentive about that.
    True mutual information can’t be negative. If its estimate turns out to be negative, it is replaced by zero.
    """

    try:
        mi = mutual_info_classif(X, y, discrete_features=False)
    except:
        print('Exception in mutual_info_classif().')
        return None, None

    mi_sorted = np.sort(mi)[::-1]  # sort in desceding order
    mi_sorted_idx = np.argsort(mi)[::-1]

    if (X.shape[1] > 50):
        plt.figure(figsize=(20, 3))
    else:
        plt.figure()  # use default fig size

    xlabels = []
    for i, v in enumerate(mi_sorted):
        xlabels.append("X"+str(mi_sorted_idx[i] + 1))
        # if (len(mi_sorted) < 20): # don't show text anno if feature number is large
        plt.text(i-0.001, v+0.001,  str(round(v, 1)))

    plt.bar(xlabels, mi_sorted, facecolor="none",
            edgecolor="black", width=0.3, hatch='/')

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


def CHISQ(X, y, show=False, save_fig=''):
    """
    Performa feature-wise chi-square test. 
    Returns an array of chi2 statistics and p-values on all the features.

    This test can be used to select the n_features features with the highest values
    for the test chi-squared statistic from X, which must contain only non-negative
    features such as booleans or frequencies (e.g., term counts in document 
    classification), relative to the classes.
    Recall that the chi-square test measures dependence between stochastic 
    variables, so using this function “weeds out” the features that are the 
    most likely to be independent of class and therefore irrelevant for 
    classification.
    """

    if (len(set(y)) < 2):
        raise Exception('The dataset must have at least two classes.')

    IMG = ''

    # chi2 test requires scaling to [0,1]
    mm_scaler = MinMaxScaler()
    X_mm_scaled = mm_scaler.fit_transform(X)

    CHI2s, ps = chi2(X_mm_scaled, y)

    if X.shape[1] > 50:
        plt.figure(figsize=(20, 3))
    else:
        plt.figure()  # use default fig size

    plt.bar(range(len(CHI2s)), CHI2s, facecolor="none",
            edgecolor="black", width=0.3, hatch='/')
    plt.title('chi squared statistics')
    # plt.xticks ([])

    if save_fig != '' and save_fig is not None:
        if save_fig.endswith('.jpg') == False:
            save_fig += '.jpg'
        plt.savefig(save_fig)
        print('figure saved to ' + save_fig)

    IMG = plt2html(plt)

    if show:
        plt.show()
    else:
        plt.close()

    return ps.tolist(), CHI2s.tolist(), IMG


def KW(X, y, verbose=False):
    '''
    A Kruskal-Wallis test is used to determine whether or not 
    there is a statistically significant difference between 
    the medians of three or more independent groups. 
    This test is the nonparametric equivalent of the one-way ANOVA 
    and is typically used when the normality assumption is violated.

    Mann-Whitney或Wilcoxon检验比较两组，而Kruskal-Wallis检验比较3组及以上。
    Kruskal-Wallis检验是一种非参数的单因素方差分析。它是基于秩（排序的，只考虑相对大小）的。

    由于KW检验考虑了样本的排序信息,而不仅仅是大于或小于中位数,因此比median test具有更大的power
    '''

    if len(set(y)) < 2:
        print('The dataset must have at least 2 classes.')
        return None, None

    ps = []
    Hs = []

    for i in range(X.shape[1]):
        Xi = X[:, i]
        Xcis = []

        for c in set(y):
            Xc = Xi[y == c]
            Xcis.append(Xc)

        # Xcis = np.array(Xcis)

        if len(set(y)) == 2:
            H, p = scipy.stats.kruskal(Xcis[0], Xcis[1], nan_policy='omit')
        elif len(set(y)) == 3:
            H, p = scipy.stats.kruskal(
                Xcis[0], Xcis[1], Xcis[2], nan_policy='omit')
        elif len(set(y)) == 4:
            H, p = scipy.stats.kruskal(
                Xcis[0], Xcis[1], Xcis[2], Xcis[3], nan_policy='omit')
        elif len(set(y)) >= 5:
            H, p = scipy.stats.kruskal(
                Xcis[0], Xcis[1], Xcis[2], Xcis[3], Xcis[4], nan_policy='omit')
            print("WARN: only the first 5 classes will be analyzed.")

        ps.append(p)
        Hs.append(H)

    if verbose:
        print('The P values for X in dimensions 1 to {}:{}'.format(
            len(X[0]), ps))
        print('The values of H statistics for X in dimensions 1 to {}:{}'.format(
            len(X[0]), Hs))

    return ps, Hs


def T_IND(X, y, verbose=False, show=False, max_plot_num=5):
    '''
    independent t test. requires two classes/groups.
    '''

    if (len(set(y)) != 2):
        print('The dataset must have 2 classes.')
        return None, None, None

    ps = []
    Ts = []
    cnt = 0
    IMG = ''

    for i in range(X.shape[1]):
        Xi = X[:, i]
        Xcis = []

        labels = []
        for c in set(y):
            Xc = Xi[y == c]
            Xcis.append(Xc)
            labels.append("$ X_" + str(i + 1) + "^{( y_" + str(c) + " )} $")

        # print(Xcis)
        # Xcis = np.array(Xcis) # will raise error: 'The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (2,) + inhomogeneous part.'

        bart = scipy.stats.bartlett(Xcis[0], Xcis[1])[1]
        lev = scipy.stats.levene(Xcis[0], Xcis[1])[1]

        if bart > 0.5 or lev > 0.5:
            T, p = scipy.stats.ttest_ind(Xcis[0], Xcis[1])
        else:
            T, p = scipy.stats.ttest_ind(Xcis[0], Xcis[1], equal_var=False)

        ps.append(p)
        Ts.append(T)

        if (cnt < max_plot_num):
            plt.figure()
            # plot ith feature of different classes
            plt.boxplot(Xcis, notch=False, labels=labels)
            test_result = "independent t test on X{}: T={},p={}".format(
                i + 1, round(T, 3), round(p, 3))
            # plt.legend(labels)
            plt.title(test_result)
            IMG += plt2html(plt)

            if show:
                plt.show()
            else:
                plt.close()
        elif cnt == max_plot_num:
            IMG += '<p>Showing the first ' + str(max_plot_num) + ' plots.</p>'
        else:
            pass  # plot no more to avoid memory cost

    if verbose:
        print('The P values of X in dimensions 1 to {}:{}'.format(
            len(X[0]), ps))
        print('The values of T statistics of X in dimensions 1 to {}:{}'.format(
            len(X[0]), Ts))

    return ps, Ts, IMG


def MedianTest(X, y, verbose=False, show=False):
    '''
    中位数检验是独立性卡方检验的一种特殊情况。

    给定k组样本，n1, n2 …… nk观测值，计算所有n1 +n2 + ……+nk观测值的中位数。
    然后构造一个2xk列联表，其中第一行包含k个样本的中位数以上的观测值，第二行包含k个样本的中位数以下或等于中位数的观测值。
    然后可以对该表应用独立性卡方检验。
    '''
    if (len(set(y)) < 2):
        print('The dataset must have at least 2 classes.')
        return None, None, None

    ps = []
    Ts = []
    MED = []  # grand median
    TBL = []  # contingency table
    IMG = ''

    for i in range(X.shape[1]):
        Xi = X[:, i]
        Xcis = []

        for c in set(y):
            Xc = Xi[y == c]
            Xcis.append(Xc)

        # Xcis = np.array(Xcis)

        if len(set(y)) == 2:
            T, p, med, tbl = scipy.stats.median_test(
                Xcis[0], Xcis[1], ties='ignore')
        elif len(set(y)) == 3:
            T, p, med, tbl = scipy.stats.median_test(
                Xcis[0], Xcis[1], Xcis[2], ties='ignore')
        elif len(set(y)) == 4:
            T, p, med, tbl = scipy.stats.median_test(
                Xcis[0], Xcis[1], Xcis[2], Xcis[3], ties='ignore')
        elif len(set(y)) >= 5:
            T, p, med, tbl = scipy.stats.median_test(
                Xcis[0], Xcis[1], Xcis[2], Xcis[3], Xcis[4], ties='ignore')
            print("WARN: only the first 5 classes will be analyzed.")

        ps.append(p)
        Ts.append(T)
        MED.append(med)
        TBL.append(tbl)

        a1 = TBL[0][:, 0]
        a2 = TBL[0][:, 1]

    if len(X[0]) == 2 and len(set(y)) == 2:

        idx = 1

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        for ax, tbl, m, p, T in zip(axes, TBL, MED, ps, Ts):

            tick_labels = ["> grand \n median", "< grand \n median"]

            im = ax.imshow(tbl, cmap='Greys')

            # # We want to show all ticks...
            ax.set_xticks(np.arange(len(tick_labels)))
            ax.set_yticks(np.arange(len(tick_labels)))
            # ... and label them with the respective list entries
            ax.set_xticklabels(tick_labels)
            ax.set_yticklabels(tick_labels)

            # Rotate the tick labels and set their alignment.
            plt.setp(ax.get_xticklabels(), rotation=0, ha="center", va="top",
                     rotation_mode="anchor")
            plt.setp(ax.get_yticklabels(), rotation=0, ha="right",
                     rotation_mode="anchor")

            # Loop over data dimensions and create text annotations.
            for i in range(len(tick_labels)):
                for j in range(len(tick_labels)):
                    _ = ax.text(j, i, tbl[i, j],
                                   ha="center", va="center", color="gray")

            ax.set_title("Median contingency table (X" + str(idx) +
                         ")\nCHI2 statistic = " + str(round(T, 3)) + ", p = " + str(round(p, 3)))
            idx += 1

        fig.tight_layout()
        IMG = plt2html(plt)

        if show:
            plt.show()
        else:
            plt.close()

    if verbose:
        print('The P value of X in dimensions 1 to {}:{}'.format(
            len(X[0]), ps))
        print('The values of the contingency table chi2 statistic of X in dimensions 1 to {}:{}'.format(
            len(X[0]), Ts))

    return Ts, ps, IMG


def ANOVA(X, y, verbose=False, show=False, max_plot_num=5):
    """
    Performa feature-wise ANOVA test. Returns an array of p-values on all the features and its minimum.

    y - support up to 5 classes
    """

    if (len(set(y)) < 2):
        raise Exception('The dataset must have at least two classes.')

    ps = []
    IMG = ''
    Fs = []

    cnt = 0

    for i in range(X.shape[1]):
        Xi = X[:, i]
        Xcis = []

        labels = []
        for c in set(y):
            Xc = Xi[y == c]
            Xcis.append(Xc)
            labels.append("$ X_"+str(i+1)+"^{( y_"+str(c)+" )} $")

        # Xcis = np.array(Xcis)

        # equal to ttest_ind() in case of 2 groups
        f, p = scipy.stats.f_oneway(Xcis[0], Xcis[1])

        if (len(set(y)) == 3):
            f, p = scipy.stats.f_oneway(Xcis[0], Xcis[1], Xcis[2])
        elif (len(set(y)) == 4):
            f, p = scipy.stats.f_oneway(Xcis[0], Xcis[1], Xcis[2], Xcis[3])
        elif (len(set(y)) >= 5):  # if there are five or more classes
            f, p = scipy.stats.f_oneway(
                Xcis[0], Xcis[1], Xcis[2], Xcis[3], Xcis[4])
            print('WARN: only the first 5 classes will be analyzed.')

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

        if (cnt < max_plot_num):

            plt.figure()
            # plot ith feature of different classes
            plt.boxplot(Xcis, notch=False, labels=labels)
            test_result = "ANOVA on X{}: f={},p={}".format(i+1, f, round(p, 3))
            # plt.legend(labels)
            plt.title(test_result)
            IMG += plt2html(plt)

            if show:
                plt.show()
            else:
                plt.close()
        elif cnt == max_plot_num:
            IMG += '<p>Showing the first ' + str(max_plot_num) + ' plots.</p>'
        else:
            pass  # plot no more to avoid memory cost

        cnt = cnt+1

        if verbose:
            print(test_result)

    IMG += '<br/>'

    return ps, Fs, IMG


def MANOVA(X, y, verbose=False):
    """
    MANOVA test of the first two features.  

    For some statisticians the MANOVA doesn’t only compare differences in mean scores between multiple groups but also assumes a cause effect relationship whereby one or more independent, controlled variables (the factors) cause the significant difference of one or more characteristics. The factors sort the data points into one of the groups causing the difference in the mean value of the groups.
    Internally, it uses multivariate regression
    """

    if (X.shape[1] <= 1):
        txt = 'There must be more than one dependent variable to fit MANOVA! Use ANOVA to substitute MANOVA.'
        anova_p, anova_F, _ = ANOVA(X, y)
        return anova_p, anova_F, txt

    X1 = X[:, 0]
    X2 = X[:, 1]
    df = pd.DataFrame({'X1': X1, 'X2': X2, 'y': y})
    # Intercept is included by default.
    mv = manova.MANOVA.from_formula('X1 + X2 ~ y', data=df)

    try:
        r = mv.mv_test()
    except:  # LinAlgError: Singular matrix
        return math.nan, math.nan, 'Exception in MANOVA'

    LOG = ''
    LOG += 'endog: ' + str(mv.endog_names) + '\n'
    LOG += 'exog: ' + str(mv.exog_names) + '\n\n'
    LOG += str(r)

    delimiters = "Wilks' lambda", "Pillai's trace"
    regexPattern = '|'.join(map(re.escape, delimiters))
    ss = re.split(regexPattern, str(r.results['y']['stat']['Pr > F']))
    manova_p = float(ss[1].strip())

    ss = re.split(regexPattern, str(r.results['y']['stat']['F Value']))
    manova_F = float(ss[1].strip())

    # print(manova_F, manova_p) # use one of the four tests (their results are the same for almost all the time)

    if (manova_p == 0):  # add a very small amount to make log legal
        manova_p = sys.float_info.epsilon
    
    if (verbose):
        print(LOG)

    return manova_p, manova_F, LOG


def MWW(X, y, verbose=False, show=False, max_plot_num=5):
    """
    Performa feature-wise MWW test. Returns an array of p-values on all the features and its minimum.

    y - support 2 classes
    """

    if (len(set(y)) != 2):
        raise Exception('The dataset must have 2 classes.')

    ps = []
    Us = []
    IMG = ''

    cnt = 0

    for i in range(X.shape[1]):
        Xi = X[:, i]
        Xcis = []

        for c in set(y):
            Xc = Xi[y == c]
            Xcis.append(Xc)

        # Xcis = np.array(Xcis)

        # Special case for ValueError: All numbers are identical in mannwhitneyu
        # Don't use np.allclose(Xcis[0], Xcis[1]) as the lengths may differ
        if len(set(list(Xcis[0]) + list(Xcis[1]))) == 1:
            # return the theoretical U max: n1*n2/2
            U = len(Xcis[0]) * len(Xcis[1]) / 2
            # theoretical max. SPSS will return p = 1.0 for identical samples.
            p = 1
        else:
            U, p = scipy.stats.mannwhitneyu(Xcis[0], Xcis[1])

        ps.append(p)
        Us.append(U)

        if cnt < max_plot_num:
            plt.figure()
            plt.hist(Xcis, bins=min(12, int(len(y)/3)), alpha=0.4, edgecolor='black', label=["$ X_"+str(
                i+1)+"^{( y_"+str(0)+")} $", "$ X_"+str(i+1)+"^{( y_"+str(1)+")} $"])  # plot ith feature of different classes
            test_result = "MWW test on X{}: U={},p={}".format(
                i+1, U, round(p, 3))
            plt.title('Feature X{} histogram on different classes\n'.format(
                i+1) + test_result)
            plt.legend()
            IMG += plt2html(plt) + '<br/>'

            if show:
                plt.show()
            else:
                plt.close()

        elif cnt == max_plot_num:
            IMG += '<p>Showing the first ' + str(max_plot_num) + ' plots.</p>'

        else:
            pass  # plot no more to avoid memory cost

        cnt = cnt + 1

        if verbose:
            print(test_result)

    IMG += '<br/>'

    return ps, Us, IMG


def es_max(X, y):
    d, _ = cohen_d(X, y)
    return d.max()


def cohen_d(X, y, show=False, save_fig=''):
    '''
    Cohen’s d is a type of effect size between two means. Cohen’s d values are also known as the standardised mean difference (SMD).
    e.g., partial eta sqaured is the percentage of variance in the dependent variable (y) explained by the independent variable (x). 

    Statistical significance (p) only tells "there is a difference not due to pure chance", while the effect size is a quantitative measure of the magnitude for the difference between two means Or the degree to which $H_0$ is false. Size of difference.   
    That's why sometimes we prefer effective size over p value.

    Effect size of Cohen's d
    ------------------------
    d > .2 : small 
    d > .5 : medium
    d > .8 : large
    '''

    labels = list(set(y))

    # only support binary classifiction. For multi-class classification, use one vs rest strategy
    assert len(labels) == 2

    Xc1 = X[y == labels[0]]
    Xc2 = X[y == labels[1]]

    n1 = len(Xc1)
    n2 = len(Xc2)
    dof = n1 + n2 - 2

    # replace 0 stds with the medium value
    pooled_std = np.sqrt(((n1-1)*np.std(Xc1, axis=0, ddof=1) ** 2
                          + (n2-1)*np.std(Xc2, axis=0, ddof=1) ** 2) / dof)
    pooled_std_median = np.median(pooled_std[pooled_std > 0])
    pooled_std[pooled_std == 0] = pooled_std_median

    d = np.abs(np.mean(Xc1, axis=0) - np.mean(Xc2, axis=0)) / pooled_std

    plt.figure()

    d_sorted = np.sort(d)[::-1]  # sort in desceding order
    d_sorted_idx = np.argsort(d)[::-1]
    xlabels = []
    for i, v in enumerate(d_sorted):
        xlabels.append("X"+str(d_sorted_idx[i] + 1))
        plt.text(i-0.01, v+0.01,  str(round(v, 1)))

    plt.bar(xlabels, d_sorted, facecolor="none",
            edgecolor="black", width=0.3, hatch="\\")
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

    return d, IMG  # d is a 1xn array. n is feature num


def correlate(X, y, verbose=False, show=False):
    """
    Performa correlation tests between each feature Xi and y.

    """

    dic = {}

    rs = []
    rhos = []
    taus = []

    prs = []
    prhos = []
    ptaus = []

    LOG = ''

    for i in range(X.shape[1]):

        Xi = X[:, i]

        LOG += '\n\n#### Correlation between X{} and y ####\n'.format(i+1)

        r, p = scipy.stats.pearsonr(Xi, y)
        LOG += '\nPearson r: {}, p-value: {}'.format(round(r, 3), round(p, 3))
        rs.append(r)
        prs.append(p)

        rho, p = scipy.stats.spearmanr(Xi, y)
        LOG += '\nSpearman rho: {}, p-value: {}'.format(
            round(rho, 3), round(p, 3))
        rhos.append(rho)
        prhos.append(p)

        tau, p = scipy.stats.kendalltau(Xi, y)
        LOG += "\nKendall's tau: {}, p-value: {}".format(
            round(tau, 3), round(p, 3))
        taus.append(tau)
        ptaus.append(p)

    if verbose:
        print(LOG)

    dic['correlation.r'] = rs
    dic['correlation.r2'] = np.power(rs, 2)  # R2, the R-squared effect size
    dic['correlation.r.p'] = prs
    dic['correlation.r.max'] = np.abs(rs).max()  # abs max
    dic['correlation.r2.max'] = np.power(rs, 2).max()  # abs max
    dic['correlation.r.p.min'] = np.min(prs)

    dic['correlation.rho'] = rhos
    dic['correlation.rho.p'] = prhos
    dic['correlation.rho.max'] = np.abs(rhos).max()  # abs max
    dic['correlation.rho.p.min'] = np.min(prhos)

    dic['correlation.tau'] = taus
    dic['correlation.tau.p'] = ptaus
    dic['correlation.tau.max'] = np.abs(taus).max()  # abs max
    dic['correlation.tau.p.min'] = np.min(ptaus)

    if (show):

        for key in ['correlation.r', 'correlation.r2', 'correlation.rho', 'correlation.tau']:
            v = dic[key]
            if (X.shape[1] > 50):
                plt.figure(figsize=(20, 3))
            else:
                plt.figure()  # use default fig size
            plt.bar(list(range(len(v))), v, facecolor="none",
                    edgecolor="black", width=0.3, hatch='/')
            # plt.ylabel(key)
            plt.title(key)
            plt.show()

    return dic, LOG


def KS(X, y, show=False, max_plot_num=5):
    """
    Performa feature-wise KS test.

    y - Because it is two-sample KS test, only support 2 classes
    """

    if (len(set(y)) != 2):
        raise Exception(
            'The dataset must have two classes. If you have more than 2 classes, use OVR (one-vs-rest) strategy.')

    ps = []
    Ds = []
    IMG = ''
    cnt = 0

    for i in range(X.shape[1]):
        Xi = X[:, i]
        Xcis = []

        for c in set(y):
            Xc = Xi[y == c]
            Xcis.append(Xc)

        # Xcis = np.array(Xcis)

        D, p = scipy.stats.ks_2samp(Xcis[0], Xcis[1])

        ps.append(p)
        Ds.append(D)

        if cnt < max_plot_num:

            plt.figure()
            plt.hist(Xcis, cumulative=True, histtype=u'step', bins=min(12, int(len(y)/3)), label=["$ CDF( X_"+str(
                i+1)+"^{(y_"+str(0)+")} ) $", "$ CDF( X_"+str(i+1)+"^{(y_"+str(1)+")} ) $"])  # plot ith feature of different classes
            test_result = "KS test on X{}: D={},p={}".format(
                i+1, D, round(p, 3))
            plt.title('Feature X{} CDF on the two classes\n'.format(
                i+1) + test_result)
            plt.legend(loc='upper left')
            IMG += plt2html(plt) + '<br/>'

            if show:
                plt.show()
            else:
                plt.close()

        elif cnt == max_plot_num:
            IMG += '<p>Showing the first ' + str(max_plot_num) + ' plots.</p>'
        else:
            pass  # plot no more to avoid memory cost

        cnt = cnt+1

    IMG += "<br/>"
    return ps, Ds, IMG


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


def ECoL_metrics(X, y):
    '''
    Use rpy2 to call ECoL R package. ECoL has implemented many metrics. 
    Returns a text report and a dict
    '''

    # ECoL requires df as input
    # robjects.numpy2ri.activate()
    # rX = robjects.r.matrix(X, nrow=X.shape[0], ncol=X.shape[1])
    # robjects.r.assign("rX", rX)
    # ry = FloatVector(y.tolist())
    # robjects.globalenv['rX'] = rX
    # robjects.globalenv['ry'] = ry

    # ys = map(lambda x : 'Class ' + str(x), y)
    # M = np.hstack((X,np.array(list(ys)).reshape(-1,1)))
    M = np.hstack((X, y.reshape(-1, 1)))
    df = pd.DataFrame(M)
    # rdf = com.convert_to_r_dataframe(df)
    # rdf = pandas2ri.py2rpy_pandasdataframe(df)
    pandas2ri.activate()  # To fix NotImplementedError in Raspbian: Conversion 'rpy2py' not defined for objects of type 'rpy2.rinterface.SexpClosure'>'
    with localconverter(robjects.default_converter + pandas2ri.converter):
        rdf = robjects.conversion.py2rpy(df)
    robjects.globalenv['rdf'] = rdf

    metrics = robjects.r('''
            # install.packages("ECoL")

            # judge and install
            packages = c("ECoL", "stats")
            package.check <- lapply(
                packages,
                FUN = function(x) {
                    if (!require(x, character.only = TRUE)) {
                        install.packages(x, dependencies = TRUE)
                        library(x, character.only = TRUE)
                    }
                }
            )

            library("ECoL") # , lib.loc = "ECoL" to use the local lib
            complexity(rdf[,1:ncol(rdf)-1], rdf[,ncol(rdf)])
            ''')

    rpt = ''
    dic = {}
    for v in zip(ECoL_METRICS, metrics):
        rpt += v[0] + "\t" + str(v[1]) + "\n"
        dic[v[0]] = v[1]

    return dic, rpt


def analyze_file(fn):
    if os.path.isfile(fn) == False:
        return 'File ' + fn + ' does not exist.'

    X, y = load_file(fn)
    return get_html(X, y)


def get_metrics(X, y):
    '''
    Addionally, we can do a PCA for high-dim data to get X beforehand.   
    We assume the covariance matrix is diagnal, i.e.   

    $\Sigma = \begin{bmatrix} \sigma^2_1 & 0 \\ 0 & \sigma^2_2 \end{bmatrix}  $

    This means x1 and x2 are linearly uncorrelated.   
    If we plot two PCs from PCA，the PCs will also be linearly uncorrelated, because they are the projections on two different orthogonal eigenvectors. 
    '''

    dic, _, _ = CLF(X, y)
    if dic is None:
        dic = {}

    ber = 1  # set maximum BER
    try:
        ber, _ = BER(X, y)
    except:
        print('Exception in GaussianNB.')
    dic['classification.BER'] = ber

    svm_width, _ = SVM_Margin_Width(X, y)
    dic['classification.SVM.Margin'] = svm_width

    ig, _ = IG(X, y)
    if ig is not None:
        dic['correlation.IG'] = ig
        dic['correlation.IG.max'] = ig.max()

    dic_cor, _ = correlate(X, y)
    dic.update(dic_cor)

    es, _ = cohen_d(X, y)
    dic['test.ES'] = es
    dic['test.ES.max'] = es.max()

    p, T, _ = T_IND(X, y)
    dic['test.student'] = p
    dic['test.student.min'] = np.min(p)
    dic['test.student.min.log10'] = np.log10(np.min(p))
    dic['test.student.T'] = T
    dic['test.student.T.max'] = np.max(T)

    p, F, _ = ANOVA(X, y)
    dic['test.ANOVA'] = p
    dic['test.ANOVA.min'] = np.min(p)
    dic['test.ANOVA.min.log10'] = np.log10(np.min(p))
    dic['test.ANOVA.F'] = F
    dic['test.ANOVA.F.max'] = np.max(F)

    p, F, log = MANOVA(X, y)
    if log == 'Exception in MANOVA':
        pass
    else:
        dic['test.MANOVA'] = p
        dic['test.MANOVA.log10'] = np.log10(p)
        dic['test.MANOVA.F'] = F

    p, U, _ = MWW(X, y)
    dic['test.MWW'] = p
    dic['test.MWW.min'] = np.min(p)
    dic['test.MWW.min.log10'] = np.log10(np.min(p))
    dic['test.MWW.U'] = U
    dic['test.MWW.U.min'] = np.min(U)

    p, D, _ = KS(X, y)
    dic['test.KS'] = p
    dic['test.KS.min'] = np.min(p)
    dic['test.KS.min.log10'] = np.log10(np.min(p))
    dic['test.KS.D'] = D
    dic['test.KS.D.max'] = np.max(D)

    p, C, _ = CHISQ(X, y)
    dic['test.CHISQ'] = p
    dic['test.CHISQ.min'] = np.min(p)
    dic['test.CHISQ.min.log10'] = np.log10(np.min(p))
    dic['test.CHISQ.CHI2'] = C
    dic['test.CHISQ.CHI2.max'] = np.max(C)

    # H follows chi2, its critical value of chi2(k-1) at 0.5
    H = [scipy.stats.chi2.ppf(.5, len(set(y))-1)] * X.shape[1]
    p = [.5] * X.shape[1]
    try:
        p, H = KW(X, y)
    except Exception as e:
        print('KW Exception: ', e)
    dic['test.KW'] = p
    dic['test.KW.min'] = np.min(p)
    dic['test.KW.min.log10'] = np.log10(np.min(p))
    dic['test.KW.H'] = H
    dic['test.KW.H.max'] = np.max(H)

    # T follows chi2, its critical value of chi2(k-1) at 0.5
    T = [scipy.stats.chi2.ppf(.5, len(set(y))-1)] * X.shape[1]
    p = [.5] * X.shape[1]
    try:
        p, T, _ = MedianTest(X, y)
    except Exception as e:
        print('MedianTest Exception: ', e)
    dic['test.Median'] = p
    dic['test.Median.min'] = np.min(p)
    dic['test.Median.min.log10'] = np.log10(np.min(p))
    dic['test.Median.CHI2'] = T
    dic['test.Median.CHI2.max'] = np.max(T)

    if ENABLE_R:
        try:
            dic_ecol, _ = ECoL_metrics(X, y)
            dic.update(dic_ecol)
        except Exception as e:
            print(e)

    dic_s = {}

    for k, v in dic.items():
        if hasattr(v, "__len__"):  # this is an np array or list
            dic[k] = list(v)
        else:  # this only contains single-value metrics
            dic_s[k] = v

    return dic, dic_s


def metrics_keys():
    X, y = mvg(md=2, nobs=10)
    dic = get_metrics(X, y)
    return list(dic[0].keys())


# Defines how each metric's polarity, i.e., how to compare each metric.
#   np.nanargmax - the metric and classifiability are positively correlated
#   np.nanargmin - the metric and classifiability are negatively correlated
metric_polarity_dict = {
    'classification.ACC': np.nanargmax,
    'classification.Kappa': np.nanargmax,
    'classification.F1_Score': np.nanargmax,
    'classification.Jaccard': np.nanargmax,  # IOU between pred and truch
    'classification.Precision': np.nanargmax,
    'classification.Recall': np.nanargmax,
    # we want see no diff between pred and truth
    'classification.McNemar': np.nanargmax,
    'classification.McNemar.CHI2': np.nanargmax,
    # we want see no diff between pred and truth
    'classification.CochranQ': np.nanargmax,
    'classification.CochranQ.T': np.nanargmin,
    'classification.CrossEntropy': np.nanargmin,
    'classification.Mean_KLD': np.nanargmin,
    'classification.Brier': np.nanargmin,
    'classification.ROC_AUC': np.nanargmax,
    'classification.PR_AUC': np.nanargmax,
    'classification.BER': np.nanargmin,
    'classification.SVM.Margin': np.nanargmax,
    'correlation.IG.max': np.nanargmax,
    'correlation.r.p': np.nanargmin,
    'correlation.r.max': np.nanargmax,
    'correlation.r2.max': np.nanargmax,
    'correlation.r.p.min': np.nanargmin,
    'correlation.rho.max': np.nanargmax,
    'correlation.rho.p.min': np.nanargmin,
    'correlation.tau.max': np.nanargmax,
    'correlation.tau.p.min': np.nanargmin,
    'test.ES.max': np.nanargmax,
    'test.student': np.nanargmin,
    'test.student.min': np.nanargmin,
    'test.student.min.log10': np.nanargmin,
    'test.student.T.max': np.nanargmax,
    'test.ANOVA.min': np.nanargmin,
    'test.ANOVA.min.log10': np.nanargmin,
    'test.ANOVA.F.max': np.nanargmax,
    'test.MANOVA': np.nanargmin,
    'test.MANOVA.log10': np.nanargmin,
    'test.MANOVA.F': np.nanargmax,
    'test.MWW.min': np.nanargmin,
    'test.MWW.min.log10': np.nanargmin,
    'test.MWW.U.min': np.nanargmin,
    'test.KS.min': np.nanargmin,
    'test.KS.min.log10': np.nanargmin,
    'test.KS.D.max': np.nanargmax,
    'test.CHISQ.min': np.nanargmin,
    'test.CHISQ.min.log10': np.nanargmin,
    'test.CHISQ.CHI2.max': np.nanargmax,
    'test.KW.min': np.nanargmin,
    'test.KW.min.log10': np.nanargmin,
    'test.KW.H.max': np.nanargmax,
    'test.Median.min': np.nanargmin,
    'test.Median.min.log10': np.nanargmin,
    'test.Median.CHI2.max': np.nanargmax,
    'overlapping.F1.mean': np.nanargmax,
    'overlapping.F1.sd': np.nanargmax,
    'overlapping.F1v.mean': np.nanargmax,
    # 'overlapping.F1v.sd':np.nanargmin,
    'overlapping.F2.mean': np.nanargmin,
    # 'overlapping.F2.sd':np.nanargmin,
    'overlapping.F3.mean': np.nanargmax,
    # 'overlapping.F3.sd':np.nanargmin,
    'overlapping.F4.mean': np.nanargmax,
    # 'overlapping.F4.sd':np.nanargmin,
    'neighborhood.N1': np.nanargmin,
    'neighborhood.N2.mean': np.nanargmin,
    # 'neighborhood.N2.sd':np.nanargmin,
    'neighborhood.N3.mean': np.nanargmin,
    # 'neighborhood.N3.sd':np.nanargmax,
    'neighborhood.N4.mean': np.nanargmin,
    # 'neighborhood.N4.sd':np.nanargmin,
    'neighborhood.T1.mean': np.nanargmax,
    # 'neighborhood.T1.sd':np.nanargmin,
    'neighborhood.LSC': np.nanargmax,
    'linearity.L1.mean': np.nanargmin,
    # 'linearity.L1.sd':np.nanargmin
}


def get_json(X, y):
    return json.dumps(get_metrics(X, y))


def get_html(X, y):
    '''
    Generate a summary report in HTML format
    '''
    html = '<table class="table table-striped">'

    tr = '<tr><th> Metric/Statistic </th><tr>'  # <th> Value </th><th> Details </th>
    html += tr

    try:
        ber, ber_img = BER(X, y, show=False)

        # tr = '<tr><td> BER </td><td>' + str(ber) + '</td><td>' + ber_img + '</td><tr>'
        tr = '<tr><td> BER = ' + str(ber) + '<br/>' + ber_img + '</td><tr>'
        html += tr
    except:
        print('Exception in GaussianNB.')

    svm_margin, svm_margin_img = SVM_Margin_Width(X, y, show=False)

    tr = '<tr><td> SVM Margin Width = ' + \
        str(svm_margin) + '<br/>' + svm_margin_img + '</td><tr>'
    html += tr

    clf, clf_img, clf_log = CLF(X, y, show=False)

    # tr = '<tr><td> ACC </td><td>' + str(acc) + '</td><td>' + acc_img + '<br/><pre>' + acc_log + '</pre></td><tr>'
    tr = '<tr><td>' + str(clf) + '<br/>' + clf_img + \
        '<br/><pre>' + clf_log + '</pre></td><tr>'
    html += tr

    ig, ig_img = IG(X, y, show=False)

    tr = '<tr><td> IG = ' + str(ig) + '<br/>' + ig_img + '</td><tr>'
    html += tr

    _, corr_log = correlate(X, y, verbose=False)
    tr = '<tr><td><pre>' + corr_log + '</pre></td><tr>'
    html += tr

    t_p, _, t_img = T_IND(X, y)

    tr = '<tr><td> Independent t-test p' + \
        str(t_p) + '<br/>' + t_img + '</td><tr>'
    html += tr

    anova_p, _, anova_img = ANOVA(X, y)

    tr = '<tr><td> ANOVA p' + str(anova_p) + '<br/>' + anova_img + '</td><tr>'
    html += tr

    manova_p, _, manova_log = MANOVA(X, y)

    if manova_log == 'Exception in MANOVA':
        pass
    else:
        tr = '<tr><td> MANOVA p = ' + \
            str(manova_p) + '<br/><pre>' + manova_log + '</pre></td><tr>'
        html += tr

    mww_p, _, mww_img = MWW(X, y)

    tr = '<tr><td> MWW p = ' + str(mww_p) + '<br/>' + mww_img + '</td><tr>'
    html += tr

    ks_p, _, ks_img = KS(X, y)

    tr = '<tr><td> K-S p = ' + str(ks_p) + '<br/>' + ks_img + '</td><tr>'
    html += tr

    chi2s_p, _, chi2s_img = CHISQ(X, y)

    tr = '<tr><td> CHISQ p = ' + \
        str(chi2s_p) + '<br/>' + chi2s_img + '</td><tr>'
    html += tr

    m_p, _, m_img = MedianTest(X, y)

    tr = '<tr><td> Median test p = ' + str(m_p) + '<br/>' + m_img + '</td><tr>'
    html += tr

    kw_p, _ = KW(X, y)

    tr = '<tr><td> Kruskal-Wallis test p = ' + str(kw_p) + '</td><tr>'
    html += tr

    es, es_img = cohen_d(X, y)

    tr = '<tr><td> ES = ' + str(es) + '<br/>' + es_img + '</td><tr>'
    html += tr

    if ENABLE_R:

        try:
            _, ecol = ECoL_metrics(X, y)
            tr = '<tr><td> ECoL metrics' + '<br/><br/><pre>' + ecol + '</pre></td><tr>'
            html += tr
        except Exception as e:
            print(e)

    # dataset summary
    tr = '<tr><th> Dataset Summary </th><tr>'
    html += tr

    tr = '<tr><td>' + str(len(y)) + ' samples, ' + str(X.shape[1]) + ' features, ' + str(len(set(
        y))) + ' classes. <br/> X shape: ' + str(X.shape) + ', y shape: ' + str(y.shape) + '</td><tr>'
    html += tr

    html += "</table>"
    # html += '<style> td { text-align:center; vertical-align:middle } </style>'

    return html


def simulate(mds, repeat=1, nobs=100, dims=2):
    '''
    Try different mds (between-group distances)

    Parameters
    ----------
    mds : an array. between-classes mean distances    
    '''

    dic = {}

    # splits (divide 1 std into how many sections) * repeat
    pbar = tqdm(total=len(mds) * repeat, position=0)

    # only do visualization when repeat == 1 and draw == True
    #detailed = (repeat == 1 and draw == True)

    for md in mds:

        raw_dic = {}

        for i in range(repeat):
            X, y = mvg(
                # mu = mu, # mean, row vector
                # s = s, # std, row vector
                nobs=nobs,  # number of observations / samples
                # distance between means, respect to std, i.e. (mu2 - mu1) / std, or how many stds is the difference.
                md=md,
                dims=dims
            )

            # if detailed:
            #    print('d = ', round(md,3))

            _, dic1 = get_metrics(X, y)
            for k, v in dic1.items():
                if k in raw_dic:
                    raw_dic[k].append(v)  # dic[k] = dic[k] + v
                else:
                    raw_dic[k] = [v]  # v

            pbar.update()
            ## End of inner iteration ##

        for k, v in raw_dic.items():

            trim_size = int(repeat / 10)

            if (repeat > 10):  # remove the max and min
                raw_dic[k] = np.mean(sorted(v)[trim_size:-trim_size])
            else:
                raw_dic[k] = np.mean(v)

        ## End of outer iteration ##

        for k, v in raw_dic.items():
            if k in dic:
                dic[k] = np.append(dic[k], v)
            else:
                dic[k] = np.array([v])

    dic['d'] = np.array(mds)

    return dic


def generate_html_for_dict(dic):
    s = '<table>'
    for k in dic:
        s += '<tr>'
        s += '<td>' + str(k) + '</td>'
        for item in dic[k]:
            # item[0] is key, item[1] is value
            s += '<td>' + str(round(item, 3)) + '</td>'
        s += '</tr>'
    s += '</table>'
    return s


def visualize_dict(dic):
    '''
    Visualize the metric dictionary returned from simulate()  
    '''

    N = len(dic) - 1

    fig = plt.figure(figsize=(48, 10*(N/10+1)))
    plt.rcParams.update({'font.size': 18})

    i = 0
    for k, v in dic.items():
        if k == 'd':
            pass
        else:
            ax = fig.add_subplot(round(N/6+1), 6, i+1)
            ax.scatter(dic['d'], v, label=k)
            ax.plot(dic['d'], v)
            ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
            ax.legend()  # loc='upper center'
            ax.axis('tight')
            i += 1

    plt.axis('tight')
    plt.show()

    plt.rcParams.update({'font.size': 10})


def visualize_corr_matrix(dic, cmap='coolwarm', threshold=0.25):

    M = dic['d']
    names = ['d']

    for k, v in dic.items():
        if k == 'd' or len(v) != len(dic['d']):
            pass
        else:
            M = np.vstack((M, v))
            names.append(k)

    plt.figure(figsize=(36, 30))
    # , columns = ['d', 'BER', 'ACC', 'TOR', 'IG', 'MAN'])
    dfM = pd.DataFrame(M.T, columns=names)

    # plot the heatmap
    # sns.heatmap(np.abs(dfM.corr()), cmap="YlGnBu", annot=True)
    # should use diverging colormap, e.g., seismic, coolwarm, bwr
    sns.heatmap(dfM.corr(), cmap=cmap, annot=True)

    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.tick_params(axis='both', which='minor', labelsize=10)
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    # plt.rcParams.update({'font.size': 10})   # restore fontsize
    # https://stackoverflow.com/questions/56942670/matplotlib-seaborn-first-and-last-row-cut-in-half-of-heatmap-plot
    # bottom, top = ax.get_ylim()
    # ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.show()

    plt.figure(figsize=(20, 7))
    corrs = np.square(dfM.corr().values[0, 1:])
    corrs = np.nan_to_num(corrs)
    plt.bar(names[1:], corrs, facecolor="none", edgecolor="black",
            width=0.8, label='R2 effect size')  # ,width = 0.6, hatch='x'
    if threshold is not None:
        plt.plot(names[1:], [threshold] * len(names[1:]), '--',
                 label='R2 threshold = ' + str(threshold))
    plt.title('R2 effect size of the correlation with between-class distance')
    plt.xticks(rotation=90)
    plt.legend(loc='lower right')
    plt.show()

    if threshold is not None:
        # print(np.where( np.abs(dfM.corr().values[0,1:])>0.9 ))
        print('Metrics above the threshold (' + str(threshold) + '): ',
              np.array(names[1:])[np.where(corrs > threshold)])


def extract_PC(dic):

    M = dic['d']
    names = []

    for k, v in dic.items():
        if k == 'd' or len(v) != len(dic['d']) or np.isnan(v).any():
            pass
        else:
            M = np.vstack((M, v))
            names.append(k)

    dfM = pd.DataFrame(M[1:].T, columns=names)
    pca = PCA()
    PCs = pca.fit_transform(dfM.values)

    plt.figure(figsize=(3*len(pca.explained_variance_ratio_), 6))
    plt.bar(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_,
            facecolor="none", edgecolor="black", width=0.4, hatch='/')
    plt.title('explained variance ratio')
    plt.xticks([])
    plt.show()

    plt.figure(figsize=(20, 7))
    plt.bar(names, pca.components_[
            0], facecolor="none", edgecolor="black", hatch='/')
    plt.title('loadings / coefficients')
    plt.xticks(rotation=90)
    plt.show()

    print("1st PC: ", PCs[:, 0])

    return pca
