'''
Extend the idea in the paper:  
A unified classifiability analysis framework based on meta-learner and its application in spectroscopic profiling data [J]. Applied Intelligence, 2021, doi: 10.1007/s10489-021-02810-8
'''

from .vis.plt2base64 import *
from .vis.plotComponents2D import *
from .vis.plotComponents1D import *
from .vis.feature_importance import *
from .vis.unsupervised_dimension_reductions import *
from .metrics import get_metrics, visualize_dict, visualize_corr_matrix, generate_html_for_dict

from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from IPython.core.display import display, HTML
import numpy as np
import pandas as pd
import scipy
from sklearn.linear_model import LinearRegression, LogisticRegression
import pickle
import joblib
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

def analyze(X,y,use_filter=True,method='meta'):
    '''
    An include-all function that trains a meta-learner model of unified single metric.
    And use that metric to evaluate the between-class and in-class classifiability.

    Parameter
    ---------
    use_filter : whether to use R2 to filter highly correlated atom metrics
    method : which method to use.
        'meta' - use linear regression as a meta-learner
        'decompose' - decomposition, e.g., PCA

    Return
    ------
    umetric_bw : between-class unified metric
    umetric_in : in-class unified metric
    pkl_file : pickle filepath for persisting the atom metric dict
    '''
    dic = calculate_atom_metrics(mu = X.mean(axis = 0), s = X.std(axis = 0),
                             mds = np.linspace(0, 6, 7+6*2),
                             repeat = 5, nobs = 100,
                             show_curve = True, show_html = True)
    pkl_file = str(datetime.now()).replace(':','').replace('-','').replace(' ','') + '.pkl'
    joblib.dump(dic, pkl_file) # later we can reload with: dic = joblib.load('x.pkl')

    _, keys, _, M = filter_metrics(dic, threshold = (0.5 if use_filter else None))
    if method == 'decompose':
        model, x_min, x_max = train_decomposer(M)
        umetric_in = []
        umetric_bw, umetric_in_ = calculate_unified_metric(X, y, model, keys, method)
        if umetric_bw > x_max:
            umetric_bw = x_max
        elif umetric_bw < x_min:
            umetric_bw = x_min
        if umetric_in_[0] > x_max:
            umetric_in_[0] = x_max
        elif umetric_in_[0] < x_min:
            umetric_in_[0] = x_min
        if umetric_in_[1] > x_max:
            umetric_in_[1] = 0.9 * x_max
        elif umetric_in_[1] < x_min:
            umetric_in_[1] = x_min
        umetric_bw = abs((umetric_bw-x_max)/(x_min-x_max))
        for i in umetric_in_:
            z = (i-x_max)/(x_min-x_max)
            z = abs(z)
            umetric_in.append(z)

    elif method == 'meta':
        model = train_metalearner(M, dic['d'])
        umetric_bw, umetric_in = calculate_unified_metric(X, y, model, keys, method)

    else:
        raise Exception('Unsupported method, must be meta or decompose')

    return umetric_bw, umetric_in, pkl_file
    # return M,keys

def mvgx(
    mu, # mean, row vector
    s, # std, row vector
    md = 2, # distance between means, respect to std, i.e. (mu2 - mu1) / std, or how many stds is the difference.
    nobs = 15 # number of observations / samples
    ):
    '''
    Generate simulated high-dim (e.g., spectroscopic profiling) data
    This is an extended version of mvg() that accepts specified mu and s vectors.

    Example
    -------
    X,y = mvgx(
    [-1,0,-1], # mean, row vector
    [1,2,0.5], # variance, row vector
    nobs = 5, # number of observations / samples
    md = 1 # distance between means, respect to std, i.e. (mu2 - mu1) / std, or how many stds is the difference.
    )
    '''

    mu = np.array(mu)
    s = np.array(s)

    N = nobs
    dims = len(mu)
    cov = np.diag(s**2)

    mu1 = mu - s * md / 2
    mu2 = mu + s * md / 2

    xc1 = np.random.multivariate_normal(mu1, cov, N)
    xc2 = np.random.multivariate_normal(mu2, cov, N)

    y = np.concatenate((np.zeros(N), np.ones(N))).astype(int)
    X = np.vstack((xc1,xc2))

    return X, y

def calculate_atom_metrics(mu, s, mds,
repeat = 3, nobs = 100,
show_curve = True, show_html = True):
    '''
    Calculate atom metric values for different mds (between-group distances)

    Parameters
    ----------
    mu : mean vector
    s : std vector
    mds : an array. between-classes mean distances (in std). e.g., np.linspace(0,1,10)
    show_curve : whether output each metric curve against the between-class distance
    show_html : whether output an inline HTML table of metrics

    Example
    -------
    dic = calculate_atom_metrics(mu = X.mean(axis = 0), s = X.std(axis = 0),
                                 mds = np.linspace(0, 1, 5),
                                 repeat = 1, nobs = 40)
    '''

    dic = {}

    ## splits (divide 1 std into how many sections) * repeat
    pbar = tqdm(total = len(mds) * repeat, position=0)

    ## only do visualization when repeat == 1 and draw == True
    #detailed = (repeat == 1 and draw == True)

    dic['d'] = np.array(mds)

    for md in mds:

        raw_dic = {}

        for i in range(repeat):
            X,y = mvgx(
                mu = mu, # mean, row vector
                s = s, # std, row vector
                nobs = nobs, # number of observations / samples
                md = md
            )

            ## if detailed:
            #    print('d = ', round(md,3))

            _, raw_dic1 = get_metrics(X,y)
            for k, v in raw_dic1.items():
                if k in raw_dic:
                    raw_dic[k].append(v) # raw_dic[k] = raw_dic[k] + v
                else:
                    raw_dic[k] = [v] # v

            pbar.update()
            ## End of inner iteration ##

        for k, v in raw_dic.items():

            trim_size = int(repeat / 10)

            if (repeat > 10): # remove the max and min
                raw_dic[k] = np.mean(sorted(v)[trim_size:-trim_size])
            else:
                raw_dic[k] = np.mean(v)

        ## End of outer iteration ##

        for k, v in raw_dic.items():
            if k in dic:
                dic[k] = np.append(dic[k], v)
            else:
                dic[k] = np.array([v])

    if show_curve:
        print('visualize_dict()')
        visualize_dict(dic)

    if show_html:
        print('generate_html_for_dict()')
        s = generate_html_for_dict(dic)
        display(HTML(s))

    return dic

def filter_metrics(dic, threshold = 0.25, display = True):
    '''
    Calculate the linear regression R2 effect size of each metric with d.
    We use the R2 to filter metrics.

    Parameter
    ---------
    threshold : R2 threshold.
        If None, will keep all the original atom metrics

    Return
    ------
    dic_r2 : R2 values
    selected_keys : selected keys
    filtered_dic : dic after filtering (R2 > threshold)
    matrix : an len(md)xlen(selected_keys) matrix of selected metrics

    Note
    ----
    For R2 effect size:
    Small: 0.01-0.09
    Medium: 0.09-0.25
    Large: 0.25 and higher
    '''
    y = dic['d']

    dic_r2 = {}
    keys = []
    filtered_dic = {}
    M = []

    if display:
        if threshold is not None:
            print('before filter')
        visualize_corr_matrix(dic, cmap = 'coolwarm', threshold = threshold)

    for k, x in dic.items():
        if k == 'd':
            pass
        else:
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x,y)
            dic_r2[k] = r_value**2
            if threshold is None or dic_r2[k] > threshold:
                keys.append(k)
                filtered_dic[k] = x
                M.append(x)

    #if display:
        # print('after filter')
        # visualize_corr_matrix(filtered_dic, cmap = 'coolwarm', threshold = 0.5)

    return dic_r2, keys, filtered_dic, np.array(M).T

def train_decomposer(M):
    umetric_in = []
    dic2 = pd.DataFrame(M)
    dic2.fillna(0, inplace=True)
    dic2.replace(np.inf, 1, inplace=True)
    dic2.replace(-np.inf, -1, inplace=True)
    # dic2 = MinMaxScaler().fit_transform(dic2)

    pca = PCA(n_components=3)
    pca_model = pca.fit(dic2)
    data1 = pca_model.transform(dic2)
    x_min = min(data1.T[0])
    x_max = max(data1.T[0])
    # print(x_min, x_max)

    a = pca.explained_variance_ratio_
    PC1 = a[0]
    PC2 = a[1]
    PC3 = a[2]

    print('The information interpretation rate of the first principal component is:', PC1)
    return pca_model,x_min,x_max


def train_metalearner(M, d, cutoff = 2):
    '''
    Train meta-learner using the atom metric matrix and the distance array.

    Parameter
    ---------
    cutoff : use this cutoff to convert d to binary values,
    as we use logistic regression as the meta-learner output.
    default is 2, i.e., if md > 2, we deem it as fairly classifiable.
    If you want more resolution in the lower range, choose a smaller cutoff.

    Note
    ----
    In the original publication, we use a linear regressor as the meta-learner.
    In thee new version, we use logistic regression, to cover the between-class distance range (0-6std).
    For d = 6std means almost no overlap, the meta-learner will return 1
    '''

    d = np.array(d) >= cutoff # np.median(d)

    clf = LogisticRegression(max_iter=1000, solver='liblinear').fit(M, d)
    print('Score: ', clf.score(M, d))
    print('Coef and Intercept: ', clf.coef_, clf.intercept_)
    return clf

def calculate_unified_metric(X, y, model, keys,method):
    '''
    First, fit a linear regression (meta-learner) model for M on d.
    Then, calcualte the between-class and in-class unified metric on real dataset X and y.

    Parameters
    ----------
    model : meta-learner model, returned by train_metalearner()
    keys : selected metric names, returned by filter_metrics()
    '''
    return AnalyzeBetweenClass(X ,y, model, keys,method), AnalyzeInClass(X, y, model, keys,method)

def AnalyzeBetweenClass(X, y, model, keys, method='decompose'):

    X_pca = PCA(n_components = 2).fit_transform(X)
    plotComponents2D(X_pca, y)

    _, new_dic = get_metrics(X, y)
    vec_metrics = []

    for key in keys:
        vec_metrics.append(new_dic[key])

    vec_metrics = np.nan_to_num(vec_metrics)
    if method == 'meta' and isinstance(model, LogisticRegression):
        umetric = model.predict_proba([vec_metrics.T])[0][1]
    elif method == 'decompose' and isinstance(model, PCA):
        x=vec_metrics.reshape(1,-1)
        # x = MinMaxScaler().fit_transform(x)
        umetric = model.transform(x)[0][0]
    else:
        raise Exception('Unsupported method, must be meta or decompose')
    # print("between-class unified metric = ", umetric[1])
    return umetric

def AnalyzeInClass(X, y, model, keys, method='decompose'):

    umetrics = []
    repeat=10
    for c in set(y):
        Xc = X[y == c]
        d = 0

        for i in range(repeat):
            yc = (np.random.rand(len(Xc)) > 0.5).astype(int) # random assign y labels
            _, new_dic = get_metrics(Xc, yc)
            vec_metrics = []
            for key in keys:
                vec_metrics.append(new_dic[key])

            vec_metrics = np.nan_to_num(vec_metrics)

            if method == 'meta' and isinstance(model, LogisticRegression):
                d += model.predict_proba([vec_metrics.T])[0][1]
            elif method == 'decompose' and isinstance(model, PCA):
                x=vec_metrics.reshape(1,-1)
                d += model.transform(x)[0][0]
            else:
                raise Exception('Unsupported method, must be meta or decompose')

        print("c = ", int(c), ", in-class unified metric = ", d/repeat)
        X_pca = PCA(n_components = 2).fit_transform(Xc)
        plotComponents2D(X_pca, y[y == c]) #, tags=range(len(X_pca)))
        umetrics.append(d/repeat)

    return umetrics