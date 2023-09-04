'''
Extend the idea in the paper:  
A unified classifiability analysis framework based on meta-learner and its application in spectroscopic profiling data [J]. Applied Intelligence, 2021, doi: 10.1007/s10489-021-02810-8
'''

import os
import math
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import IPython.core.display
import numpy as np
import pandas as pd
import scipy
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
import scipy.optimize
import joblib

from .vis.plotComponents2D import plotComponents2D
from .metrics import get_metrics, visualize_atom_dict, visualize_corr_matrix, generate_html_for_dict

def analyze(X, y, mds=np.linspace(0, 4, 9), cutoff=2, filter_threshold=.5, methods=['meta.logistic', 'decompose.pca'], pkl=None, verbose=False):
    '''
    An include-all function that trains a meta-learner model of unified single metric.
    And use that metric to evaluate the between-class and in-class classifiability.

    Parameter
    ---------
    mds : a list of float, defines a range from not classifiable to most classifiable. 
    cutoff : a float, defines the cutoff of classifiability.
    use_filter : whether to use R2 to filter highly correlated atom metrics
    methods : a list of methods to use:
        'meta.linear' - use linear regression as a meta-learner
        'meta.logistic' - use logistic regression as a meta-learner
        'decompose.pca' - decomposition using PCA
        'decompose.lda' - decomposition using LDA
    pkl : a pickle file of pre-computed atom metrics to load. If None, will generate on site.

    Return
    ------
    umetric_bw : between-class unified metric
    umetric_in : in-class unified metric
    pkl_file : pickle filepath for persisting the atom metric dict
    '''

    if pkl and os.path.isfile(pkl):
        dic = joblib.load(pkl) # load an existing pkl file
        pkl_file = pkl
        print('Load atom metrics from', pkl_file)
    else:
        dic = calculate_atom_metrics(mu = X.mean(axis = 0), s = X.std(axis = 0),
                            mds = mds, # defines a range from not classifiable to most classifiable.
                            repeat = 3, nobs = len(X),
                            show_html = True, verbose = verbose)
        pkl_file = str(datetime.now()).replace(':','').replace('-','').replace(' ','') + '.pkl'
        joblib.dump(dic, pkl_file) # later we can reload with: dic = joblib.load('x.pkl')
        print('Save atom metrics to', pkl_file)

    visualize_atom_dict(dic)
    _, keys, _, M = filter_metrics(dic, filter_threshold)

    # as different metrics may have quite different value ranges, perform a standardization. 
    scaler = StandardScaler()
    M = scaler.fit_transform(M)

    dic_result = {}

    for method in methods:

        print()
        print('----- Atom metric synthesis method:', method, '-----')

        if method == 'decompose.pca':
            # model, x_min, x_max, slope = train_decomposer_pca(M, dic['d'])
            model, A, B, C = train_decomposer_pca(M, dic['d'])
            umetric_bw, umetric_in = calculate_unified_metric(X, y, model, keys, method, scaler, verbose=verbose)

            if True:

                # maps to the [0,1] range
                print('PC1 before mapping: ', umetric_bw, umetric_in)

            '''
            np.interp(x, xp, fp, left=None, right=None)

            left : optional float or complex corresponding to fp
                Value to return for `x < xp[0]`, default is `fp[0]`.

            right : optional float or complex corresponding to fp
                Value to return for `x > xp[-1]`, default is `fp[-1]`.
            
            Because the default left and right params, we dont need to do extra out-of-range (>max or <min) treatments.
            '''
            umetric_bw = f_sigmoid(umetric_bw, A, B, C) # np.interp(umetric_bw,[x_min,x_max],[0,1] if slope else [1,0])
            umetric_in = f_sigmoid(umetric_in, A, B, C) # np.interp(umetric_in,[x_min,x_max],[0,1] if slope else [1,0])
            
            if verbose:
                print('After mapping: ', umetric_bw, umetric_in)

        elif method == 'meta.logistic':
            model = train_metalearner_logistic(M, dic['d'], cutoff=cutoff, verbose=verbose)
            umetric_bw, umetric_in = calculate_unified_metric(X, y, model, keys, method, scaler, verbose=verbose)
        
        elif method == 'decompose.lda':
            model, x_min, x_max, slope = train_decomposer_lda(M, dic['d'], cutoff=cutoff)
            umetric_bw, umetric_in = calculate_unified_metric(X, y, model, keys, method, scaler, verbose=verbose)

            # maps to the [0,1] range
            print('before scaling: ', umetric_bw, umetric_in)
            print('C1 range: ', x_min, x_max)

            umetric_bw = np.interp(umetric_bw, [x_min, x_max], [0, 1] if slope else [1,0])
            umetric_in = np.interp(umetric_in, [x_min, x_max], [0, 1] if slope else [1,0])
            print('after scaling: ', umetric_bw, umetric_in)

        elif method == 'meta.linear':
            model = train_metalearner_linear(M, dic['d'], verbose = verbose)
            umetric_bw, umetric_in = calculate_unified_metric(X, y, model, keys, method, scaler, verbose=verbose)

        else:
            raise Exception('Unsupported method ' + method )
        
        dic_result[method] = {'umetric_bw': umetric_bw, 'umetric_in': umetric_in}

        print('----- End of', method, '-----')
        print()

    print('between-class and in-class scores:')
    print(dic_result)
    print()

    return dic_result, pkl_file

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
    cov = np.diag(s**2)

    mu1 = mu - s * md / 2
    mu2 = mu + s * md / 2

    xc1 = np.random.multivariate_normal(mu1, cov, N)
    xc2 = np.random.multivariate_normal(mu2, cov, N)

    y = np.concatenate((np.zeros(N), np.ones(N))).astype(int)
    X = np.vstack((xc1,xc2))

    return X, y

def calculate_atom_metrics(mu, s, mds, repeat = 3, nobs = 100,
                           show_html = True, verbose = False):
    '''
    Calculate atom metric values for different mds (between-group distances)

    Parameters
    ----------
    mu : mean vector
    s : std vector
    mds : an array. between-classes mean distances (in std). e.g., np.linspace(0,6,10)
    show_html : whether output an inline HTML table of metrics

    Example
    -------
    dic = calculate_atom_metrics(mu = X.mean(axis = 0), s = X.std(axis = 0),
                                 mds = np.linspace(0, 1.6, 9),
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

            _, raw_dic1 = get_metrics(X,y,verbose=verbose)
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

    if show_html:
        print('generate_html_for_dict()')
        s = generate_html_for_dict(dic)
        IPython.core.display.display(IPython.core.display.HTML(s))

    return dic

def filter_metrics(dic, threshold = 0.25, display = True):
    '''
    Calculate the linear regression R2 effect size of each metric with d.
    We use the R2 to filter metrics.

    Parameter
    ---------
    threshold : R2 threshold.
        If None or 0, will keep all the original atom metrics

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
        if threshold is not None and threshold>0:
            print('before filter')
        try:
            visualize_corr_matrix(dic, cmap = 'coolwarm', threshold = threshold)
        except Exception as e:
            print('Exception in visualize_corr_matrix()', e)

    for k, x in dic.items():
        if k == 'd' or len(np.unique(x))==1:
            pass
        else:
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x,y)
            dic_r2[k] = r_value**2
            if threshold is None or dic_r2[k] >= threshold:
                keys.append(k)
                filtered_dic[k] = x
                M.append(x)

    #if display:
        # print('after filter')
        # visualize_corr_matrix(filtered_dic, cmap = 'coolwarm', threshold = 0.5)

    return dic_r2, keys, filtered_dic, np.array(M).T

def train_decomposer_lda(M, d, cutoff = 2):
    d = np.array(d) >= cutoff
    df = pd.DataFrame(M)

    df.fillna(0, inplace=True)
    df.replace(np.inf, 1, inplace=True)
    df.replace(-np.inf, -1, inplace=True)

    lda = LDA(n_components=1)
    X_lda = lda.fit_transform(df, d)
    C1_min = min(X_lda.T[0])
    C1_max = max(X_lda.T[0])

    plt.scatter(d, X_lda.T[0])
    plt.title('C1 ~ d')
    plt.show()

    slope = X_lda.T[0][-1] > X_lda.T[0][0]  # 1 # /-1

    print('Explained Variance Ratios for the first component', lda.explained_variance_ratio_[0])
    return lda, C1_min, C1_max, slope

# 需要拟合的函数, d = f(PC1)
def f_log(x, A, B, C):
    return A * np.exp(x+B) + C
def f_sigmoid(x, A, B, C):
    return A / (1 + np.exp(- np.array(x)*B + C))

def train_decomposer_pca(M, d):
    '''
    Train a PCA decomposer using the atom metric matrix.

    Return
    ------
    decomposer : the decomposition model. default is PCA
    PC1_min, PC1_max : the reference range of first PC.
    slope : Boolean. whether the PC1 is positively or negatively related to between-class distance.
    '''

    df = pd.DataFrame(M)

    df.fillna(0, inplace=True)
    df.replace(np.inf, 1, inplace=True)
    df.replace(-np.inf, -1, inplace=True)
    # df = MinMaxScaler().fit_transform(df)

    decomposer = PCA(n_components=3)
    X_pca = decomposer.fit_transform(df)
    PC1_min = min(X_pca.T[0])
    PC1_max = max(X_pca.T[0])

    # 得到返回的A，B值
    A, B, C = scipy.optimize.curve_fit(f_sigmoid, X_pca.T[0], d)[0]

    plt.scatter(d, X_pca.T[0])
    plt.title('PC1 ~ d')
    plt.xlabel('d')
    plt.ylabel('PC1(' + str(round(decomposer.explained_variance_ratio_[0], 3)) + ')')
    # plt.xticks([])
    # plt.yticks([])
    plt.show()

    slope = X_pca.T[0][-1] > X_pca.T[0][0] # 1 # /-1

    print('Explained Variance Ratios for the first three PCs', decomposer.explained_variance_ratio_[:3])
    return decomposer, A, B, C #PC1_min, PC1_max, slope


def train_metalearner_linear(M, d, verbose = False):
    d = np.array(d) # np.median(d)
    dic2 = pd.DataFrame(M)
    dic2.fillna(0, inplace=True)
    dic2.replace(np.inf, 1, inplace=True)
    dic2.replace(-np.inf, -1, inplace=True)
    lr = LinearRegression().fit(dic2, d)
    # print('Score: ', lr.score(M, d))

    if verbose:
        print('Coef and Intercept: ', lr.coef_, lr.intercept_)

    return lr

def train_metalearner_logistic(M, d, cutoff = 2, verbose = False):
    '''
    Train meta-learner using the atom metric matrix and the distance array.

    Parameter
    ---------
    cutoff : use this cutoff to convert d to binary values,
    as we use logistic regression as the meta-learner output.
    We deem it as fairly classifiable if d > cutoff. 

    Note
    ----
    In the original publication, we use a linear regressor as the meta-learner.
    In the new version, we use logistic regression, to cover the between-class distance range (0-6std).
    For d = 6 std means almost no overlap, the meta-learner will return 1
    '''

    d = np.array(d) >= cutoff # np.median(d)

    clf = LogisticRegressionCV(max_iter=2000, solver='liblinear').fit(M, d)

    if verbose:
        # print('Score: ', clf.score(M, d))
        print('Coef and Intercept: ', clf.coef_, clf.intercept_)

    return clf

def calculate_unified_metric(X, y, model, keys, method, scaler, verbose = False):
    '''
    First, fit a linear regression (meta-learner) model for M on d.
    Then, calcualte the between-class and in-class unified metric on real dataset X and y.

    Parameters
    ----------
    model : meta-learner model, returned by train_metalearner()
    keys : selected metric names, returned by filter_metrics()
    '''
    return AnalyzeBetweenClass(X ,y, model, keys, method, scaler, verbose=verbose), AnalyzeInClass(X, y, model, keys,method, scaler, verbose=verbose)

def AnalyzeBetweenClass(X, y, model, keys, method, scaler, verbose = False):

    if verbose:

        pca = PCA(n_components = 2)
        X_pca = pca.fit_transform(X)
        plotComponents2D(X_pca, y)
        plt.title('Scatter plot of entire dataset after PCA DR')
        plt.xlabel('PC1(' + str(round(pca.explained_variance_ratio_[0], 3)) + ')')
        plt.ylabel('PC2(' + str(round(pca.explained_variance_ratio_[1], 3)) + ')')
        plt.show()

    _, new_dic = get_metrics(X, y, verbose=verbose)
    vec_metrics = []

    for key in keys:
        vec_metrics.append(new_dic[key] if key in new_dic else 0)

    vec_metrics = np.nan_to_num(vec_metrics,nan=0,posinf=1000,neginf=-1000)

    M = vec_metrics.reshape(1,-1)
    M = scaler.transform(M)
    
    if method == 'meta.logistic' and (isinstance(model, LogisticRegression) or isinstance(model, LogisticRegressionCV)):
        umetric = model.predict_proba(M)[0][1]
    elif method == 'decompose.pca' and isinstance(model, PCA):        
        umetric = model.transform(M)[0][0]
    elif method == 'decompose.lda' and isinstance(model, LDA):
        umetric = model.transform(M)[0][0]
    elif method == 'meta.linear' and isinstance(model, LinearRegression):
        umetric = model.predict(M)
    else:
        raise Exception('Unsupported method ' + method )
    
    if verbose:
        print("between-class unified metric = ", umetric)

    return umetric

def AnalyzeInClass(X, y, model, keys, method, scaler, repeat = 3, verbose = False):
    '''
    Parameters
    ----------
    repeat : to get in-class metrics, samples of each class are randomly assigned different labels.
        This controls how many times to run to get the averaged result.
    '''

    umetrics = []
    for c in set(y):
        Xc = X[y == c]
        d = 0

        for i in range(repeat):
            yc = (np.random.rand(len(Xc)) > 0.5).astype(int) # random assign y labels
            _, new_dic = get_metrics(Xc, yc, verbose=verbose)
            vec_metrics = []
            for key in keys:
                vec_metrics.append(new_dic[key] if key in new_dic else 0)

            vec_metrics = np.nan_to_num(vec_metrics,nan=0,posinf=1000,neginf=-1000)
            
            M = vec_metrics.reshape(1,-1)
            M = scaler.transform(M)

            if method == 'meta.logistic' and isinstance(model, LogisticRegression):
                d += model.predict_proba(M)[0][1]
            elif method == 'decompose.pca' and isinstance(model, PCA):
                d += model.transform(M)[0][0]
            elif method == 'decompose.lda' and isinstance(model, LDA):
                d += model.transform(M)[0][0]
            elif method == 'meta.linear' and isinstance(model, LinearRegression):
                d += model.predict(M)
            else:
                raise Exception('Unsupported method ' + method )

        if verbose:
            print("c = ", int(c), ", in-class unified metric = ", d/repeat)

            pca = PCA(n_components = 2)
            X_pca = pca.fit_transform(Xc)
            plotComponents2D(X_pca, y[y == c]) #, tags=range(len(X_pca)))

            plt.title('Scatter plot of Class ' + str(int(c)) + ' after PCA DR')
            plt.xlabel('PC1(' + str(round(pca.explained_variance_ratio_[0], 3)) + ')')
            plt.ylabel('PC2(' + str(round(pca.explained_variance_ratio_[1], 3)) + ')')
            plt.show()

        umetrics.append(d/repeat)

    return umetrics