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

def calculate_atom_metrics(mu, s, mds, repeat = 1, nobs = 100, 
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

    dic_r2 = []
    keys = []
    filtered_dic = []
    M = []

    for k, x in dic.items():
        if k == 'd':
            pass
        else:
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x,y)
            dic_r2[k] = r_value**2
            if dic_r2[k] > threshold:
                keys.append(k)
                filtered_dic[k] = x
                M.append(x)

    if display:
        print('before filter')
        visualize_corr_matrix(dic, cmap = 'coolwarm', threshold = threshold)
        print('after filter')
        visualize_corr_matrix(filtered_dic, cmap = 'coolwarm', threshold = 0.5)

    return dic_r2, keys, filtered_dic, np.array(M).T