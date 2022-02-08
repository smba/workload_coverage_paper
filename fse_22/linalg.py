#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression, ElasticNet, Lasso
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures

from pycosa.util import remove_multicollinearity, get_vif
from pycosa.plotting import mirrored_histogram
import scipy.cluster.hierarchy as spc
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
import networkx as nx
import logging
import math
import itertools
import numpy as np
from scipy.stats import ks_2samp, pearsonr, kendalltau, wilcoxon
import matplotlib.cm as cm
from cliffs_delta import cliffs_delta

from matplotlib import gridspec

import sys

data = pd.read_csv('datax/h2_measurements.csv')
da = data[data['workload'] == 'tpcc-2']['throughput']
db = data[data['workload'] == 'tpcc-8']['throughput']

#mirrored_histogram(db, da, 'scale factor 8', 'scale factor 2', bandwith=0.02, figsize=(6, 2), export_name='h2_motivation.eps', xlabel='Transactions / Sec.')
#sys.exit()

def outlier_treatment(datacolumn):
    sorted(datacolumn)
    Q1,Q3 = np.percentile(datacolumn , [25,75])
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)
    return lower_range,upper_range

systems = {
    'h2': ['throughput'],
    'kanzi': ['time', 'ratio', 'max-resident-size'],
    'batik': ['time', 'max-resident-size'],
    'jump3r': ['time', 'max-resident-size'],
    'jadx': ['time', 'max-resident-size'],
    'dconvert': ['time', 'max-resident-size'],    
}


# CONSTANTS
STDEV_THRESHOLD = 0.05

for system in systems:
    data = pd.read_csv('datax/{}_measurements.csv'.format(system))

    for kpi in systems[system]:
        coefficients = []
        swarm_coefficients = []
        workloads = []
        for w, df in data.groupby('workload'):
            workloads.append(w)
            X = df.drop(columns=['config', 'workload'] + systems[system])
            X = remove_multicollinearity(X)
            
            for xcol in X.columns:
              X[xcol] = MinMaxScaler().fit_transform(X[xcol].values.reshape(-1, 1))
            
            ss = StandardScaler()
            y = ss.fit_transform(df[kpi].values.reshape(-1, 1))

            lm = LinearRegression()
            lm.fit(X, y)
            #print(lm.coef_)
            coefs = {c: lm.coef_[0][i] for i, c in enumerate(X.columns)}
            coefficients.append(coefs)
            
            for option, coefficient in coefs.items():
                swarm_coefficients.append(
                    {
                        'workload': w,
                        'option': option,
                        'influence': coefficient
                    }    
                )
            
        # get worklaod with GREATEST absolute influence
        options = coefficients[0].keys()
        ginis = {}
        covs = {}
        means = {}
        nr_outlier = {}
        for opt in options:
            influences = []
            for coefs in coefficients:
                influences.append(coefs[opt])
            means[opt] = np.median(influences)
            covs[opt] = np.std(influences) 

                
        pivot = dict(sorted(means.items(), key=lambda item: item[1], reverse=True))

        option_order = reversed(pivot.keys())
        option_order = [o for o in option_order]
        swarm_coefficients = pd.DataFrame(swarm_coefficients)
        
        if kpi in ['time', 'throughput']:
            plt.figure(figsize=(8,4.5))
            plt.grid(axis='y')
            sns.stripplot(y="option", x="influence", data=swarm_coefficients, size=5,  order=option_order, marker='d', color='0.0')
            #sns.boxplot(y="option", x="influence", data=swarm_coefficients, order=option_order, color='green', linewidth=0.8, boxprops={'alpha':0.7})
            plt.axvline(0, color='black', linestyle='-', linewidth=0.5)
            for i, o in enumerate(option_order):
                plt.axhline(o, color='black', linewidth=0.4, alpha=0.25)
                mi = swarm_coefficients[swarm_coefficients['option'] == o]['influence'].min()
                ma = swarm_coefficients[swarm_coefficients['option'] == o]['influence'].max()
                #print(o, mi, ma)
                plt.plot([mi, ma], [o, o], color='black',linewidth=1.4)
                if mi < 0:
                    plt.barh([o], [mi], color='green', alpha=0.3)
                if ma > 0:
                    plt.barh([o], [ma], color='red', alpha=0.3)
            plt.title('{} ({}; {} workloads)'.format(system, kpi, len(workloads)),fontweight='bold')
            plt.xlabel('Relative Performance Influence')
            plt.ylabel('')
            plt.draw()
            plt.savefig('{}_rq2.pdf'.format(system), bbox_inches='tight')
            #plt.show()
'''
# build elastic model
for system in systems:
    data = pd.read_csv('datax/{}_measurements.csv'.format(system))

    for kpi in systems[system]:
        if kpi not in ['time', 'throughput']:
            continue
        coefficients = []
        swarm_coefficients = []
        workloads = []
        for w, df in data.groupby('workload'):
            workloads.append(w)
            X = df.drop(columns=['config', 'workload'] + systems[system])
            X = remove_multicollinearity(X)
            
            for xcol in X.columns:
              X[xcol] = MinMaxScaler().fit_transform(X[xcol].values.reshape(-1, 1))
            
            poly = PolynomialFeatures(degree=2, interaction_only = True)
            X_ = poly.fit_transform(X)
            X = pd.DataFrame(X_, columns = poly.get_feature_names(X.columns))
            X = remove_multicollinearity(X)
            
            print('here')
            
            ss = StandardScaler()
            y = ss.fit_transform(df[kpi].values.reshape(-1, 1))

            lm = Lasso()
            lm.fit(X, y)
            #print(lm.coef_)
            coefs = {c: lm.coef_[i] for i, c in enumerate(X.columns)}
            coefficients.append(coefs)
            
            for option, coefficient in coefs.items():
                swarm_coefficients.append(
                    {
                        'workload': w,
                        'option': option,
                        'influence': coefficient
                    }    
                )
            
        # get worklaod with GREATEST absolute influence
        options = coefficients[0].keys()
        ginis = {}
        covs = {}
        means = {}
        nr_outlier = {}
        for opt in options:
            influences = []
            for coefs in coefficients:
                influences.append(coefs[opt])
            means[opt] = np.median(influences)
            covs[opt] = np.std(influences) 

                
        pivot = dict(sorted(means.items(), key=lambda item: item[1], reverse=True))

        option_order = reversed(pivot.keys())
        option_order = [o for o in option_order]
        swarm_coefficients = pd.DataFrame(swarm_coefficients)
        
        for option in option_order:
            minimum = swarm_coefficients[swarm_coefficients['option'] == option]['influence'].min()
            maximum = swarm_coefficients[swarm_coefficients['option'] == option]['influence'].max()
            
            if abs(minimum) < 0.1 or abs(maximum) < 0.1:
                swarm_coefficients = swarm_coefficients[swarm_coefficients['option'] != option]
        
        print(swarm_coefficients)
        remaining = swarm_coefficients.option.unique()
        option_order = list(filter(lambda o: o in remaining, option_order))
        #plt.hist(swarm_coefficients['influence'], bins=100)
        
        if kpi in ['time', 'throughput']:
            #plt.figure(figsize=(8,4.5))
            plt.grid(axis='y')
            sns.stripplot(y="option", x="influence", data=swarm_coefficients, size=5,  order=option_order, marker='d', color='0.0')
            #sns.boxplot(y="option", x="influence", data=swarm_coefficients, order=option_order, color='green', linewidth=0.8, boxprops={'alpha':0.7})
            plt.axvline(0, color='black', linestyle='-', linewidth=0.5)
            for i, o in enumerate(option_order):
                plt.axhline(o, color='black', linewidth=0.4, alpha=0.25)
                mi = swarm_coefficients[swarm_coefficients['option'] == o]['influence'].min()
                ma = swarm_coefficients[swarm_coefficients['option'] == o]['influence'].max()

                plt.plot([mi, ma], [o, o], color='black',linewidth=1.4)
                if mi < 0:
                    plt.barh([o], [mi], color='green', alpha=0.3)
                if ma > 0:
                    plt.barh([o], [ma], color='red', alpha=0.3)
            plt.title('{} ({}; {} workloads)'.format(system, kpi, len(workloads)),fontweight='bold')
            plt.xlabel('Relative Performance Influence')
            plt.ylabel('')
            #plt.draw()
            #plt.savefig('{}_rq2.pdf'.format(system), bbox_inches='tight')
            plt.show()
'''

def a12(lst1,lst2,rev=True):
    "how often is x in lst1 more than y in lst2?"
    more = same = 0.0
    for x in lst1:
        for y in lst2:
            if   x==y : same += 1
            elif rev     and x > y : more += 1
            elif not rev and x < y : more += 1
    return (more + 0.5*same)  / (len(lst1)*len(lst2))

def distribution_analysis(dist1, dist2):
    
    # thresholds
    t_kendall = 0.6
    t_pearson = 0.6
    t_ks = 0.05
    t_a = (0.3, 0.7)
    
    # check for similar analysis
    # kolmogorov smirnov
    # A12 effect size
    cliff, res = cliffs_delta(dist1, dist2)

    test_statistic, p_value = wilcoxon(dist1, dist2)
    
    pearson = pearsonr(dist1, dist2)[0]
    kendall = kendalltau(dist1, dist2)[0]
    
    categorization = np.zeros(4)
    
    # are they similar?
    if p_value > t_ks and cliff > 0.147:
        categorization[0] = 1
    else:
        categorization[0] = 0
    
    if np.abs(pearson) > t_pearson:
        categorization[1] = 1
    else:
        categorization[1] = 0
        
    if np.abs(pearson) < t_pearson and np.abs(kendall) > t_kendall:
        categorization[2] = 1
    else:
        categorization[2] = 0
        
    if np.abs(pearson) < t_pearson and np.abs(kendall) < t_kendall:
        categorization[3] = 1
    else:
        categorization[3] = 0
        
    return categorization
'''
dist_a = np.random.laplace(0,1, size=500)
dist_b = np.random.laplace(0,1, size=500)
distribution_analysis(dist_a, dist_b)

for system in systems:
    data = pd.read_csv('datax/{}_measurements.csv'.format(system))

    cats = []
    for kpi in systems[system]:
        if kpi in ['time', 'throughput']:
            
            workloads = data.workload.unique()
            
            for w1, w2 in itertools.combinations(workloads, 2):
                d1 = data[data['workload'] == w1].sort_values(by='config')[kpi].values
                d2 = data[data['workload'] == w2].sort_values(by='config')[kpi].values
                
                cat = distribution_analysis(d1, d2)
                cats.append(cat)
                
    cats = np.vstack(cats)
    plt.pcolormesh(cats)
    plt.title(system)
    plt.show()
    
    print(system, cats.shape[0])
    print('SD', cats.sum(axis=0)[0], round(100*cats.sum(axis=0)[0] / cats.shape[0], 2))
    print('LT', cats.sum(axis=0)[1], round(100*cats.sum(axis=0)[1] / cats.shape[0], 2))
    print('XMT', cats.sum(axis=0)[2], round(100*cats.sum(axis=0)[2] / cats.shape[0], 2))
    print('NMT', cats.sum(axis=0)[3], round(100*cats.sum(axis=0)[3] / cats.shape[0], 2))
    print('----------')
            
            
'''    