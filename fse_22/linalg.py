#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pycosa.util import remove_multicollinearity, get_vif
import scipy.cluster.hierarchy as spc

import networkx as nx
import logging
import math
import numpy as np

from matplotlib import gridspec

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
            
            if system == 'h2':
                plt.figure(figsize=(4,2))
                sns.histplot(df[kpi].values.reshape(-1, 1), bins=75)
                plt.xlabel('Throughput (transactions/sec.)')
                plt.ylabel('Frequency')
                plt.title(w)
                plt.savefig('h2_{}.eps'.format(w), bbox_inches='tight')
            
            coefficients.append(coefs)
            
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
            
            lowerb, upperb = outlier_treatment(influences)
            outliers = len(list(filter(lambda x: x < lowerb or x > upperb, influences)))
            
            if covs[opt] > STDEV_THRESHOLD:
                nr_outlier[opt] = outliers
            else:
                nr_outlier[opt] = 0
                
        pivot = dict(sorted(means.items(), key=lambda item: item[1], reverse=True))

        option_order = pivot.keys()
        
        fig, ax = plt.subplots(nrows=1, ncols=2, sharex='col', sharey=True,
                               gridspec_kw={'width_ratios': [3, 0.7]},
                               figsize=(9, 6))
        
        
        for z, coefs in enumerate(coefficients):
            ax[0].plot(
                [coefs[k] for k in option_order],
                option_order,label=workloads[z], marker='.')
         
        ax[1].barh(list(option_order), [covs[opt] for opt in option_order], color='black', alpha=0.75)
        #ax[2].barh(list(option_order), [nr_outlier[opt] for opt in option_order], color='brown', alpha=0.75)
        
        ax[0].legend(ncol=3, bbox_to_anchor=(0.8, -0.11))
        plt.suptitle(system + ' -- ' + kpi)
        #plt.xticks(rotation=45, rotation_mode='anchor')
        ax[0].set_xlabel(r'Relative performance influence $ \beta_i$')
        ax[0].axvline(0, linestyle='--', linewidth=1, color='grey')    
        #ax[1].axvline(STDEV_THRESHOLD, linestyle='--', linewidth=1, color='brown')   
        ax[1].set_xlabel('StDev $\sigma$')
        #ax[2].set_xlabel('#Outliers exceeding\n threshold StDev')
        #ax[2].set_xlim(0, len(workloads))
        
        plt.show()
    
# Correlation Analysis
for system in systems:
    data = pd.read_csv('datax/{}_measurements.csv'.format(system))

    for kpi in systems[system]:
        workloads = data.workload.unique()
        
        df = pd.DataFrame()
        for w in workloads:
            df[w] = data[data['workload'] == w].sort_values(by='config')[kpi].values
            
        corr = df.corr(method='pearson')
                
        # Generate a mask for the upper triangle        
        g = sns.clustermap(corr, cmap='RdBu_r', vmin=0, vmax=1.0, linewidth=3, 
                           cbar_kws={ "orientation": "horizontal" },
                           cbar_pos = (0.15, 0.83, 0.7, 0.01))
        g.ax_heatmap.yaxis.set_ticks_position("left")

        g.ax_row_dendrogram.set_visible(False)
        g.ax_col_dendrogram.set_visible(False)
        
        plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), fontsize=10)
        plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), fontsize=10)
        
        mask = np.triu(np.ones_like(corr))
        values = g.ax_heatmap.collections[0].get_array().reshape(corr.shape)
        new_values = np.ma.array(values, mask=mask)
        g.ax_heatmap.collections[0].set_array(new_values)
        
        plt.title('{}_{}_kendall.eps'.format(system, kpi))
        plt.show()
        plt.savefig('grafix/{}_{}_kendall.eps'.format(system, kpi), bbox_inches='tight')
        plt.clf()
        