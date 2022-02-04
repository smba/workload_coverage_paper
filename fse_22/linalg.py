#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pycosa.util import remove_multicollinearity, get_vif
from pycosa.plotting import mirrored_histogram
import scipy.cluster.hierarchy as spc
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
import networkx as nx
import logging
import math
import numpy as np
import matplotlib.cm as cm

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
            
            lowerb, upperb = outlier_treatment(influences)
            outliers = len(list(filter(lambda x: x < lowerb or x > upperb, influences)))
            
            if covs[opt] > STDEV_THRESHOLD:
                nr_outlier[opt] = outliers
            else:
                nr_outlier[opt] = 0
                
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
                print(o, mi, ma)
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
        '''
        workloads = swarm_coefficients.workload.unique()

        cmap = plt.get_cmap('hsv')
        colors = list([cmap(1.*i/len(workloads)) for i in range(len(workloads))])
        colors = [list(c) for c in colors]
        #for i in range(len(colors)):
        #    colors[i][3] = 1.0
        
        
        fig = plt.figure(figsize=(4.5, 4.5))
        for j, o in enumerate(option_order):
            plt.axhline(j, linewidth=0.3, alpha=0.5,color='black', linestyle=':')
            #influences = np.random.laplace(0,3,size=len(workloads))
            
            influences = swarm_coefficients[swarm_coefficients['option'] == o].sort_values(by='workload')['influence'].values
            
            # diverse enough -> show full color spectrum
            if True:
                
                median = np.median(influences)
                std = np.std(influences)
                
                n_std = 2
                indices = np.where( (influences < median - n_std*std) | (influences > median + n_std*std) )[0]
                #print(indices)
                colors_new = [[0.1,0.1,0.1,0.1] for i in range(len(colors))]
                for i in indices:
                    colors_new[i] = colors[i]
            plt.scatter(influences, [o for i in range(len(workloads))], color=colors_new, s=22, marker='D')
        plt.axvline(0, linewidth=0.4, alpha=0.5,color='black')
        #plt.savefig('grafik.pdf', bbox_inches='tight')
        #plt.clf()
        plt.title('{} -- {}'.format(system, kpi))
        plt.xlabel('Relative Performance Influence')
        
        legend_elements = []
        
        for i, w in enumerate(workloads):
            legend_elements.append(
                Line2D([0], [0], marker='D', color='w', label=w,
                          markerfacecolor=colors[i], markersize=5))
        fig.legend(handles=legend_elements, loc='lower center',ncol=3,bbox_to_anchor=(0.5, -0.3))
        
        #fig.subplots_adjust(bottom=0.25)
        plt.show()
        '''
        
        
        '''
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
        '''
'''
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
'''       