import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import iqr

def cluster_corr(corr_array, inplace=False):
    """
    Rearranges the correlation matrix, corr_array, so that groups of highly 
    correlated variables are next to eachother 
    
    Parameters
    ----------
    corr_array : pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix 
        
    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix with the columns and rows rearranged
    """
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max()/2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, 
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)
    
    if not inplace:
        corr_array = corr_array.copy()
    
    if isinstance(corr_array, pd.DataFrame):
        return corr_array.iloc[idx, :].T.iloc[idx, :]
    return corr_array[idx, :][:, idx]

def levenshtein(a,b):
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a,b = b,a
        n,m = m,n
        
    current = range(n+1)
    for i in range(1,m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1,n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)
            
    return current[n]
'''
#sns.set_context("paper")
sns.set_palette("colorblind")
    
# Set the font to be serif, rather than sans
sns.set(font='sans')

systems = ['kanzi']

for i, system in enumerate(systems):
    df = pd.read_csv('auswertung/performance_similarity/{}/{}_performance_similarity_cache.csv'.format(system, system))
    
    metric = 'throughput' if system == 'h2' else 'time' 
    df = df[df['nfp'] == metric]

    sim = df[(df['wilcoxon_p'] > 0.05)].shape[0]
    lin = df[(df['wilcoxon_p'] < 0.05) & (df['pearsonr'] > 0.6)].shape[0]
    mon = df[(df['kendalltau'] > 0.6) & (df['pearsonr'] < 0.6)].shape[0]
    dis = df.shape[0] - sum([sim, lin, mon])
    
    print(system)
    print(sim, sim / df.shape[0]*100)
    print(lin, lin / df.shape[0]*100)
    print(mon, mon / df.shape[0]*100)
    print(dis, dis / df.shape[0]*100)
    print('-----')

    
    workloads = set(df.workload1.unique().tolist()) | set(df.workload2.unique().tolist())
    cor = pd.DataFrame(np.zeros((len(workloads), len(workloads))),index=workloads,columns=workloads)
    for w in workloads:
        cor.loc[w][w] = 1.0
    print(workloads)
    for row in df.iterrows():
        w1 = row[1]['workload1']
        w2 = row[1]['workload2']
        corr = row[1]['kendalltau']
        cor.loc[w1][w2] = float(corr)
        cor.loc[w2][w1] = float(corr)
        
    cor = cluster_corr(cor)
    
    mask = np.triu(np.ones_like(cor, dtype=bool))
    
    #plt.figure(figsize=(15,15))
    gg = sns.heatmap(cor, cmap='bwr', vmin=0, vmax=1, mask=mask,linewidths=.1,square=True, cbar_kws={"shrink": .5})
    gg.set_facecolor("lightgrey")
    plt.xticks(rotation=45)
    #plt.yticks(rotation=45)
    #plt.draw()
    plt.savefig('kendall_{}.pdf'.format(system), bbox_inches='tight')
    plt.clf()
'''

def transform(a):
    max_abs = np.max(np.abs(a))
    return np.array(a) / max_abs

print(1)
import itertools

stdss = {}
for system in ['kanzi', 'jump3r', 'batik', 'jadx', 'h2', 'dconvert']:
    #perf = pd.read_csv('{}_optimized/{}_performance_median.csv'.format(system, system))
    #plt.hist(perf['time'], bins=100)
    #plt.yscale('log')
    #plt.ylabel('frequency')
    #plt.xlabel('performance [seconds]')
    metric = 'throughput' if system == 'h2' else 'time' 
    df = pd.read_csv('auswertung/performance_models/{}/{}_models.csv'.format(system, system))
    df.drop(columns=['Unnamed: 0'], inplace=True)
    df = df[df['metric'] == metric]
    #print(df)
    df_lm = df[df['model'] == 'LR']
    
    values = []
    for w, wgroup in df_lm.groupby('workload'):
        ggroup = wgroup.sort_values(by=['term'])
        ggroup['value_standardized'] = MinMaxScaler().fit_transform(ggroup['value'].values.reshape(-1, 1))
        
        ggroup['value_relative'] = transform(ggroup['value'])
        
        values.append(ggroup['value_relative'])
        
    values = np.vstack(values).T
    print(values.shape)
    stds = []
    for row in values:
        stds.append(np.std(row))
    stdss[system] = stds

labels, data = [*zip(*stdss.items())]  # 'transpose' items to parallel key, value lists

# or backwards compatable    
labels, data = stdss.keys(), stdss.values()

plt.violinplot(data)
plt.xticks(range(1, len(labels) + 1), labels)

plt.ylabel(r"standard deviation of relative " + "\n" + "performance influences")
plt.savefig('performance_influence_variability.pdf', bbox_inches='tight')