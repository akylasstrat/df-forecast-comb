# -*- coding: utf-8 -*-
"""
Plots for data pooling

@author: akylas.stratigakos@minesparisl.psl.eu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os
#from pathlib import Path
import pickle
import gurobipy as gp
#import pickle
from sklearn.preprocessing import MinMaxScaler

cd = os.path.dirname(__file__)  #Current directory
sys.path.append(cd)
#project_dir=Path(cd).parent.__str__()   #project_directory
plt.rcParams['figure.dpi'] = 600

from sklearn.ensemble import ExtraTreesRegressor
from EnsemblePrescriptiveTree import *
from scipy.stats import norm, gaussian_kde
from scipy import interpolate
from scipy.special import ndtr
from QR_regressor import *
from sklearn.neighbors import KNeighborsRegressor
from EnsemblePrescriptiveTree_OOB import *
from optimization_functions import *
from newsvendor import *
from matplotlib.ticker import FormatStrFormatter

# IEEE plot parameters (not sure about mathfont)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 600
plt.rcParams['figure.figsize'] = (3.5, 2) # Height can be changed
plt.rcParams['font.size'] = 7
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams["mathtext.fontset"] = 'dejavuserif'

def mse(pred, actual):                
    return np.square(pred.reshape(-1) - actual.reshape(-1)).mean()

def newsvendor_loss(pred, actual, q = 0.5):                
    return np.maximum(q*(actual - pred), (q-1)*(actual - pred)).mean()
        
def projection(predictions):
    predictions[predictions<0] = 0
    predictions[predictions>1] = 1
    return predictions

def tree_params():
    ''' Hyperparameters for tree algorithms'''
    params = {}
    params['n_estimators'] = 100
    params['n_min'] = 2
    params['max_features'] = 'auto'
    return params

def params():
    ''' Set up the experiment parameters'''

    params = {}
    # Either pv or wind
    params['target'] = 'wind'
    params['start_date'] = '2019-01-01'
    params['split_date'] = '2020-01-01' # Defines train/test split
    params['end_date'] = '2020-05-01'
    params['n_base'] = 60 # number of base series
    
    params['parallel'] = False # If True, then trees are grown in parallel
    params['save'] = False # If True, then saves models and results
    params['train_brc'] = False # If True, then saves models and results
    params['train_trees'] = False # If True, then train models, else tries to load previous runs
    
    # Experiment parameters
    if params['target'] == 'wind':
        params['n_corrupt_nodes'] = [5, 10, 20] # number of nodes with erroneous measurements
    elif params['target'] == 'pv':
        params['n_corrupt_nodes'] = [2, 5, 10] # number of base series
    params['percentage'] = [.05, .10, .20, .50]  # percentage of corrupted datapoints
    params['iterations'] = 5 # per pair of (n_nodes,percentage)
    
    # other experiment parameters
    params['contamination'] = False
    params['problem'] = 'newsvendor' # mse or newsvendor
    params['N_sample'] = [10, 50, 100, 200, 500]
    params['iterations'] = 5

    return params

#%%%%%%%%%%%%%%%%%
# Results for a single run (fixed S)

config = params()
problem = 'newsvendor'
config['save'] = False
hyperparam = tree_params()

case_id = 'smart4res'
case_path = f'{cd}\\{case_id}-case-study'
results = pd.read_csv(f'{case_path}\\results\\{problem}_results.csv', index_col = 0)


# example of barycenters
with open(f'{case_path}\\results\\w_brc_{200}.sav', 'rb') as handle: w_brc = pickle.load(handle)
with open(f'{case_path}\\results\\l2_brc_{200}.sav', 'rb') as handle: l2_brc = pickle.load(handle)

plt.matshow(w_brc)
plt.show()

#%%
N_sample = results['N'].unique()

marker = ['o', 'd', 's', 'v', '*']
m_plot = ['Local', 'Pooled', 'W-BRC', 'L2-BRC', 'Combined']
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:brown', 'tab:purple']
labels = ['$\mathtt{Local}$', '$\mathtt{Pool}$', '$\mathtt{W-BRC}$', '$\ell_2-\mathtt{BRC}$', '$\mathtt{Comb}$']

if problem == 'mse':
    fig, ax  = plt.subplots()
    for i, m in enumerate(m_plot):
        #std = 100*results.groupby(['N'])[[m]].mean().values.reshape(-1)
        std = 100*(results.groupby(['N', 'iteration'])[[m]].mean()).groupby('N').std().values.reshape(-1)
        y_val = 100*results.groupby(['N'])[[m]].mean().values.reshape(-1)
        (100*results.groupby(['N'])[[m]].mean()).plot(ax = ax, marker = marker[i], color = colors[i], label = labels[i], 
                                                      legend=False)
        
        plt.fill_between(N_sample, y_val-std, y_val+std, alpha = 0.25)
        
    plt.ylabel('MSE')
    plt.legend(labels)
    plt.xlabel('Number of observations')
    plt.xticks(N_sample, N_sample)
    if config['save']: plt.savefig(f'{case_path}\\plots\\{case_id}_{problem}_aggregate.pdf')
    plt.show()

elif problem == 'newsvendor':
    fig, ax  = plt.subplots(nrows = 3, sharex = True, figsize = (3.5, 4))
    for j, q in enumerate([.1, .5, .9]):
        # set current plot axis
        plt.axes(ax[j])
        for i, m in enumerate(m_plot):
            y_val = (100*results.query(f'Quantile=={q}').groupby(['N'])[m].mean())
            plt.plot(N_sample, y_val, color = colors[i], label = labels[i], marker = marker[i])

        ax[j].annotate(f'crit. quant: {q}', (0.2, 0.85), color = 'black', 
                            xycoords = 'axes fraction', fontsize = 8, bbox=dict(facecolor='none', edgecolor='black', boxstyle='square'))
        plt.ylabel('Pinball loss')
    plt.xticks(N_sample, N_sample)
    plt.legend(labels, ncol = 2)
    plt.xlabel('Number of observations')
    if config['save']: plt.savefig(f'{case_path}\\plots\\\{case_id}_{problem}_aggregate.pdf')
    plt.show()

#%% Box plots

fig, ax = plt.subplots(constrained_layout=True)

N_sample = results['N'].unique()

mid_positions = np.arange(len(m_plot)/2-0.5, 2*len(N_sample)*len(m_plot), 2*len(N_sample))
dx = 0.25
width = .8

if problem == 'mse':
    
    for j, n_sample in enumerate(N_sample):    
        original_x_pos = np.arange(len(m_plot)*2*j, (2*j+1)*len(m_plot))
        x_pos = [pos+(pos-mid_positions[j])*dx for pos in original_x_pos]

        temp_data = results.query(f'N=={n_sample}')[m_plot].copy()
        
        std = 100*temp_data.std().values.reshape(-1)
        y_val = 100*temp_data.mean().values.reshape(-1)
        
        Lines = []
        for i, m in enumerate(m_plot):
            l = plt.errorbar(x_pos[i], y_val[i], yerr=std[i], linestyle='',
                         marker = marker[i], color = colors[i], elinewidth = 1, 
                         markersize = 5, label = m)
            
            Lines.append(l)
        '''
        box1 = plt.boxplot( temp_data, positions = x_pos, widths  = width,
                           patch_artist=True, showfliers=False, whis = 1.5)
        #['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']
        for element in [ 'medians']:
            plt.setp(box1[element], color = 'y', linewidth = .9)
        for i, patch in enumerate(box1['boxes']):
            patch.set(facecolor = colors[i]) 
        #for element in [ 'means']:
        #    plt.setp(box1[element], marker = 's', linewidth = .9)
        '''
        
    plt.xticks(ticks= mid_positions, labels= N_sample)
    plt.xlim([-2, x_pos[-1]+0.5])
    plt.xlabel('$N_{s}$')
    plt.ylabel('MSE (%)')
    #plt.annotate('$B=25, K=d_x/2$', xy=(0.7, 1.05), xycoords='axes fraction', bbox=dict(boxstyle='square', fc='white'))
    plt.legend([l[0] for l in Lines], labels, loc = [0.5, .75], 
               ncol=2, fontsize = 6)
    
    #plt.legend([box for box in box1["boxes"]], labels, loc = [0.5, .75], 
    #           ncol=2, fontsize = 6)
    if config['save']: plt.savefig(f'{case_path}\\plots\\{case_id}_{problem}_boxplot.pdf')
    plt.show()

elif problem == 'newsvendor':
    
    fig, ax  = plt.subplots(nrows = 3, sharex = True, figsize = (3.5, 3*1.8))
    for nfig, q in enumerate([.1, .5, .9]):

        plt.axes(ax[nfig])
        for j, n_sample in enumerate(N_sample):    
            original_x_pos = np.arange(len(m_plot)*2*j, (2*j+1)*len(m_plot))
            x_pos = [pos+(pos-mid_positions[j])*dx for pos in original_x_pos]
            
            temp_data = 100*results.query(f'N=={n_sample} and Quantile=={q}')[m_plot].copy()
            
            std = temp_data.std().values.reshape(-1)
            y_val = temp_data.mean().values.reshape(-1)
            
            Lines = []
            for i, m in enumerate(m_plot):
                l = plt.errorbar(x_pos[i], y_val[i], yerr=std[i], linestyle='',
                             marker = marker[i], color = colors[i], elinewidth = 1, 
                             markersize = 5, label = m)
                
                Lines.append(l)

            '''
            box1 = plt.boxplot( temp_data, positions = x_pos, widths  = width,
                               patch_artist=True, showfliers=False)
            #['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']
            for element in [  'medians']:
                plt.setp(box1[element], color = 'y', linewidth = 1.2)
            for i, patch in enumerate(box1['boxes']):
                patch.set(facecolor = colors[i]) 
            '''
            
        ax[nfig].annotate(f'crit. quant: {q}', (0.2, 0.85), color = 'black', 
                            xycoords = 'axes fraction', fontsize = 8, bbox=dict(facecolor='none', edgecolor='black', boxstyle='square'))
        
        plt.ylabel('Pinball loss')
    
    plt.xticks(ticks= mid_positions, labels= N_sample)
    plt.xlim([-2, x_pos[-1]+0.5])
    plt.xlabel('$N_{s}$')
    
    #lgd = fig.legend([box for box in box1["boxes"]], labels, 
    #                 fontsize=6, ncol=3, loc = (1, .8), 
    #                 bbox_to_anchor=(0.2, -0.1))    
    
    lgd = plt.legend([l[0] for l in Lines], labels, loc = [0.5, .75], 
               ncol=2, fontsize = 6)

    #plt.annotate('$B=25, K=d_x/2$', xy=(0.7, 1.05), xycoords='axes fraction', bbox=dict(boxstyle='square', fc='white'))
    if config['save']: plt.savefig(f'{case_path}\\plots\\{case_id}_{problem}_boxplot.pdf', 
                                   bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()
    

#%%%%%%%%%%%%%%%%%
# Full results for smart4res/vestas case study (w.r.t. to varying N_s, S, and model quality)

config = params()
quant = 0.2
problem = 'reg_trad'
qual = False
config['save'] = True
hyperparam = tree_params()

case_id = 'smart4res-mult'
case_path = f'{cd}\\{case_id}-case-study'
#results_path = f'{case_path}\\new-results-2010'
results_path = f'{case_path}\\new-results-2010'

if problem in ['newsvendor', 'reg_trad']: problem = f'{problem}_{quant}'


if qual:
    results = pd.read_csv(f'{results_path}\\{problem}_results.csv', index_col = 0)
    fig_name = f'{case_id}_{problem}_quality_boxplot.pdf'
else:
    results = pd.read_csv(f'{results_path}\\{problem}_results.csv', index_col = 0)
    fig_name = f'{case_id}_{problem}_boxplot.pdf'


N_sample = results['N'].unique()
N_assets = results['N_assets'].unique()

marker = ['o', 'd', 's', '*', 'v', '2','3']
#models = ['Local', 'Pooled', 'W-BRC', 'Combined', 'L2-BRC']
#m_to_plot = ['Local', 'Pooled', 'W-BRC', 'Combined']

models = ['Local', 'Pooled', 'WBC', 'optWBC', 'ConvComb', 'Interpol', 'L2-BRC']
m_plot = ['Local', 'Pooled', 'WBC', 'Interpol']

model_index = [models.index(m) for m in m_plot]

results[models] = 100*results[models]

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:red', 'tab:gray']
labels = ['$\mathtt{Local}$', '$\mathtt{Pool-Naive}$', '$\mathtt{Pool-OT}$', '$\mathtt{optWBC}$', '$\mathtt{ConvComb}$', 
          '$\mathtt{Interp}$', '$\ell_2-\mathtt{BRC}$']


#%%
cart_inter = []
rf_inter = []
cart_local = []
rf_local = []
inter = []
for n_asset in results['N_assets'].unique()[:3]:
    for n in results['N'].unique():
        for iter_ in results['iteration'].unique():
            temp = results.query(f'N_assets=={n_asset} and iteration=={iter_}and N=={n}')
            
            cart_inter.append(temp['Interpol'].iloc[:int((len(temp['Park_id'])/2))+1].mean())
            rf_inter.append(temp['Interpol'].iloc[int((len(temp['Park_id'])/2))+1:].mean())
            cart_local.append(temp['Local'].iloc[:int((len(temp['Park_id'])/2))+1].mean())
            rf_local.append(temp['Local'].iloc[int((len(temp['Park_id'])/2))+1:].mean())

            inter.append(temp['Interpol'].mean())
            
inter = np.array(inter)
cart_inter = np.array(cart_inter)
rf_inter = np.array(rf_inter)

plt.plot(cart_inter, label = 'CART-inter')
plt.plot(rf_inter, label = 'CART-local')
plt.plot(inter, '--')
plt.legend()
plt.show()

#%% Single plot for presentation slides
'''
if problem == 'mse':
    fig_name = f'{problem}_defense_slide.pdf'
    marker = ['^', 'd', 's', '*', 'v', '2','3']
    
    N_sample_to_plot = [50, 100, 200, 500]
    mid_positions = np.arange(len(m_plot)/2-0.5, 2*len(N_sample_to_plot)*len(m_plot), 2*len(m_plot))
    dx = 0.25 
    
    fig, ax = plt.subplots(1, constrained_layout = True, figsize = (3.5, 1.8))
        
    for row,n_assets in enumerate([20]):        
    
        for j, n_sample in enumerate(N_sample_to_plot):    
            original_x_pos = np.arange(len(m_plot)*2*j, (2*j+1)*len(m_plot))
            #mid_positions.append(np.median(original_x_pos))
            x_pos = [pos+(pos-mid_positions[j])*dx for pos in original_x_pos]
            
            temp_data = results.query(f'N_assets=={n_assets} and N=={n_sample}')[m_plot].copy()
    
            #temp_data[m_plot] = (temp_data['Local'].values.reshape(-1,1) - temp_data[m_plot])/temp_data['Local'].values.reshape(-1,1)            
            std = results.query(f'N_assets=={n_assets} and N=={n_sample}').groupby('iteration').mean()[m_plot].std()
            #std = temp_data[m_to_plot].std().values.reshape(-1)
            y_val = temp_data[m_plot].mean().values.reshape(-1)
            
            Lines = []
            
            for i, ind in enumerate(model_index):                
                l = plt.errorbar(x_pos[i], y_val[i], yerr=std[i], linestyle='',
                             marker = marker[ind], color = colors[ind], elinewidth = 1, 
                             markersize = 5, label = m)
                
                Lines.append(l)
                
        plt.xticks(ticks= mid_positions, labels= N_sample_to_plot)
        plt.xlim([-2, x_pos[-1]+0.5])
        plt.ylim([3.0, 4.7])
        ax.annotate(f'$S={int(n_assets)}$', (0.2, 0.8), color = 'black', 
                            xycoords = 'axes fraction', fontsize = 8,
                       bbox=dict(facecolor='none', edgecolor='black', boxstyle='square', linewidth = 0.5))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        
        plt.xlabel('$N_{s}$')
        plt.ylabel('MSE (%)')
        
    lgd = fig.legend([l[0] for l in Lines], [labels[ind] for ind in model_index], bbox_to_anchor=(0.95, 0.95), 
               ncol=1, fontsize = 6)
    
    fig.savefig(f'{case_path}\\plots\\{fig_name}', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()
'''    
#%% Box plots for all experiments

N_sample_to_plot = [50, 100, 200, 500]

mid_positions = np.arange(len(m_plot)/2-0.5, 2*len(N_sample_to_plot)*len(m_plot), 2*len(m_plot))

dx = 0.25 

# dictionary for subplots
ax_lbl = [[N_assets[0], 10], 
          [20, 50]]

# axis ratio
gs_kw = dict(width_ratios=[1, 1], height_ratios=[1,1])

fig, ax = plt.subplot_mosaic(ax_lbl, constrained_layout = True, figsize = (5, 1.5*2), 
                             gridspec_kw = gs_kw, sharex=True, sharey = True)

if problem == 'mse':
    
    #fig, ax  = plt.subplots(nrows = len(N_assets), sharex = True, figsize = (3.5, 1.8*len(N_assets)))
    for row,n_assets in enumerate(N_assets):        
        plt.sca( ax[n_assets] )

        for j, n_sample in enumerate(N_sample_to_plot):    
            original_x_pos = np.arange(len(m_plot)*2*j, (2*j+1)*len(m_plot))
            #mid_positions.append(np.median(original_x_pos))
            x_pos = [pos+(pos-mid_positions[j])*dx for pos in original_x_pos]
            
            temp_data = results.query(f'N_assets=={n_assets} and N=={n_sample}')[m_plot].copy()

            #temp_data[m_plot] = (temp_data['Local'].values.reshape(-1,1) - temp_data[m_plot])/temp_data['Local'].values.reshape(-1,1)            
            std = results.query(f'N_assets=={n_assets} and N=={n_sample}').groupby('iteration').mean()[m_plot].std()
            #std = temp_data[m_to_plot].std().values.reshape(-1)
            y_val = temp_data[m_plot].mean().values.reshape(-1)
            
            Lines = []
            
            for i, ind in enumerate(model_index):                
                l = plt.errorbar(x_pos[i], y_val[i], yerr=std[i], linestyle='',
                             marker = marker[ind], color = colors[ind], elinewidth = 1, 
                             markersize = 5)
                
                Lines.append(l)
            
            '''    
            box1 = plt.boxplot( temp_data, positions = x_pos, widths  = width,
                               patch_artist=True, showfliers=False)
            #['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']
            for element in [ 'medians']:
                plt.setp(box1[element], color = 'y', linewidth = 1)
            for element in [ 'whiskers']:
                plt.setp(box1[element], linewidth = .75)
            for i, patch in enumerate(box1['boxes']):
                patch.set(facecolor = colors[i], linewidth = .75) 
            '''

        plt.xticks(ticks= mid_positions, labels= N_sample_to_plot)
        plt.xlim([-2, x_pos[-1]+0.5])
        plt.ylim([3.0, 4.7])
        ax[n_assets].annotate(f'$S={int(n_assets)}$', (0.2, 0.8), color = 'black', 
                            xycoords = 'axes fraction', fontsize = 8,
                       bbox=dict(facecolor='none', edgecolor='black', boxstyle='square', linewidth = 0.5))
        ax[n_assets].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        
    #plt.annotate('$B=25, K=d_x/2$', xy=(0.7, 1.05), xycoords='axes fraction', bbox=dict(boxstyle='square', fc='white'))
        plt.xlabel('$N_{s}$')
        plt.ylabel('MSE (%)')
        #plt.grid(axis = 'y')
        

    #lgd = fig.legend([box for box in box1["boxes"]], labels, bbox_to_anchor=(0.75, -0.0), 
    #           ncol=5, fontsize = 6)
    
    lgd = fig.legend([l[0] for l in Lines], [labels[ind] for ind in model_index], bbox_to_anchor=(0.77, -0.0), 
               ncol=len(m_plot), fontsize = 6)
    
    if config['save']: 
        fig.savefig(f'{case_path}\\plots\\{fig_name}', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()

elif ('newsvendor' in problem)or('reg_trad' in problem):
        
    for row,n_assets in enumerate(N_assets):        
        plt.sca( ax[n_assets] )

        for j, n_sample in enumerate(N_sample_to_plot):    
            original_x_pos = np.arange(len(m_plot)*2*j, (2*j+1)*len(m_plot))
            x_pos = [pos+(pos-mid_positions[j])*dx for pos in original_x_pos]
            
            temp_data = results.query(f'N_assets=={n_assets} and N=={n_sample}').copy()
            #temp_data[m_plot] = (temp_data['Local'].values.reshape(-1,1) - temp_data[m_plot])/temp_data['Local'].values.reshape(-1,1)            
            std = results.query(f'N_assets=={n_assets} and N=={n_sample}').groupby('iteration').mean()[m_plot].std()/np.sqrt(5)
            #std = temp_data[m_to_plot].std().values.reshape(-1)
            #std = temp_data.groupby('iteration').mean().std()

            y_val = temp_data[m_plot].mean().values.reshape(-1)
            Lines = []
            for i, ind in enumerate(model_index):
                l = plt.errorbar(x_pos[i], y_val[i], yerr=std[i], linestyle='',
                             marker = marker[ind], color = colors[ind], elinewidth = 1, 
                             markersize = 5, label = labels[ind])
                
                Lines.append(l)
            
            plt.xticks(ticks= mid_positions[:len(N_sample_to_plot)], labels= N_sample_to_plot)
            plt.xlim([-2, x_pos[-1]+0.5])
    
    
        ax[n_assets].annotate(f'$S={int(n_assets)}$', (0.2, 0.85), color = 'black', 
                            xycoords = 'axes fraction', fontsize = 8,
                       bbox=dict(facecolor='none', edgecolor='black', boxstyle='square', linewidth = 0.5))
        ax[n_assets].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        #plt.annotate('$B=25, K=d_x/2$', xy=(0.7, 1.05), xycoords='axes fraction', bbox=dict(boxstyle='square', fc='white'))
        plt.xlabel('$N_{s}$')
        plt.ylabel('Task loss')
        #plt.grid(axis = 'y')
            

        #lgd = fig.legend([box for box in box1["boxes"]], labels, bbox_to_anchor=(0.75, -0.0), 
        #           ncol=5, fontsize = 6)
        
        lgd = fig.legend([l[0] for l in Lines], [labels[ind] for ind in model_index], bbox_to_anchor=(0.75, -0.0), 
                   ncol=6, fontsize = 6)
    

    #plt.annotate('$B=25, K=d_x/2$', xy=(0.7, 1.05), xycoords='axes fraction', bbox=dict(boxstyle='square', fc='white'))
    if config['save']: plt.savefig(f'{case_path}\\plots\\{fig_name}', 
                                   bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()
   

#%%%%%%%% Results for random number of observations per problem

case_id = 'smart4res-mult'
case_path = f'{cd}\\{case_id}-case-study'

mse_results = pd.read_csv(f'{case_path}\\new-results-2010\\mse_randomN_results.csv', index_col = 0)

mse_results_aggr = mse_results.groupby(['N_assets', 'iteration']).mean()
mse_results_aggr[m_plot] = (mse_results_aggr['Local'].values.reshape(-1,1) -  mse_results_aggr[m_plot])/mse_results_aggr['Local'].values.reshape(-1,1)

mse_mean = 100*mse_results_aggr.groupby(['N_assets'])[m_plot].mean()
mse_se = 100*mse_results_aggr.groupby(['N_assets'])[m_plot].mean().std()
mse_se = 100*mse_results_aggr.groupby(['N_assets', 'iteration'])[m_plot].mean().groupby('N_assets').std()/np.sqrt(5)


table_tex = mse_mean.round(2).astype(str)+' '+' ('+mse_se.round(2).astype(str)+')'

#mse_se = 100*mse_results_aggr.groupby(['N_assets'])[m_plot].std()/np.sqrt(10)

#mse_results[m_plot] = (mse_results['Local'].values.reshape(-1,1)-mse_results[m_plot])/mse_results['Local'].values.reshape(-1,1)
#mse_mean = 100*mse_results.groupby(['N_assets'])[m_plot].mean()
#mse_se = 100*results.groupby(['N_assets', 'iteration'])[m_plot].mean().std()/np.sqrt(10)
#mse_se = 100*mse_results.groupby(['N_assets', 'iteration'])[m_plot].mean().groupby(['N_assets']).std()/np.sqrt(10)

#%%
nvd_results = pd.read_csv(f'{case_path}\\new-results-2010\\newsvendor_0.2_randomN_results.csv', index_col = 0)

nvd_results_aggr = nvd_results.groupby(['N_assets', 'iteration']).mean()
nvd_results_aggr[m_plot] = (nvd_results_aggr['Local'].values.reshape(-1,1) -  nvd_results_aggr[m_plot])/nvd_results_aggr['Local'].values.reshape(-1,1)

nvd_mean = 100*nvd_results_aggr.groupby(['N_assets'])[m_plot].mean()
nvd_se = 100*nvd_results_aggr.groupby(['N_assets'])[m_plot].std()/np.sqrt(10)
nvd_se = 100*nvd_results_aggr.groupby(['N_assets', 'iteration'])[m_plot].mean().groupby('N_assets').std()/np.sqrt(5)
table_tex = nvd_mean.round(2).astype(str)+' '+' ('+nvd_se.round(2).astype(str)+')'

#%%
nvd_results = pd.read_csv(f'{case_path}\\results\\random_N_newsvendor_results.csv', index_col = 0)
#nvd_results[m_plot] = (nvd_results['Local'].values.reshape(-1,1)-nvd_results[m_plot])/nvd_results['Local'].values.reshape(-1,1)

xx = (nvd_results.groupby(['N_assets']).mean()['Local'].values.reshape(-1,1)- nvd_results.groupby(['N_assets']).mean()[m_plot])/nvd_results.groupby(['N_assets']).mean()['Local'].values.reshape(-1,1)

nvd_mean = 100*nvd_results.query('Quantile == 0.8').groupby(['N_assets'])[m_plot].mean()
nvd_se = 100*nvd_results.query('Quantile == 0.8').groupby(['N_assets', 'iteration'])[m_plot].mean().groupby(['N_assets']).std()/np.sqrt(10)
    
#%% Plots for synthetic data
stop_here
config = params()
problem = 'mse'
config['save'] = False
locshift = False
hyperparam = tree_params()

case_id = 'synthetic-data'
case_path = f'{cd}\\{case_id}-case-study'
if locshift:
    results = pd.read_csv(f'{case_path}\\results\\aggr_results_shift.csv', index_col = 0)
else:
    results = pd.read_csv(f'{case_path}\\results\\aggr_results_noshift.csv', index_col = 0)

#%%

N_sample = results['N'].unique()

marker = ['o', 'd', 's', '+', '2']
m_plot = ['Local', 'Pooled', 'W-BRC', 'l2-BRC', 'Combined']
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:brown', 'tab:purple']
labels = ['$\mathtt{Local}$', '$\mathtt{Pool}$', '$\mathtt{W-BRC}$', '$\ell_2-\mathtt{BRC}$', '$\mathtt{Comb}$']

fig, ax  = plt.subplots()
for i, m in enumerate(m_plot):
    std = 100*results.groupby(['N'])[[m]].std().values.reshape(-1)
    y_val = (results.groupby(['N'])[[m]].mean()).values.reshape(-1)
    (results.groupby(['N_assets', 'N'])[[m]].mean()).plot(ax = ax, marker = marker[i], color = colors[i], label = labels[i], legend=False)
    
    #plt.fill_between(N_sample, y_val-std, y_val+std, alpha = 0.25)
    
plt.ylabel('MSE')
plt.legend(labels)
plt.xlabel('Number of observations')
plt.xticks(N_sample, N_sample)
if config['save']: plt.savefig(f'{case_path}\\plots\\{case_id}_{problem}_aggregate.pdf')
plt.show()