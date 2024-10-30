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

cd = os.path.dirname(__file__)  #Current directory
sys.path.append(cd)
#project_dir=Path(cd).parent.__str__()   #project_directory

from matplotlib.ticker import FormatStrFormatter

# IEEE plot parameters (not sure about mathfont)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 600
plt.rcParams['figure.figsize'] = (3.5, 2) # Height can be changed
plt.rcParams['font.size'] = 7
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams["mathtext.fontset"] = 'dejavuserif'

def params():
    ''' Set up the experiment parameters'''
    params = {}    
    params['save'] = True # If True, then saves models and results
    return params

# Dictionary that stores the labels used in the paper
models_to_labels_dict = {'Ave':'OLP', 'CRPS':'CRPSL', 'knn':'kNN$', 'cart':'CART', 'rf':'RF', 
              'invW-0':'invW-0','invW-0.1':'invW-0.1','invW-1':'invW-1', 'invW-0.001':'$invW-0.001$','invW-inf':'invW-\infty',
              'DF_0':'DFL-0', 'DF_0.1':'DFL-0.1', 'DF_1':'DFL-1', 
              'Expert1':'Expert 1', 'Expert2':'Expert 2', 
              'DF-LR_0':'DFL-LR-0','DF-LR_0.1':'DFL-LR-0.1', 'DF-LR_1':'DFL-LR-1',
              'DF-MLP_0':'DFL-NN-0','DF-MLP_0.1':'DFL-NN-0.1', 'DF-MLP_1':'DFL-NN-1',
              'DF-LR_0':'DFL-LR-0','DF-LR_0.1':'DFL-LR-0.1', 'DF-LR_1':'DFL-LR-1',
              'CRPS-LR':'CRPSL-LR','CRPS-MLP':'CRPSL-NN', 
              'cart_date':'CART'}

#%%

# Directory to results folder
results_path = f'{cd}\\results'

#%% Synthetic Data Example

models = ['Expert1', 'Expert2', 'Ave', 'invW-0', 'invW-0.1', 'invW-1', 'invW-inf', 'CRPS','DF_0', 'DF_0.1', 'DF_1']

# Table 1, Section 3.2
qs_df = pd.read_csv(f'{results_path}\\synthetic_data\\synthetic_QS_mean_softmax.csv', index_col = 0)
cost_df = pd.read_csv(f'{results_path}\\synthetic_data\\synthetic_Decision_cost_softmax.csv', index_col = 0)
lambda_static = pd.read_csv(f'{results_path}\\synthetic_data\\lambda_static_softmax.csv', index_col = 0)

table_df = pd.DataFrame(data = [], index = [models_to_labels_dict[m] for m in models], columns = ['lambda_1', 'lambda_2', 'Regret', 'CRPS'])

for m in models:
    index_label = models_to_labels_dict[m]
    table_df.loc[index_label]['lambda_1'] = np.round(lambda_static[m][0],3)
    table_df.loc[index_label]['lambda_2'] = np.round(lambda_static[m][1],3)
    table_df.loc[index_label]['Regret'] = np.round(cost_df[m].values[0],3)
    table_df.loc[index_label]['CRPS'] = np.round(qs_df[m].values[0],3)

print('Table 1, Section 3.2')
print(table_df)

#%% Grid scheduling results// Table 3, Section 3.4

models = ['knn', 'cart', 'rf', 'Ave', 'invW-0', 'invW-0.1', 'invW-1', 'invW-inf', 'CRPS','DF_0', 'DF_0.1', 'DF_1']

config = params()
zone = 'Z1' # do not change this
# Cases = ['pglib_opf_case14_ieee.m', 'pglib_opf_case57_ieee.m']
case = 'pglib_opf_case14_ieee.m'

label_dict = {'Ave':'$\mathtt{OLP}$', 'CRPS':'$\mathtt{CRPSL}$', 
              'knn':'$\mathtt{kNN}$', 'cart':'$\mathtt{CART}$', 'rf':'$\mathtt{RF}$', 
              'invW-0':'$\mathtt{invW-0}$','invW-0.1':'$\mathtt{invW-0.1}$', 
              'invW-1':'$\mathtt{invW-1}$', 'invW-0.001':'$\mathtt{invW-0.001}$',
              'invW-inf':'$\mathtt{invW-\infty}$',
              'DF_0':'$\mathtt{DFL}$-0', 'DF_0.1':'$\mathtt{DFL}$-0.1', 'DF_1':'$\mathtt{DFL}$-1'}

da_cost = pd.read_csv(f'{results_path}\\grid_scheduling\\NEWRESULTS_{zone}_{case}_wind_DA_cost.csv', index_col = 0)
rt_cost = pd.read_csv(f'{results_path}\\grid_scheduling\\NEWRESULTS_{zone}_{case}_wind_RT_cost.csv', index_col = 0)
crps_df = pd.read_csv(f'{results_path}\\grid_scheduling\\NEWRESULTS_{zone}_{case}_wind_mean_QS.csv', index_col = 0)
lambda_static = pd.read_csv(f'{results_path}\\grid_scheduling\\NEWRESULTS_{zone}_{case}_wind_lambda_static.csv', index_col = 0)

total_regret = da_cost + rt_cost - da_cost['Perfect'].values[0]

table_df = pd.DataFrame(data = [], index = [models_to_labels_dict[m] for m in models], columns = ['lambda_1', 'lambda_2', 'lambda_3', 'Regret', 'CRPS'])

for m in models:
    
    index_label = models_to_labels_dict[m]
    
    table_df.loc[index_label]['lambda_1'] = np.round(lambda_static[m][0],3)
    table_df.loc[index_label]['lambda_2'] = np.round(lambda_static[m][1],3)
    table_df.loc[index_label]['lambda_3'] = np.round(lambda_static[m][2],3) 
    table_df.loc[index_label]['Regret'] = np.round(total_regret[m].values[0],3)
    table_df.loc[index_label]['CRPS'] = np.round(crps_df[m].values[0],3)


print('\n')
print('Table 3, Section 3.4')
print(table_df)

#### Figure is not included in the paper
models_to_plot = ['Ave', 'invW-0', 'invW-inf', 'CRPS', 'DF_0', 'DF_0.1', 'DF_1']
fig, ax  = plt.subplots()
lambda_static[models_to_plot].plot(kind='bar', ax = ax)
plt.xticks([0,1,2], ['$\mathtt{kNN}$', '$\mathtt{CART}$', '$\mathtt{RF}$'], rotation = 0)
plt.legend([label_dict[key] for key in models_to_plot], ncol = 2)
plt.xlabel('Component forecasts')
plt.ylabel('Combination weights $\mathtt{\lambda}$')
#plt.savefig(f'{cd}\\plots\\lambda_barplot_wind_grid_sched.pdf')
plt.show()


#%%%%%%%%%%%% Solar trading results// Table 3, Section 3.4

config = params()
config['save'] = False
target_prob = 'reg_trad' # Do not change
#crit_fract = 0.9
results_folder = ''

decision_cost = []
qs_cost = []

# Load results for all solar farms
for z in ['Z1','Z2', 'Z3']:
    decision_cost.append(pd.read_csv(f'{results_path}\\solar_trading_results_softmax\\{z}_{target_prob}__Decision_cost.csv', index_col = 0))
    qs_cost.append(pd.read_csv(f'{results_path}\\solar_trading_results_softmax\\{z}_{target_prob}__mean_QS.csv', index_col = 0))

decision_cost = pd.concat(decision_cost)
decision_cost.reset_index(inplace = True)
qs_cost = pd.concat(qs_cost)
qs_cost.reset_index(inplace = True)

################# Table 2, Section 3.3
component_models = ['knn', 'cart_date', 'rf']
static_models = ['Ave', 'invW-0', 'invW-0.1', 'invW-1', 'invW-inf', 'CRPS','DF_0', 'DF_0.1', 'DF_1']
adapt_models = ['CRPS-LR', 'CRPS-MLP', 'DF-LR_0','DF-MLP_0', 'DF-LR_0.1', 'DF-MLP_0.1', 'DF-LR_1', 'DF-MLP_1']
all_models = component_models + static_models + adapt_models

target_zone = 2
target_quantile = '0.2'

lambda_static = pd.read_csv(f'{results_path}\\solar_trading_results_softmax\\Z{target_zone}_reg_trad__{target_quantile}_lambda_static.csv', index_col = 0)

# Use only results for specific experiment
target_cost_df = decision_cost.query(f'Target=={target_zone} and Quantile == {target_quantile}')
target_qs_df = qs_cost.query(f'Target=={target_zone} and Quantile == {target_quantile}')

table_df = pd.DataFrame(data = [], index = [models_to_labels_dict[m] for m in all_models], columns = ['lambda_1', 'lambda_2', 'lambda_3', 'Regret', 'CRPS'])

for m in all_models:
    
    index_label = models_to_labels_dict[m]

    table_df.loc[index_label]['Regret'] = np.round(target_cost_df[m].values[0],3)
    table_df.loc[index_label]['CRPS'] = np.round(target_qs_df[m].values[0],3)
    
    if m in adapt_models:
        continue
    else:
        table_df.loc[index_label]['lambda_1'] = np.round(lambda_static[m][0],3)
        table_df.loc[index_label]['lambda_2'] = np.round(lambda_static[m][1],3)
        table_df.loc[index_label]['lambda_3'] = np.round(lambda_static[m][2],3) 

print('\n')
print('Table 2, Section 3.3')
print(table_df)

# Plot lambdas for specific combination (Plot is not included in the paper)
models_to_plot = ['Ave', 'invW-0', 'CRPS', 'DF_0', 'DF_0.1', 'DF_1']
fig, ax  = plt.subplots()
lambda_static[models_to_plot].plot(kind='bar', ax = ax)
plt.xticks([0,1,2], ['$\mathtt{kNN}$', '$\mathtt{CART}$', '$\mathtt{RF}$'], rotation = 0)
plt.legend([label_dict[key] for key in models_to_plot], ncol = 2)
plt.xlabel('Component forecasts')
plt.ylabel('Combination weights $\mathtt{\lambda}$')
# if config['save']:  plt.savefig(f'{cd}\\plots\\lambda_barplot_trading.pdf')
plt.show()


################# Figure 2, Section 3.3
gamma = [0, 0.1, 1]

# Step 1: Find relative improvement over OLP/Ave (linear pooling with equal weights) in terms of Decision regret and CRPS
relative_cost_df = decision_cost.copy()
relative_cost_df[static_models + adapt_models] = (relative_cost_df['Ave'].values.reshape(-1,1)-relative_cost_df[static_models + adapt_models])/relative_cost_df['Ave'].values.reshape(-1,1)

relative_crps_df = qs_cost.copy()
relative_crps_df[static_models + adapt_models] = (relative_crps_df['Ave'].values.reshape(-1,1)-relative_crps_df[static_models + adapt_models])/relative_crps_df['Ave'].values.reshape(-1,1)

# Step 2: Create plot
color = ['tab:blue', 'tab:green', 'tab:brown', 'tab:orange', 'tab:purple', 'black']
marker = ['s', 'o', 'd', '+', '1', '2']

static_models_plot = ['CRPS'] + [f'DF_{g}' for g in gamma] 
static_labels = ['$\mathtt{CRPSL}$'] + ['$\mathtt{DFL}$-0', '$\mathtt{DFL}$-0.1', '$\mathtt{DFL}$-1']

lr_models_plot = ['CRPS-LR'] + [f'DF-LR_{g}' for g in gamma] 
mlp_models_plot = ['CRPS-MLP'] + [f'DF-MLP_{g}' for g in gamma] 

lr_labels = ['$\mathtt{CRPSL-LR}$'] + ['$\mathtt{DFL-LR}-$0', '$\mathtt{DFL-LR}-$0.1', '$\mathtt{DFL-LR}-$1']
mlp_labels = ['$\mathtt{CRPSL-MLP}$'] + ['$\mathtt{DFL-MLP}-$0', '$\mathtt{DFL-MLP}-$0.1', '$\mathtt{DFL-MLP}-$1']

fig, ax  = plt.subplots()

plt.xlabel('Regret improvement over $\mathtt{OLP}$ (%)')
plt.ylabel('CRPS improvement over $\mathtt{OLP}$ (%)')

for i,m in enumerate(lr_models_plot):
    plt.scatter(100*relative_cost_df[m].mean(), 100*relative_crps_df[m].mean(), marker = 's', color = color[i])
for i,m in enumerate(mlp_models_plot):
    plt.scatter(100*relative_cost_df[m].mean(), 100*relative_crps_df[m].mean(), marker = 'd', color = color[i])
for i,m in enumerate(static_models_plot):
    ax.scatter(100*relative_cost_df[m].mean(), 100*relative_crps_df[m].mean(), marker = 'o', label = static_labels[i], color=color[i])

ax2 = ax.twinx()
ax2.scatter(np.NaN, np.NaN, marker = 's', label = '$\mathtt{LR}$', color='gray', alpha = 0.5)
ax2.scatter(np.NaN, np.NaN, marker = 'd', label = '$\mathtt{NN}$', color='gray', alpha = 0.5)

ax2.get_yaxis().set_visible(False)

lgd1 = ax.legend(loc=(0.01, 0.6))
lgd2 = ax2.legend(loc=(0.25, 0.75))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
if config['save']:  
    plt.savefig(f'{cd}//plots//adaptive_reg_trad_cost_CRPS_tradeoff_softmax.pdf', bbox_extra_artists=(lgd1,lgd2), bbox_inches='tight')
plt.show()
