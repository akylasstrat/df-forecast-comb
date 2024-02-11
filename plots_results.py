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
plt.rcParams['figure.dpi'] = 600

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

    params['start_date'] = '2012-01-01'
    params['split_date'] = '2013-01-01' # Defines train/test split
    params['end_date'] = '2013-12-30'
    
    params['save'] = True # If True, then saves models and results
    
    # Experimental setup parameters
    params['problem'] = 'reg_trad' # {mse, newsvendor, cvar, reg_trad}
    params['N_experts'] = 9
    params['iterations'] = 5
    params['target_zones'] = ['Z1', 'Z2', 'Z3', 'Z4', 'Z5',
                              'Z6', 'Z7', 'Z8', 'Z9', 'Z10']
    
    
    params['crit_quant'] = [0.8]
    params['risk_aversion'] = [0.5]
    
    # approaches to map data to decisions
    # LR: linear regression, DecComb: combination of perfect-foresight decisions (both maintain convexity)
    # DT: decision tree, NN: neural network (both results in MIPs)
    params['decision_rules'] = ['LR', 'JMBench', 'SalvaBench'] 

    return params

#%%%%%%%%%%%%%%%%%
# Results for a single run (fixed S)

config = params()
config['save'] = False
target_prob = 'reg_trad'
crit_fract = 0.9

decision_cost = []
qs_cost = []
for z in ['Z1', 'Z2', 'Z3']:
    decision_cost.append(pd.read_csv(f'{cd}\\results\\solar_different_prob_models\\{z}_{target_prob}__Decision_cost.csv', index_col = 0))
    qs_cost.append(pd.read_csv(f'{cd}\\results\\solar_different_prob_models\\{z}_{target_prob}__mean_QS.csv', index_col = 0))

decision_cost = pd.concat(decision_cost)
decision_cost.reset_index(inplace = True)
qs_cost = pd.concat(qs_cost)
qs_cost.reset_index(inplace = True)

#%%
gamma = [0, 0.1, 1]
static_models = ['knn', 'cart','cart_date', 'rf', 'Ave', 'Insample', 'SalvaBench', 'CRPS'] + [f'DF_{g}' for g in gamma]
solo_models = ['knn', 'cart','cart_date', 'rf']

fig, ax  = plt.subplots()

decision_cost.groupby('Quantile')[['SalvaBench','CRPS'] + [f'DF_{g}' for g in gamma]].mean().plot(kind = 'bar', ax=ax)
plt.ylim()
#%% Relative values compared to naive linear pooling (equal weights)
rel_cost = decision_cost.copy()
rel_cost[static_models] = (rel_cost['Ave'].values.reshape(-1,1)-rel_cost[static_models])/rel_cost['Ave'].values.reshape(-1,1)
#%%
fig, ax  = plt.subplots()
rel_cost.query(f'Target==3 and risk_aversion == 0.2').groupby(['Quantile'])[['CRPS'] + [f'DF_{g}' for g in gamma]].mean().plot(kind = 'bar', ax=ax)
plt.ylim()
#%%
fig, ax  = plt.subplots()
rel_cost.query(f'Target==3 and risk_aversion == 0.2').groupby(['Quantile'])[['CRPS'] + [f'DF_{g}' for g in gamma]].mean().plot(kind = 'bar', ax=ax)
plt.ylim()
#%%
rel_crps = qs_cost.copy()
rel_crps[static_models] = (rel_crps['Ave'].values.reshape(-1,1)-rel_crps[static_models])/rel_crps['Ave'].values.reshape(-1,1)

farm = [1,2,3]
rho = 0.2
models_plot = ['CRPS'] + [f'DF_{g}' for g in gamma]

temp_crps_df = rel_crps.query(f'Target=={farm} and risk_aversion == {rho}')
fig, ax  = plt.subplots()
temp_crps_df.groupby(['Quantile'])[models_plot].mean().plot(kind = 'bar', ax=ax)
plt.ylim([ temp_crps_df.groupby(['Quantile'])[models_plot].mean().values.min()-.01, 
          temp_crps_df.groupby(['Quantile'])[models_plot].mean().values.max()+.01])
plt.title('CRPS Improvement')
plt.show()

temp_cost_df = rel_cost.query(f'Target=={farm} and risk_aversion == {rho}')
fig, ax  = plt.subplots()
temp_cost_df.groupby(['Quantile'])[models_plot].mean().plot(kind = 'bar', ax=ax)
plt.ylim([ temp_cost_df.groupby(['Quantile'])[models_plot].mean().values.min()-.01, 
          temp_cost_df.groupby(['Quantile'])[models_plot].mean().values.max()+.01])
plt.title('Decision Cost Improvement')
plt.show()

#%%
color = ['tab:blue', 'tab:green', 'tab:brown', 'tab:orange', 'tab:purple', 'black']
marker = ['s', 'o', 'd', '+', '1', '2']
models_plot = ['CRPS'] + [f'DF_{g}' for g in gamma]
labels = ['$\mathtt{CRPSL}$'] + ['$\mathtt{DFL}-$0', '$\mathtt{DFL}-$0.1', '$\mathtt{DFL}-$1']

for i,m in enumerate(models_plot):
    plt.scatter(100*temp_cost_df[m].mean(), 100*temp_crps_df[m].mean(), c = color[i], label = labels[i], marker = marker[i])
plt.legend()    
plt.xlabel('Decision Cost Improvement (%)')
plt.ylabel('CRPS Improvement (%)')
plt.savefig(f'{cd}//plots//cost_CRPS_tradeoff.pdf')
plt.show()
#%%

fig, ax  = plt.subplots()
qs_cost.groupby(['Target', 'Quantile'])[solo_models].mean().plot(ax=ax)
plt.ylim()

#%%
decision_cost = decision_cost.dropna()
qs_cost = qs_cost.dropna()

single_models = [f'Model-{i}' for i in range(9)]
single_models = ['knn', 'cart', 'rf']
# extra column that contains the best-performing single expert
decision_cost['TopExpert'] = decision_cost[single_models].min(1)
qs_cost['TopExpert'] = qs_cost['knn']

static_models_all = single_models + ['TopExpert', 'Ave','Insample', 'SalvaBench', 'CRPS'] + [f'DF_{g}' for g in gamma]

static_models = ['TopExpert', 'Ave','Insample', 'SalvaBench', 'CRPS'] + [f'DF_{g}' for g in gamma]

adaptive_models_lr = ['CRPS-LR', 'SalvaBench-LR'] + [f'DF-LR_{g}' for g in gamma]
adaptive_models_mlp = ['CRPS-MLP', 'SalvaBench-MLP'] + [f'DF-MLP_{g}' for g in gamma]

#%%
fig, ax  = plt.subplots()

decision_cost[static_models + adaptive_models_mlp].mean().plot(yerr =  decision_cost[static_models + adaptive_models_mlp].std()/np.sqrt(9))

plt.xticks(rotation = 45)

for i in range(len(decision_cost)):
    col_ind = decision_cost[static_models].iloc[i].argmin()
    print(decision_cost[static_models].columns[col_ind])

#%%
mean_QS_df = decision_cost.copy()
import re

for m in static_models + adaptive_models_lr + adaptive_models_mlp:
    if m == 'TopExpert': continue
    for i in range(mean_QS_df[m].shape[0]):
        temp = re.findall(r'\d+.\d+', qs_cost[m].iloc[i])
        temp = np.array(temp).astype(float)
        mean_QS_df[m].iloc[i] = temp.mean()

mean_QS_df['TopExpert'] = mean_QS_df[single_models].min(1)