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
problem = 'newsvendor'
config['save'] = False
target_prob = 'reg_trad'
crit_fract = 0.9

decision_cost = pd.read_csv(f'{cd}\\results\\different_prob_models\\{target_prob}_{crit_fract}_Decision_cost.csv', index_col = 0)
qs_cost = pd.read_csv(f'{cd}\\results\\different_prob_models\\{target_prob}_{crit_fract}_QS.csv', index_col = 0)


#%%
decision_cost = decision_cost.dropna()
qs_cost = qs_cost.dropna()

gamma = [0, 0.1, 1]
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