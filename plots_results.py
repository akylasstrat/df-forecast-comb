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

#%% Grid scheduling results
config = params()
zone = 'Z2' # do not change this
Cases = ['pglib_opf_case14_ieee.m', 'pglib_opf_case57_ieee.m']
case = Cases[1]

results_path = f'{cd}\\results\\grid_scheduling'

da_cost = pd.read_csv(f'{results_path}\\NEWRESULTS_{zone}_{case}_wind_DA_cost.csv', index_col = 0)
rt_cost = pd.read_csv(f'{results_path}\\NEWRESULTS_{zone}_{case}_wind_RT_cost.csv', index_col = 0)
crps = pd.read_csv(f'{results_path}\\NEWRESULTS_{zone}_{case}_wind_mean_QS.csv', index_col = 0)
lambda_static = pd.read_csv(f'{results_path}\\NEWRESULTS_{zone}_{case}_wind_lambda_static.csv', index_col = 0)

total_regret = da_cost + rt_cost - da_cost['Perfect'].values[0]

print(total_regret.mean().round(3))
print(crps.mean().round(3))

fig, ax  = plt.subplots()
lambda_static[['Ave', 'Insample', 'CRPS', 'DF_0.001', 'DF_0.01']].plot(kind='bar', ax = ax)
plt.xticks([0,1,2], ['$\mathtt{kNN}$', '$\mathtt{CART}$', '$\mathtt{RF}$'], rotation = 0)
plt.legend(['$\mathtt{OLP}$', '$\mathtt{invW}$', '$\mathtt{CRPSL}$', '$\mathtt{DFL}$-0.001', '$\mathtt{DFL}$-0.01'], ncol = 2)
plt.xlabel('Component forecasts')
plt.ylabel('Combination weights $\mathtt{\lambda}$')
#plt.savefig(f'{cd}\\plots\\lambda_barplot_wind_grid_sched.pdf')
plt.show()


#%% Trading results
config = params()
config['save'] = False
target_prob = 'reg_trad'
#crit_fract = 0.9
results_folder = 'solar_trading_results_projection'

decision_cost = []
qs_cost = []
for z in ['Z1','Z2', 'Z3']:
    decision_cost.append(pd.read_csv(f'{cd}\\results\\{results_folder}\\{z}_{target_prob}__Decision_cost.csv', index_col = 0))
    qs_cost.append(pd.read_csv(f'{cd}\\results\\{results_folder}\\{z}_{target_prob}__mean_QS.csv', index_col = 0))


decision_cost = pd.concat(decision_cost)
decision_cost.reset_index(inplace = True)
qs_cost = pd.concat(qs_cost)
qs_cost.reset_index(inplace = True)
#%%
target = 'Z3'

cost_df = pd.read_csv(f'{cd}\\results\\solar_trading_results_projection\\{target}_{target_prob}__Decision_cost.csv', index_col = 0)
crps_df = pd.read_csv(f'{cd}\\results\\solar_trading_results_projection\\{target}_{target_prob}__mean_QS.csv', index_col = 0)

cost_softmax = pd.read_csv(f'{cd}\\results\\solar_trading_results_softmax\\{target}_{target_prob}__Decision_cost.csv', index_col = 0)
crps_softmax = pd.read_csv(f'{cd}\\results\\solar_trading_results_softmax\\{target}_{target_prob}__mean_QS.csv', index_col = 0)

for s in ['CRPS', 'DF_0', 'DF_0.1', 'DF_1']:
    cost_df[f'{s}_softmax'] = cost_softmax[s]
    crps_df[f'{s}_softmax'] = crps_softmax[s]

#%% Plot lambdas for specific combination

lambda_static = pd.read_csv(f'{cd}\\results\\{results_folder}\\Z2_reg_trad__0.2_lambda_static.csv', index_col = 0)

fig, ax  = plt.subplots()
lambda_static[['Ave', 'invW-0', 'CRPS', 'DF_0', 'DF_0.1', 'DF_1']].plot(kind='bar', ax = ax)
plt.xticks([0,1,2], ['$\mathtt{kNN}$', '$\mathtt{CART}$', '$\mathtt{RF}$'], rotation = 0)
plt.legend(['$\mathtt{OLP}$', '$\mathtt{invW}$', '$\mathtt{CRPSL}$', '$\mathtt{DFL}$-0', '$\mathtt{DFL}$-0.1', 
            '$\mathtt{DFL}$-1'], ncol = 2)
plt.xlabel('Component forecasts')
plt.ylabel('Combination weights $\mathtt{\lambda}$')
if config['save']:  plt.savefig(f'{cd}\\plots\\lambda_barplot_trading.pdf')
plt.show()

#%%
gamma = [0, 0.1, 1]
static_models = ['knn','cart_date', 'rf', 'Ave', 'invW-0', 'SalvaBench', 'CRPS'] + [f'DF_{g}' for g in gamma]
adaptive_models = ['CRPS-LR', 'CRPS-MLP', 'SalvaBench-LR','SalvaBench-MLP', 'DF-LR_0', 'DF-MLP_0', 'DF-LR_0.1', 'DF-MLP_0.1','DF-LR_1', 'DF-MLP_1']
solo_models = ['knn', 'cart','cart_date', 'rf']

fig, ax  = plt.subplots()

decision_cost.groupby('Quantile')[['SalvaBench','CRPS'] + [f'DF_{g}' for g in gamma]].mean().plot(kind = 'bar', ax=ax)
plt.ylim()
#%% Relative values compared to naive linear pooling (equal weights)
rel_cost = decision_cost.copy()
rel_cost[static_models + adaptive_models] = (rel_cost['Ave'].values.reshape(-1,1)-rel_cost[static_models + adaptive_models])/rel_cost['Ave'].values.reshape(-1,1)
#%%
fig, ax  = plt.subplots()
rel_cost.query(f'Target==2 and risk_aversion == 0.2').groupby(['Quantile'])[['CRPS'] + [f'DF_{g}' for g in gamma]].mean().plot(kind = 'bar', ax=ax)
plt.ylim()
#%%
fig, ax  = plt.subplots()
rel_cost.query(f'Target==2 and risk_aversion == 0.2').groupby(['Quantile'])[['CRPS'] + [f'DF_{g}' for g in gamma]].mean().plot(kind = 'bar', ax=ax)
plt.ylim()
#%%
rel_crps = qs_cost.copy()
rel_crps[static_models + adaptive_models] = (rel_crps['Ave'].values.reshape(-1,1)-rel_crps[static_models + adaptive_models])/rel_crps['Ave'].values.reshape(-1,1)
#%%
farm = [1, 2, 3]
rho = 0.2
models_plot = static_models

temp_crps_df = rel_crps.query(f'Target=={farm} and risk_aversion == {rho}')

temp_crps_df.groupby(['Quantile'])[models_plot].mean().plot(kind = 'bar', ax=ax)

fig, ax  = plt.subplots()
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
#models_plot = ['CRPS-LR'] + [f'DF-LR_{g}' for g in gamma] 
#models_plot = ['CRPS-MLP'] + [f'DF-MLP_{g}' for g in gamma] 
labels = ['$\mathtt{CRPSL}$'] + ['$\mathtt{DFL}$-0', '$\mathtt{DFL}$-0.1', '$\mathtt{DFL}$-1']

for i,m in enumerate(models_plot):
    plt.scatter(100*temp_cost_df[m].mean(), 100*temp_crps_df[m].mean(), label = labels[i], marker = marker[i], color = color[i])
plt.legend()    
plt.xlabel('Decision Cost Improvement (%)')
plt.ylabel('CRPS Improvement (%)')
#plt.savefig(f'{cd}//plots//reg_trad_cost_CRPS_tradeoff.pdf')
plt.show()
#%%

color = ['tab:blue', 'tab:green', 'tab:brown', 'tab:orange', 'tab:purple', 'black']
marker = ['s', 'o', 'd', '+', '1', '2']
lr_models_plot = ['CRPS-LR'] + [f'DF-LR_{g}' for g in gamma] 
mlp_models_plot = ['CRPS-MLP'] + [f'DF-MLP_{g}' for g in gamma] 
lr_labels = ['$\mathtt{CRPSL-LR}$'] + ['$\mathtt{DFL-LR}-$0', '$\mathtt{DFL-LR}-$0.1', '$\mathtt{DFL-LR}-$1']
mlp_labels = ['$\mathtt{CRPSL-MLP}$'] + ['$\mathtt{DFL-MLP}-$0', '$\mathtt{DFL-MLP}-$0.1', '$\mathtt{DFL-MLP}-$1']

fig, ax  = plt.subplots()

plt.xlabel('Cost improvement over $\mathtt{OLP}$ (%)')
plt.ylabel('CRPS improvement over $\mathtt{OLP}$ (%)')

for i,m in enumerate(lr_models_plot):
    plt.scatter(100*temp_cost_df[m].mean(), 100*temp_crps_df[m].mean(), marker = 's', color = color[i])
for i,m in enumerate(mlp_models_plot):
    plt.scatter(100*temp_cost_df[m].mean(), 100*temp_crps_df[m].mean(), marker = 'd', color = color[i])
for i,m in enumerate(models_plot):
    ax.scatter(100*temp_cost_df[m].mean(), 100*temp_crps_df[m].mean(), marker = 'o', label = labels[i], color=color[i])

#plt.legend()    

ax2 = ax.twinx()
ax2.scatter(np.NaN, np.NaN, marker = 's', label = '$\mathtt{LR}$', color='gray', alpha = 0.5)
ax2.scatter(np.NaN, np.NaN, marker = 'd', label = '$\mathtt{NN}$', color='gray', alpha = 0.5)

ax2.get_yaxis().set_visible(False)

lgd1 = ax.legend(loc=(0.01, 0.6))
lgd2 = ax2.legend(loc=(0.25, 0.75))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
if config['save']:  plt.savefig(f'{cd}//plots//adaptive_reg_trad_cost_CRPS_tradeoff_projection.pdf', bbox_extra_artists=(lgd1,lgd2), bbox_inches='tight')
plt.show()
