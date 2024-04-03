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
    params['N_experts'] = 3
    params['iterations'] = 5
    params['target_zones'] = ['Z1', 'Z2', 'Z3']
    
    
    params['crit_quant'] = [0.8]
    params['risk_aversion'] = [0.5]
    
    # approaches to map data to decisions
    # LR: linear regression, DecComb: combination of perfect-foresight decisions (both maintain convexity)
    # DT: decision tree, NN: neural network (both results in MIPs)
    params['decision_rules'] = ['LR', 'JMBench', 'SalvaBench'] 

    return params

#%%%%%%%%%%%%%%%%%
# Results for a single run (fixed S)

'''
target_prob = 'reg_trad'

cols = ['Quantile', 'risk_aversion', 'Target', 'knn', 'cart', 'cart_date', 'rf', 'Ave',
 'Insample', 'SalvaBench',
 'CRPS', 'DF_0', 'DF_0.1', 'DF_1']
cols2 = ['risk_aversion', 'Target', 'knn', 'cart', 'cart_date', 'rf', 'Ave',
 'Insample', 'SalvaBench',
 'CRPS', 'DF_0', 'DF_0.1', 'DF_1']

for z in ['Z1', 'Z2', 'Z3']:
    
    decision_cost = pd.read_csv(f'{cd}\\results\\solar_different_prob_models\\{z}_{target_prob}__Decision_cost.csv', index_col = 0)
    qs_cost = pd.read_csv(f'{cd}\\results\\solar_different_prob_models\\{z}_{target_prob}__mean_QS.csv', index_col = 0)
    pinball_loss = pd.read_csv(f'{cd}\\results\\solar_different_prob_models\\{z}_{target_prob}__QS.csv', index_col = 0)
    pinball_loss['Quantile'] = decision_cost['Quantile']
    
    adapt_decision_cost = pd.read_csv(f'{cd}\\results\\solar_different_prob_models\\adaptive_{z}_{target_prob}__Decision_cost.csv', index_col = 0)
    adaptive_qs_cost = pd.read_csv(f'{cd}\\results\\solar_different_prob_models\\adaptive_{z}_{target_prob}__mean_QS.csv', index_col = 0)
    adaptive_pinball_loss = pd.read_csv(f'{cd}\\results\\solar_different_prob_models\\adaptive_{z}_{target_prob}__QS.csv', index_col = 0)
    adaptive_pinball_loss['Quantile'] = adapt_decision_cost['Quantile']

    decision_cost = decision_cost.merge(adapt_decision_cost, on = ['Quantile', 'risk_aversion', 'Target'], how = 'left')
    qs_cost = qs_cost.merge(adaptive_qs_cost, on = ['Quantile', 'risk_aversion', 'Target'], how = 'left')
    pinball_loss = pinball_loss.merge(adaptive_pinball_loss, on = ['Quantile', 'risk_aversion', 'Target'], how = 'left')
    
    decision_cost.to_csv(f'{cd}\\results\\solar_different_prob_models\\{z}_{target_prob}__Decision_cost.csv')
    qs_cost.to_csv(f'{cd}\\results\\solar_different_prob_models\\{z}_{target_prob}__mean_QS.csv')
    pinball_loss.to_csv(f'{cd}\\results\\solar_different_prob_models\\{z}_{target_prob}__QS.csv')
'''

#%% Grid scheduling results
config = params()
zone = 'Z1'
Cases = ['pglib_opf_case14_ieee.m', 'pglib_opf_case57_ieee.m']
case = Cases[1]

results_path = f'{cd}\\results\\grid_scheduling'

da_cost = pd.read_csv(f'{results_path}\\{zone}_{case}_wind_DA_cost.csv', index_col = 0)
rt_cost = pd.read_csv(f'{results_path}\\{zone}_{case}_wind_RT_cost.csv', index_col = 0)
crps = pd.read_csv(f'{results_path}\\{zone}_{case}_wind_mean_QS.csv', index_col = 0)
lambda_static = pd.read_csv(f'{results_path}\\{zone}_{case}_wind_lambda_static.csv', index_col = 0)

total_cost = da_cost + rt_cost

print(total_cost.mean().round(3))
print(crps.mean().round(3))

fig, ax  = plt.subplots()
lambda_static[['Ave', 'Insample', 'CRPS', 'DF_0', 'DF_0.1']].plot(kind='bar', ax = ax)
plt.xticks([0,1,2], ['$\mathtt{kNN}$', '$\mathtt{CART}$', '$\mathtt{RF}$'], rotation = 0)
plt.legend(['$\mathtt{OLP}$', '$\mathtt{invW}$', '$\mathtt{CRPSL}$', '$\mathtt{DFL}$-0', '$\mathtt{DFL}$-0.1'], ncol = 2)
plt.xlabel('Component forecasts')
plt.ylabel('Combination weights $\mathtt{\lambda}$')
plt.savefig(f'{cd}\\plots\\lambda_barplot_grid_sched.pdf')
plt.show()


#%% Trading results
config = params()
config['save'] = False
target_prob = 'reg_trad'
crit_fract = 0.9

decision_cost = []
qs_cost = []
for z in ['Z1','Z2', 'Z3']:
    decision_cost.append(pd.read_csv(f'{cd}\\results\\solar_new_results\\{z}_{target_prob}__Decision_cost.csv', index_col = 0))
    qs_cost.append(pd.read_csv(f'{cd}\\results\\solar_new_results\\{z}_{target_prob}__mean_QS.csv', index_col = 0))


decision_cost = pd.concat(decision_cost)
decision_cost.reset_index(inplace = True)
qs_cost = pd.concat(qs_cost)
qs_cost.reset_index(inplace = True)
#%% Plot lambdas for specific combination

lambda_static = pd.read_csv(f'{cd}\\results\\solar_new_results\\Z2_reg_trad__0.2_lambda_static.csv', index_col = 0)

fig, ax  = plt.subplots()
lambda_static[['Ave', 'Insample', 'CRPS', 'DF_0', 'DF_0.1', 'DF_1']].plot(kind='bar', ax = ax)
plt.xticks([0,1,2], ['$\mathtt{kNN}$', '$\mathtt{CART}$', '$\mathtt{RF}$'], rotation = 0)
plt.legend(['$\mathtt{OLP}$', '$\mathtt{invW}$', '$\mathtt{CRPSL}$', '$\mathtt{DFL}$-0', '$\mathtt{DFL}$-0.1', 
            '$\mathtt{DFL}$-1'], ncol = 2)
plt.xlabel('Component forecasts')
plt.ylabel('Combination weights $\mathtt{\lambda}$')
plt.savefig(f'{cd}\\plots\\lambda_barplot_trading.pdf')
plt.show()

#%%
gamma = [0, 0.1, 1]
static_models = ['knn','cart_date', 'rf', 'Ave', 'Insample', 'SalvaBench', 'CRPS'] + [f'DF_{g}' for g in gamma]
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
lgd2 = ax2.legend(loc=(0.25, 0.6))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

plt.savefig(f'{cd}//plots//adaptive_reg_trad_cost_CRPS_tradeoff.pdf', bbox_extra_artists=(lgd1,lgd2), bbox_inches='tight')
plt.show()

#%%
config = params()
target_prob = 'pwl'

decision_cost = []
qs_cost = []
for z in ['Z1', 'Z2']:
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
rel_cost.query(f'Target==2 and risk_aversion == 0.2').groupby(['Quantile'])[['CRPS'] + [f'DF_{g}' for g in gamma]].mean().plot(kind = 'bar', ax=ax)
plt.ylim()
#%%
fig, ax  = plt.subplots()
rel_cost.query(f'Target==2 and risk_aversion == 0.2').groupby(['Quantile'])[['CRPS'] + [f'DF_{g}' for g in gamma]].mean().plot(kind = 'bar', ax=ax)
plt.ylim()
#%%
rel_crps = qs_cost.copy()
rel_crps[static_models ] = (rel_crps['Ave'].values.reshape(-1,1)-rel_crps[static_models ])/rel_crps['Ave'].values.reshape(-1,1)
#%%
farm = [1,2]
rho = 0.2
models_plot = static_models

temp_crps_df = rel_crps.query(f'Target=={farm} and risk_aversion == {rho}')
temp_crps_df.groupby(['Quantile'])[models_plot].mean().plot(kind = 'bar', ax=ax)

temp_cost_df = rel_cost.query(f'Target=={farm} and risk_aversion == {rho}')
temp_cost_df.groupby(['Quantile'])[models_plot].mean().plot(kind = 'bar', ax=ax)


color = ['tab:blue', 'tab:green', 'tab:brown', 'tab:orange', 'tab:purple', 'black']
marker = ['s', 'o', 'd', '+', '1', '2']
models_plot = ['CRPS'] + [f'DF_{g}' for g in gamma] 
labels = ['$\mathtt{CRPSL}$'] + ['$\mathtt{DFL}$-0', '$\mathtt{DFL}$-0.1', '$\mathtt{DFL}$-1']

fig, ax  = plt.subplots()
for i,m in enumerate(models_plot):
    plt.scatter(100*temp_cost_df[m].mean(), 100*temp_crps_df[m].mean(), label = labels[i], marker = marker[i], color = color[i])
plt.legend()    
plt.xlabel('Decision Cost Improvement (%)')
plt.ylabel('CRPS Improvement (%)')
plt.savefig(f'{cd}//plots//pwl_cost_CRPS_tradeoff.pdf')
plt.show()
#%%

color = ['tab:blue', 'tab:green', 'tab:brown', 'tab:orange', 'tab:purple', 'black']
marker = ['s', 'o', 'd', '+', '1', '2']
lr_models_plot = ['CRPS-LR'] + [f'DF-LR_{g}' for g in gamma] 
mlp_models_plot = ['CRPS-MLP'] + [f'DF-MLP_{g}' for g in gamma] 
lr_labels = ['$\mathtt{CRPSL-LR}$'] + ['$\mathtt{DFL-LR}-$0', '$\mathtt{DFL-LR}-$0.1', '$\mathtt{DFL-LR}-$1']
mlp_labels = ['$\mathtt{CRPSL-MLP}$'] + ['$\mathtt{DFL-MLP}-$0', '$\mathtt{DFL-MLP}-$0.1', '$\mathtt{DFL-MLP}-$1']

fig, ax  = plt.subplots()

for i,m in enumerate(lr_models_plot):
    plt.scatter(100*temp_cost_df[m].mean(), 100*temp_crps_df[m].mean(), marker = 's', color = color[i])
for i,m in enumerate(mlp_models_plot):
    plt.scatter(100*temp_cost_df[m].mean(), 100*temp_crps_df[m].mean(), marker = 'd', color = color[i])
for i,m in enumerate(models_plot):
    ax.scatter(np.NaN, np.NaN, marker = 'o', label = labels[i], color=color[i])

#plt.legend()    

ax2 = ax.twinx()
ax2.scatter(np.NaN, np.NaN, marker = 's', label = 'Linear', color='gray', alpha = 0.5)
ax2.scatter(np.NaN, np.NaN, marker = 'd', label = 'NN', color='gray', alpha = 0.5)

ax2.get_yaxis().set_visible(False)

ax.legend(loc='upper left')
ax2.legend(loc='center left')

plt.xlabel('Decision Cost Improvement (%)')
plt.ylabel('CRPS Improvement (%)')
plt.savefig(f'{cd}//plots//adaptive_reg_trad_cost_CRPS_tradeoff.pdf')
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