# -*- coding: utf-8 -*-
"""
Grid scheduling example

@author: astratig
"""

import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import sys, os
#import pickle
import gurobipy as gp
import torch

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from matpowercaseframes import CaseFrames


cd = os.path.dirname(__file__)  #Current directory
sys.path.append(cd)

from gurobi_ml import add_predictor_constr
from sklearn.preprocessing import MinMaxScaler

from utility_functions import *
from optimal_transport_functions import *
from torch_layers_functions import *
from torch.utils.data import Dataset, DataLoader
import torch

# IEEE plot parameters (not sure about mathfont)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 600
plt.rcParams['figure.figsize'] = (3.5, 2) # Height can be changed
plt.rcParams['font.size'] = 7
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams["mathtext.fontset"] = 'dejavuserif'

def crps_learning_combination(target_y, prob_vectors, support = np.arange(0, 1.01, .01).round(2), verbose = 0):

    'Linear pool minimizing the CRPS, returns the combination weights'
    
    #### set optimization problem
    n_obs = prob_vectors[0].shape[0]
    n_locs = len(support)
    n_experts = len(prob_vectors)
    
    m = gp.Model()
    if verbose == 0: 
        m.setParam('OutputFlag', 0)
    m.setParam('BarHomogeneous', 1)
        
    # Decision variables
    lambdas = m.addMVar(n_experts, vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1)
    p_comb = m.addMVar(prob_vectors[0].shape, vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1)
    CDF_comb = m.addMVar(prob_vectors[0].shape, vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1)
    
    lambdas.Start = (1/n_experts)*np.ones(n_experts)
    
    crps_i = m.addMVar(n_obs, vtype = gp.GRB.CONTINUOUS, lb = 0)
    # Heavyside function for each observation
    H_i = 1*np.repeat(support.reshape(1,-1), n_obs, axis = 0) >= target_y.reshape(-1,1)

    # Linear pool
    m.addConstr( p_comb == sum([prob_vectors[i]*lambdas[i] for i in range(n_experts)]) )
    m.addConstr( lambdas.sum() == 1 )

    # PDF to CDF
    m.addConstrs( CDF_comb[:,j] == p_comb[:,:j+1].sum(1) for j in range(n_locs))
    print('check1')
    # CRPS for each observation i
    
    #m.addConstrs( crps_i[i] >= (CDF_comb[i] - H_i[i])@(CDF_comb[i] - H_i[i]) for i in range(n_obs))
    print('check2')
    
    #crps_i = (CDF_comb[i] - H_i[i])@(CDF_comb[i] - H_i[i])
    #print(crps_i)
    m.setObjective( sum([(CDF_comb[i] - H_i[i])@(CDF_comb[i] - H_i[i]) for i in range(n_obs)])/n_obs, gp.GRB.MINIMIZE)
    m.optimize()
    
    return lambdas.X

def tree_params():
    ''' Hyperparameters for tree algorithms'''
    params = {}
    params['n_estimators'] = 50
    params['n_min'] = 2
    params['max_features'] = 1
    return params

def insample_weight_tuning(grid, target_y, train_z_opt, 
                           support = np.arange(0, 1.01, .01).round(2), verbose = 0, **kwargs):
    ''' For each observation and each expert, solve the stochastic problem, find expected in-sample decision cost, 
        set weights based on inverse cost (or could use softmax activation)
        - Args:
            target_y: realizations of uncertainty
            prob_vectors: list of predictive PDFs
            crit_fract: critical fractile for newsvendor problem
            support: support locations
        - Output:
            lambdas_inv: weights based on inverse in-sample performance'''
    
    n_experts = len(train_z_opt)    
    risk_aversion = kwargs['risk_aversion']
    ### Find optimal decisions under perfect foresight information
    
    print('Estimate in-sample task loss...')
    #z_opt = np.zeros((n_obs, n_models))
    insample_cost = np.zeros((n_experts))
    insample_inverse_cost = np.zeros((n_experts))

    for j in range(n_experts):        
        insample_cost[j] = scheduling_task_loss(grid, train_z_opt[j], target_y.reshape(-1))
        insample_inverse_cost[j] = 1/insample_cost[j]

    lambdas_inv = insample_inverse_cost/insample_inverse_cost.sum()
    lambdas_softmax = np.exp(insample_inverse_cost)/sum(np.exp(insample_inverse_cost))

    return lambdas_inv, lambdas_softmax
                
def scheduling_task_loss(grid, da_prescriptions, actual, regularization = 0):
    'Estimates aggregated DA+RT cost'

    prescr_copy = da_prescriptions.copy()
    actual_copy = actual.copy().reshape(-1)
    n_samples = len(prescr_copy)
    Task_loss = []
    
    # Solve the RT scheduling problem    
    rt_sched = gp.Model()
    rt_sched.setParam('OutputFlag', 0)

    # DA Variables
    p_DA = rt_sched.addMVar((grid['n_unit']), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'p_G')
    
    # cost variables
    da_cost = rt_sched.addMVar((1), vtype = gp.GRB.CONTINUOUS, lb = 0)
    rt_cost = rt_sched.addMVar((1), vtype = gp.GRB.CONTINUOUS, lb =  -gp.GRB.INFINITY)

    # RT Variables
    w_rt = rt_sched.addMVar((1), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
    r_up = rt_sched.addMVar((grid['n_unit']), vtype = gp.GRB.CONTINUOUS, lb = 0)
    r_down = rt_sched.addMVar((grid['n_unit']), vtype = gp.GRB.CONTINUOUS, lb = 0)

    G_shed = rt_sched.addMVar((grid['n_unit']), vtype = gp.GRB.CONTINUOUS, lb = 0)
    L_shed = rt_sched.addMVar((grid['n_loads']), vtype = gp.GRB.CONTINUOUS, lb = 0)

    # Add generator constraints, DA & RT
    rt_sched.addConstr( p_DA + r_up <= grid['Pmax'].reshape(-1))    
    rt_sched.addConstr( r_down <= p_DA)    

    rt_sched.addConstr( r_up <= grid['R_u_max'].reshape(-1))
    rt_sched.addConstr( r_down <= grid['R_d_max'].reshape(-1))

    rt_sched.addConstr( G_shed <= grid['Pmax'].reshape(-1))
    rt_sched.addConstr( L_shed <= grid['Pd'].reshape(-1))
        
    # RT balancing constraint    
    rt_sched.addConstr( p_DA.sum() + r_up.sum() - r_down.sum() - G_shed.sum() + grid['w_capacity']*w_rt.sum() + L_shed.sum() == grid['Pd'].sum())
    
    # Expected decision cost for DA + RT scheduling
    rt_sched.addConstr( da_cost == grid['Cost']@p_DA)
    rt_sched.addConstr( rt_cost == (grid['C_up']-grid['Cost'])@r_up + (grid['Cost']-grid['C_down'])@r_down + grid['VOLL']*L_shed.sum() + grid['VOLL']*G_shed.sum())

    rt_sched.setObjective(rt_cost)
    
    for i in range(n_samples):

        # Solve DA market with predictions, save results
        c1 = rt_sched.addConstr(p_DA == prescr_copy[i])
        c2 = rt_sched.addConstr(w_rt == actual_copy[i])
        
        rt_sched.optimize()
        
        for c in [c1,c2]: rt_sched.remove(c)
        
        Task_loss.append(rt_sched.ObjVal)
        
    Task_loss = np.array(Task_loss)
    
    return Task_loss.mean()

def solve_stoch_sched(grid, scenarios, weights, regularization = 0):
    ''' Solves stochastic scheduling problem
        -Args
            scenarios: sampled scenarios (e.g., locations of PDF)
            weights: weight of each sceanrio (e.g., probability of each PDF location)
            problem: string that contains the problem description {mse, newsvendor, reg_trad, cvar}
            kwargs: additional arguments for each problem
        scenarios: support/ fixed locations
        weights: the learned probabilities'''
    
    if scenarios.ndim>1:
        target_scen = scenarios.copy().reshape(-1)
    else:
        target_scen = scenarios.copy()
        
    n_scen = len(target_scen)


    stoch_market = gp.Model()
    stoch_market.setParam('OutputFlag', 0)

    # DA Variables
    p_DA = stoch_market.addMVar((grid['n_unit']), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'p_G')
    slack_DA = stoch_market.addMVar((grid['n_loads']), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'slack')
    exp_w = stoch_market.addMVar((1), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
    # cost variables
    da_cost = stoch_market.addMVar((1), vtype = gp.GRB.CONTINUOUS, lb = 0)
    rt_cost = stoch_market.addMVar((n_scen), vtype = gp.GRB.CONTINUOUS, lb =  -gp.GRB.INFINITY)
    t_aux = stoch_market.addMVar((1), vtype = gp.GRB.CONTINUOUS, lb =  -gp.GRB.INFINITY)

    # RT Variables
    w_rt = stoch_market.addMVar((1, n_scen), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
    r_up = stoch_market.addMVar((grid['n_unit'], n_scen), vtype = gp.GRB.CONTINUOUS, lb = 0)
    r_down = stoch_market.addMVar((grid['n_unit'], n_scen), vtype = gp.GRB.CONTINUOUS, lb = 0)

    G_shed = stoch_market.addMVar((grid['n_unit'], n_scen), vtype = gp.GRB.CONTINUOUS, lb = 0)
    L_shed = stoch_market.addMVar((grid['n_loads'], n_scen), vtype = gp.GRB.CONTINUOUS, lb = 0)

    # Add generator constraints, DA & RT
    stoch_market.addConstrs( p_DA + r_up[:,s] <= grid['Pmax'].reshape(-1) for s in range(n_scen))
    stoch_market.addConstrs( r_down[:,s] <= p_DA for s in range(n_scen))
    
    stoch_market.addConstr( r_up <= np.tile(grid['R_u_max'].reshape(-1,1), n_scen))
    stoch_market.addConstr( r_down <= np.tile(grid['R_d_max'].reshape(-1,1), n_scen))

    stoch_market.addConstr( G_shed <= np.tile(grid['Pmax'].reshape(-1,1), n_scen))
    stoch_market.addConstr( L_shed <= np.tile(grid['Pd'].reshape(-1,1), n_scen))
        
    # DA balancing constraint
    #stoch_market.addConstr( p_DA.sum() + exp_w.sum() + slack_DA.sum() == grid['Pd'].sum())

    # RT balancing constraint
    #stoch_market.addConstrs( p_DA.sum() + exp_w.sum() + slack_DA.sum() == grid['Pd'].sum() for s in range(n_scen))
    
    stoch_market.addConstrs( p_DA.sum() + r_up[:,s].sum() - r_down[:,s].sum()
                            -G_shed[:,s].sum() + grid['w_capacity']*w_rt[:,s].sum() + L_shed[:,s].sum() == grid['Pd'].sum() for s in range(n_scen))

        

    # Expected decision cost for DA + RT scheduling
    stoch_market.addConstr( da_cost == grid['Cost']@p_DA + grid['VOLL']*slack_DA.sum())
    stoch_market.addConstrs( rt_cost[s] == (grid['C_up'])@r_up[:,s] + (-grid['C_down'])@r_down[:,s] + grid['VOLL']*L_shed[:,s].sum() + grid['VOLL']*G_shed[:,s].sum()
                            for s in range(n_scen))

    

    # solve for multiple test observations/ declares gurobi model once for speed up
    n_test_obs = len(weights)
    DA_prescriptions = np.zeros((n_test_obs, grid['n_unit']))

    stoch_market.addConstr( w_rt == target_scen)

    stoch_market.setObjective( da_cost + t_aux)

    
    for row in range(len(weights)):
        if row%1000 == 0: print(row)
        # expected renewable production
        c1 = stoch_market.addConstr( exp_w == target_scen@weights[row])
        c2 = stoch_market.addConstr( t_aux >= rt_cost@weights[row])
    
        stoch_market.optimize()
        
        for constr in [c1,c2]: stoch_market.remove(constr)        
        stoch_market.remove(c1)
        
        DA_prescriptions[row] = p_DA.X
        
    return DA_prescriptions
    
def grid_dict(path, save = False):
    ''' reads .m file with matpowercaseframes, returns dictionary with problem matrices'''

    matgrid = CaseFrames(path)
    # set cardinalities
    gen_mask = matgrid.gen.PMAX > 0
        
    num_nodes = len(matgrid.bus)
    num_lines = len(matgrid.branch)
    
    num_gen = len(matgrid.gen[gen_mask]) 
    num_load = len(matgrid.bus)  # assume demand at each node

    # Construct incidence matrix
    A = np.zeros((num_lines, num_nodes))
    
    for l in range(num_lines):
        temp_line = matgrid.branch.iloc[l]
        #A[l, temp_line['F_BUS'].astype(int)-1] = 1
        #A[l, temp_line['T_BUS'].astype(int)-1] = -1
        A[l, np.where(matgrid.bus.BUS_I == temp_line['F_BUS'])[0]] = 1
        A[l, np.where(matgrid.bus.BUS_I == temp_line['T_BUS'])[0]] = -1
        
    # Construct diagonal reactance matrix
    react = 1/matgrid.branch['BR_X'].values
    b_diag = np.diag(react)
    
    # Bus susceptance matrix
    B_susc = A.T@b_diag@A
    
    B_line = b_diag@A
    B_inv = np.zeros(B_susc.shape)
    B_inv[1:,1:] = np.linalg.inv(B_susc[1:,1:])
    PTDF = B_line@B_inv
    
    node_G = np.zeros((num_nodes, num_gen))
    #print(matgrid.gen)
    for i in range(len(matgrid.gen[gen_mask])):
        node_G[np.where(matgrid.bus.BUS_I == matgrid.gen[gen_mask].GEN_BUS.iloc[i])[0], i] = 1
        
    node_L = np.diag(np.ones(num_nodes))
    Line_cap = matgrid.branch.RATE_A.values
    
    grid = {}
    grid['Pd'] = matgrid.bus['PD'].values
    grid['Pmax'] = matgrid.gen['PMAX'].values[gen_mask]
    grid['Pmin'] = matgrid.gen['PMIN'].values[gen_mask]
    grid['Cost'] = matgrid.gencost['COST_1'].values[gen_mask]
    
    grid['Line_Capacity'] = Line_cap
    grid['node_G'] = node_G
    grid['node_L'] = node_L
    grid['B_susc'] = B_susc
    grid['A'] = A
    grid['b_diag'] = b_diag
    grid['B_line'] = B_line
    grid['PTDF'] = PTDF
    
    # Cardinality of sets
    grid['n_nodes'] = num_nodes
    grid['n_lines'] = num_lines
    grid['n_unit'] = num_gen
    grid['n_loads'] = num_load
    
    #Other parameters set by user
    grid['VOLL'] = 500   #Value of Lost Load
    grid['VOWS'] = 35   #Value of wind spillage
    grid['gshed'] = 200   #Value of wind spillage
    
    grid['B_line'] = grid['b_diag']@grid['A']
    B_inv = np.zeros(grid['B_susc'].shape)
    B_inv[1:,1:] = np.linalg.inv(grid['B_susc'][1:,1:])
    grid['PTDF'] = grid['B_line']@B_inv
    
    #if save:  
    #    pickle.dump(grid, open(cd+'\\data\\'+network.split('.')[0]+'.sav', 'wb'))
    return grid

def load_grid_data(case_name, pglib_path):
    case_name_prefix = case_name.split('.')[0]    
    #matgrid = CaseFrames(pglib_path + case)
    grid = grid_dict(pglib_path + case_name)
    
    print(case_name_prefix)

    grid['C_up'] = 1.8*grid['Cost']
    grid['C_down'] = 0.9*grid['Cost']
    grid['w_capacity'] = w_cap_dict[case_name]
    grid['w_bus'] = w_bus_dict[case_name]
    
    # wind incidence matrix
    grid['node_Wind'] = np.zeros((grid['n_nodes'], 1))
    grid['node_Wind'][grid['w_bus']] = 1
    
    
    R_u_max = np.ones(grid['n_unit'])
    R_u_max[grid['Cost'] < 10] = (grid['Pmax'][grid['Cost'] < 10]/5).round(2)
    R_u_max[ (grid['Cost'] >= 10)*(grid['Cost'] < 20) ] = ( grid['Pmax'][(grid['Cost'] >= 10)*(grid['Cost'] < 20)] /2 ).round(2)
    R_u_max[ grid['Cost'] >= 20 ] = grid['Pmax'][grid['Cost'] >= 20]
    
    R_d_max = R_u_max
    grid['VOLL'] = 200
    grid['R_u_max'] = R_u_max
    grid['R_d_max'] = R_d_max
    return grid

def nn_params():
    'NN hyperparameters'
    nn_params = {}
    nn_params['patience'] = 10
    nn_params['batch_size'] = 512  
    nn_params['num_epochs'] = 1500
    nn_params['learning_rate'] = 1e-2
    nn_params['apply_softmax'] = True
    return nn_params

def params():
    ''' Set up the experiment parameters'''

    params = {}

    params['start_date'] = '2012-01-01'
    params['split_date_prob'] = '2013-01-01' # Defines train/test split
    params['split_date_comb'] = '2014-01-01' # Defines train/test split
    params['end_date'] = '2014-07-01'
    
    params['save'] = True # If True, then saves models and results
    
    # Experimental setup parameters
    params['problem'] = 'sched' # {mse, newsvendor, cvar, reg_trad, pwl}
    params['gamma_list'] = [0, 0.1, 1]
    params['target_zone'] = [2]
    
    
    params['crit_quant'] = np.arange(0.1, 0.4, 0.1).round(2)
    params['risk_aversion'] = [0.2]

    params['train_static'] = True
    params['train_adaptive'] = False
    
    return params

#%%
config = params()
hyperparam = tree_params()
nn_hparam = nn_params()

results_path = f'{cd}\\results\\grid_scheduling'
data_path = f'{cd}\\data'
pglib_path =  'C:/Users/astratig/pglib-opf/'


aggr_df = pd.read_csv(f'{data_path}\\gefcom2014-solar.csv', index_col = 0, parse_dates=True)

# load grid data
Cases = ['pglib_opf_case14_ieee.m', 'pglib_opf_case57_ieee.m', 'pglib_opf_case118_ieee.m', 
         'pglib_opf_case24_ieee_rts.m', 'pglib_opf_case39_epri.m', 'pglib_opf_case73_ieee_rts.m']

w_bus = [13, 37, 36, 14, 5, 40]
w_bus_dict = {}

w_cap = [100, 600, 500, 1000, 1500, 1000]
w_cap_dict = {}

for i, case in enumerate(Cases):
    w_bus_dict[case] = w_bus[i]
    w_cap_dict[case] = w_cap[i]

target_case = Cases[3]
grid = load_grid_data(target_case, pglib_path)

#%% Data pre-processing
zone_target = config['target_zone']
aggr_df = aggr_df.query(f'ZONEID=={zone_target}')
target_problem = config['problem']
risk_aversion = config['risk_aversion']

filename_prefix = f'Z{zone_target[0]}_{target_case}_'
        
del aggr_df['ZONEID']

aggr_df['Hour'] = aggr_df.index.hour
aggr_df['Month'] = aggr_df.index.month

weather_dict = {'VAR78':'tclw', 'VAR79': 'tciw', 'VAR134':'SP', 'VAR157':'rh', 
                'VAR164':'tcc', 'VAR165':'10u', 'VAR166':'10v', 'VAR167':'2T', 
                'VAR169':'SSRD', 'VAR175':'STRD', 'VAR178':'TSR', 'VAR228':'TP'}

aggr_df = aggr_df.rename(columns = weather_dict)

#solar_df['diurnal_1'] = np.sin(2*np.pi*(solar_df.index.hour+1)/24)
#solar_df['diurnal_2'] = np.cos(2*np.pi*(solar_df.index.hour+1)/24)
#solar_df['diurnal_3'] = np.sin(4*np.pi*(solar_df.index.hour+1)/24)
#solar_df['diurnal_4'] = np.cos(4*np.pi*(solar_df.index.hour+1)/24)
#%%
for col in ['SSRD', 'STRD', 'TSR', 'TP']:
    if col != 'TP':
        aggr_df[col] = aggr_df[col].diff()/3600
        aggr_df[col][aggr_df[col]<0] = np.nan
    else:
        aggr_df[col] = aggr_df[col].diff()
        aggr_df[col][aggr_df[col]<0] = np.nan
        
aggr_df = aggr_df.interpolate()
aggr_df = aggr_df.dropna()


#target_scaler = MinMaxScaler()
#pred_scaler = MinMaxScaler()

#%%

aggr_df['Hour'] = aggr_df.index.hour
aggr_df['Month'] = aggr_df.index.month

# remove hours with zero production (timestamp is UTC/ plants are in Australia)
bool_ind = aggr_df.groupby('Hour').mean()['POWER'] == 0
zero_hour = bool_ind.index.values[bool_ind.values]
aggr_df = aggr_df.query(f'Hour < {zero_hour.min()} or Hour>{zero_hour.max()}')
#%%

#!!!! Add randomization based on the iteration counter here
critical_fractile = config['crit_quant'][0]

tuple_list = [tup for tup in itertools.product(zone_target, config['risk_aversion'])]

#%%

step = .01
y_supp = np.arange(0, 1+step, step).round(2)
nlocations = len(y_supp)
predictor_names = list(aggr_df.columns)
predictor_names.remove('POWER')

weather_variables = ['tclw','tciw','SP','rh','tcc','10u','10v','2T','SSRD','STRD','TSR','TP']
calendar_variables = ['Hour', 'Month']
all_variables = weather_variables + calendar_variables

#%%

feat_scaler = MinMaxScaler()
feat_scaler.fit(aggr_df[all_variables][config['start_date']:config['split_date_comb']])
aggr_df[all_variables] = feat_scaler.transform(aggr_df[all_variables])

### Create train/test sets for all series
trainY = aggr_df['POWER'][config['start_date']:config['split_date_prob']].round(2)
comb_trainY = aggr_df['POWER'][config['split_date_prob']:config['split_date_comb']].round(2)
testY = aggr_df['POWER'][config['split_date_comb']:].round(2)

trainX_weather = aggr_df[weather_variables][config['start_date']:config['split_date_prob']]
comb_trainX_weather = aggr_df[weather_variables][config['split_date_prob']:config['split_date_comb']]
testX_weather = aggr_df[weather_variables][config['split_date_comb']:]

trainX_date = aggr_df[calendar_variables][config['start_date']:config['split_date_prob']]
comb_trainX_date = aggr_df[calendar_variables][config['split_date_prob']:config['split_date_comb']]
testX_date = aggr_df[calendar_variables][config['split_date_comb']:]

n_obs = len(comb_trainY)
n_test_obs = len(testY)

#%%
### NN hyperparameters
patience = nn_hparam['patience']
batch_size = nn_hparam['batch_size']
num_epochs = nn_hparam['num_epochs']
learning_rate = nn_hparam['learning_rate']
apply_softmax = nn_hparam['apply_softmax']
row_counter = 0

#try:
#    Decision_cost = pd.read_csv(f'{results_path}\\{filename_prefix}_Decision_cost.csv', index_col = 0)
#    QS_df = pd.read_csv(f'{results_path}\\{filename_prefix}_QS.csv', index_col = 0)
#    mean_QS = pd.read_csv(f'{results_path}\\{filename_prefix}_mean_QS.csv', index_col = 0)
#except:
#    Decision_cost = pd.DataFrame()
#    QS_df = pd.DataFrame()
#    mean_QS = pd.DataFrame()

RT_cost = pd.DataFrame()
DA_cost = pd.DataFrame()

#QS_df = pd.DataFrame()
mean_QS = pd.DataFrame()

#%%

for tup in tuple_list[row_counter:]:

    target_zone = tup[0]    
    risk_aversion = tup[1]
    
    # !!! Add case here
    print(f'Quantile:{critical_fractile}, zone:{target_zone}')
    
    np.random.seed(1234)

    if row_counter == 0:        
        #### Train different probabilistic forecasting models/ only train for first iteration    
    
        # store predictions
        train_w_dict = {}
        test_w_dict = {}
        probabilistic_models = {}

        ## kNN
        parameters = {'n_neighbors':[5, 10, 50, 100]}
        
        # cross-validation for hyperparamter tuning and model training
        knn_model_cv = GridSearchCV(KNeighborsRegressor(), parameters)
        knn_model_cv.fit(trainX_weather, trainY.values)    
        best_n_neighbors = knn_model_cv.best_estimator_.get_params()['n_neighbors']
        # find the weights for training/ comb training/ test set
            
        train_w_dict['knn'] = knn_model_cv.best_estimator_.kneighbors_graph(comb_trainX_weather).toarray()*(1/best_n_neighbors)
        test_w_dict['knn'] = knn_model_cv.best_estimator_.kneighbors_graph(testX_weather).toarray()*(1/best_n_neighbors)
            
        # cross-validation for hyperparamter tuning and model training
        knn_model_cv = GridSearchCV(KNeighborsRegressor(), parameters)
        knn_model_cv.fit(trainX_weather, trainY.values)    
        best_n_neighbors = knn_model_cv.best_estimator_.get_params()['n_neighbors']
    
        probabilistic_models['knn'] = knn_model_cv.best_estimator_

        # CART 1: weather predictors
        cart_parameters = {'max_depth':[5, 10, 20, 50, 100], 'min_samples_leaf':[1, 2, 5, 10]}
        cart_model_cv = GridSearchCV(DecisionTreeRegressor(), cart_parameters)
        
        cart_model_cv.fit(trainX_weather, trainY.values)    
            
        cart_model = cart_model_cv.best_estimator_
        probabilistic_models['cart'] = cart_model_cv.best_estimator_
        
        train_w_dict['cart'] = cart_find_weights(trainX_weather, comb_trainX_weather, cart_model)
        test_w_dict['cart'] = cart_find_weights(trainX_weather, testX_weather, cart_model)
        
        #%%
        # CART 2: date predictors
        
        cart_parameters = {'max_depth':[5, 10, 50, 100], 'min_samples_leaf':[1, 2, 5, 10]}
        cart_model_cv = GridSearchCV(DecisionTreeRegressor(), cart_parameters)
        
        trainX_cart2 = trainX_date.copy()
        comb_trainX_cart2 = comb_trainX_date.copy()
        testX_cart2 = testX_date.copy()
        
        trainX_cart2 = aggr_df[calendar_variables+weather_variables][config['start_date']:config['split_date_prob']]
        comb_trainX_cart2 = aggr_df[calendar_variables+weather_variables][config['split_date_prob']:config['split_date_comb']]
        testX_cart2 = aggr_df[calendar_variables+weather_variables][config['split_date_comb']:]
                
        cart_model_cv.fit(trainX_cart2, trainY.values)    
            
        cart_model = cart_model_cv.best_estimator_
        
        probabilistic_models['cart_date'] = cart_model_cv.best_estimator_
        
        train_w_dict['cart_date'] = cart_find_weights(trainX_cart2, comb_trainX_cart2, cart_model)
        test_w_dict['cart_date'] = cart_find_weights(trainX_cart2, testX_cart2, cart_model)
        
        #%%
        # Random Forest
    
        rf_parameters = {'min_samples_leaf':[2, 5, 10],'n_estimators':[100], 
                      'max_features':[1, 2, 4, len(trainX_weather.columns)]}
    
        rf_model_cv = GridSearchCV(ExtraTreesRegressor(), rf_parameters)
        rf_model_cv.fit(trainX_weather, trainY.values)    
            
        rf_model = rf_model_cv.best_estimator_
        probabilistic_models['rf'] = rf_model_cv.best_estimator_
    
        knn_point_pred = knn_model_cv.best_estimator_.predict(testX_weather)
        rf_point_pred = rf_model.predict(testX_weather)
        cart_point_pred = cart_model.predict(testX_cart2)

        train_w_dict['rf'] = forest_find_weights(trainX_weather, comb_trainX_weather, rf_model)
        test_w_dict['rf'] = forest_find_weights(trainX_weather, testX_weather, rf_model)
        #%%
        #
        ## Climatology forecast
        #train_w_dict['clim'] = np.ones((comb_trainY.shape[0], trainY.shape[0]))*(1/len(trainY))
        #test_w_dict['clim'] = np.ones((testY.shape[0], trainY.shape[0]))*(1/len(trainY))
        
        # Translate weighted observations to discrete PDFs
        train_p_list = []
        test_p_list = []
    
        all_learners = list(train_w_dict.keys())
        
        for i, learner in enumerate(all_learners):
            if learner == 'clim':
                # find for a single row, then repeat for each observation
                temp_p = wemp_to_support(train_w_dict[learner][0:1], trainY.values, y_supp).repeat(len(train_w_dict[learner]),0)
                train_p_list.append(temp_p)
                
                temp_p = wemp_to_support(test_w_dict[learner][0:1], trainY.values, y_supp).repeat(len(test_w_dict[learner]),0)
                test_p_list.append(temp_p)
            else:            
                train_p_list.append(wemp_to_support(train_w_dict[learner], trainY.values, y_supp))
                test_p_list.append(wemp_to_support(test_w_dict[learner], trainY.values, y_supp))

        # estimate CRPS
        print('CRPS')
        for j, m in enumerate(all_learners): 
            temp_CDF = test_p_list[j].cumsum(1)
            H_i = 1*np.repeat(y_supp.reshape(1,-1), len(testY), axis = 0)>=testY.values.reshape(-1,1)
            
            CRPS = np.square(temp_CDF - H_i).mean()
    
            print(f'{m}:{CRPS}')
        #%%
        # estimate QS
        print('QS')
        target_quant = np.arange(.01, 1, .01)
        for j,m in enumerate(all_learners):
            temp_pdf = test_p_list[j]
    
            temp_q_forecast = np.array([inverted_cdf(target_quant, y_supp, temp_pdf[i]) for i in range(n_test_obs)])            
            temp_qs = 100*pinball(temp_q_forecast, testY.values, target_quant).round(4)
            print(m)
            plt.plot(temp_qs, label = m)
        #plt.plot(100*pinball(test_q_pred, testY[target_zone].values, target_quant).round(4), label = 'QR reg')
        plt.legend()
        plt.ylabel('Pinball loss')
        plt.xlabel('Quantile')
        plt.xticks(np.arange(10, 100, 10), np.arange(0.1, 1, .1).round(2))
        plt.show()
        #%%
        #% Visualize some prob. forecasts for sanity check
        #%
        # step 1: find inverted CDFs
        F_inv = [np.array([inverted_cdf([.05, .10, .90, .95] , trainY.values, train_w_dict[learner][i]) for i in range(500)]) 
                 for j,learner in enumerate(all_learners)]
        
        plt.plot(comb_trainY[200:250].values)
        for i, learner in enumerate(all_learners):
            #plt.fill_between(np.arange(100), F_inv[i][200:300,0], F_inv[i][200:300,-1], alpha = .3, color = 'red')
            plt.fill_between(np.arange(50), F_inv[i][200:250,0], F_inv[i][200:250,-1], alpha = .3, label = learner)
        plt.legend()
        plt.show()
        
        N_experts = len(all_learners)
    #%%
    train_targetY = comb_trainY.values.reshape(-1)
    
    # Supervised learning set as tensors for PyTorch
    valid_obs = 1000
    tensor_trainY = torch.FloatTensor(train_targetY[:-valid_obs])
    tensor_train_p = torch.FloatTensor(np.column_stack((train_p_list)))
    tensor_train_p_list = [torch.FloatTensor(train_p_list[i][:-valid_obs]) for i in range(N_experts)]

    tensor_validY = torch.FloatTensor(train_targetY[-valid_obs:])
    tensor_valid_p_list = [torch.FloatTensor(train_p_list[i][-valid_obs:]) for i in range(N_experts)]
        
    tensor_trainX = torch.FloatTensor(comb_trainX_date[:-valid_obs].values)
    tensor_validX = torch.FloatTensor(comb_trainX_date[-valid_obs:].values)
    tensor_testX = torch.FloatTensor(testX_date.values)
    
    train_data = torch.utils.data.TensorDataset(tensor_train_p_list[0], tensor_train_p_list[1], tensor_train_p_list[2], tensor_trainY)
    
    n_train_obs = len(train_targetY)
    n_test_obs = len(testY)
    
    
    trainZopt = N_experts*[np.zeros((n_train_obs, grid['n_unit']))]
    testZopt = N_experts*[np.zeros((n_test_obs, grid['n_unit']))]

    print('Finding optimal decisions in training set')
    for j in range(N_experts):
        temp_z_opt = solve_stoch_sched(grid, y_supp, train_p_list[j], regularization = risk_aversion)
        trainZopt[j] = temp_z_opt
        
    #%%
    ###########% Static forecast combinations
    lambda_static_dict = {}
        
    for i,learner in enumerate(all_learners):
        temp_ind = np.zeros(N_experts)
        temp_ind[i] = 1
        lambda_static_dict[f'{learner}'] = temp_ind
        
    lambda_static_dict['Ave'] = (1/N_experts)*np.ones(N_experts)
                
    
    # Set weights to in-sample performance
    lambda_tuned_inv, _ = insample_weight_tuning(grid, train_targetY, trainZopt, support = y_supp, risk_aversion = risk_aversion)
    
    # Benchmark/ Salva's suggestion/ weighted combination of in-sample optimal (stochastic) decisions
    #lambda_ = averaging_decisions(grid, train_targetY, trainZopt, support = y_supp, bounds = False, risk_aversion = risk_aversion)

    lambda_static_dict['Insample'] = lambda_tuned_inv    
    #lambda_static_dict['SalvaBench'] = lambda_

    #%%   
    #% CRPS learning
    
    train_data_loader = create_data_loader(tensor_train_p_list + [tensor_trainY], batch_size = batch_size)
    valid_data_loader = create_data_loader(tensor_valid_p_list + [tensor_validY], batch_size = batch_size)
    
    if row_counter == 0:
        #### CRPS minimization/ with torch layer
        lpool_crps_model = LinearPoolCRPSLayer(num_inputs=N_experts, support = torch.FloatTensor(y_supp),
                                               apply_softmax = True)
        optimizer = torch.optim.Adam(lpool_crps_model.parameters(), lr = learning_rate)
        lpool_crps_model.train_model(train_data_loader, optimizer, epochs = num_epochs, patience = patience, 
                                     projection = True)
        print(to_np(torch.nn.functional.softmax(lpool_crps_model.weights)))

        if apply_softmax:
            lambda_crps = to_np(torch.nn.functional.softmax(lpool_crps_model.weights))
        else:
            lambda_crps = to_np(lpool_crps_model.weights)
    
        
        #lambda_crps = crps_learning_combination(comb_trainY.values, train_p_list, support = y_supp, verbose = 1)
        
    lambda_static_dict['CRPS'] = lambda_crps
    
    #%%
    ##### Decision-focused combination for different values of gamma     
    from torch_layers_functions import * 
    
    for gamma in config['gamma_list']:
        
        lpool_sched_model = LinearPoolSchedLayer(num_inputs=N_experts, support = torch.FloatTensor(y_supp), 
                                                 grid = grid, gamma = gamma, apply_softmax = True, clearing_type = 'det')
        
        optimizer = torch.optim.Adam(lpool_sched_model.parameters(), lr = learning_rate)
        
        lpool_sched_model.train_model(train_data_loader, valid_data_loader, optimizer, epochs = num_epochs, 
                                          patience = patience, projection = False, validation = False, relative_tolerance = 1e-5)
        if apply_softmax:
            lambda_static_dict[f'DF_{gamma}'] = to_np(torch.nn.functional.softmax(lpool_sched_model.weights))
        else:
            lambda_static_dict[f'DF_{gamma}'] = to_np(lpool_sched_model.weights)
            
        if config['save']:
            #Prescriptions.to_csv(f'{results_path}\\{target_problem}_{critical_fractile}_{target_zone}_Prescriptions.csv')
            lamda_static_df = pd.DataFrame.from_dict(lambda_static_dict)
            lamda_static_df.to_csv(f'{results_path}\\{filename_prefix}_lambda_static.csv')

#%%
    for m in list(lambda_static_dict.keys())[N_experts:]:
        plt.plot(lambda_static_dict[m], label = m)
    plt.legend()
    plt.show()
    
        
    ### Adaptive combination model    
    # i) fix val_loader, ii) train for gamma = 0.1, iii) add to dictionary
    # iv) create one for CRPS only (gamma == +inf)
    
    train_adapt_data_loader = create_data_loader(tensor_train_p_list + [tensor_trainX, tensor_trainY], batch_size = batch_size)
    valid_adapt_data_loader = create_data_loader(tensor_valid_p_list + [tensor_validX, tensor_validY], batch_size = batch_size)
            
    #tensor_trainZopt = torch.FloatTensor(trainZopt[:-valid_obs])
    #tensor_validZopt = torch.FloatTensor(trainZopt[-valid_obs:])
    #tensor_testZopt = torch.FloatTensor(testZopt)
    
    adaptive_models_dict = {}
    
    adapt_models = False
    #!!!! Update these
    if adapt_models:
        ### CRPS/ Linear Regression
        lr_lpool_crps_model = AdaptiveLinearPoolCRPSLayer(input_size = tensor_trainX.shape[1], hidden_sizes = [], output_size = N_experts, 
                                                          support = torch.FloatTensor(y_supp))        
        optimizer = torch.optim.Adam(lr_lpool_crps_model.parameters(), lr = learning_rate)        
        lr_lpool_crps_model.train_model(train_adapt_data_loader, valid_adapt_data_loader, optimizer, epochs = num_epochs, 
                                          patience = patience, projection = False)
    
        adaptive_models_dict['CRPS-LR'] = lr_lpool_crps_model
    
        ### CRPS/ MLP learner
        
        mlp_lpool_crps_model = AdaptiveLinearPoolCRPSLayer(input_size = tensor_trainX.shape[1], hidden_sizes = [20, 20, 20], output_size = N_experts, support = torch.FloatTensor(y_supp))        
        optimizer = torch.optim.Adam(mlp_lpool_crps_model.parameters(), lr = learning_rate)        
        mlp_lpool_crps_model.train_model(train_adapt_data_loader, valid_adapt_data_loader, optimizer, epochs = num_epochs, 
                                          patience = patience, projection = False)
    
        adaptive_models_dict['CRPS-MLP'] = mlp_lpool_crps_model
    
        ### Decision Combination/ LR
        train_dec_data_loader = create_data_loader([tensor_trainZopt, tensor_trainX, tensor_trainY], batch_size = batch_size)
        valid_dec_data_loader = create_data_loader([tensor_validZopt, tensor_validX, tensor_validY], batch_size = batch_size)
    
    
        lr_lpool_decision_model = AdaptiveLinearPoolDecisions(input_size = tensor_trainX.shape[1], hidden_sizes = [], output_size = N_experts, support = torch.FloatTensor(y_supp))        
        optimizer = torch.optim.Adam(lr_lpool_decision_model.parameters(), lr = 1e-3)        
        lr_lpool_decision_model.train_model(train_dec_data_loader, valid_dec_data_loader, optimizer, epochs = num_epochs, 
                                          patience = patience, projection = False)
    
        adaptive_models_dict['SalvaBench-LR'] = lr_lpool_decision_model
    
        mlp_lpool_decision_model = AdaptiveLinearPoolDecisions(input_size = tensor_trainX.shape[1], hidden_sizes = [20, 20, 20], 
                                                               output_size = N_experts, support = torch.FloatTensor(y_supp))        
        optimizer = torch.optim.Adam(mlp_lpool_decision_model.parameters(), lr = 1e-3)        
        mlp_lpool_decision_model.train_model(train_dec_data_loader, valid_dec_data_loader, optimizer, epochs = num_epochs, 
                                          patience = patience, projection = False)
    
        adaptive_models_dict['SalvaBench-MLP'] = mlp_lpool_decision_model
        
        #%
        for gamma in config['gamma_list']:
                        
            lr_lpool_newsv_model = AdaptiveLinearPoolNewsvendorLayer(input_size = tensor_trainX.shape[1], hidden_sizes = [], 
                                                                     output_size = N_experts, support = torch.FloatTensor(y_supp), gamma = gamma, critic_fract = critical_fractile, 
                                                                     risk_aversion = risk_aversion, apply_softmax = True, regularizer=None)
            
            optimizer = torch.optim.Adam(lr_lpool_newsv_model.parameters(), lr = 0.5e-2)
            lr_lpool_newsv_model.train_model(train_adapt_data_loader, valid_adapt_data_loader, 
                                                  optimizer, epochs = 1000, patience = patience, projection = False, relative_tolerance = 0)
            
            
            adaptive_models_dict[f'DF-LR_{gamma}'] = lr_lpool_newsv_model
    
            mlp_lpool_newsv_model = AdaptiveLinearPoolNewsvendorLayer(input_size = tensor_trainX.shape[1], hidden_sizes = [20,20,20], 
                                                                     output_size = N_experts, support = torch.FloatTensor(y_supp), 
                                                                     gamma = gamma, critic_fract = critical_fractile, risk_aversion = risk_aversion, apply_softmax = True, regularizer=None)
            
            optimizer = torch.optim.Adam(mlp_lpool_newsv_model.parameters(), lr = 0.5e-2)
            mlp_lpool_newsv_model.train_model(train_adapt_data_loader, valid_adapt_data_loader, 
                                                  optimizer, epochs = 1000, patience = patience, projection = False, relative_tolerance = 0)
    
            adaptive_models_dict[f'DF-MLP_{gamma}'] = mlp_lpool_newsv_model
            
    #%%
        lamda_static_df = pd.read_csv(f'{results_path}\\{filename_prefix}_lambda_static.csv', index_col = 0)
    lambda_static_dict = lamda_static_df.to_dict('list')
    adaptive_models_dict = {}
    #%%
    #% Evaluate performance for all models
    #%
    static_models = list(lambda_static_dict) 
    adaptive_models = list(adaptive_models_dict.keys())
    all_models = static_models + adaptive_models

    lambda_adapt_dict = {}
    Prescriptions = {}
    
    #pd.DataFrame(data = np.zeros((n_test_obs, len(all_models))), columns = all_models)
    
    # Store pinball loss and Decision cost for task-loss
    #temp_QS = pd.DataFrame()
    #temp_QS['Target'] = [target_zone]
    #temp_QS['risk_aversion'] = risk_aversion
    
    temp_RT_cost = pd.DataFrame()
    temp_RT_cost['risk_aversion'] = [risk_aversion]
    temp_RT_cost['Target'] = [target_zone]
    temp_DA_cost = temp_RT_cost.copy()
    temp_mean_QS = temp_RT_cost.copy()

    target_quant = np.arange(0.1, 1, 0.1).round(2)
    print('Estimating out-of-sample performance...')
    for j, m in enumerate(all_models):
        print(m)
        if m in static_models:                
            # Combine PDFs for each observation
            temp_pdf = sum([lambda_static_dict[m][j]*test_p_list[j] for j in range(N_experts)])            
        elif m in adaptive_models:
            # For each model, predict combination weights
            lambda_adapt_dict[m] = adaptive_models_dict[m].predict_weights(tensor_testX)
            # Combine PDFs for each observation
            temp_pdf = np.array([sum([lambda_adapt_dict[m][i,j]*test_p_list[j][i] for j in range(N_experts)]) for i in range(n_test_obs)])    
          
        temp_prescriptions = solve_stoch_sched(grid, y_supp, temp_pdf, regularization = risk_aversion)

        Prescriptions[m] = temp_prescriptions
        print(m)
            
        # Estimate task-loss for specific model
        #%
        
        temp_RT_cost[m] = scheduling_task_loss(grid, Prescriptions[m], testY.values)
        temp_DA_cost[m] = (Prescriptions[m]@grid['Cost']).mean()
        #%
        print(m)
        
        # Evaluate QS (approximation of CRPS) for each model
        # find quantile forecasts
        temp_q_forecast = np.array([inverted_cdf(target_quant, y_supp, temp_pdf[i]) for i in range(n_test_obs)])            
        temp_qs = 100*pinball(temp_q_forecast, testY.values, target_quant).round(4)
        print(m)

        #temp_QS[m] = [temp_qs]
        
        #temp_CDF = temp_pdf.cumsum(1)
        #H_i = 1*np.repeat(y_supp.reshape(1,-1), len(testY), axis = 0)>=testY.values.reshape(-1,1)
        #CRPS = 100*np.square(temp_CDF - H_i).mean()            

        temp_mean_QS[m] = temp_qs.mean()

    #    if m in ['Ave', 'SalvaBench', 'CRPS', 'DF_0.1', 'DF_1']:
    #        plt.plot(temp_qs, label = m)
    #plt.legend()
    #plt.ylabel('Pinball loss')
    #plt.xticks(np.arange(len(target_quant)), target_quant)
    #plt.xlabel('Quantile')
    #plt.show()
    #%
    print('DA Decision Cost')
    print(temp_DA_cost[all_models].mean().round(4))

    print('RT Decision Cost')
    print(temp_RT_cost[all_models].mean().round(4))

    print('Total Decision Cost')
    print((temp_DA_cost + temp_RT_cost)[all_models].mean().round(4))
    
    print('CRPS')
    print(temp_mean_QS[all_models].mean().round(4))
    
    
    try:
        DA_cost = pd.concat([DA_cost, temp_DA_cost], ignore_index = True)            
        RT_cost = pd.concat([RT_cost, temp_RT_cost], ignore_index = True)            
        mean_QS = pd.concat([mean_QS, temp_mean_QS], ignore_index = True)        
    except:
        DA_cost = temp_DA_cost.copy()
        RT_cost = temp_RT_cost.copy()
        mean_QS = temp_mean_QS.copy()

    if config['save']:
        DA_cost.to_csv(f'{results_path}\\{filename_prefix}_DA_cost.csv')
        RT_cost.to_csv(f'{results_path}\\{filename_prefix}_RT_cost.csv')
        #QS_df.to_csv(f'{results_path}\\{filename_prefix}_QS.csv')
        mean_QS.to_csv(f'{results_path}\\{filename_prefix}_mean_QS.csv')

        #Prescriptions.to_csv(f'{results_path}\\{target_problem}_{critical_fractile}_{target_zone}_Prescriptions.csv')
        lamda_static_df = pd.DataFrame.from_dict(lambda_static_dict)
        lamda_static_df.to_csv(f'{results_path}\\{filename_prefix}_lambda_static.csv')

    row_counter += 1        