# -*- coding: utf-8 -*-
"""
Wind production/ Grid scheduling example/ New test case

@author: astratig
"""

import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import sys, os
import pickle
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

def insample_weight_tuning(grid, target_y, train_z_opt, train_prob_list, 
                           support = np.arange(0, 1.01, .01).round(2), regularization_gamma = 0, verbose = 0, **kwargs):
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
    # risk_aversion = kwargs['risk_aversion']
    ### Find optimal decisions under perfect foresight information
    
    print('Estimate in-sample task loss...')
    #z_opt = np.zeros((n_obs, n_models))
    insample_da_cost = np.zeros((n_experts))
    insample_rt_cost = np.zeros((n_experts))
    in_sample_CRPS = np.zeros((n_experts))

    insample_cost_regularized = np.zeros((n_experts))
    insample_inverse_cost = np.zeros((n_experts))

    for j in range(n_experts):        
        # In-sample scheduling costs
        temp_da_cost, temp_rt_cost = scheduling_task_loss(grid, train_z_opt[j], target_y.reshape(-1))
        insample_da_cost[j] = temp_da_cost
        insample_rt_cost[j] = temp_rt_cost
        
        # Estimate in-sample CRPS        
        temp_CDF = train_prob_list[j].cumsum(1)
        H_i = 1*np.repeat(y_supp.reshape(1,-1), len(comb_trainY), axis = 0)>=comb_trainY.values.reshape(-1,1)
        in_sample_CRPS[j] =  np.square(temp_CDF - H_i).mean()

        # In-sample regularized cost
        if regularization_gamma =='inf':
            insample_cost_regularized[j] = in_sample_CRPS[j]
        else:
            insample_cost_regularized[j] = (insample_da_cost[j] + insample_rt_cost[j]) + regularization_gamma*in_sample_CRPS[j]            

        insample_inverse_cost[j] = 1/insample_cost_regularized[j]

    lambdas_inv = insample_inverse_cost/insample_inverse_cost.sum()
    lambdas_softmax = np.exp(insample_inverse_cost)/sum(np.exp(insample_inverse_cost))

    return lambdas_inv, lambdas_softmax

def perfect_scheduling_cost(grid, actual, regularization = 0):
    'Estimates the dispatch cost of the perfect foresight solution'

    actual_copy = actual.copy().reshape(-1)
    n_samples = len(actual)
    Task_loss = []
    
    # Solve the RT scheduling problem    
    model = gp.Model()
    model.setParam('OutputFlag', 0)

    # DA Variables
    p_DA = model.addMVar((grid['n_unit']), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'p_G')
    
    # cost variables
    da_cost = model.addMVar((1), vtype = gp.GRB.CONTINUOUS, lb = 0)

    # RT Variables
    w_rt = model.addMVar((1), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)

    # Add generator constraints, DA & RT
    model.addConstr( p_DA <= grid['Pmax'].reshape(-1))    
        
    # RT balancing constraint    
    model.addConstr( p_DA.sum() + grid['w_capacity']*w_rt.sum() == grid['Pd'].sum())
    
    # Expected decision cost for DA + RT scheduling
    model.addConstr( da_cost == grid['Cost']@p_DA)

    model.setObjective(da_cost)
    
    for i in range(n_samples):

        # Solve DA market with predictions, save results
        c2 = model.addConstr(w_rt == actual_copy[i])
        model.optimize()
        for c in [c2]: model.remove(c)
        
        Task_loss.append(model.ObjVal)
        
    Task_loss = np.array(Task_loss)
    return Task_loss.mean()

def scheduling_task_loss(grid, da_prescriptions, actual, regularization = 0):
    'Estimates aggregated DA+RT cost'

    prescr_copy = da_prescriptions.copy()
    actual_copy = actual.copy().reshape(-1)
    n_samples = len(prescr_copy)
    Task_loss = []
    DA_loss = []
    RT_loss = []
    # Solve the RT scheduling problem    
    rt_sched = gp.Model()
    rt_sched.setParam('OutputFlag', 0)

    # DA Variables
    p_DA = rt_sched.addMVar((grid['n_unit']), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'p_G')
    
    # cost variables
    da_cost = rt_sched.addMVar((1), vtype = gp.GRB.CONTINUOUS, lb = 0)
    rt_cost = rt_sched.addMVar((1), vtype = gp.GRB.CONTINUOUS, lb =  -gp.GRB.INFINITY)

    # RT Variables
    w_rt = rt_sched.addMVar((1), vtype = gp.GRB.CONTINUOUS, lb = 0)
    r_up = rt_sched.addMVar((grid['n_unit']), vtype = gp.GRB.CONTINUOUS, lb = 0)
    r_down = rt_sched.addMVar((grid['n_unit']), vtype = gp.GRB.CONTINUOUS, lb = 0)

    G_shed = rt_sched.addMVar((1), vtype = gp.GRB.CONTINUOUS, lb = 0)
    L_shed = rt_sched.addMVar((1), vtype = gp.GRB.CONTINUOUS, lb = 0)

    # Add generator constraints, DA & RT
    rt_sched.addConstr( p_DA + r_up <= grid['Pmax'].reshape(-1))    
    rt_sched.addConstr( r_down <= p_DA)    
    
    # Ramping constraints
    rt_sched.addConstr( r_up <= grid['R_u_max'].reshape(-1))
    rt_sched.addConstr( r_down <= grid['R_d_max'].reshape(-1))

    # rt_sched.addConstr( G_shed <= grid['Pmax'].reshape(-1))
    # rt_sched.addConstr( L_shed <= grid['Pd'].reshape(-1))
        
    # RT balancing constraint    
    rt_sched.addConstr( p_DA.sum() + r_up.sum() - r_down.sum() - G_shed.sum() + grid['w_capacity']*w_rt.sum() + L_shed.sum() == grid['Pd'].sum())
    
    # Expected decision cost for DA + RT scheduling
    rt_sched.addConstr( da_cost == grid['Cost']@p_DA)
    rt_sched.addConstr( rt_cost == (grid['C_up'])@r_up + (-grid['C_down'])@r_down + grid['VOLL']*(L_shed.sum() + G_shed.sum()))

    rt_sched.setObjective(rt_cost)
    
    for i in range(n_samples):

        # Solve DA market with predictions, save results
        c1 = rt_sched.addConstr(p_DA == prescr_copy[i])
        c2 = rt_sched.addConstr(w_rt == actual_copy[i])

        rt_sched.optimize()
        
        for c in [c1,c2]: rt_sched.remove(c)
        
        Task_loss.append(rt_sched.ObjVal)

        DA_loss.append(da_cost.X)
        RT_loss.append(rt_cost.X)
        
    Task_loss = np.array(Task_loss)
    DA_loss = np.array(DA_loss)
    RT_loss = np.array(RT_loss)

    return DA_loss.mean(), RT_loss.mean()

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
    #slack_DA = stoch_market.addMVar((grid['n_loads']), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'slack')
    #exp_w = stoch_market.addMVar((1), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
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
    
    # Slacks for real-time
    # stoch_market.addConstr( G_shed <= np.tile(grid['Pmax'].reshape(-1,1), n_scen))
    # stoch_market.addConstr( L_shed <= np.tile(grid['Pd'].reshape(-1,1), n_scen))
        
    # DA balancing constraint
    #stoch_market.addConstr( p_DA.sum() + exp_w.sum() + slack_DA.sum() == grid['Pd'].sum())
    # RT balancing constraint
    #stoch_market.addConstrs( p_DA.sum() + exp_w.sum() + slack_DA.sum() == grid['Pd'].sum() for s in range(n_scen))
    
    stoch_market.addConstrs( p_DA.sum() + r_up[:,s].sum() - r_down[:,s].sum()
                            -G_shed[:,s].sum() + grid['w_capacity']*w_rt[:,s].sum() + L_shed[:,s].sum() == grid['Pd'].sum() for s in range(n_scen))

        

    # Expected decision cost for DA + RT scheduling
    stoch_market.addConstr( da_cost == grid['Cost']@p_DA)
    stoch_market.addConstrs( rt_cost[s] == (grid['C_up'])@r_up[:,s] + (-grid['C_down'])@r_down[:,s] + grid['VOLL']*(L_shed[:,s].sum() + G_shed[:,s].sum())
                            for s in range(n_scen))

    # Declare model once & solve for multiple test observations (speed up)
    n_test_obs = len(weights)
    DA_prescriptions = np.zeros((n_test_obs, grid['n_unit']))
    stoch_market.addConstr( w_rt == target_scen)
    stoch_market.setObjective( da_cost + t_aux)
    
    for row in range(len(weights)):
        if row%1000 == 0: print(row)
        # expected renewable production
        #c1 = stoch_market.addConstr( exp_w == target_scen@weights[row])
        c2 = stoch_market.addConstr( t_aux >= rt_cost@weights[row])
        stoch_market.optimize()        
        for constr in [c2]: stoch_market.remove(constr)        
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
    
    np.random.seed(1234)    
    #grid['C_up'] = (1 + np.random.uniform(0.5, 0.9, len(grid['Cost'])))*grid['Cost']
    #grid['C_down'] = (1 - np.random.uniform(0.1, 0.3, len(grid['Cost'])))*grid['Cost']
    
    
    grid['w_capacity'] = w_cap_dict[case_name]
    grid['w_bus'] = w_bus_dict[case_name]
    
    # wind incidence matrix
    grid['node_Wind'] = np.zeros((grid['n_nodes'], 1))
    grid['node_Wind'][grid['w_bus']] = 1
    
    
    grid['C_up'] = 5*grid['Cost']
    grid['C_down'] = 0.9*grid['Cost']

    R_u_max = np.ones(grid['n_unit'])
    
    R_u_max[grid['Cost'] < 10] = 0.1*grid['Pmax'][grid['Cost'] < 10]
    R_u_max[ (grid['Cost'] >= 10)*(grid['Cost'] < 20) ] = 0.3*grid['Pmax'][(grid['Cost'] >= 10)*(grid['Cost'] < 20)]
    R_u_max[ grid['Cost'] >= 20 ] = 1*grid['Pmax'][grid['Cost'] >= 20]
    
    R_d_max = R_u_max
    grid['VOLL'] = 200
    grid['R_u_max'] = R_u_max
    grid['R_d_max'] = R_d_max
    return grid

def gd_params():
    'Gradient-descent hyperparameters'
    gd_params = {}
    gd_params['patience'] = 10
    gd_params['batch_size'] = 512  
    gd_params['num_epochs'] = 500
    gd_params['learning_rate'] = 1e-2
    gd_params['feasibility_method'] = 'softmax'
    return gd_params

def params():
    ''' Set up the experiment parameters'''

    params = {}

    params['start_date'] = '2012-01-01'
    params['split_date_prob'] = '2013-01-01' # Defines train/test split
    params['split_date_comb'] = '2013-07-01' # Defines train/test split
    params['end_date'] = '2014-01-01'

    
    params['save'] = True # If True, then saves models and results
    
    # Experimental setup parameters
    params['problem'] = 'sched' # {mse, newsvendor, cvar, reg_trad, pwl}
    params['risk_aversion'] = [0.2] # {mse, newsvendor, cvar, reg_trad, pwl}

    params['dataset'] = 'wind' # !!! Do not change
    params['gamma_list'] = [0, 0.1, 1]
    params['target_zone'] = [2] # !!! Do not change
    params['target_ieee_case'] = 1
    
    params['train_static'] = True
    
    return params


#%%
config = params()
hyperparam = tree_params()
nn_hparam = gd_params()

# results_path = f'{cd}\\results\\grid_scheduling'
results_path = f'{cd}\\results\\grid_scheduling'
data_path = f'{cd}\\data'
pglib_path =  'C:/Users/astratig/pglib-opf/'

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

target_case = Cases[config['target_ieee_case']]
grid = load_grid_data(target_case, pglib_path)
#grid['Pd'][0] = grid['Pd'][0]  + 100
grid['Pd'] = 1.5*grid['Pd']

grid['Pd'] = np.array( grid['w_capacity'] + 100)
grid['Pmax'] = (grid['Pmax']/grid['Pmax'].sum())*(np.array( grid['w_capacity'] + 150))

#%%
np.random.seed(0)    
grid['C_up'] = (1 + np.random.uniform(0.1, 5, len(grid['Cost'])))*grid['Cost']
grid['C_down'] = (1 - np.random.uniform(0.8, 0.95, len(grid['Cost'])))*grid['Cost']

dataset = config['dataset']

step = .01
y_supp = np.arange(0, 1+step, step).round(2)
nlocations = len(y_supp)
#%%
zone_target = config['target_zone']
filename_prefix = f'NEWRESULTS_Z{zone_target[0]}_{target_case}_{dataset}'

if dataset == 'wind':
    # Load wind data and pre-processing
    
    config['start_date'] = '2012-01-01'
    config['split_date_prob'] = '2013-01-01' # Defines train/test split
    config['split_date_comb'] = '2013-07-01' # Defines train/test split
    config['end_date'] = '2014-01-01'

    aggr_df = pd.read_csv(f'{data_path}\\GEFCom2014-processed.csv', index_col = 0, header = [0,1])
    

    target_problem = config['problem']
    expert_zone = ['Z7', 'Z8', 'Z9']
    risk_aversion = config['risk_aversion']


    tuple_list = [tup for tup in itertools.product(zone_target, config['risk_aversion'])]

    # keep only relevant data
    zone_df = aggr_df.copy()['Z1']

    feat_scaler = MinMaxScaler()
    
    ### Create train/test sets for all series
    trainY = zone_df['POWER'][config['start_date']:config['split_date_prob']].round(2)
    comb_trainY = zone_df['POWER'][config['split_date_prob']:config['split_date_comb']].round(2)
    testY = zone_df['POWER'][config['split_date_comb']:].round(2)

    expert_zones = ['Z2', 'Z3', 'Z4']
    
    trainX_v1 = aggr_df[expert_zones[0]][['wspeed10', 'wdir10_rad', 'wspeed100', 'wdir100_rad']][config['start_date']:config['split_date_prob']]
    comb_trainX_v1 = aggr_df[expert_zones[0]][['wspeed10', 'wdir10_rad', 'wspeed100', 'wdir100_rad']][config['split_date_prob']:config['split_date_comb']]
    testX_v1 = aggr_df[expert_zones[0]][['wspeed10', 'wdir10_rad', 'wspeed100', 'wdir100_rad']][config['split_date_comb']:]
    
    trainX_v2 = aggr_df[expert_zones[1]][['wspeed10', 'wdir10_rad', 'wspeed100', 'wdir100_rad']][config['start_date']:config['split_date_prob']]
    comb_trainX_v2 = aggr_df[expert_zones[1]][['wspeed10', 'wdir10_rad', 'wspeed100', 'wdir100_rad']][config['split_date_prob']:config['split_date_comb']]
    testX_v2 = aggr_df[expert_zones[1]][['wspeed10', 'wdir10_rad', 'wspeed100', 'wdir100_rad']][config['split_date_comb']:]
    
    trainX_v3 = aggr_df[expert_zones[2]][['wspeed10', 'wdir10_rad', 'wspeed100', 'wdir100_rad']][config['start_date']:config['split_date_prob']]
    comb_trainX_v3 = aggr_df[expert_zones[2]][['wspeed10', 'wdir10_rad', 'wspeed100', 'wdir100_rad']][config['split_date_prob']:config['split_date_comb']]
    testX_v3 = aggr_df[expert_zones[2]][['wspeed10', 'wdir10_rad', 'wspeed100', 'wdir100_rad']][config['split_date_comb']:]
    
n_obs = len(comb_trainY)
n_test_obs = len(testY)

#%%
### NN hyperparameters
patience = nn_hparam['patience']
batch_size = nn_hparam['batch_size']
num_epochs = nn_hparam['num_epochs']
learning_rate = nn_hparam['learning_rate']
# apply_softmax = nn_hparam['apply_softmax']
row_counter = 0

RT_cost = pd.DataFrame()
DA_cost = pd.DataFrame()

#QS_df = pd.DataFrame()
mean_QS = pd.DataFrame()

#%%

for tup in tuple_list[row_counter:]:

    target_zone = tup[0]    
    risk_aversion = tup[1]
        
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
        knn_model_cv.fit(trainX_v1, trainY.values)    
        best_n_neighbors = knn_model_cv.best_estimator_.get_params()['n_neighbors']
        # find the weights for training/ comb training/ test set
            
        train_w_dict['knn'] = knn_model_cv.best_estimator_.kneighbors_graph(comb_trainX_v1).toarray()*(1/best_n_neighbors)
        test_w_dict['knn'] = knn_model_cv.best_estimator_.kneighbors_graph(testX_v1).toarray()*(1/best_n_neighbors)
                
        probabilistic_models['knn'] = knn_model_cv.best_estimator_

        # CART 1: weather predictors
        cart_parameters = {'max_depth':[5, 10, 20, 50, 100], 'min_samples_leaf':[1, 2, 5, 10]}
        cart_model_cv = GridSearchCV(DecisionTreeRegressor(), cart_parameters)
        
        cart_model_cv.fit(trainX_v2, trainY.values)    
            
        cart_model = cart_model_cv.best_estimator_
        probabilistic_models['cart'] = cart_model_cv.best_estimator_
        
        train_w_dict['cart'] = cart_find_weights(trainX_v2, comb_trainX_v2, cart_model)
        test_w_dict['cart'] = cart_find_weights(trainX_v2, testX_v2, cart_model)
                        
        rf_model = ExtraTreesRegressor(min_samples_leaf = 10, max_features = 2).fit(trainX_v3, trainY.values)
        
        probabilistic_models['rf'] = rf_model

        train_w_dict['rf'] = forest_find_weights(trainX_v3, comb_trainX_v3, rf_model)
        test_w_dict['rf'] = forest_find_weights(trainX_v3, testX_v3, rf_model)
    
        rf_point_pred = rf_model.predict(testX_v3)
        knn_point_pred = knn_model_cv.best_estimator_.predict(testX_v1)
        cart_point_pred = cart_model.predict(testX_v2)
        
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
        
        # estimate QS
        print('QS')
        target_quant = np.arange(.01, 1, .01)
        for j,m in enumerate(all_learners):
            temp_pdf = test_p_list[j]
    
            temp_q_forecast = np.array([inverted_cdf(target_quant, y_supp, temp_pdf[i]) for i in range(n_test_obs)])            
            temp_qs = 100*pinball(temp_q_forecast, testY.values, target_quant).round(4)
            print(m)
            plt.plot(temp_qs, label = m)
        plt.ylabel('Quantile Score')
        plt.xlabel('Quantile')
        
        plt.legend(['$k$$\mathtt{NN}$', '$\mathtt{CART}$', '$\mathtt{RF}$'])
        plt.xticks(np.arange(10, 100, 10), np.arange(0.1, 1, .1).round(2))
        # if dataset == 'wind':
        #     plt.savefig(f'{cd}\\plots\\quantile_score_wind_forecast.pdf')
        plt.xticks(np.arange(10, 100, 10), np.arange(0.1, 1, .1).round(2))
        plt.show()

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
    valid_obs = round(0.15*len(comb_trainY))
    tensor_trainY = torch.FloatTensor(train_targetY[:-valid_obs])
    tensor_train_p = torch.FloatTensor(np.column_stack((train_p_list)))
    tensor_train_p_list = [torch.FloatTensor(train_p_list[i][:-valid_obs]) for i in range(N_experts)]
    
    # full training data sets (no validation)
    tensor_trainY_full = torch.FloatTensor(train_targetY)
    tensor_train_p_list_full = [torch.FloatTensor(train_p_list[i]) for i in range(N_experts)]

    tensor_validY = torch.FloatTensor(train_targetY[-valid_obs:])
    tensor_valid_p_list = [torch.FloatTensor(train_p_list[i][-valid_obs:]) for i in range(N_experts)]
            
    train_data = torch.utils.data.TensorDataset(tensor_train_p_list[0], tensor_train_p_list[1], tensor_train_p_list[2], tensor_trainY)
    train_data_full = torch.utils.data.TensorDataset(tensor_train_p_list_full[0], tensor_train_p_list_full[1], tensor_train_p_list_full[2], 
                                                     tensor_trainY_full)
    
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
                
    #### Inverse Performance-based weights (invW in the paper)
    for g in ([0, 0.1, 1] + ['inf']):
        lambda_tuned_inv, _ = insample_weight_tuning(grid, train_targetY, trainZopt, train_p_list, regularization_gamma=g, problem = target_problem, 
                                                     support = y_supp)
        
        lambda_static_dict[f'invW-{g}'] = lambda_tuned_inv    
    #%%
    #% CRPS learning
    from torch_layers_functions import * 

    train_data_loader = create_data_loader(tensor_train_p_list + [tensor_trainY], batch_size = 100, shuffle = False)
    train_data_loader_full = create_data_loader(tensor_train_p_list_full + [tensor_trainY_full], batch_size = 100, shuffle = False)
    valid_data_loader = create_data_loader(tensor_valid_p_list + [tensor_validY], batch_size = 100, shuffle = False)
    
    if row_counter == 0:
        #### CRPS minimization/ with torch layer
        lpool_crps_model = LinearPoolCRPSLayer(num_inputs=N_experts, support = torch.FloatTensor(y_supp), 
                                               feasibility_method = nn_hparam['feasibility_method'])
        optimizer = torch.optim.Adam(lpool_crps_model.parameters(), lr = learning_rate)
        lpool_crps_model.train_model(train_data_loader_full, valid_data_loader, optimizer, epochs = 500, patience = 25, 
                                     validation = False)

        #lambda_crps = crps_learning_combination(comb_trainY.values, train_p_list, support = y_supp, verbose = 1)  
    lambda_static_dict['CRPS'] = lpool_crps_model.get_weights()
    # lambda_static_dict['CRPS'] = lpool_crps_model.weights.detach().numpy()
    print(lambda_static_dict['CRPS'])
    
    #%%
    ##### Decision-focused combination for different values of gamma     
    from torch_layers_functions import * 
    patience = 5

    #lambda_static_dict['DF_0'] = [0.32049093, 0.3465582, 0.33295092]    
    config['gamma_list'] = [0, 0.1, 1]
    if config['train_static']:
        
        for gamma in config['gamma_list']:
            
            lpool_sched_model = LinearPoolSchedLayer(num_inputs = N_experts, support = torch.FloatTensor(y_supp), 
                                                     grid = grid, gamma = gamma, clearing_type = 'stoch', feasibility_method = 'softmax')
            
            optimizer = torch.optim.Adam(lpool_sched_model.parameters(), lr = learning_rate)
            
            lpool_sched_model.train_model(train_data_loader_full, valid_data_loader, optimizer, epochs = 50, 
                                              patience = patience, validation = False, relative_tolerance = 1e-5)
            print(f'Learned weights:{lpool_sched_model.get_weights()}')
            lambda_static_dict[f'DF_{gamma}'] = lpool_sched_model.get_weights()
                    
            if config['save']:
                lamda_static_df = pd.DataFrame.from_dict(lambda_static_dict)
                lamda_static_df.to_csv(f'{results_path}\\{filename_prefix}_lambda_static.csv')
        
                with open(f'{results_path}\\{filename_prefix}_lambda_static_dict.pickle', 'wb') as handle:
                    pickle.dump(lambda_static_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(f'{results_path}\\{filename_prefix}_lambda_static_dict.pickle', 'rb') as handle:
            lambda_static_dict = pickle.load(handle)
        
    for m in list(lambda_static_dict.keys())[N_experts:]:
        plt.plot(lambda_static_dict[m], label = m)
    plt.legend()
    plt.show()
    
    #### Evaluate performance on test set
    
    static_models = list(lambda_static_dict) 
    all_models = static_models
    
    Prescriptions = {}
        
    temp_RT_cost = pd.DataFrame()
    temp_RT_cost['risk_aversion'] = [risk_aversion]
    temp_RT_cost['Target'] = [target_zone]
    temp_DA_cost = temp_RT_cost.copy()
    temp_mean_QS = temp_RT_cost.copy()
    
    target_quant = np.arange(0.1, 1, 0.1).round(2)
    print('Estimating out-of-sample performance...')
    
    # Estimate cost of perfect foresight solution
    perfect_foresight_DA_cost = perfect_scheduling_cost(grid, testY.values)
    temp_DA_cost['Perfect'] = perfect_foresight_DA_cost
    temp_RT_cost['Perfect'] = 0
    #all_models = all_models + ['Perfect']

    for j, m in enumerate(all_models):
        print(m)
        # Combine PDFs for each observation
        temp_pdf = sum([lambda_static_dict[m][j]*test_p_list[j] for j in range(N_experts)])            

        temp_prescriptions = solve_stoch_sched(grid, y_supp, temp_pdf, regularization = risk_aversion)

        Prescriptions[m] = temp_prescriptions
            
        # Estimate task-loss for specific model
        #%
        
        temp_da_cost_out, temp_rt_cost_out = scheduling_task_loss(grid, Prescriptions[m], testY.values)
        temp_RT_cost[m] = temp_rt_cost_out
        temp_DA_cost[m] = temp_da_cost_out

        # Evaluate QS (approximation of CRPS) for each model
        # find quantile forecasts
        temp_q_forecast = np.array([inverted_cdf(target_quant, y_supp, temp_pdf[i]) for i in range(n_test_obs)])            
        temp_qs = 100*pinball(temp_q_forecast, testY.values, target_quant).round(4)
        
        temp_mean_QS[m] = temp_qs.mean()
        
        print('Regret')
        print( ((temp_DA_cost + temp_RT_cost)[m] - temp_DA_cost['Perfect'].values[0]).mean().round(4))

    print('DA Decision Cost')
    print(temp_DA_cost[all_models].mean().round(4))

    print('RT Decision Cost')
    print(temp_RT_cost[all_models].mean().round(4))

    print('Total Decision Cost')
    print((temp_DA_cost + temp_RT_cost)[all_models].mean().round(4))

    print('Regret')
    print( ((temp_DA_cost + temp_RT_cost)[all_models] - temp_DA_cost['Perfect'].values[0]).mean().round(4))

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
    
    config['save'] = True
    if config['save']:
        DA_cost.to_csv(f'{results_path}\\{filename_prefix}_DA_cost.csv')
        RT_cost.to_csv(f'{results_path}\\{filename_prefix}_RT_cost.csv')
        #QS_df.to_csv(f'{results_path}\\{filename_prefix}_QS.csv')
        mean_QS.to_csv(f'{results_path}\\{filename_prefix}_mean_QS.csv')

        #Prescriptions.to_csv(f'{results_path}\\{target_problem}_{critical_fractile}_{target_zone}_Prescriptions.csv')
        lamda_static_df = pd.DataFrame.from_dict(lambda_static_dict)
        lamda_static_df.to_csv(f'{results_path}\\{filename_prefix}_lambda_static.csv')

    row_counter += 1        