# -*- coding: utf-8 -*-
"""
Testing optimal weight allocation for quantile averaging
"""

import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import sys, os
import pickle
import gurobipy as gp

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from gurobi_ml import add_predictor_constr
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler

from gurobi_ml.sklearn import add_decision_tree_regressor_constr, add_random_forest_regressor_constr

cd = os.path.dirname(__file__)  #Current directory
sys.path.append(cd)
#project_dir=Path(cd).parent.__str__()   #project_directory

from EnsemblePrescriptiveTree import *
from EnsemblePrescriptiveTree_OOB import *
from optimization_functions import *

from LinearDecisionTree import *
from sklearn.neural_network import MLPRegressor

import copy

from utility_functions import *
from optimal_transport_functions import *

import cvxpy as cp
import torch
from torch import nn
from cvxpylayers.torch import CvxpyLayer

def to_np(x):
    return x.detach().numpy()

# IEEE plot parameters (not sure about mathfont)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 600
plt.rcParams['figure.figsize'] = (3.5, 2) # Height can be changed
plt.rcParams['font.size'] = 7
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams["mathtext.fontset"] = 'dejavuserif'

def insample_weight_tuning(target_y, prob_vectors, brc_predictor = [], type_ = 'convex_comb', 
                                crit_fract = 0.5, support = np.arange(0, 1.01, .01).round(2), bounds = False, verbose = 0):
    ''' FOr each observation and expert, solve the stochastic problem, find expected in-sample decision cost, 
        set weights based on inverse cost (could use softmax activation)'''

    n_obs = prob_vectors[0].shape[0]
    n_models = len(prob_vectors)

    ### Find optimal decisions under perfect foresight information
    
    print('Solve in-sample stochastic problems...')
    
    z_opt = np.zeros((n_obs, n_models))
    insample_cost = np.zeros((n_models))
    insample_inverse_cost = np.zeros((n_models))
    
    for j in range(n_models):
        for i in range(n_obs):
            # Solve stochastic problem, find decision
            z_opt[i,j] = inverted_cdf([crit_fract], support, prob_vectors[j][i])
        # Estimate decision cost (regret)
                
        insample_cost[j] = newsvendor_loss(z_opt[:,j], target_y.reshape(-1), q = crit_fract)
        insample_inverse_cost[j] = 1/insample_cost[j]

    lambdas = insample_inverse_cost/insample_inverse_cost.sum()
    return lambdas

def averaging_decisions(target_y, prob_vectors, brc_predictor = [], type_ = 'convex_comb', 
                                crit_fract = 0.5, support = np.arange(0, 1.01, .01).round(2), bounds = False, verbose = 0):
    ''' (Salva's benchmark) Solve the stochastic problem in-sample for each observations and each expert, combine the decisions'''

    n_obs = prob_vectors[0].shape[0]
    n_models = len(prob_vectors)

    ### Find optimal decisions under perfect foresight information
    
    print('Solve in-sample stchastic problems...')
    
    z_opt = np.zeros((n_obs, n_models))

    for i in range(n_obs):
        for j in range(n_models):
            z_opt[i,j] = inverted_cdf([crit_fract], support, prob_vectors[j][i])
    
    #### set optimization problem
    
    m = gp.Model()
    if verbose == 0: 
        m.setParam('OutputFlag', 0)
    # Decision variables
    lambdas = m.addMVar(n_models, vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1)
    z_comb = m.addMVar(n_obs, vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1)
    
    lambdas.Start = (1/n_models)*np.ones(n_models)
    
    error = m.addMVar(n_obs, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
    loss_i = m.addMVar(n_obs, vtype = gp.GRB.CONTINUOUS, lb = 0)

    m.addConstr( lambdas.sum() == 1)
    
    m.addConstr( z_comb == sum([z_opt[:,j]*lambdas[j] for j in range(n_models)]) )
    
    # Task-loss function
    m.addConstr( error == target_y - z_comb)
    m.addConstr( loss_i >= crit_fract*error)
    m.addConstr( loss_i >= (crit_fract-1)*error)

    m.setObjective( loss_i.sum()/n_obs, gp.GRB.MINIMIZE)
    m.optimize()
    
    return lambdas.X

def wsum_tune_combination_newsvendor(target_y, prob_vectors, brc_predictor = [], type_ = 'convex_comb', 
                                crit_fract = 0.5, support = np.arange(0, 1.01, .01).round(2), bounds = False, verbose = 0):

    'Takes input prob. vectors, optimizes the forecast combination parameters, returns list of coordinates, simple weighted combination of problem'

    ### Find optimal decisions under perfect foresight information
    print('Derive decisions under perfect information...')
    z0_opt = []
    for y0 in support:
        # solve problem with perfect information forecast
        # !!!! This should call a generic opt. function
        z0_opt.append(y0)
    z0_opt = np.array(z0_opt)
    
    #### set optimization problem
    n_obs = prob_vectors[0].shape[0]
    n_models = len(prob_vectors)
    
    m = gp.Model()
    if verbose == 0: 
        m.setParam('OutputFlag', 0)
    # Decision variables
    lambdas = m.addMVar(n_models, vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1)
    p_comb = m.addMVar(prob_vectors[0].shape, vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1)
    
    lambdas.Start = (1/n_models)*np.ones(n_models)
    
    error = m.addMVar(n_obs, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
    loss_i = m.addMVar(n_obs, vtype = gp.GRB.CONTINUOUS, lb = 0)

    z = m.addMVar(n_obs, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
    if bounds:
        m.addConstr( z <= 1)
        m.addConstr( z >= 0)
        
    m.addConstr( lambdas.sum() == 1)
    # constraints on probability
    #m.addConstr( p_comb.sum(1) == 1)
    
    if type_ == 'convex_comb':
        m.addConstr( p_comb == sum([prob_vectors[i]*lambdas[i] for i in range(n_models)]) )
        
    elif type_ == 'barycenter':
        # create inverted cdfs
        target_quantiles = np.arange(0,1+.01,.01)
        prob_inv = [np.array([inverted_cdf(target_quantiles, support, w = p[i]) for i in range(n_obs)]) for p in prob_vectors ]

        # upper/lower bound for inverse correspond to bound on wind production, not quantiles
        p_comb_inv = m.addMVar(prob_inv[0].shape, vtype = gp.GRB.CONTINUOUS, lb = 0)


        ### quantile averaging
        m.addConstr( p_comb.sum(1) == 1)
        m.addConstr( p_comb_inv == sum([prob_inv[i]*lambdas[i] for i in range(n_models)]))
        
        ### mapping from inverse c.d.f. (quantile function) to prob. vector (p.d.f.)
        add_predictor_constr(m, brc_predictor, p_comb_inv, p_comb, epsilon = 1e-6)
        
    ### mapping probability vectors to decisions (replace with trained ML predictor for complex problems)/ simple weighted sum of decisions
    m.addConstr( error == target_y - z)
    
    m.addConstr( z == p_comb@z0_opt)

    # Task-loss function
    m.addConstr( error == target_y - z)
    m.addConstr( loss_i >= crit_fract*error)
    m.addConstr( loss_i >= (crit_fract-1)*error)

    m.setObjective( loss_i.sum()/n_obs, gp.GRB.MINIMIZE)
    m.optimize()
    
    return lambdas.X

def adapt_combination_newsvendor(target_y, X, prob_vectors, ml_predictor, brc_predictor = [], type_ = 'convex_comb', 
                                crit_fract = 0.5, support = np.arange(0, 1.01, .01).round(2), bounds = False, verbose = 0):

    'Learns a policy that maps contextual information to forecast combination weights'
    #### set optimization problem
    n_obs = prob_vectors[0].shape[0]
    n_feat = X.shape[1]
    n_models = len(prob_vectors)
    
    m = gp.Model()
    if verbose == 0: 
        m.setParam('OutputFlag', 0)
    # Decision variables
    
    lambdas = m.addMVar((n_obs, n_models), vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1)
    
    coef_ = m.addMVar((n_models, n_feat), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
    inter_ = m.addMVar(n_models, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)

    p_comb = m.addMVar(prob_vectors[0].shape, vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1)
    
    #lambdas.Start = (1/n_models)*np.ones(n_models)

    error = m.addMVar(n_obs, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
    loss_i = m.addMVar(n_obs, vtype = gp.GRB.CONTINUOUS, lb = 0)

    z = m.addMVar(n_obs, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
    if bounds:
        m.addConstr( z <= 1)
        m.addConstr( z >= 0)
        
    m.addConstr( lambdas.sum(1) == 1)
    
    m.addConstrs( lambdas[i] == coef_@X[i] + inter_ for i in range(n_obs))

    # constraints on probability
    #m.addConstr( p_comb.sum(1) == 1)
    
    if type_ == 'convex_comb':
        m.addConstrs( p_comb[i] == sum([prob_vectors[j][i]*lambdas[i,j] for j in range(n_models)])  for i in range(n_obs) )
        
    elif type_ == 'barycenter':
        # create inverted cdfs
        target_quantiles = np.arange(0,1+.01,.01)
        prob_inv = [np.array([inverted_cdf(target_quantiles, support, w = p[i]) for i in range(n_obs)]) for p in prob_vectors ]

        # upper/lower bound for inverse correspond to bound on wind production, not quantiles
        p_comb_inv = m.addMVar(prob_inv[0].shape, vtype = gp.GRB.CONTINUOUS, lb = 0)


        ### quantile averaging
        m.addConstr( p_comb.sum(1) == 1)
        m.addConstr( p_comb_inv == sum([prob_inv[i]*lambdas[i] for i in range(n_models)]))
        
        ### mapping from inverse c.d.f. (quantile function) to prob. vector (p.d.f.)
        add_predictor_constr(m, brc_predictor, p_comb_inv, p_comb, epsilon = 1e-6)
        
    ### mapping probability vectors to decisions (replace with trained ML predictor for complex problems)
    pred_constr = add_predictor_constr(m, ml_predictor, p_comb, z, epsilon = 1e-6)
    #pred_constr = add_decision_tree_regressor_constr(m, dt_model, p_comb, z, epsilon = .1e-6)
    pred_constr.print_stats()

    # Task-loss function
    m.addConstr( error == target_y - z)
    m.addConstr( loss_i >= crit_fract*error)
    m.addConstr( loss_i >= (crit_fract-1)*error)

    m.setObjective( loss_i.sum()/n_obs, gp.GRB.MINIMIZE)
    m.optimize()
    
    return coef_.X, inter_.X

def tree_params():
    ''' Hyperparameters for tree algorithms'''
    params = {}
    params['n_estimators'] = 50
    params['n_min'] = 2
    params['max_features'] = 1
    return params
                            
def params():
    ''' Set up the experiment parameters'''

    params = {}
    # Either pv or wind
    params['target_zone'] = 'Z1'
        
    params['start_date'] = '2012-01-01'
    params['split_date'] = '2013-01-01' # Defines train/test split
    params['end_date'] = '2013-12-30'
    
    params['save'] = False # If True, then saves models and results
    params['train_brc'] = False # If True, then saves models and results
    
    # Experimental setup parameters
    params['problem'] = 'newsvendor' # {mse, newsvendor, cvar, reg_trad}
    params['N_sample'] = [10, 50, 100, 200, 500]
    params['N_experts'] = 3
    params['iterations'] = 5
    
    params['critical_fractile'] = np.arange(0.1, 1, 0.1).round(2)
    
    # approaches to map data to decisions
    # LR: linear regression, DecComb: combination of perfect-foresight decisions (both maintain convexity)
    # DT: decision tree, NN: neural network (both results in MIPs)
    params['decision_rules'] = ['LR', 'JMBench', 'SalvaBench'] 

    return params
    
#%%
config = params()
hyperparam = tree_params()

results_path = f'{cd}\\results'
data_path = f'{cd}\\data'

aggr_wind_df = pd.read_csv(f'{data_path}\\GEFCom2014-processed.csv', index_col = 0, header = [0,1])
#%%

#!!!! Add randomization based on the iteration counter here

target_problem = config['problem']
row_counter = 0

for critical_fractile, iter_ in itertools.product(config['critical_fractile'], range(config['iterations'])):

    all_zones = [f'Z{i}' for i in range(1,11)]
    np.random.seed(row_counter)
    
    print(f'Quantile:{critical_fractile}, iteration:{iter_}')
    #target_zone = config['target_zone']
    #expert_zones = ['Z2', 'Z4', 'Z8', 'Z9']
    
    target_zone = np.random.choice(all_zones)
    expert_zones = all_zones.copy()
    expert_zones.remove(target_zone)
    expert_zones = list(np.random.choice(expert_zones, config['N_experts'], replace = False))
    
    pred_col = ['wspeed10', 'wdir10_rad', 'wspeed100', 'wdir100_rad']
    #%%
    # number of forecasts to combine
    N_experts = config['N_experts']
    
    # number of observations to train prob. forecasting model
    N_sample = len(aggr_wind_df)//4
    
    step = .01
    y_supp = np.arange(0, 1+step, step).round(2)
    nlocations = len(y_supp)
    
    #%%
    ### Create train/test sets for all series
    
    trainY = aggr_wind_df.xs('POWER', axis=1, level=1)[[target_zone] + expert_zones][config['start_date']:config['split_date']][:N_sample].round(2)
    comb_trainY = aggr_wind_df.xs('POWER', axis=1, level=1)[[target_zone] + expert_zones][config['start_date']:config['split_date']][N_sample:].round(2)
    testY = aggr_wind_df.xs('POWER', axis=1, level=1)[[target_zone] + expert_zones][config['split_date']:].round(2)
    
    # feature data for target location/zone
    
    local_features_df = aggr_wind_df[target_zone].copy()
    local_features_df['wspeed10_sq'] = np.power(local_features_df['wspeed10'],2)
    local_features_df['wspeed10_cb'] = np.power(local_features_df['wspeed10'],3)
    
    local_features_df['wspeed100_sq'] = np.power(local_features_df['wspeed100'],2)
    local_features_df['wspeed100_cb'] = np.power(local_features_df['wspeed100'],3)
    
    local_features_df[['diurnal_1', 'diurnal_2', 'diurnal_3', 'diurnal_4']] = aggr_wind_df[['diurnal_1', 'diurnal_2', 'diurnal_3', 'diurnal_4']]
    
    local_feat = ['wspeed10', 'wspeed100', 'wdir10_rad', 'wdir100_rad', 'diurnal_1', 'diurnal_2', 'diurnal_3', 'diurnal_4']
    trainX_local = local_features_df[local_feat][config['start_date']:config['split_date']][:N_sample].round(2)
    comd_trainX_local = local_features_df[local_feat][config['start_date']:config['split_date']][N_sample:].round(2)
    testX_local = local_features_df[local_feat][config['split_date']:].round(2)
    
    # supervised sets for experts    
    trainX_exp = aggr_wind_df[expert_zones][config['start_date']:config['split_date']][:N_sample].round(2)
    comb_trainX_exp = aggr_wind_df[expert_zones][config['start_date']:config['split_date']][N_sample:].round(2)
    testX_exp = aggr_wind_df[expert_zones][config['split_date']:].round(2)
    
    # number of training observations for the combination model
    n_obs = len(comb_trainY)
    n_test_obs = len(testY)
    
    #%% Train experts, i.e., probabilistic forecasting models in adjacent locations
    
    # data conditioned on wind speed
    
    prob_models = []
    
    for i, zone in enumerate(expert_zones[:N_experts]):
        print(f'Training model {i}')
        
        #temp_model = EnsemblePrescriptiveTree_OOB(n_estimators = 10, max_features = 1, type_split = 'quant')
        #temp_model.fit(trainX_exp[zone][pred_col].values.round(2), trainY[zone].values, y_supp, y_supp, bootstrap = False, quant = np.arange(.01, 1, .01), problem = 'mse') 
     
        temp_model = EnsemblePrescriptiveTree(n_estimators = 30, max_features = len(pred_col), type_split = 'quant' )
        temp_model.fit(trainX_exp[zone][pred_col].values, trainY[zone].values, quant = np.arange(.01, 1, .01), problem = 'mse') 
        prob_models.append(temp_model)
    
    #%% Generate predictions for train/test set for forecast combination
    # find local weights for meta-training set/ map weights to support locations
    print('Generating prob. forecasts for train/test set...')
    
    train_w_list = []
    train_p_list = []
    for i, p in enumerate(expert_zones[:N_experts]):
        train_w_list.append(prob_models[i].find_weights(comb_trainX_exp[p][pred_col].values, trainX_exp[p][pred_col].values))
        train_p_list.append(wemp_to_support(train_w_list[i], trainY[p].values, y_supp))
        
    test_w_list = []
    test_p_list = []
    for i, p in enumerate(expert_zones[:N_experts]):
        test_w_list.append(prob_models[i].find_weights(testX_exp[p][pred_col].values, trainX_exp[p][pred_col].values))
        test_p_list.append(wemp_to_support(test_w_list[i], trainY[p].values, y_supp))
        
    train_targetY = comb_trainY[target_zone].values.reshape(-1)
    
    # Supervised learning set for torch
    tensor_trainY = torch.FloatTensor(train_targetY)
    tensor_train_p = torch.FloatTensor(np.column_stack((train_p_list)))
    tensor_train_p_list = [torch.FloatTensor(train_p_list[i]) for i in range(N_experts)]
    
    feat_scaler = MinMaxScaler()
    feat_scaler.fit(comd_trainX_local)
    tensor_trainX = torch.FloatTensor(feat_scaler.transform(comd_trainX_local))
    tensor_testX = torch.FloatTensor(feat_scaler.transform(testX_local))
    
    train_data = torch.utils.data.TensorDataset(tensor_train_p_list[0], tensor_train_p_list[1], tensor_train_p_list[2], tensor_trainY)
    
    n_train_obs = len(train_targetY)
    n_test_obs = len(testY)
    
#%%%%%%%%%%%%%%% Newsvendor experiment

    if target_problem == 'newsvendor':

        lambda_cc_dict = {}
        
        # Benchmark/ Salva's suggestion/ weighted combination of in-sample optimal (stochastic) decisions
        lambda_bench = averaging_decisions(train_targetY, train_p_list, crit_fract = critical_fractile, support = y_supp, bounds = False)
        #%%
        lambda_tuned = insample_weight_tuning(train_targetY, train_p_list, crit_fract = critical_fractile, support = y_supp, bounds = False)
        
        print(f'Lambda SBench:{lambda_bench}')
        print(f'Lambda insample tune:{lambda_tuned}')
            
        #%% Testing all methods
        
        
        ##### Static Forecast Combinations
        
        # turn weights to distributions, find average distribution
        p_ave = sum(test_p_list)/N_experts
        p_brc = np.zeros((n_test_obs, nlocations))
        p_brc_opt = np.zeros((n_test_obs, nlocations))
        p_cc_tune = sum([lambda_tuned[i]*test_p_list[i] for i in range(N_experts)])

        '''
        # Barycenter with average & optimal coordinates
        for i in range(n_test_obs):
            temp_p_list = [p[i] for p in test_p_list]

            # Barycenter with uniform weights
            temp_p_brc, _, _ = wass2_barycenter_1D(N_experts*[y_supp], temp_p_list, lambda_coord = N_experts*[1/N_experts], support = y_supp, p = 2, 
                                       prob_dx = .01)
            p_brc[i] = temp_p_brc
        
            # Barycenter with optimal weights (Papayanis, Yannacopoulos)
            temp_p_brc, _, _ = wass2_barycenter_1D(N_experts*[y_supp], temp_p_list, lambda_coord = lambda_brc_opt, support = y_supp, p = 2, 
                                       prob_dx = .01)
            p_brc_opt[i] = temp_p_brc
        '''
        
        # turn probability vectors to decisions/ closed-form solution for newsvendor problem    
        models = [f'Model-{i}' for i in range(N_experts)] + ['CC-Ave', 'CC-Tune']
        
        Prescriptions = pd.DataFrame(data = np.zeros((n_test_obs, len(models))), columns = models)
        for i in range(N_experts):
            Prescriptions[f'Model-{i}'] = np.array([inverted_cdf([critical_fractile], y_supp, test_p_list[i][k]) for k in range(n_test_obs)]).reshape(-1)
        
        Prescriptions['CC-Ave'] = np.array([inverted_cdf([critical_fractile], y_supp, p_ave[i]) for i in range(n_test_obs)]).reshape(-1)
        Prescriptions['CC-Tune'] = np.array([inverted_cdf([critical_fractile], y_supp, p_cc_tune[i]) for i in range(n_test_obs)]).reshape(-1)
        
        temp_output = pd.DataFrame()
        temp_output['Quantile'] = [critical_fractile]
        temp_output['Iteration'] = iter_
        
        for m in models:
            print(f'{m}:{100*newsvendor_loss(Prescriptions[m].values, testY[target_zone], q = critical_fractile).round(4)}')
            
            temp_output[m] = 100*newsvendor_loss(Prescriptions[m].values, testY[target_zone], q = critical_fractile).round(4)
        
        if row_counter == 0: 
            Output = temp_output.copy()
        else:
            Output = pd.concat([Output, temp_output])        
        Output.to_csv(f'{cd}\\results\\tuned_weight_allocation_tests.csv')
        row_counter += 1