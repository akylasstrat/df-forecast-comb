# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Decision-Focused Forecast Combination/ main script/ second experimental setup: different probabilistic forecasting models for the same location/ solar plants example

@author: a.stratigakos
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


cd = os.path.dirname(__file__)  #Current directory
sys.path.append(cd)

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

def averaging_decisions(target_y, train_z_opt, problem, 
                        crit_fract = 0.5, support = np.arange(0, 1.01, .01).round(2), bounds = False, verbose = 0, 
                        **kwargs):            
    ''' (Salva's benchmark) Solve the stochastic problem in-sample for each observations and each expert, combine the decisions'''

    n_obs = train_z_opt.shape[0]
    n_models = train_z_opt.shape[1]
    risk_aversion = kwargs['risk_aversion']

    ### Find optimal decisions under perfect foresight information
    
    print('Solve in-sample stchastic problems...')
    
    #z_opt = np.zeros((n_obs, n_models))
    insample_cost = np.zeros(n_models)
    insample_inverse_cost = np.zeros(n_models)
    
    for j in range(n_models):
        '''
        if problem == 'newsvendor':
            for i in range(n_obs):
                # Solve stochastic problem, find decision
                z_opt[i,j] = inverted_cdf([crit_fract], support, prob_vectors[j][i])
            # Estimate decision cost (regret)
        elif problem == 'reg_trad':
            temp_z_opt = solve_opt_prob(support, prob_vectors[j], problem, risk_aversion = risk_aversion, 
                                        crit_quant = crit_fract)
            z_opt[:,j] = temp_z_opt
        '''
        
    #### set optimization problem    
    m = gp.Model()
    if verbose == 0: 
        m.setParam('OutputFlag', 0)
    # Decision variables
    lambdas = m.addMVar(n_models, vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1)
    z_comb = m.addMVar(n_obs, vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1)
    
    lambdas.Start = (1/n_models)*np.ones(n_models)
    
    error = m.addMVar(n_obs, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
    pinball_loss_i = m.addMVar(n_obs, vtype = gp.GRB.CONTINUOUS, lb = 0)

    m.addConstr( lambdas.sum() == 1)
    
    m.addConstr( z_comb == sum([train_z_opt[:,j]*lambdas[j] for j in range(n_models)]) )
    
    # Task-loss function
    m.addConstr( error == target_y - z_comb)
    
    if (problem == 'reg_trad') or (problem == 'newsvendor'):
        m.addConstr( pinball_loss_i >= crit_fract*error)
        m.addConstr( pinball_loss_i >= (crit_fract-1)*error)    
        m.setObjective( (1-risk_aversion)*pinball_loss_i.sum() + risk_aversion*(error@error) , gp.GRB.MINIMIZE)
    elif problem == 'pwl':
        m.addConstr( pinball_loss_i >= crit_fract*error)
        m.addConstr( pinball_loss_i >= (-0.5)*error)
        m.addConstr( pinball_loss_i >= (crit_fract - 1)*(error + 0.1))
    
        m.setObjective( pinball_loss_i.sum() + risk_aversion*(error@error), gp.GRB.MINIMIZE)
        
    m.optimize()
    
    return lambdas.X

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

def tune_combination_newsvendor(target_y, prob_vectors, ml_predictor, brc_predictor = [], type_ = 'convex_comb', 
                                crit_fract = 0.5, support = np.arange(0, 1.01, .01).round(2), bounds = False, verbose = 0):

    'Takes input prob. vectors, optimizes the forecast combination parameters, returns list of coordinates'
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

        ########################################
        ### piecewise linear constraint for inverse cdf function
        '''
        y_supp_aux = m.addMVar(support.shape, vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1)
        p_comb_cdf = m.addMVar(p_comb.shape, vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1)
        
        aux_cdf = [[m.addVar(vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1) for j in range(len(support))] for i in range(nobs)]
        aux_inv = [[m.addVar(vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1) for j in range(len(support))] for i in range(nobs)]
        
        for i in range(nobs):
            for j in range(len(support)):
                m.addGenConstrPWL(y_supp_aux[j], aux_cdf[i][j], aux_inv[i], target_quantiles, "myPWLConstr")    
            
                m.addConstr( aux_cdf[i][j] == p_comb_cdf[i,j] )
                m.addConstr( aux_inv[i][j] == p_comb_inv[i,j] )

        m.addConstr( y_supp_aux == support)
        
        ### turn c.d.f. to prob. vector
        m.addConstrs( p_comb[:,j] == p_comb_cdf[:,j]-p_comb_cdf[:,j-1] for j in range(1,len(support)))
        m.addConstr( p_comb[:,0] == p_comb_cdf[:,0])
        '''
        ########################################
        
        ### mapping from inverse c.d.f. (quantile function) to prob. vector (p.d.f.)
        add_predictor_constr(m, brc_predictor, p_comb_inv, p_comb, epsilon = 1e-6)
        
    ### mapping probability vectors to decisions (replace with trained ML predictor for complex problems)
    #pred_constr = add_random_forest_regressor_constr(m, rf_model, p_comb, z, epsilon = 1e-6)
    #pred_constr = add_predictor_constr(m, rf_model, p_comb, z, epsilon = 1e-6)
    pred_constr = add_predictor_constr(m, ml_predictor, p_comb, z, epsilon = 1e-6)
    #pred_constr = add_decision_tree_regressor_constr(m, dt_model, p_comb, z, epsilon = .1e-6)
    pred_constr.print_stats()

    # Task-loss function
    m.addConstr( error == target_y - z)
    m.addConstr( loss_i >= crit_fract*error)
    m.addConstr( loss_i >= (crit_fract-1)*error)

    m.setObjective( loss_i.sum()/n_obs, gp.GRB.MINIMIZE)
    m.optimize()
    
    return lambdas.X

def tree_params():
    ''' Hyperparameters for tree algorithms'''
    params = {}
    params['n_estimators'] = 50
    params['n_min'] = 2
    params['max_features'] = 1
    return params

#def create_data_loader(X, Y, batch_size = 64, shuffle = True):
#    dataset = torch.utils.data.TensorDataset(X,Y)
#    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def insample_weight_tuning(target_y, train_z_opt, problem = 'newsvendor', 
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
    
    n_obs = train_z_opt
    n_models = train_z_opt.shape[1]
    
    risk_aversion = kwargs['risk_aversion']
    crit_quant = kwargs['crit_quant']
    ### Find optimal decisions under perfect foresight information
    
    print('Solve in-sample stochastic problems...')
    
    #z_opt = np.zeros((n_obs, n_models))
    insample_cost = np.zeros((n_models))
    insample_inverse_cost = np.zeros((n_models))

    for j in range(n_models):
        '''
        if problem == 'newsvendor':
            for i in range(n_obs):
                # Solve stochastic problem, find decision
                z_opt[i,j] = inverted_cdf([crit_fract], support, prob_vectors[j][i])
            # Estimate decision cost (regret)
        elif problem == 'reg_trad':
            
            temp_w_mat = np.array(prob_vectors[j])

            temp_z_opt = solve_opt_prob(support, temp_w_mat, problem, risk_aversion = risk_aversion, 
                                        crit_quant = crit_quant)
            z_opt[:,j] = temp_z_opt
        '''
        #insample_cost[j] = newsvendor_loss(z_opt[:,j], target_y.reshape(-1), q = crit_fract)
        insample_cost[j] = task_loss(train_z_opt[:,j], target_y.reshape(-1), problem, risk_aversion = risk_aversion, 
                                     crit_quant = crit_quant)
        insample_inverse_cost[j] = 1/insample_cost[j]

    lambdas_inv = insample_inverse_cost/insample_inverse_cost.sum()
    lambdas_softmax = np.exp(insample_inverse_cost)/sum(np.exp(insample_inverse_cost))

    return lambdas_inv, lambdas_softmax
                
def task_loss(pred, actual, problem, **kwargs):
    'Estimates task loss for different problems'

    pred_copy = pred.copy().reshape(-1)
    actual_copy = actual.copy().reshape(-1)
    
    
    if problem == 'mse':
        return np.square(actual_copy-pred_copy).mean()
    elif problem == 'newsvendor':

        return np.maximum(kwargs['crit_quant']*(actual_copy - pred_copy), (kwargs['crit_quant']-1)*(actual_copy - pred_copy)).mean()
    elif problem == 'cvar':
        pinball_loss = np.maximum(kwargs['crit_quant']*(actual_copy - pred_copy), (kwargs['crit_quant']-1)*(actual_copy - pred_copy))    
        #profit = 2e3*(27*actual_copy - pinball_loss)
        cvar_mask = pinball_loss >= np.quantile(pinball_loss, 1-kwargs['epsilon'])

        task_loss = ((1-kwargs['risk_aversion'])*pinball_loss.mean() + kwargs['risk_aversion']*pinball_loss[cvar_mask].mean()) 
        
        return task_loss
    
    elif problem == 'pwl':

        deviation = actual_copy - pred_copy
        square_loss = np.square(deviation)          

        p1 = kwargs['crit_quant']*deviation
        p2 = -0.5*deviation        
        p3 = (kwargs['crit_quant']-1)*(deviation + 0.1)        
        pwl_loss = np.maximum.reduce([p1, p2, p3])        
        return pwl_loss.mean() + kwargs['risk_aversion']*square_loss.mean()


    elif problem == 'reg_trad':

        deviation = actual_copy - pred_copy
        pinball_loss = np.maximum(kwargs['crit_quant']*deviation, (kwargs['crit_quant']-1)*deviation)  
        square_loss = np.square(deviation)          
        
        return (1- kwargs['risk_aversion'])*pinball_loss.mean() + kwargs['risk_aversion']*square_loss.mean()
        
def solve_opt_prob(scenarios, weights, problem, **kwargs):
    ''' Solves stochastic optimization problem
        -Args
            scenarios: sampled scenarios (e.g., locations of PDF)
            weights: weight of each sceanrio (e.g., probability of each PDF location)
            problem: string that contains the problem description {mse, newsvendor, reg_trad, cvar}
            kwargs: additional arguments for each problem
        scenarios: support/ fixed locations
        weights: the learned probabilities'''
    
    risk_aversion = kwargs['risk_aversion']

    #e = kwargs['epsilon']
    crit_quant = kwargs['crit_quant']

    if scenarios.ndim>1:
        target_scen = scenarios.copy().reshape(-1)
    else:
        target_scen = scenarios.copy()
        
    n_scen = len(target_scen)
    
    if problem == 'cvar':
        # CVaR tail probability
        
        m = gp.Model()
        m.setParam('OutputFlag', 0)
        ################################################
        #CVaR: auxiliary parameters
        # Multi-temporal: Minimize average Daily costs
        ### Variables
        offer = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1)
        deviation = m.addMVar(n_scen, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
        loss = m.addMVar(n_scen, vtype = gp.GRB.CONTINUOUS, lb = 0)
        profit = m.addMVar(n_scen, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)

        ### CVaR variables (follows Georghiou, Kuhn, et al.)
        beta = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name='VaR')
        zeta = m.addMVar(n_scen, vtype = gp.GRB.CONTINUOUS, lb = 0)  # Aux
        cvar = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
         
        m.addConstr( deviation == target_scen - offer)
        
        m.addConstr( loss >= (crit_quant)*deviation)
        m.addConstr( loss >= (crit_quant-1)*deviation)

        m.addConstr( profit == (target_scen*27 - loss) )            
        m.addConstr( zeta >= beta - profit)

        #m.addConstr( zeta >=  -beta + loss )
        #m.addConstr( cvar == beta + (1/e)*(zeta@weights))
        m.addConstr( cvar == beta - (1/(1-e))*(zeta@weights) )            
        m.setObjective( (1-risk_aversion)*(profit@weights) + risk_aversion*(cvar), gp.GRB.MAXIMIZE )
        
        #m.setObjective( 0, gp.GRB.MINIMIZE)
        
        m.optimize()
        return offer.X[0]
    
    elif problem == 'reg_trad':
        # regularized newsvendor problem
        if weights.ndim == 1:
            # solve for a single problem instance
            m = gp.Model()
            
            m.setParam('OutputFlag', 0)

            # target variable
            offer = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'offer')
            deviation = m.addMVar(n_scen, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'offer')
            loss = m.addMVar(n_scen, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'aux')

            m.addConstr(deviation == (target_scen - offer) )
            
            m.addConstr(loss >= crit_quant*(deviation) )
            m.addConstr(loss >= (crit_quant-1)*(deviation) )
            
            m.setObjective( (1-risk_aversion)*(weights@loss) + risk_aversion*(deviation@(deviation*weights)), gp.GRB.MINIMIZE)
            m.optimize()
                
            return offer.X
        
        else:
            # solve for multiple test observations/ declares gurobi model once for speed up
            n_test_obs = len(weights)
            Prescriptions = np.zeros((n_test_obs))
            m = gp.Model()            
            m.setParam('OutputFlag', 0)

            # variables
            offer = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'offer')
            deviation = m.addMVar(n_scen, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
            loss = m.addMVar(n_scen, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'aux')
            t = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'aux')
            
            # constraints
            m.addConstr(deviation == (target_scen - offer) )
            m.addConstr(loss >= crit_quant*deviation )
            m.addConstr(loss >= (crit_quant-1)*deviation )
            m.setObjective( t, gp.GRB.MINIMIZE)
            
            for row in range(len(weights)):
                c1 = m.addConstr(t >= 2*(1-risk_aversion)*(weights[row]@loss) 
                               + 2*risk_aversion*(deviation@(deviation*weights[row])))

                m.optimize()
                
                m.remove(c1)
                
                Prescriptions[row] = offer.X[0]
                
            return Prescriptions
    
    elif (problem == 'reg_trad') or (problem == 'newsvendor'):
        # regularized newsvendor problem
        if weights.ndim == 1:
            # solve for a single problem instance
            m = gp.Model()
            
            m.setParam('OutputFlag', 0)

            # target variable
            offer = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'offer')
            deviation = m.addMVar(n_scen, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'offer')
            loss = m.addMVar(n_scen, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'aux')

            m.addConstr(deviation == (target_scen - offer) )
            
            m.addConstr(loss >= crit_quant*(deviation) )
            m.addConstr(loss >= (crit_quant-1)*(deviation) )
            
            m.setObjective( (1-risk_aversion)*(weights@loss) + risk_aversion*(deviation@(deviation*weights)), gp.GRB.MINIMIZE)
            m.optimize()
                
            return offer.X
        
        else:
            # solve for multiple test observations/ declares gurobi model once for speed up
            n_test_obs = len(weights)
            Prescriptions = np.zeros((n_test_obs))
            m = gp.Model()            
            m.setParam('OutputFlag', 0)

            # variables
            offer = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'offer')
            deviation = m.addMVar(n_scen, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
            loss = m.addMVar(n_scen, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'aux')
            t = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'aux')
            
            # constraints
            m.addConstr(deviation == (target_scen - offer) )
            m.addConstr(loss >= crit_quant*deviation )
            m.addConstr(loss >= (crit_quant-1)*deviation )
            m.setObjective( t, gp.GRB.MINIMIZE)
            
            for row in range(len(weights)):
                c1 = m.addConstr(t >= 2*(1-risk_aversion)*(weights[row]@loss) 
                               + 2*risk_aversion*(deviation@(deviation*weights[row])))

                m.optimize()
                
                m.remove(c1)
                
                Prescriptions[row] = offer.X[0]
                
            return Prescriptions
    elif problem == 'pwl':
        # regularized newsvendor problem
        if weights.ndim == 1:
            # solve for a single problem instance
            m = gp.Model()
            
            m.setParam('OutputFlag', 0)

            # target variable
            offer = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'offer')
            deviation = m.addMVar(n_scen, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'offer')
            loss = m.addMVar(n_scen, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'aux')

            m.addConstr(deviation == (target_scen - offer) )
            
            m.addConstr(loss >= crit_quant*(deviation) )
            m.addConstr(loss >= -0.5*(deviation) )
            m.addConstr(loss >= (crit_quant-1)*(deviation + 0.1) )
            
            m.setObjective( (weights@loss) + risk_aversion*(deviation@(deviation*weights)), gp.GRB.MINIMIZE)
            m.optimize()
                
            return offer.X
        
        else:
            # solve for multiple test observations/ declares gurobi model once for speed up
            n_test_obs = len(weights)
            Prescriptions = np.zeros((n_test_obs))
            m = gp.Model()            
            m.setParam('OutputFlag', 0)

            # variables
            offer = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'offer')
            deviation = m.addMVar(n_scen, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
            loss = m.addMVar(n_scen, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'aux')
            t = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'aux')
            
            # constraints
            m.addConstr(deviation == (target_scen - offer) )

            m.addConstr(loss >= crit_quant*(deviation) )
            m.addConstr(loss >= -0.5*(deviation) )
            m.addConstr(loss >= (crit_quant-1)*(deviation + 0.1) )
            m.setObjective( t, gp.GRB.MINIMIZE)
            
            for row in range(len(weights)):
                c1 = m.addConstr(t >= (weights[row]@loss) + risk_aversion*(deviation@(deviation*weights[row])))

                m.optimize()                
                m.remove(c1)                
                Prescriptions[row] = offer.X[0]        
            return Prescriptions
        
    elif problem =='mse':
        return (target_scen@weights)
 
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
    params['problem'] = 'reg_trad' # {mse, newsvendor, cvar, reg_trad, pwl}
    params['gamma_list'] = [0, 0.1, 1]
    params['target_zone'] = [3]
        
    params['crit_quant'] = np.arange(0.2, 1, 0.1).round(2)
    params['risk_aversion'] = [0.2]

    params['train_static'] = True
    params['train_adaptive'] = True
    
    return params

#%%
    
config = params()
hyperparam = tree_params()
nn_hparam = nn_params()

results_path = f'{cd}\\results\\solar_new_results'
data_path = f'{cd}\\data'


aggr_df = pd.read_csv(f'{data_path}\\gefcom2014-solar.csv', index_col = 0, parse_dates=True)
#%% Data pre-processing
zone_target = config['target_zone']
aggr_df = aggr_df.query(f'ZONEID=={zone_target}')
target_problem = config['problem']
risk_aversion = config['risk_aversion']

filename_prefix = f'Z{zone_target[0]}_{target_problem}_'
        
del aggr_df['ZONEID']

aggr_df['Hour'] = aggr_df.index.hour
aggr_df['Month'] = aggr_df.index.month

weather_dict = {'VAR78':'tclw', 'VAR79': 'tciw', 'VAR134':'SP', 'VAR157':'rh', 
                'VAR164':'tcc', 'VAR165':'10u', 'VAR166':'10v', 'VAR167':'2T', 
                'VAR169':'SSRD', 'VAR175':'STRD', 'VAR178':'TSR', 'VAR228':'TP'}

aggr_df = aggr_df.rename(columns = weather_dict)

#aggr_df['diurnal_2'] = np.cos(2*np.pi*(aggr_df.index.hour+1)/24)
#aggr_df['diurnal_3'] = np.sin(4*np.pi*(aggr_df.index.hour+1)/24)
#aggr_df['diurnal_4'] = np.cos(4*np.pi*(aggr_df.index.hour+1)/24)
aggr_df['diurnal'] = np.maximum(np.sin(2*np.pi*(aggr_df.index.hour+3)/24), np.zeros(len(aggr_df)))
aggr_df['month_cos'] = np.cos(2*np.pi*(aggr_df.index.month+1)/12)

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


tuple_list = [tup for tup in itertools.product(zone_target, config['crit_quant'], 
                                               config['risk_aversion'])]
#%%

step = .01
y_supp = np.arange(0, 1+step, step).round(2)
nlocations = len(y_supp)
predictor_names = list(aggr_df.columns)
predictor_names.remove('POWER')

weather_variables = ['tclw','tciw','SP','rh','tcc','10u','10v','2T','SSRD','STRD','TSR','TP']
calendar_ordinal_variables = ['Hour', 'Month']
calendar_sine_variables = ['diurnal', 'month_cos']

all_variables = weather_variables + calendar_ordinal_variables + calendar_sine_variables

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

trainX_date = aggr_df[calendar_ordinal_variables][config['start_date']:config['split_date_prob']]
comb_trainX_date = aggr_df[calendar_ordinal_variables][config['split_date_prob']:config['split_date_comb']]
testX_date = aggr_df[calendar_ordinal_variables][config['split_date_comb']:]

trainX_date_sine = aggr_df[calendar_sine_variables][config['start_date']:config['split_date_prob']]
comb_trainX_date_sine = aggr_df[calendar_sine_variables][config['split_date_prob']:config['split_date_comb']]
testX_date_sine = aggr_df[calendar_sine_variables][config['split_date_comb']:]

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

try:
    Decision_cost = pd.read_csv(f'{results_path}\\{filename_prefix}_Decision_cost.csv', index_col = 0)
    QS_df = pd.read_csv(f'{results_path}\\{filename_prefix}_QS.csv', index_col = 0)
    mean_QS = pd.read_csv(f'{results_path}\\{filename_prefix}_mean_QS.csv', index_col = 0)
except:
    Decision_cost = pd.DataFrame()
    QS_df = pd.DataFrame()
    mean_QS = pd.DataFrame()

#%%

for tup in tuple_list[row_counter:]:

    target_zone = tup[0]    
    critical_fractile = tup[1]
    risk_aversion = tup[2]
        
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

        # CART 2: date predictors
        
        cart_parameters = {'max_depth':[5, 10, 50, 100], 'min_samples_leaf':[1, 2, 5, 10]}
        cart_model_cv = GridSearchCV(DecisionTreeRegressor(), cart_parameters)
        
        trainX_cart2 = trainX_date.copy()
        comb_trainX_cart2 = comb_trainX_date.copy()
        testX_cart2 = testX_date.copy()
        
        trainX_cart2 = aggr_df[calendar_ordinal_variables+weather_variables][config['start_date']:config['split_date_prob']]
        comb_trainX_cart2 = aggr_df[calendar_ordinal_variables+weather_variables][config['split_date_prob']:config['split_date_comb']]
        testX_cart2 = aggr_df[calendar_ordinal_variables+weather_variables][config['split_date_comb']:]
                
        cart_model_cv.fit(trainX_cart2, trainY.values)    
            
        cart_model = cart_model_cv.best_estimator_
        
        probabilistic_models['cart_date'] = cart_model_cv.best_estimator_
        
        train_w_dict['cart_date'] = cart_find_weights(trainX_cart2, comb_trainX_cart2, cart_model)
        test_w_dict['cart_date'] = cart_find_weights(trainX_cart2, testX_cart2, cart_model)
        
        # Random Forest
    
        rf_parameters = {'min_samples_leaf':[2, 5, 10],'n_estimators':[50], 
                      'max_features':[1, 2, 4, len(trainX_weather.columns)]}
    
        rf_model_cv = GridSearchCV(ExtraTreesRegressor(), rf_parameters)
        rf_model_cv.fit(trainX_weather, trainY.values)    
            
        rf_model = rf_model_cv.best_estimator_
        probabilistic_models['rf'] = rf_model_cv.best_estimator_
    
        knn_point_pred = knn_model_cv.best_estimator_.predict(testX_weather)
        rf_point_pred = rf_model.predict(testX_weather)
    
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
        plt.savefig(f'{cd}\\plots\\pinball_loss_model.pdf')
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
        
        
        ### Define the rest of the supervised learning parameters     
        # projection step (used for gradient-based methods)
        
        #y_proj = cp.Variable(N_experts)
        #y_hat = cp.Parameter(N_experts)
        #proj_problem = cp.Problem(cp.Minimize(0.5*cp.sum_squares(y_proj-y_hat)), [y_proj >= 0, y_proj.sum()==1])
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
    
    #tensor_trainX = torch.FloatTensor(comb_trainX_date[:-valid_obs].values)
    #tensor_validX = torch.FloatTensor(comb_trainX_date[-valid_obs:].values)
    #tensor_testX = torch.FloatTensor(testX_date.values)

    tensor_trainX = torch.FloatTensor(comb_trainX_date_sine[:-valid_obs].values)
    tensor_validX = torch.FloatTensor(comb_trainX_date_sine[-valid_obs:].values)
    tensor_testX = torch.FloatTensor(testX_date_sine.values)

    #tensor_trainX = torch.FloatTensor(comb_trainX_onehot[:-valid_obs])
    #tensor_validX = torch.FloatTensor(comb_trainX_onehot[-valid_obs:])
    #tensor_testX = torch.FloatTensor(testX_onehot)

    train_data = torch.utils.data.TensorDataset(tensor_train_p_list[0], tensor_train_p_list[1], tensor_train_p_list[2], tensor_trainY)
    
    n_train_obs = len(train_targetY)
    n_test_obs = len(testY)
    
    
    trainZopt = np.zeros((n_train_obs, len(train_p_list)))
    testZopt = np.zeros((n_test_obs, len(test_p_list)))
        
    print('Finding optimal decisions in training set')
    for j in range(N_experts):
        temp_z_opt = solve_opt_prob(y_supp, train_p_list[j], target_problem, risk_aversion = risk_aversion, 
                                    crit_quant = critical_fractile)
        trainZopt[:,j] = temp_z_opt

    ###########% Static forecast combinations
    lambda_static_dict = {}
        
    for i,learner in enumerate(all_learners):
        temp_ind = np.zeros(N_experts)
        temp_ind[i] = 1
        lambda_static_dict[f'{learner}'] = temp_ind
        
    lambda_static_dict['Ave'] = (1/N_experts)*np.ones(N_experts)
                
    
    # Set weights to in-sample performance
    lambda_tuned_inv, _ = insample_weight_tuning(train_targetY, trainZopt, problem = target_problem,
                                                 crit_quant = critical_fractile, 
                                                 support = y_supp, risk_aversion = risk_aversion)
    
    # Benchmark/ Salva's suggestion/ weighted combination of in-sample optimal (stochastic) decisions
    lambda_ = averaging_decisions(train_targetY, trainZopt, target_problem, crit_fract = critical_fractile,
                                  support = y_supp, bounds = False, risk_aversion = risk_aversion)

    lambda_static_dict['Insample'] = lambda_tuned_inv    
    lambda_static_dict['SalvaBench'] = lambda_

    
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
    
        
        lambda_crps = crps_learning_combination(comb_trainY.values, train_p_list, support = y_supp, verbose = 1)
        
    lambda_static_dict['CRPS'] = lambda_crps
    
    #%%
    ##### Decision-focused combination for different values of gamma  
    from torch_layers_functions import *
      
    train_data_loader = create_data_loader(tensor_train_p_list + [tensor_trainY], batch_size = 512)
    valid_data_loader = create_data_loader(tensor_valid_p_list + [tensor_validY], batch_size = 512)
    
    num_epochs = 100
    patience = 10
    
    for gamma in config['gamma_list']:
        
        lpool_newsv_model = LinearPoolNewsvendorLayer(num_inputs=N_experts, support = torch.FloatTensor(y_supp),
                                                    gamma = gamma, problem = target_problem, critic_fract = critical_fractile, risk_aversion = risk_aversion,
                                                    apply_softmax = True, regularizer=None)
        
        optimizer = torch.optim.Adam(lpool_newsv_model.parameters(), lr = learning_rate)
        
        lpool_newsv_model.train_model(train_data_loader, valid_data_loader, optimizer, epochs = num_epochs, 
                                          patience = patience, projection = False, validation = False, relative_tolerance = 1e-5)
        if apply_softmax:
            lambda_static_dict[f'DF_{gamma}'] = to_np(torch.nn.functional.softmax(lpool_newsv_model.weights))
        else:
            lambda_static_dict[f'DF_{gamma}'] = to_np(lpool_newsv_model.weights)

    for m in list(lambda_static_dict.keys())[N_experts:]:
        plt.plot(lambda_static_dict[m], label = m)
    plt.legend()
    plt.show()
    
    if config['save']:
        #Prescriptions.to_csv(f'{results_path}\\{target_problem}_{critical_fractile}_{target_zone}_Prescriptions.csv')
        lamda_static_df = pd.DataFrame.from_dict(lambda_static_dict)
        lamda_static_df.to_csv(f'{results_path}\\{filename_prefix}_{critical_fractile}_lambda_static.csv')

    #%%
    ### Adaptive combination model    
    # i) fix val_loader, ii) train for gamma = 0.1, iii) add to dictionary
    # iv) create one for CRPS only (gamma == +inf)
    
    train_adapt_data_loader = create_data_loader(tensor_train_p_list + [tensor_trainX, tensor_trainY], batch_size = batch_size)
    valid_adapt_data_loader = create_data_loader(tensor_valid_p_list + [tensor_validX, tensor_validY], batch_size = batch_size)
            
    tensor_trainZopt = torch.FloatTensor(trainZopt[:-valid_obs])
    tensor_validZopt = torch.FloatTensor(trainZopt[-valid_obs:])
    tensor_testZopt = torch.FloatTensor(testZopt)
    
    adaptive_models_dict = {}
    
    adapt_models = config['train_adaptive']
    num_epochs = 500

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
            
            optimizer = torch.optim.Adam(lr_lpool_newsv_model.parameters(), lr = 1e-2)
            lr_lpool_newsv_model.train_model(train_adapt_data_loader, valid_adapt_data_loader, 
                                                  optimizer, epochs = 500, patience = 10, projection = False, relative_tolerance = 0)
            
            
            adaptive_models_dict[f'DF-LR_{gamma}'] = lr_lpool_newsv_model
    
            mlp_lpool_newsv_model = AdaptiveLinearPoolNewsvendorLayer(input_size = tensor_trainX.shape[1], hidden_sizes = [20,20,20], 
                                                                     output_size = N_experts, support = torch.FloatTensor(y_supp), 
                                                                     gamma = gamma, critic_fract = critical_fractile, risk_aversion = risk_aversion, apply_softmax = True, regularizer=None)
            
            optimizer = torch.optim.Adam(mlp_lpool_newsv_model.parameters(), lr = 1e-2)
            mlp_lpool_newsv_model.train_model(train_adapt_data_loader, valid_adapt_data_loader, 
                                                  optimizer, epochs = 500, patience = 10, projection = False, relative_tolerance = 0)
    
            adaptive_models_dict[f'DF-MLP_{gamma}'] = mlp_lpool_newsv_model
    
    #% Evaluate performance for all models
    #%
    static_models = list(lambda_static_dict) 
    adaptive_models = list(adaptive_models_dict.keys())
    all_models = static_models + adaptive_models

    lambda_adapt_dict = {}
    Prescriptions = pd.DataFrame(data = np.zeros((n_test_obs, len(all_models))), columns = all_models)
    
    # Store pinball loss and Decision cost for task-loss
    temp_QS = pd.DataFrame()
    temp_QS['Target'] = [target_zone]
    temp_QS['risk_aversion'] = risk_aversion
    
    temp_Decision_cost = pd.DataFrame()
    temp_Decision_cost['Quantile'] = [critical_fractile]
    temp_Decision_cost['risk_aversion'] = risk_aversion
    temp_Decision_cost['Target'] = target_zone
    temp_mean_QS = temp_Decision_cost.copy()

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

        temp_prescriptions = solve_opt_prob(y_supp, temp_pdf, target_problem, risk_aversion = risk_aversion, 
                                            crit_quant = critical_fractile)
           
        Prescriptions[m] = temp_prescriptions
        print(m)
            
        # Estimate task-loss for specific model
        #%
        temp_Decision_cost[m] = 100*task_loss(Prescriptions[m].values, testY.values, 
                                          target_problem, crit_quant = critical_fractile, risk_aversion = risk_aversion)
        #%
        print(m)
        
        # Evaluate QS (approximation of CRPS) for each model
        # find quantile forecasts
        temp_q_forecast = np.array([inverted_cdf(target_quant, y_supp, temp_pdf[i]) for i in range(n_test_obs)])            
        temp_qs = 100*pinball(temp_q_forecast, testY.values, target_quant).round(4)
        print(m)

        temp_QS[m] = [temp_qs]
        
        temp_CDF = temp_pdf.cumsum(1)
        H_i = 1*np.repeat(y_supp.reshape(1,-1), len(testY), axis = 0)>=testY.values.reshape(-1,1)
        
        CRPS = 100*np.square(temp_CDF - H_i).mean()            
        temp_mean_QS[m] = CRPS

    print('Decision Cost')
    print(temp_Decision_cost[all_models].mean().round(4))

    print('CRPS')
    print(temp_mean_QS[all_models].mean().round(4))
            
    try:
        Decision_cost = pd.concat([Decision_cost, temp_Decision_cost], ignore_index = True)            
        QS_df = pd.concat([QS_df, temp_QS], ignore_index = True)        
        mean_QS = pd.concat([mean_QS, temp_mean_QS], ignore_index = True)        
    except:
        Decision_cost = temp_Decision_cost.copy()
        QS_df = temp_QS.copy()            
        mean_QS = temp_mean_QS.copy()

    if config['save']:
        Decision_cost.to_csv(f'{results_path}\\{filename_prefix}_Decision_cost.csv')
        QS_df.to_csv(f'{results_path}\\{filename_prefix}_QS.csv')
        mean_QS.to_csv(f'{results_path}\\{filename_prefix}_mean_QS.csv')

        #Prescriptions.to_csv(f'{results_path}\\{target_problem}_{critical_fractile}_{target_zone}_Prescriptions.csv')
    
    row_counter += 1        