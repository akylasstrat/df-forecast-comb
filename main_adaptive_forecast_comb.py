# -*- coding: utf-8 -*-
"""
Decision-Focused Forecast Combination/ main script/ adaptive combination only

@author: a.stratigakos
"""

import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import sys, os
import pickle
import gurobipy as gp
import torch

cd = os.path.dirname(__file__)  #Current directory
sys.path.append(cd)

#from sklearn.linear_model import LinearRegression, Ridge, Lasso
from gurobi_ml import add_predictor_constr
#from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
#from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler

#from gurobi_ml.sklearn import add_decision_tree_regressor_constr, add_random_forest_regressor_constr


#project_dir=Path(cd).parent.__str__()   #project_directory

from EnsemblePrescriptiveTree import *
from EnsemblePrescriptiveTree_OOB import *
from optimization_functions import *

from LinearDecisionTree import *
from sklearn.neural_network import MLPRegressor

from utility_functions import *
from optimal_transport_functions import *
from torch_layers_functions import *

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
    m.addConstr( pinball_loss_i >= crit_fract*error)
    m.addConstr( pinball_loss_i >= (crit_fract-1)*error)

    m.setObjective( (1-risk_aversion)*pinball_loss_i.sum() + risk_aversion*(error@error) , gp.GRB.MINIMIZE)
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

from torch.utils.data import Dataset, DataLoader

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
    
    elif problem == 'reg_trad':

        deviation = actual_copy - pred_copy
        pinball_loss = np.maximum(kwargs['crit_quant']*deviation, (kwargs['crit_quant']-1)*deviation)  
        square_loss = np.square(deviation)          
        
        return (1- kwargs['risk_aversion'])*pinball_loss.mean() +  kwargs['risk_aversion']*square_loss.mean()
        
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
            
            # constraints
            m.addConstr(deviation == (target_scen - offer) )            
            m.addConstr(loss >= crit_quant*deviation )
            m.addConstr(loss >= (crit_quant-1)*deviation )
            
            for row in range(len(weights)):
                
                m.setObjective( (1-risk_aversion)*(weights[row]@loss) 
                               + risk_aversion*(deviation@(deviation*weights[row])), gp.GRB.MINIMIZE)

                m.optimize()
                Prescriptions[row] = offer.X[0]
                
            return Prescriptions
    
    elif problem =='mse':
        return (target_scen@weights)
    elif problem == 'newsvendor':
        
        m = gp.Model()
        m.setParam('OutputFlag', 0)

        # target variable
        offer = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'offer')
        loss = m.addMVar(n_scen, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'aux')
                                
        m.addConstr(loss >= crit_quant*(target_scen - offer) )
        m.addConstr(loss >= (crit_quant-1)*(target_scen - offer) )
        
        m.setObjective( weights@loss, gp.GRB.MINIMIZE)
        m.optimize()
        
        return offer.X        

def nn_params():
    'NN hyperparameters'
    nn_params = {}
    nn_params['patience'] = 10
    nn_params['batch_size'] = 512
    nn_params['num_epochs'] = 1000
    nn_params['learning_rate'] = 1e-2
    nn_params['apply_softmax'] = True
    return nn_params

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
    
#%%
config = params()
hyperparam = tree_params()
nn_hparam = nn_params()

results_path = f'{cd}\\results'
data_path = f'{cd}\\data'

aggr_wind_df = pd.read_csv(f'{data_path}\\GEFCom2014-processed.csv', index_col = 0, header = [0,1])
#%%

#!!!! Add randomization based on the iteration counter here

target_problem = config['problem']

train_forecast_model = False
generate_forecasts = True

try:
    Decision_cost = pd.read_csv(f'{cd}\\results\\fix_{target_problem}_total_linearpool_Decision_cost.csv', index_col = 0)
    QS_df = pd.read_csv(f'{cd}\\results\\fix_{target_problem}_total_linear_pool_QS.csv', index_col = 0)
    row_counter = len(Decision_cost)
except: 
    row_counter = 0

if target_problem == 'newsvendor':
    config['risk_aversion'] = [0]
    tuple_list = [tup for tup in itertools.product(config['target_zones'], config['crit_quant'])]
elif target_problem == 'reg_trad':
    tuple_list = [tup for tup in itertools.product(config['target_zones'], config['crit_quant'], 
                                                   config['risk_aversion'])]

#%%
# Set up some problem parameters
all_zones = [f'Z{i}' for i in range(1,11)]
#target_zone = np.random.choice(all_zones)
# number of forecasts to combine
N_experts = config['N_experts']

# number of observations to train prob. forecasting model
N_sample = len(aggr_wind_df)//4

step = .01
y_supp = np.arange(0, 1+step, step).round(2)
nlocations = len(y_supp)
pred_col = ['wspeed10', 'wdir10_rad', 'wspeed100', 'wdir100_rad']

### Create train/test sets for all series

trainY = aggr_wind_df.xs('POWER', axis=1, level=1)[all_zones][config['start_date']:config['split_date']][:N_sample].round(2)
comb_trainY = aggr_wind_df.xs('POWER', axis=1, level=1)[all_zones][config['start_date']:config['split_date']][N_sample:].round(2)
testY = aggr_wind_df.xs('POWER', axis=1, level=1)[all_zones][config['split_date']:].round(2)

trainX_allzones = aggr_wind_df[all_zones][config['start_date']:config['split_date']][:N_sample].round(2)
comb_trainX_allzones = aggr_wind_df[all_zones][config['start_date']:config['split_date']][N_sample:].round(2)    
testX_allzones = aggr_wind_df[all_zones][config['split_date']:].round(2)

### NN hyperparameters
patience = nn_hparam['patience']
batch_size = nn_hparam['batch_size']
num_epochs = nn_hparam['num_epochs']
learning_rate = nn_hparam['learning_rate']
apply_softmax = nn_hparam['apply_softmax']

for tup in tuple_list[row_counter:]:

    target_zone = tup[0]    
    critical_fractile = tup[1]
    risk_aversion = tup[2]
                
    np.random.seed(row_counter)
    
    print(f'Quantile:{critical_fractile}, zone:{target_zone}')
    #target_zone = config['target_zone']
    #expert_zones = ['Z2', 'Z4', 'Z8', 'Z9']
        
    
    expert_zones = all_zones.copy()
    expert_zones.remove(target_zone)
    expert_zones = expert_zones[:config['N_experts']]
    #expert_zones = list(np.random.choice(expert_zones, config['N_experts'], replace = False))
    #expert_zones = list(np.random.choice(expert_zones, config['N_experts'], replace = False))

    ### Feature data for target location/zone 
    local_features_df = aggr_wind_df[target_zone].copy()
    local_features_df['wspeed10_sq'] = np.power(local_features_df['wspeed10'],2)
    local_features_df['wspeed10_cb'] = np.power(local_features_df['wspeed10'],3)

    local_features_df['wspeed100_sq'] = np.power(local_features_df['wspeed100'],2)
    local_features_df['wspeed100_cb'] = np.power(local_features_df['wspeed100'],3)
    
    local_features_df[['diurnal_1', 'diurnal_2', 'diurnal_3', 'diurnal_4']] = aggr_wind_df[['diurnal_1', 'diurnal_2', 'diurnal_3', 'diurnal_4']]
    
    local_feat = ['wspeed10', 'wspeed100', 'wdir10_rad', 'wdir100_rad', 'diurnal_1', 'diurnal_2', 'diurnal_3', 'diurnal_4']
    trainX_local = local_features_df[local_feat][config['start_date']:config['split_date']][:N_sample].round(2)
    comb_trainX_local = local_features_df[local_feat][config['start_date']:config['split_date']][N_sample:].round(2)
    testX_local = local_features_df[local_feat][config['split_date']:].round(2)

    # supervised sets for experts    
    #comb_trainX_exp = aggr_wind_df[expert_zones][config['start_date']:config['split_date']][N_sample:].round(2)    
    #trainX_exp = aggr_wind_df[expert_zones][config['start_date']:config['split_date']][:N_sample].round(2)
    #testX_exp = aggr_wind_df[expert_zones][config['split_date']:].round(2)


    # number of training observations for the combination model
    n_obs = len(comb_trainY)
    n_test_obs = len(testY)
    
    if train_forecast_model:
        #% Train experts, i.e., probabilistic forecasting models in adjacent locations    
        prob_models = {}
        
        for i, zone in enumerate(all_zones):
            print(f'Training model for zone:{zone}')
                     
            temp_model = EnsemblePrescriptiveTree(n_estimators = 50, 
                                                  max_features = len(pred_col), type_split = 'quant' )
            temp_model.fit(trainX_allzones[zone][pred_col].values, trainY[zone].values,
                           quant = np.arange(.01, 1, .01), problem = 'mse') 
            
            prob_models[zone] = temp_model
        
        pickle.dump(prob_models, open(f'{cd}\\results\\prob_forecast_models.sav', 'wb'))
    else:
        # load the model from disk
        prob_models = pickle.load(open(f'{cd}\\results\\prob_forecast_models.sav', 'rb'))
    #%%
    #% Generate predictions for training/test set for forecast combination
    # find local weights for meta-training set/ map weights to support locations
    if ('train_w_dict' not in locals()) or ('test_w_dict' not in locals()):
        print('Generating prob. forecasts for train/test set...')
        train_w_dict = {}
        test_w_dict = {}
        
        for i, zone in enumerate(all_zones):
            train_w_dict[zone] = prob_models[zone].find_weights(comb_trainX_allzones[zone][pred_col].values, 
                                                               trainX_allzones[zone][pred_col].values) 
            test_w_dict[zone] = prob_models[zone].find_weights(testX_allzones[zone][pred_col].values, 
                                                              trainX_allzones[zone][pred_col].values)

    # Translate weighted observations to discrete PDFs
    train_p_list = []
    test_p_list = []
    
    for i, zone in enumerate(all_zones):
        if zone == target_zone: continue
        train_p_list.append(wemp_to_support(train_w_dict[zone], trainY[zone].values, y_supp))
        test_p_list.append(wemp_to_support(test_w_dict[zone], trainY[zone].values, y_supp))


    #% Visualize some prob. forecasts
    #%
    # step 1: find inverted CDFs
    F_inv = [np.array([inverted_cdf([.05, .10, .90, .95] , trainY[zone].values, train_w_dict[zone][i]) for i in range(500)]) for j,zone in enumerate(expert_zones)]
    
    plt.plot(comb_trainY[target_zone][200:300])
    for i in [0,2]:
        #plt.fill_between(np.arange(100), F_inv[i][200:300,0], F_inv[i][200:300,-1], alpha = .3, color = 'red')
        
        plt.fill_between(np.arange(100), F_inv[i][200:300,0], F_inv[i][200:300,-1], alpha = .3)
    plt.show()
    
    ### Define the rest of the supervised learning parameters     
    # projection step (used for gradient-based methods)
    
    #y_proj = cp.Variable(N_experts)
    #y_hat = cp.Parameter(N_experts)
    #proj_problem = cp.Problem(cp.Minimize(0.5*cp.sum_squares(y_proj-y_hat)), [y_proj >= 0, y_proj.sum()==1])
    
    train_targetY = comb_trainY[target_zone].values.reshape(-1)
    
    # Supervised learning set as tensors for PyTorch
    tensor_trainY = torch.FloatTensor(train_targetY[:-800])
    tensor_train_p = torch.FloatTensor(np.column_stack((train_p_list)))
    tensor_train_p_list = [torch.FloatTensor(train_p_list[i][:-800]) for i in range(N_experts)]

    tensor_validY = torch.FloatTensor(train_targetY[-800:])
    tensor_valid_p_list = [torch.FloatTensor(train_p_list[i][-800:]) for i in range(N_experts)]
    
    feat_scaler = MinMaxScaler()
    feat_scaler.fit(comb_trainX_local)
    
    tensor_trainX = torch.FloatTensor(feat_scaler.transform(comb_trainX_local[:-800]))
    tensor_validX = torch.FloatTensor(feat_scaler.transform(comb_trainX_local[-800:]))
    tensor_testX = torch.FloatTensor(feat_scaler.transform(testX_local))
    
    train_data = torch.utils.data.TensorDataset(tensor_train_p_list[0], tensor_train_p_list[1], tensor_train_p_list[2], tensor_trainY)
    
    n_train_obs = len(train_targetY)
    n_test_obs = len(testY)
    
    #%%
    if target_problem in ['newsvendor', 'reg_trad']:
        
        trainZopt = np.zeros((n_train_obs, len(train_p_list)))
        testZopt = np.zeros((n_test_obs, len(test_p_list)))
        
        print('Finding optimal decisions in training set')
        for j in range(N_experts):
            temp_z_opt = solve_opt_prob(y_supp, train_p_list[j], target_problem, risk_aversion = risk_aversion, 
                                        crit_quant = critical_fractile)
            trainZopt[:,j] = temp_z_opt

        ###########% Static forecast combinations
        lambda_static_dict = {}
        
        for i in range(N_experts):
            temp_ind = np.zeros(N_experts)
            temp_ind[i] = 1
            lambda_static_dict[f'Model-{i}'] = temp_ind
            
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


        #% PyTorch layers
        #%%
        train_data_loader = create_data_loader(tensor_train_p_list + [tensor_trainY], batch_size = 512)
        valid_data_loader = create_data_loader(tensor_valid_p_list + [tensor_validY], batch_size = 512)

        #### CRPS minimization/ with torch layer
        lpool_crps_model = LinearPoolCRPSLayer(num_inputs=N_experts, support = torch.FloatTensor(y_supp),
                                               apply_softmax = True)
        optimizer = torch.optim.Adam(lpool_crps_model.parameters(), lr = learning_rate)
        lpool_crps_model.train_model(train_data_loader, optimizer, epochs = num_epochs, patience = patience, 
                                     projection = True)
        
        if apply_softmax:
            lambda_static_dict['CRPS'] = to_np(torch.nn.functional.softmax(lpool_crps_model.weights))
        else:
            lambda_static_dict['CRPS'] = to_np(lpool_crps_model.weights)
        #%%
        from torch_layers_functions import *
        ##### Decision-focused combination for different values of gamma        
        for gamma in [0, 0.1, 1]:
            
            lpool_newsv_model = LinearPoolNewsvendorLayer(num_inputs=N_experts, support = torch.FloatTensor(y_supp), 
                                                        gamma = gamma, critic_fract = critical_fractile, risk_aversion = risk_aversion,
                                                        apply_softmax = True, regularizer=None)
            
            optimizer = torch.optim.Adam(lpool_newsv_model.parameters(), lr = learning_rate)
            
            lpool_newsv_model.train_model(train_data_loader, valid_data_loader, optimizer, epochs = num_epochs, 
                                              patience = patience, projection = False, validation = True, relative_tolerance = 0.001)
            if apply_softmax:
                lambda_static_dict[f'DF_{gamma}'] = to_np(torch.nn.functional.softmax(lpool_newsv_model.weights))
            else:
                lambda_static_dict[f'DF_{gamma}'] = to_np(lpool_newsv_model.weights)
        #%%
        print(lambda_static_dict)
        
        for m in list(lambda_static_dict.keys())[N_experts:]:
            plt.plot(lambda_static_dict[m], label = m)
        plt.legend()
        plt.show()
        
        
        ### Adaptive combination model
        
        # i) fix val_loader, ii) train for gamma = 0.1, iii) add to dictionary
        # iv) create one for CRPS only (gamma == +inf)
        
        train_adapt_data_loader = create_data_loader(tensor_train_p_list + [tensor_trainX, tensor_trainY], batch_size = batch_size)
        valid_adapt_data_loader = create_data_loader(tensor_valid_p_list + [tensor_validX, tensor_validY], batch_size = batch_size)
                
        tensor_trainZopt = torch.FloatTensor(trainZopt[:-800])
        tensor_validZopt = torch.FloatTensor(trainZopt[-800:])
        tensor_testZopt = torch.FloatTensor(testZopt)
        
        adaptive_models_dict = {}
        
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
        
        
        #%%
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
        
        #%%
        from torch_layers_functions import *

        for gamma in [0, 0.1, 1]:
                        
            lr_lpool_newsv_model = AdaptiveLinearPoolNewsvendorLayer(input_size = tensor_trainX.shape[1], hidden_sizes = [], 
                                                                     output_size = N_experts, support = torch.FloatTensor(y_supp), gamma = gamma, critic_fract = critical_fractile, 
                                                                     risk_aversion = risk_aversion, apply_softmax = True, regularizer=None)
            
            optimizer = torch.optim.Adam(lr_lpool_newsv_model.parameters(), lr = learning_rate)
            lr_lpool_newsv_model.train_model(train_adapt_data_loader, valid_adapt_data_loader, 
                                                  optimizer, epochs = 1000, patience = patience, projection = False, relative_tolerance = 0.001)
            
            
            adaptive_models_dict[f'DF-LR_{gamma}'] = lr_lpool_newsv_model

            mlp_lpool_newsv_model = AdaptiveLinearPoolNewsvendorLayer(input_size = tensor_trainX.shape[1], hidden_sizes = [20,20,20], 
                                                                     output_size = N_experts, support = torch.FloatTensor(y_supp), 
                                                                     gamma = gamma, critic_fract = critical_fractile, risk_aversion = risk_aversion, apply_softmax = True, regularizer=None)
            
            optimizer = torch.optim.Adam(mlp_lpool_newsv_model.parameters(), lr = learning_rate)
            mlp_lpool_newsv_model.train_model(train_adapt_data_loader, valid_adapt_data_loader, 
                                                  optimizer, epochs = 1000, patience = patience, projection = False)

            adaptive_models_dict[f'DF-MLP_{gamma}'] = mlp_lpool_newsv_model


        #%%
        # turn probability vectors to decisions/ closed-form solution for newsvendor problem    
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

        target_quant = np.arange(0.1, 1, 0.1).round(2)

        for j, m in enumerate(all_models):
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
                
            # Estimate task-loss for specific model
            temp_Decision_cost[m] = 100*task_loss(Prescriptions[m].values, testY[target_zone].values, 
                                              target_problem, crit_quant = critical_fractile, risk_aversion = risk_aversion)
            
            # Evaluate QS (approximation of CRPS) for each model
            # find quantile forecasts
            temp_q_forecast = np.array([inverted_cdf(target_quant, y_supp, temp_pdf[i]) for i in range(n_test_obs)])            
            temp_qs = 100*pinball(temp_q_forecast, testY[target_zone].values, target_quant).round(4)
            
            temp_QS[m] = [temp_qs]
            
            if m in ['Ave', 'SalvaBench', 'CRPS', 'DF_0.1', 'DF_1']:
                plt.plot(temp_qs, label = m)
        plt.legend()
        plt.ylabel('Pinball loss')
        plt.xticks(np.arange(len(target_quant)), target_quant)
        plt.xlabel('Quantile')
        plt.show()

        print('Decision Cost')
        print(temp_Decision_cost[all_models].mean().round(4))

        if row_counter == 0: 
            Decision_cost = temp_Decision_cost.copy()
        else:
            Decision_cost = pd.concat([Decision_cost, temp_Decision_cost], ignore_index = True)
            
        if row_counter == 0: 
            QS_df = temp_QS.copy()
        else:
            QS_df = pd.concat([QS_df, temp_QS], ignore_index = True)        
        
        if config['save']:
            Decision_cost.to_csv(f'{cd}\\results\\fix_{target_problem}_{critical_fractile}_total_linearpool_Decision_cost.csv')
            QS_df.to_csv(f'{cd}\\results\\fix_{target_problem}_{critical_fractile}_total_linear_pool_QS.csv')
            Prescriptions.to_csv(f'{cd}\\results\\fix_{target_problem}_{critical_fractile}_{target_zone}_Prescriptions.csv')
        
        row_counter += 1
        
        ### Adaptive Forecast Combinations
        
        #lambda_adapt_lr = lr_adapt_Newsv_Comb_Model.predict_weights(tensor_testX)    
        #lambda_adapt_mlp = mlp_adapt_Newsv_Comb_Model.predict_weights(tensor_testX)
        
        #p_adapt_cc_lr = np.array([sum([lambda_adapt_lr[i,j]*test_p_list[j][i] for j in range(N_experts)]) for i in range(n_test_obs)])    
        #p_adapt_cc_mlp = np.array([sum([lambda_adapt_mlp[i,j]*test_p_list[j][i] for j in range(N_experts)]) for i in range(n_test_obs)])
        
        
        #Prescriptions['BRC-DT'] = np.array([inverted_cdf([critical_fractile], y_supp, p_brc_dt[i]) for i in range(len(test_localY))]).reshape(-1)
            
        
#%%
mean_QS_df = Decision_cost.copy()

for m in models:
    for i in range(mean_QS_df[m].shape[0]):
        mean_QS_df[m].iloc[i] = QS_df[m].iloc[i].mean()


