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
import pickle

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor


cd = os.path.dirname(__file__)  #Current directory
sys.path.append(cd)

from sklearn.preprocessing import MinMaxScaler

from utility_functions import *
# from optimal_transport_functions import *
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
        
        m.setObjective( ((1-risk_aversion)*pinball_loss_i.sum() + risk_aversion*(error@error))/n_obs, gp.GRB.MINIMIZE)

    elif problem == 'pwl':
        m.addConstr( pinball_loss_i >= crit_fract*error)
        m.addConstr( pinball_loss_i >= (-0.5)*error)
        m.addConstr( pinball_loss_i >= (crit_fract - 1)*(error + 0.1))
    
        m.setObjective( ((1-risk_aversion)*pinball_loss_i.sum() + risk_aversion*(error@error))/n_obs, gp.GRB.MINIMIZE)
        
    m.optimize()
    print('In-sample Task loss')
    print(f'{m.ObjVal}')
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

def insample_weight_tuning(target_y, train_z_opt, train_prob_list, problem = 'newsvendor', 
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
    
    n_obs = train_z_opt
    n_models = train_z_opt.shape[1]
    
    risk_aversion = kwargs['risk_aversion']
    crit_quant = kwargs['crit_quant']
    ### Find optimal decisions under perfect foresight information
    
    print('Solve in-sample stochastic problems...')
    
    #z_opt = np.zeros((n_obs, n_models))
    insample_inverse_cost = np.zeros((n_models))
    insample_regret = np.zeros((n_models))
    in_sample_CRPS = np.zeros((n_models))
    insample_cost_regularized = np.zeros((n_models))
    
    for j in range(n_models):
        
        # Estimate in-sample Regret/ Task loss
        insample_regret[j] = task_loss(train_z_opt[:,j], target_y.reshape(-1), problem, risk_aversion = risk_aversion, 
                                     crit_quant = crit_quant)
        
        # Estimate in-sample CRPS        
        temp_CDF = train_prob_list[j].cumsum(1)
        H_i = 1*np.repeat(y_supp.reshape(1,-1), len(comb_trainY), axis = 0)>=comb_trainY.values.reshape(-1,1)
        in_sample_CRPS[j] =  np.square(temp_CDF - H_i).mean()
        
        # In-sample regularized cost
        if regularization_gamma =='inf':
            insample_cost_regularized[j] = in_sample_CRPS[j]
        else:
            insample_cost_regularized[j] = insample_regret[j] + regularization_gamma*in_sample_CRPS[j]            

        insample_inverse_cost[j] = 1/insample_cost_regularized[j]
    
    # Find lambdas
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

            print(f'In-sample cost:{m.ObjVal}')
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
 

def tree_params():
    ''' Hyperparameters for tree algorithms'''
    params = {}
    params['n_estimators'] = 50
    params['n_min'] = 2
    params['max_features'] = 1
    return params

def gd_params():
    'Gradient-descent hyperparameters'
    nn_params = {}
    nn_params['patience'] = 10
    nn_params['batch_size'] = 512  
    nn_params['num_epochs'] = 500
    nn_params['learning_rate'] = 1e-2
    nn_params['feasibility_method'] = 'softmax'
    return nn_params

def params():
    ''' Set up the experiment parameters'''

    params = {}

    params['start_date'] = '2012-01-01'
    params['split_date_prob'] = '2013-01-01' # Defines train/test split
    params['split_date_comb'] = '2014-01-01' # Defines train/test split
    params['end_date'] = '2014-07-01'
    
    # Experimental setup parameters
    params['problem'] = 'reg_trad' # {mse, newsvendor, cvar, reg_trad, pwl}// Do not change
    params['gamma_list'] = [0, 0.1, 1]
    params['target_zone'] = [2] # [1,2,3] This variable selects the respective solar plant from the GEFCom2014 data [Z1, Z2, Z3]
    
    # Problem parameters        
    params['crit_quant'] = np.arange(0.1, 1, 0.1).round(2) # Critical quantile, each value runs a different experiment
    params['risk_aversion'] = [0.2] # Trading risk-aversion// Do not change

    params['save'] = True # If True, then saves trained models and results
    params['train_static'] = True # If True, then trains static combination models
    params['train_adaptive'] = True # If True, then trains adaptive combination models
    
    return params

#%%
    
config = params()
hyperparam = tree_params()
gd_hparam = gd_params()

results_path = f'{cd}\\results\\solar_trading_results_' + gd_hparam['feasibility_method']
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
### Gradient algorithm and NN-model hyperparameters
patience = gd_hparam['patience']
batch_size = gd_hparam['batch_size']
num_epochs = gd_hparam['num_epochs']
learning_rate = gd_hparam['learning_rate']
# apply_softmax = nn_hparam['apply_softmax']
row_counter = 0

try:
    # Load and update existing results
    Decision_cost = pd.read_csv(f'{results_path}\\{filename_prefix}_Decision_cost.csv', index_col = 0)
    QS_df = pd.read_csv(f'{results_path}\\{filename_prefix}_QS.csv', index_col = 0)
    mean_QS = pd.read_csv(f'{results_path}\\{filename_prefix}_mean_QS.csv', index_col = 0)
except:
    Decision_cost = pd.DataFrame()
    QS_df = pd.DataFrame()
    mean_QS = pd.DataFrame()
#%%
# Iterate over combinations of problem parameters (for a single power plant)
for tup in tuple_list[row_counter:]:

    target_zone = tup[0]    
    critical_fractile = tup[1]
    risk_aversion = tup[2]
    
    print(f'Quantile:{critical_fractile}, zone:{target_zone}')
    
    np.random.seed(1234)

    if row_counter == 0:        
        
        ###### Train expert forecasting models, derive component forecasts (only required for the first iteration)
        print('Train prob. forecasting models and generate out-of-sample forecasts (only required in first iteration)')
        # store predictions
        train_w_dict = {}
        test_w_dict = {}
        probabilistic_models = {}

        ## kNN
        print('Training kNN')
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
        print('Training CART')
        
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
        print('Training RF')
    
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

        ## Climatology forecast
        #train_w_dict['clim'] = np.ones((comb_trainY.shape[0], trainY.shape[0]))*(1/len(trainY))
        #test_w_dict['clim'] = np.ones((testY.shape[0], trainY.shape[0]))*(1/len(trainY))
        
        # Map weighted historical observations to weights of discrete PDF support
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
        
        
        # In-sample performance estimate CRPS
        print('Evaluate in-sample CRPS')
        for j, m in enumerate(all_learners): 
            
            temp_CDF = train_p_list[j].cumsum(1)
            H_i = 1*np.repeat(y_supp.reshape(1,-1), len(comb_trainY), axis = 0)>=comb_trainY.values.reshape(-1,1)
            
            CRPS = np.square(temp_CDF - H_i).mean()
    
            print(f'{m}:{CRPS}')

        temp_CDF = (train_p_list[0].cumsum(1) + train_p_list[1].cumsum(1) + train_p_list[2].cumsum(1))/3
        H_i = 1*np.repeat(y_supp.reshape(1,-1), len(comb_trainY), axis = 0)>=comb_trainY.values.reshape(-1,1)
        
        CRPS = np.square(temp_CDF - H_i).mean()
        print(f'OLP:{CRPS}')
        
        #% In-sample task loss performance     
        print('Evaluate in-sample task loss')
        for j, m in enumerate(all_learners):
            # Combine PDFs for each observation
            temp_prescriptions = solve_opt_prob(y_supp, train_p_list[j], target_problem, risk_aversion = risk_aversion, crit_quant = critical_fractile)

            # Estimate task-loss for specific model
            temp_decision_cost = 100*task_loss(temp_prescriptions, comb_trainY.values, target_problem, crit_quant = critical_fractile, risk_aversion = risk_aversion)
            print(f'Model:{m}')        
            print(f'Task loss:{temp_decision_cost}')        

        # Combine PDFs for each observation
        ave_pdf = (train_p_list[0] + train_p_list[1] + train_p_list[2])/3
        
        temp_prescriptions = solve_opt_prob(y_supp, ave_pdf, target_problem, risk_aversion = risk_aversion, crit_quant = critical_fractile)
        # Estimate task-loss for specific model
        temp_decision_cost = 100*task_loss(temp_prescriptions, comb_trainY.values, target_problem, crit_quant = critical_fractile, risk_aversion = risk_aversion)

        print('Model:OLP')        
        print(f'Task loss:{temp_decision_cost}')        


        #
        # Evaluate probabilistic forecast using Quantile Score, generate Figure 1
        print('Evaluate out-of-sample Quantile Score and generate Figure 1')
        target_quant = np.arange(.01, 1, .01)
        for j,m in enumerate(all_learners):
            temp_pdf = test_p_list[j]
    
            temp_q_forecast = np.array([inverted_cdf(target_quant, y_supp, temp_pdf[i]) for i in range(n_test_obs)])            
            temp_qs = 100*pinball(temp_q_forecast, testY.values, target_quant).round(4)
            plt.plot(temp_qs, label = m)
        #plt.plot(100*pinball(test_q_pred, testY[target_zone].values, target_quant).round(4), label = 'QR reg')
        plt.legend(['$k$$\mathtt{NN}$', '$\mathtt{CART}$', '$\mathtt{RF}$'])
        plt.ylabel('Quantile Score')
        plt.xlabel('Quantile')
        plt.xticks(np.arange(10, 100, 10), np.arange(0.1, 1, .1).round(2))
        plt.savefig(f'{cd}\\plots\\quantile_score_solar_forecast.pdf')
        plt.show()
        #%
        #% Visualize some prob. forecasts for sanity check
        #%
        print('Visualize prob. forecasts for sanity check')

        # step 1: find inverted CDFs
        F_inv = [np.array([inverted_cdf([.05, .10, .90, .95] , trainY.values, train_w_dict[learner][i]) for i in range(500)]) 
                 for j,learner in enumerate(all_learners)]
        
        plt.plot(comb_trainY[200:250].values)
        for i, learner in enumerate(all_learners):
            #plt.fill_between(np.arange(100), F_inv[i][200:300,0], F_inv[i][200:300,-1], alpha = .3, color = 'red')
            plt.fill_between(np.arange(50), F_inv[i][200:250,0], F_inv[i][200:250,-1], alpha = .3, label = learner)
        plt.legend()
        plt.ylabel('p.u.')
        plt.xlabel('Timesteps')
        plt.title('Prob. Forecasts Example')
        plt.show()
        

        N_experts = len(all_learners)
    
    print('Learn Static Combinations')
    #% ########### Static combinations 
    train_targetY = comb_trainY.values.reshape(-1)
    
    # Supervised learning set as tensors for PyTorch
    valid_obs = round(0.15*len(train_targetY))
    
    # Training data sets without validation
    tensor_trainY_full = torch.FloatTensor(train_targetY)
    tensor_train_p_full = torch.FloatTensor(np.column_stack((train_p_list)))
    tensor_train_p_list_full = [torch.FloatTensor(train_p_list[i]) for i in range(N_experts)]
    
    # Training data when considering validation
    tensor_trainY = torch.FloatTensor(train_targetY[:-valid_obs])
    tensor_train_p = torch.FloatTensor(np.column_stack((train_p_list)))
    tensor_train_p_list = [torch.FloatTensor(train_p_list[i][:-valid_obs]) for i in range(N_experts)]

    tensor_validY = torch.FloatTensor(train_targetY[-valid_obs:])
    tensor_valid_p_list = [torch.FloatTensor(train_p_list[i][-valid_obs:]) for i in range(N_experts)]
    
    tensor_trainX = torch.FloatTensor(comb_trainX_date_sine[:-valid_obs].values)
    tensor_validX = torch.FloatTensor(comb_trainX_date_sine[-valid_obs:].values)
    tensor_testX = torch.FloatTensor(testX_date_sine.values)

    train_data = torch.utils.data.TensorDataset(tensor_train_p_list[0], tensor_train_p_list[1], tensor_train_p_list[2], tensor_trainY)
    
    n_train_obs = len(train_targetY)
    n_test_obs = len(testY)
    
    trainZopt = np.zeros((n_train_obs, len(train_p_list)))
    testZopt = np.zeros((n_test_obs, len(test_p_list)))
        
    # print('Finding optimal decisions in training set')
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
    
    # Ordinary linear pooling
    lambda_static_dict['Ave'] = (1/N_experts)*np.ones(N_experts)
    
    #!!!!!!!!!!!! Add regularization here 
    #%
    # Inverse Performance-based weights (invW in the paper)
    for g in (config['gamma_list'] + ['inf']):
        lambda_tuned_inv, _ = insample_weight_tuning(train_targetY, trainZopt, train_p_list, regularization_gamma=g, problem = target_problem,
                                                     crit_quant = critical_fractile, support = y_supp, risk_aversion = risk_aversion)
        
        lambda_static_dict[f'invW-{g}'] = lambda_tuned_inv    
        
    #%
    # Benchmark/ Salva's suggestion/ weighted combination of in-sample optimal (stochastic) decisions
    # *** This method is not presented in the paper ***
    lambda_ = averaging_decisions(train_targetY, trainZopt, target_problem, crit_fract = critical_fractile,
                                  support = y_supp, bounds = False, risk_aversion = risk_aversion)

    lambda_static_dict['SalvaBench'] = lambda_

    #%
    ###### CRPS learning (optimized once)
    train_data_loader_full = create_data_loader(tensor_train_p_list_full + [tensor_trainY_full], batch_size = 512, shuffle = False)
    train_data_loader = create_data_loader(tensor_train_p_list + [tensor_trainY], batch_size = 512, shuffle = False)
    valid_data_loader = create_data_loader(tensor_valid_p_list + [tensor_validY], batch_size = 512, shuffle = False)
    
    if row_counter == 0:
        #### CRPS minimization using gradient-based approach
        lpool_crps_model = LinearPoolCRPSLayer(num_inputs=N_experts, support = torch.FloatTensor(y_supp), 
                                               feasibility_method = gd_hparam['feasibility_method'])
        optimizer = torch.optim.Adam(lpool_crps_model.parameters(), lr = 1e-2)
        lpool_crps_model.train_model(train_data_loader_full, valid_data_loader, optimizer, epochs = 500, patience = 25)
        
        #%
        # Sanity check: assess in-sample perfomrance
        # lambda_crps = crps_learning_combination(comb_trainY.values, train_p_list, support = y_supp)

    #%
    lambda_static_dict['CRPS'] = lpool_crps_model.get_weights()

    #%
    ##### Decision-focused learning combination for different values of gamma       
    train_data_loader_full = create_data_loader(tensor_train_p_list_full + [tensor_trainY_full], batch_size = 512, shuffle= False)    
    train_data_loader = create_data_loader(tensor_train_p_list + [tensor_trainY], batch_size = 512, shuffle= False)
    valid_data_loader = create_data_loader(tensor_valid_p_list + [tensor_validY], batch_size = 512, shuffle= False)
    
    num_epochs = 100
    patience = 5
    
    # Iterate over values of hyperparameter \gamma
    print('Learn decision-focused static combinations')
    for gamma in config['gamma_list']:
        
        lpool_newsv_model = LinearPoolNewsvendorLayer(num_inputs=N_experts, support = torch.FloatTensor(y_supp),
                                                    gamma = gamma, problem = target_problem, critic_fract = critical_fractile, risk_aversion = risk_aversion, 
                                                    feasibility_method = gd_hparam['feasibility_method'])
        
        optimizer = torch.optim.Adam(lpool_newsv_model.parameters(), lr = 1e-2)
        
        lpool_newsv_model.train_model(train_data_loader_full, valid_data_loader, optimizer, epochs = num_epochs, 
                                          patience = patience, validation = False, relative_tolerance = 1e-5)

        
        lambda_static_dict[f'DF_{gamma}'] = lpool_newsv_model.get_weights()

        # print('Weights')
        # print(lambda_static_dict[f'DF_{gamma}'])

    for m in list(lambda_static_dict.keys())[N_experts:]:
        plt.plot(lambda_static_dict[m], label = m)
    plt.legend()
    plt.title('Static Combination Weights')
    plt.show()
    
    # save static combination weights as dataframe
    if config['save']:
        lamda_static_df = pd.DataFrame.from_dict(lambda_static_dict)
        lamda_static_df.to_csv(f'{results_path}\\{filename_prefix}_{critical_fractile}_lambda_static.csv')

        with open(f'{results_path}\\{filename_prefix}_{critical_fractile}_lambda_static_dict.pickle', 'wb') as handle:
            pickle.dump(lambda_static_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # with open(f'{results_path}\\{filename_prefix}_{critical_fractile}_lambda_static_dict.pickle', 'rb') as handle:
        #     lambda_static_dict = pickle.load(handle)

    #%
    ### Adaptive/ Conditional combinations: linear pool weights adapt to contextual information
    
    train_adapt_data_loader = create_data_loader(tensor_train_p_list + [tensor_trainX, tensor_trainY], batch_size = batch_size, shuffle = False)
    valid_adapt_data_loader = create_data_loader(tensor_valid_p_list + [tensor_validX, tensor_validY], batch_size = batch_size, shuffle = False)
            
    tensor_trainZopt = torch.FloatTensor(trainZopt[:-valid_obs])
    tensor_validZopt = torch.FloatTensor(trainZopt[-valid_obs:])
    tensor_testZopt = torch.FloatTensor(testZopt)
    
    adaptive_models_dict = {}
    
    num_epochs = 500
    #%
    ####### Train Conditional/Adaptive Combination models

    if config['train_adaptive'] == True:
        
        print('Learn Conditional Combination Models')
        ### CRPS Learning - Linear Regression
        torch.manual_seed(0)
        lr_lpool_crps_model = AdaptiveLinearPoolCRPSLayer(input_size = tensor_trainX.shape[1], hidden_sizes = [], output_size = N_experts, 
                                                          support = torch.FloatTensor(y_supp))        
        optimizer = torch.optim.Adam(lr_lpool_crps_model.parameters(), lr = 1e-2)        
        lr_lpool_crps_model.train_model(train_adapt_data_loader, valid_adapt_data_loader, optimizer, epochs = num_epochs, 
                                          patience = 10, validation = True)
    

        adaptive_models_dict['CRPS-LR'] = lr_lpool_crps_model
        
        ### CRPS Learning - Neural Net/ MLP
        torch.manual_seed(0)        
        mlp_lpool_crps_model = AdaptiveLinearPoolCRPSLayer(input_size = tensor_trainX.shape[1], hidden_sizes = [20, 20, 20], output_size = N_experts, support = torch.FloatTensor(y_supp))        
        optimizer = torch.optim.Adam(mlp_lpool_crps_model.parameters(), lr = 1e-3)        
        mlp_lpool_crps_model.train_model(train_adapt_data_loader, valid_adapt_data_loader, optimizer, epochs = num_epochs, 
                                          patience = 10, validation = True)
    
        adaptive_models_dict['CRPS-MLP'] = mlp_lpool_crps_model
        if config['save']:
            with open(f'{results_path}\\{filename_prefix}_{critical_fractile}_adaptive_models_dict.pickle', 'wb') as handle:
                pickle.dump(adaptive_models_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        # with open(f'{results_path}\\{filename_prefix}_{critical_fractile}_adaptive_models_dict.pickle', 'rb') as handle:
        #     adaptive_models_dict = pickle.load(handle)
        #%
        ### Conditional combination of weighted decisions ***these do not appear in the paper***
        train_dec_data_loader = create_data_loader([tensor_trainZopt, tensor_trainX, tensor_trainY], batch_size = batch_size, shuffle = False)
        valid_dec_data_loader = create_data_loader([tensor_validZopt, tensor_validX, tensor_validY], batch_size = batch_size, shuffle = False)
    
        torch.manual_seed(0)        
        lr_lpool_decision_model = AdaptiveLinearPoolDecisions(input_size = tensor_trainX.shape[1], hidden_sizes = [], output_size = N_experts, support = torch.FloatTensor(y_supp))        
        optimizer = torch.optim.Adam(lr_lpool_decision_model.parameters(), lr = 1e-3)        
        lr_lpool_decision_model.train_model(train_dec_data_loader, valid_dec_data_loader, optimizer, epochs = num_epochs, 
                                          patience = patience, validation = True)
    
        adaptive_models_dict['SalvaBench-LR'] = lr_lpool_decision_model

        torch.manual_seed(0)        
        
        mlp_lpool_decision_model = AdaptiveLinearPoolDecisions(input_size = tensor_trainX.shape[1], hidden_sizes = [20, 20, 20], 
                                                               output_size = N_experts, support = torch.FloatTensor(y_supp))        
        optimizer = torch.optim.Adam(mlp_lpool_decision_model.parameters(), lr = 1e-3)        
        mlp_lpool_decision_model.train_model(train_dec_data_loader, valid_dec_data_loader, optimizer, epochs = num_epochs, 
                                          patience = patience, validation = True)
    
        adaptive_models_dict['SalvaBench-MLP'] = mlp_lpool_decision_model
        if config['save']:
            with open(f'{results_path}\\{filename_prefix}_{critical_fractile}_adaptive_models_dict.pickle', 'wb') as handle:
                pickle.dump(adaptive_models_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        #%
        ### Decision-focused Learning - Linear Regression + MLP models over all values of gamma        
        for gamma in config['gamma_list']:
            
            torch.manual_seed(0)        

            lr_lpool_newsv_model = AdaptiveLinearPoolNewsvendorLayer(input_size = tensor_trainX.shape[1], hidden_sizes = [], 
                                                                     output_size = N_experts, support = torch.FloatTensor(y_supp), gamma = gamma, critic_fract = critical_fractile, 
                                                                     risk_aversion = risk_aversion, apply_softmax = True)
            
            optimizer = torch.optim.Adam(lr_lpool_newsv_model.parameters(), lr = 1e-2)
            lr_lpool_newsv_model.train_model(train_adapt_data_loader, valid_adapt_data_loader, 
                                                  optimizer, epochs = 250, patience = 10, relative_tolerance = 0, validation = True)
            
            
            adaptive_models_dict[f'DF-LR_{gamma}'] = lr_lpool_newsv_model

            torch.manual_seed(0)        

            mlp_lpool_newsv_model = AdaptiveLinearPoolNewsvendorLayer(input_size = tensor_trainX.shape[1], hidden_sizes = [20,20,20], 
                                                                     output_size = N_experts, support = torch.FloatTensor(y_supp), 
                                                                     gamma = gamma, critic_fract = critical_fractile, risk_aversion = risk_aversion)
            
            optimizer = torch.optim.Adam(mlp_lpool_newsv_model.parameters(), lr = 1e-2)
            mlp_lpool_newsv_model.train_model(train_adapt_data_loader, valid_adapt_data_loader, 
                                                  optimizer, epochs = 250, patience = 10, relative_tolerance = 0, validation = True)
    
            adaptive_models_dict[f'DF-MLP_{gamma}'] = mlp_lpool_newsv_model

        if config['save']:
            with open(f'{results_path}\\{filename_prefix}_{critical_fractile}_adaptive_models_dict.pickle', 'wb') as handle:
                pickle.dump(adaptive_models_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(f'{results_path}\\{filename_prefix}_{critical_fractile}_adaptive_models_dict.pickle', 'rb') as handle:
            adaptive_models_dict = pickle.load(handle)
    
    #% Performance evaluation
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
        print(f'Model:{m}')
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
        temp_Decision_cost[m] = 100*task_loss(Prescriptions[m].values, testY.values, 
                                          target_problem, crit_quant = critical_fractile, risk_aversion = risk_aversion)

        # Evaluate Quantile Score and CRPS for combined forecasts
        # find quantile forecasts
        temp_q_forecast = np.array([inverted_cdf(target_quant, y_supp, temp_pdf[i]) for i in range(n_test_obs)])            
        temp_qs = 100*pinball(temp_q_forecast, testY.values, target_quant).round(4)

        temp_QS[m] = [temp_qs]
        
        temp_CDF = temp_pdf.cumsum(1)
        H_i = 1*np.repeat(y_supp.reshape(1,-1), len(testY), axis = 0)>=testY.values.reshape(-1,1)
        
        CRPS = 100*np.square(temp_CDF - H_i).mean()            
        temp_mean_QS[m] = CRPS

    print('Out-of-sample Decision Cost')
    print(temp_Decision_cost[all_models].mean().round(4))

    print('Out-of-sample CRPS')
    print(temp_mean_QS[all_models].mean().round(4))
    
    # save results
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