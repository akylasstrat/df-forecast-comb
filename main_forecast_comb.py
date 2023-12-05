# -*- coding: utf-8 -*-
"""
Decision-Focused Forecast Combination/ main script

@author: a.stratigakos
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import pickle
import gurobipy as gp

cd = os.path.dirname(__file__)  #Current directory
sys.path.append(cd)
#project_dir=Path(cd).parent.__str__()   #project_directory
plt.rcParams['figure.dpi'] = 600

from sklearn.ensemble import ExtraTreesRegressor
from EnsemblePrescriptiveTree import *
from optimization_functions import *

from utility_functions import *
from optimal_transport_functions import *
from gurobi_ml import add_predictor_constr

# IEEE plot parameters (not sure about mathfont)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 600
plt.rcParams['figure.figsize'] = (3.5, 2) # Height can be changed
plt.rcParams['font.size'] = 7
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams["mathtext.fontset"] = 'dejavuserif'

def tune_combination_newsvendor(target_y, prob_vectors, ml_predictor, brc_predictor = [], type_ = 'convex_comb', 
                                crit_fract = 0.5, support = np.arange(0, 1.01, .01).round(2), bounds = False, 
                                verbose = 0):
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

def params():
    ''' Set up the experiment parameters'''

    params = {}
    # Either pv or wind
    params['target'] = 'Z1'
        
    params['start_date'] = '2012-01-01'
    params['split_date'] = '2013-01-01' # Defines train/test split
    params['end_date'] = '2013-12-30'
    
    params['save'] = False # If True, then saves models and results
    params['train_brc'] = False # If True, then saves models and results
    
    # Experimental setup parameters
    params['quality'] = False
    params['problem'] = 'reg_trad' # {mse, newsvendor, cvar, reg_trad}
    params['N_sample'] = [10, 50, 100, 200, 500]
    params['N_assets'] = [5]
    params['iterations'] = 5
    params['critical_fractile'] = 0.2
    
    return params

#%%
config = params()
hyperparam = tree_params()

results_path = f'{cd}\\results'
data_path = f'{cd}\\data'

aggr_wind_df = pd.read_csv(f'{data_path}\\GEFCom2014-processed.csv', index_col = 0, header = [0,1])

#%%
target_zone = config['target']
expert_zones = ['Z2', 'Z4', 'Z8', 'Z9']

pred_col = ['wspeed10', 'wdir10_rad', 'wspeed100', 'wdir100_rad']
# number of observations to train prob. forecasting model
N_sample = len(trainY)//2
# number of forecasts to combine
N_experts = 3


trainY = aggr_wind_df.xs('POWER', axis=1, level=1)[[target_zone] + expert_zones][config['start_date']:config['split_date']][:N_sample].round(2)
comb_trainY = aggr_wind_df.xs('POWER', axis=1, level=1)[[target_zone] + expert_zones][config['start_date']:config['split_date']][N_sample:].round(2)
testY = aggr_wind_df.xs('POWER', axis=1, level=1)[[target_zone] + expert_zones][config['split_date']:].round(2)

# feature data for target location/zone
trainX_local = aggr_wind_df[target_zone][['wspeed10', 'wdir10_rad']][config['start_date']:config['split_date']][:N_sample]
comd_trainX_local = aggr_wind_df[target_zone][['wspeed10', 'wdir10_rad']][config['start_date']:config['split_date']][N_sample:]
testX_local = aggr_wind_df[target_zone][['wspeed10', 'wdir10_rad']][config['split_date']:]

# supervised sets for experts    
trainX_exp = aggr_wind_df[expert_zones][config['start_date']:config['split_date']][:N_sample]
comb_trainX_exp = aggr_wind_df[expert_zones][config['start_date']:config['split_date']][N_sample:]
testX_exp = aggr_wind_df[expert_zones][config['split_date']:]

#%% Train experts, i.e., probabilistic forecasting models in adjacent locations

# data conditioned on wind speed
step = .01
y_supp = np.arange(0, 1+step, step).round(2)

prob_models = []

for i, zone in enumerate(expert_zones[:N_experts]):
    print(f'Training model {i}')
    temp_model = EnsemblePrescriptiveTree(n_estimators = 100, max_features = len(pred_col), type_split = 'quant' )
    temp_model.fit(trainX_exp[zone][pred_col].values, trainY[zone].values, quant = np.arange(.01, 1, .01), problem = 'mse') 
    prob_models.append(temp_model)

#%%
# find local weights for meta-training set/ map weights to support locations
w_list = []
p_list = []
for i, p in enumerate(expert_zones[:N_experts]):
    w_list.append(prob_models[i].find_weights(comb_trainX_exp[p][pred_col].values, trainX_exp[p][pred_col].values))
    p_list.append(wemp_to_support(w_list[i], trainY[p].values, y_supp))
    
#%%
# find inverted CDFs
F_inv = [np.array([inverted_cdf([.05, .10, .90, .95] , trainY[zone].values, w_list[j][i]) for i in range(500)]) for j in range(N_experts)]
#%%
# Visualize some probabilistic forecasts
plt.plot(comb_trainY[target_zone][200:300])
for i in [0,2]:
    plt.fill_between(np.arange(100), F_inv[i][200:300,0], F_inv[i][200:300,-1], alpha = .3, color = 'red')
    plt.fill_between(np.arange(100), F_inv[i][200:300,1], F_inv[i][200:300,-2], alpha = .3, color = 'blue')
plt.show()

#%% Example: point forecasting (MSE) & convex combination

# 1-1 mapping: decisions are the weighted sum
nobs = p_list[0].shape[0]
nlocations = p_list[0].shape[1]

# set up the SAA of the target optimization problem
m = gp.Model()
#m.setParam('OutputFlag', 1)
# Decision variables

targetY = comb_trainY[target_zone].values.reshape(-1)

lambda_coord = m.addMVar(N_experts, vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1)
p_comb = m.addMVar((nobs, nlocations), vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1)
z = m.addMVar(nobs, vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1)

m.addConstr( lambda_coord.sum() == 1)
# constraints on probability
m.addConstr( p_comb.sum(1) == 1)
m.addConstr( p_comb == sum([lambda_coord[i]*p_list[i] for i in range(N_experts)]))

# mapping probability vectors to decisions (replace with trained ML predictor for more complex problems)
m.addConstr( z == p_comb@y_supp)

# Task-loss function
m.setObjective( (targetY-z)@(targetY-z), gp.GRB.MINIMIZE)
m.optimize()

#%%
# Testing

# find weights for OOS observations
w_test_list = [prob_models[i].find_weights(testX_exp[p][pred_col].values, trainX_exp[p][pred_col].values) for i, p in enumerate(expert_zones[:N_experts])]
#%%
# evaluate all models
y_hat_local = [lambda_coord[i].X*w_test_list[i]@trainY[target_zone].values for i in range(N_experts)]
y_hat_comb = sum([lambda_coord[i].X*w_test_list[i] for i in range(N_experts)])@trainY[target_zone].values

for i in range(N_experts):
    print(f'Model {i}:{eval_point_pred(y_hat_local[i], testY[target_zone])[0]}')
print(f'ConvComb:{eval_point_pred(y_hat_comb, testY[target_zone])[0]}')
print(f'Ave:{eval_point_pred(sum(y_hat_local)/N_experts, testY[target_zone])[0]}')

#%%%%%%%%%%%%%%% Newsvendor experiment

from sklearn.linear_model import LinearRegression, Ridge
from gurobi_ml import add_predictor_constr
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from gurobi_ml.sklearn import add_decision_tree_regressor_constr, add_random_forest_regressor_constr


# problem parameters
critical_fractile = .7

# optimal solutions to create training set for decision rule
z_opt = []
for j in range(N_experts):
    z_opt.append( np.array([inverted_cdf([critical_fractile] , y_supp, p_list[j][i]) for i in range(nobs)]) )

z_opt = np.row_stack(z_opt)
p_stacked = np.row_stack((p_list))
#%%
#### Train ML predictor to map prob. vectors (p.d.f.) to optimal decisions (linear, tree-based methods)

lr_model = Ridge().fit(p_stacked, z_opt)
rf_model = RandomForestRegressor(n_estimators = 10).fit(p_stacked, z_opt)
dt_model = DecisionTreeRegressor().fit(p_stacked, z_opt)
#%%
plt.plot(z_opt[-100:])
plt.plot(lr_model.predict(p_stacked)[-100:], '--')
plt.plot(dt_model.predict(p_stacked)[-100:], '-.')
plt.plot(rf_model.predict(p_stacked)[-100:], 'd', markersize = 2)
plt.legend(['z_opt', 'LR', 'DT', 'RF'])
plt.show()

#%% 

# Case 1: Convex Combination/ LR decision rule
lambda_cc_lr = tune_combination_newsvendor(targetY, p_list, lr_model, 
                                           crit_fract = critical_fractile, support = y_supp)

# Case 2: Convex Combination/ DT decision rule
lambda_cc_dt = tune_combination_newsvendor(targetY, p_list, dt_model, 
                                           crit_fract = critical_fractile, support = y_supp, verbose = 1)

print(f'Decision rule solution:{lambda_cc_dt}')

'''
# Differentiable optimization
m = N_experts
n = nobs
k = len(y_supp)

# forward pass
# Problem 1: forecast combination
lambda_ = cp.Parameter((m))
p_comb = cp.Variable((n,k))
constr_FC = [p_comb >=0, p_comb <=1, p_comb.sum(1) == 1, p_comb == sum([p_list[i]*lambda_[i] for i in range(m)])]

# problem 2: find wSAA decisions 
z = cp.Variable(n)
error = cp.Variable(n)
loss = cp.Variable(n)

constr_wSAA = [z >= 0, z <= 1, z == sum([p_list[i]*lambda_[i] for i in range(m)])@y_supp, 
               error == z-valid_localY.reshape(-1), 
               loss >= critical_fractile*error, loss >= (1-critical_fractile)*error]

objective = cp.Minimize(0.5*cp.square(error).sum())
problem = cp.Problem(objective, constr_wSAA)


# projection problem
y = cp.Variable((m))
y_hat = cp.Parameter((m))
constr_Proj = [y >=0, y <=1, y.sum() == 1]
projection_problem = cp.Problem(cp.Minimize(cp.sum_squares(y-y_hat)), constr_Proj)

# gradient descent/ initialize parameters
L_t = [[1, 0]]
lambda_.value = L_t[0]
Loss = []
eta = .1
for iter_ in range(50):
    print(f'iteration:{iter_}')
    # solve forward pass
    problem.solve(requires_grad = True)
    Loss.append(objective.value)
    
    # backward pass
    z.gradient = error.value    
    problem.backward()
    
    print(f'gradients:{lambda_.gradient}')
    
    #update parameter values
    lambda_t_1 = L_t[-1] + eta*lambda_.gradient

    # projection onto the prob. simplex
    y_hat.value = lambda_t_1
    projection_problem.solve()
    
    # update with new projected lambda values
    L_t.append( y.value )    
    lambda_.value = L_t[-1]
    
    plt.plot(Loss)
    plt.show()
    
    if cp.norm(L_t[-1] - L_t[-2], p=2).value <=1e-5: break

plt.plot(L_t)
plt.show()

print(f'Gradient-based solution:{L_t[-1].round(2)}')
print(f'Optimal Obj. value:{objective.value.round(2)}')
'''
#%%
# Case 3: Barycentric interpolation/ DT

# Train additional ML predictor to learn mapping from quantile function to p.d.f. (the true is piecewise constant)
target_quantiles = np.arange(0,1.01,.01)

p_inv_list = [np.array([inverted_cdf(target_quantiles, y_supp, w = p_list[j][i]) for i in range(nobs)]) for j in range(N_experts)]

dt_model_inv = DecisionTreeRegressor(max_depth=5).fit(np.row_stack(p_inv_list), np.row_stack(p_list))
lr_model_inv = LinearRegression().fit(np.row_stack(p_inv_list), np.row_stack(p_list))

#%%
t = 10

plt.plot(dt_model_inv.predict(np.row_stack(p_inv_list))[t], label = 'DT-apprx')
plt.plot(lr_model_inv.predict(np.row_stack(p_inv_list))[t], label = 'LR-apprx')
plt.plot(np.row_stack(p_list)[t])
plt.show()
#%%
#lambda_brc_dt = tune_combination_newsvendor(valid_localY, p_list, dt_model, brc_predictor= dt_model_inv, 
#                                            type_ = 'barycenter', crit_fract = critical_fractile, support = y_supp, verbose = 1)


# Case 1: Convex Combination/ LR decision rule
'''
m = gp.Model()
#m.setParam('OutputFlag', 1)
# Decision variables
lambda_1_cclr = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1)
lambda_2_cclr = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1)
p_comb = m.addMVar(p1.shape, vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1)
error = m.addMVar(p1.shape[0], vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
loss_i = m.addMVar(p1.shape[0], vtype = gp.GRB.CONTINUOUS, lb = 0)

z = m.addMVar(p1.shape[0], vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)

m.addConstr( 1 == lambda_1_cclr + lambda_2_cclr)
# constraints on probability
m.addConstr( p_comb.sum(1) == 1)
m.addConstr( p_comb == lambda_1_cclr*(p1) + lambda_2_cclr*(p2))

# mapping probability vectors to decisions (replace with trained ML predictor for complex problems)
#m.addConstr( z == (lambda_1_cclr*(p1) + lambda_2_cclr*(p2))@lr_model.coef_.reshape(-1) + lr_model.intercept_)

pred_constr = add_predictor_constr(m, lr_model, p_comb, z)
#pred_constr.print_stats()

# Task-loss function
m.addConstr( error == valid_localY - z)
m.addConstr( loss_i >= critical_fractile*error)
m.addConstr( loss_i >= (critical_fractile-1)*error)

m.setObjective( loss_i.sum()/(p1.shape[1]), gp.GRB.MINIMIZE)
m.optimize()
'''

# Case 2: Convex Combination/ DT decision rule
'''
#### set optimization problem
m_cc_dt = gp.Model()
#m.setParam('OutputFlag', 1)
# Decision variables
lambda_1_ccdt = m_cc_dt.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1)
lambda_2_ccdt = m_cc_dt.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1)
p_comb = m_cc_dt.addMVar(p1.shape, vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1)
error = m_cc_dt.addMVar(p1.shape[0], vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
loss_i = m_cc_dt.addMVar(p1.shape[0], vtype = gp.GRB.CONTINUOUS, lb = 0)

z = m_cc_dt.addMVar(p1.shape[0], vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)

m_cc_dt.addConstr( 1 == lambda_1_ccdt + lambda_2_ccdt)
# constraints on probability
#m.addConstr( p_comb.sum(1) == 1)
m_cc_dt.addConstr( p_comb == lambda_1_ccdt*(p1) + lambda_2_ccdt*(p2))

# mapping probability vectors to decisions (replace with trained ML predictor for complex problems)
#pred_constr = add_random_forest_regressor_constr(m, rf_model, p_comb, z, epsilon = 1e-6)
#pred_constr = add_predictor_constr(m, rf_model, p_comb, z, epsilon = 1e-6)


pred_constr = add_predictor_constr(m, dt_model, p_comb, z, epsilon = 1e-6)
#pred_constr = add_decision_tree_regressor_constr(m, dt_model, p_comb, z, epsilon = .1e-6)
pred_constr.print_stats()

# Task-loss function
m.addConstr( error == valid_localY - z)
m.addConstr( loss_i >= critical_fractile*error)
m.addConstr( loss_i >= (critical_fractile-1)*error)

m.setObjective( loss_i.sum()/(p1.shape[1]), gp.GRB.MINIMIZE)
m.optimize()
'''

# Case 3: Barycentric interpolation/ DT

'''
# optimize the barycentric coordinates
m = gp.Model()
#m.setParam('OutputFlag', 1)
# Decision variables
lambda_1_brc_dt = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1)
lambda_2_brc_dt = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1)
p_comb = m.addMVar(p1.shape, vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1)
y_supp_aux = m.addMVar(y_supp.shape, vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1)

# upper/lower bound for inverse correspond to bound on wind production, not quantiles
p_comb_inv = m.addMVar(p1_inv.shape, vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1)
p_comb_cdf = m.addMVar(p1.shape, vtype = gp.GRB.CONTINUOUS, lb = 0)

error = m.addMVar(p1.shape[0], vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
loss_i = m.addMVar(p1.shape[0], vtype = gp.GRB.CONTINUOUS, lb = 0)

z = m.addMVar(p1.shape[0], vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)

m.addConstr( 1 == lambda_1_brc_dt + lambda_2_brc_dt)

# quantile averaging
#m.addConstr( p_comb.sum(1) == 1)
m.addConstr( p_comb_inv == lambda_1_brc_dt*(p1_inv) + lambda_2_brc_dt*(p2_inv))

# piecewise linear constraint for inverse cdf function
#for i in range(p1.shape[0]):
#    m.addGenConstrPWL(y_supp_aux, p_comb_cdf[i], target_quantiles, p_comb_inv[i], "myPWLConstr")    
#m.addConstr( y_supp_aux == y_supp)

# mapping from inverse c.d.f. (quantile function) to prob. vector (p.d.f.)
add_predictor_constr(m, dt_model_inv, p_comb_inv, p_comb, epsilon = 1e-6)


# mapping probability vectors to decisions (replace with trained ML predictor for complex problems)
pred_constr = add_predictor_constr(m, dt_model, p_comb, z, epsilon = 1e-6)
#pred_constr = add_decision_tree_regressor_constr(m, dt_model, p_comb, z, epsilon = .1e-6)
pred_constr.print_stats()

# Task-loss function
m.addConstr( error == valid_localY - z)
m.addConstr( loss_i >= critical_fractile*error)
m.addConstr( loss_i >= (critical_fractile-1)*error)

m.setObjective( loss_i.sum()/(p1.shape[1]), gp.GRB.MINIMIZE)
m.optimize()
'''

#%% Testing all methods

# turn weights to distributions, find average distribution
n_test_obs = len(w_test_list[0])
p_hat_test_list = [wemp_to_support(w, train_localY, y_supp) for w in w_test_list]
p_ave = sum(p_hat_test_list)/N_experts
p_brc = np.zeros((n_test_obs, nlocations))

# Barycenter with average coordinates
for i in range(n_test_obs):
    temp_p_list = [p[i] for p in p_hat_test_list]

    temp_p_brc, _, _ = wass2_barycenter_1D(N_experts*[y_supp], temp_p_list, lambda_coord = N_experts*[1/N_experts], support = y_supp, p = 2, 
                               prob_dx = .001)
    p_brc[i] = temp_p_brc
#%%
p_cc_lr = sum([lambda_cc_lr[i]*p_hat_test_list[i] for i in range(N_experts)])
p_cc_dt = sum([lambda_cc_dt[i]*p_hat_test_list[i] for i in range(N_experts)])
p_brc_dt = np.zeros((n_test_obs, nlocations))

#p_cc_lr = lambda_cc_lr[0]*p1_hat + lambda_cc_lr[1]*p2_hat
#p_cc_dt = lambda_cc_dt[0]*p1_hat + lambda_cc_dt[1]*p2_hat
#p_brc_dt = np.zeros(p1_hat.shape)

# Barycenter with tuned coordinates
'''
for i in range(n_test_obs):
    temp_p_list = [p[i] for p in p_hat_test_list]
    
    temp_p_brc, _, _ = wass2_barycenter_1D(N_experts*[y_supp], temp_p_list, lambda_coord = lambda_brc_dt, support = y_supp, p = 2, 
                               prob_dx = .01)
    p_brc_dt[i] = temp_p_brc
    '''
#%%
# turn probability vectors to decisions/ closed-form solution for newsvendor problem

models = [f'Model-{i}' for i in range(N_experts)] + ['CC-Ave', 'BRC-Ave', 'CC-LR', 'CC-DT', 'BRC-DT']

Prescriptions = pd.DataFrame(data = np.zeros((len(test_localY), len(models))), columns = models)
for i in range(N_experts):
    Prescriptions[f'Model-{i}'] = np.array([inverted_cdf([critical_fractile], y_supp, p_hat_test_list[i][k]) for k in range(len(test_localY))]).reshape(-1)

Prescriptions['CC-Ave'] = np.array([inverted_cdf([critical_fractile], y_supp, p_ave[i]) for i in range(len(test_localY))]).reshape(-1)
Prescriptions['BRC-Ave'] = np.array([inverted_cdf([critical_fractile], y_supp, p_brc[i]) for i in range(len(test_localY))]).reshape(-1)
Prescriptions['CC-LR'] = np.array([inverted_cdf([critical_fractile], y_supp, p_cc_lr[i]) for i in range(len(test_localY))]).reshape(-1)
Prescriptions['CC-DT'] = np.array([inverted_cdf([critical_fractile], y_supp, p_cc_dt[i]) for i in range(len(test_localY))]).reshape(-1)
#Prescriptions['BRC-DT'] = np.array([inverted_cdf([critical_fractile], y_supp, p_brc_dt[i]) for i in range(len(test_localY))]).reshape(-1)

#%%

for m in models:
    print(f'{m}:{100*newsvendor_loss(Prescriptions[m].values, test_localY, q = critical_fractile).round(4)}')
    
# optimal solution/ recover critical fractile from inverted cdf
#y_hat_brc_comb = np.array([inverted_cdf([critical_fractile], y_supp, p_brc_comb[i]) for i in range(len(test_localY))])

#y_hat_1 = np.array([inverted_cdf([critical_fractile], y_supp, p1_hat[i]) for i in range(len(test_localY))])
#y_hat_2 = np.array([inverted_cdf([critical_fractile], y_supp, p2_hat[i]) for i in range(len(test_localY))])
#y_hat_ave = np.array([inverted_cdf([critical_fractile], y_supp, p_ave[i]) for i in range(len(test_localY))])
#y_hat_LR_comb = np.array([inverted_cdf([critical_fractile], y_supp, p_ave_hat_lr[i]) for i in range(len(test_localY))])

#print(f'Model 1:{newsvendor_loss(y_hat_1, test_localY, q = critical_fractile)}')
#print(f'Model 2:{newsvendor_loss(y_hat_2, test_localY, q = critical_fractile)}')
#print(f'Average:{newsvendor_loss(y_hat_ave, test_localY, q = critical_fractile)}')
#print(f'ConvComb-LR:{newsvendor_loss(y_hat_LR_comb, test_localY, q = critical_fractile)}')

