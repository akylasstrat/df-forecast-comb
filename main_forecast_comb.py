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
N_sample = len(aggr_wind_df)//4
# number of forecasts to combine
N_experts = 3

trainY = aggr_wind_df.xs('POWER', axis=1, level=1)[[target_zone] + expert_zones][config['start_date']:config['split_date']][:N_sample].round(2)
comb_trainY = aggr_wind_df.xs('POWER', axis=1, level=1)[[target_zone] + expert_zones][config['start_date']:config['split_date']][N_sample:].round(2)
testY = aggr_wind_df.xs('POWER', axis=1, level=1)[[target_zone] + expert_zones][config['split_date']:].round(2)

# feature data for target location/zone
trainX_local = aggr_wind_df[target_zone][['wspeed10', 'wdir10_rad']][config['start_date']:config['split_date']][:N_sample].round(2)
comd_trainX_local = aggr_wind_df[target_zone][['wspeed10', 'wdir10_rad']][config['start_date']:config['split_date']][N_sample:].round(2)
testX_local = aggr_wind_df[target_zone][['wspeed10', 'wdir10_rad']][config['split_date']:].round(2)

# supervised sets for experts    
trainX_exp = aggr_wind_df[expert_zones][config['start_date']:config['split_date']][:N_sample].round(2)
comb_trainX_exp = aggr_wind_df[expert_zones][config['start_date']:config['split_date']][N_sample:].round(2)
testX_exp = aggr_wind_df[expert_zones][config['split_date']:].round(2)


#%% Train experts, i.e., probabilistic forecasting models in adjacent locations

# data conditioned on wind speed
step = .01
y_supp = np.arange(0, 1+step, step).round(2)

prob_models = []
from EnsemblePrescriptiveTree_OOB import *

for i, zone in enumerate(expert_zones[:N_experts]):
    print(f'Training model {i}')
    
    #temp_model = EnsemblePrescriptiveTree_OOB(n_estimators = 10, max_features = 1, type_split = 'quant')
    #temp_model.fit(trainX_exp[zone][pred_col].values.round(2), trainY[zone].values, y_supp, y_supp, bootstrap = False, quant = np.arange(.01, 1, .01), problem = 'mse') 
 
    temp_model = EnsemblePrescriptiveTree(n_estimators = 10, max_features = len(pred_col), type_split = 'quant' )
    temp_model.fit(trainX_exp[zone][pred_col].values, trainY[zone].values, quant = np.arange(.01, 1, .01), problem = 'mse') 
    prob_models.append(temp_model)

#%% Predictions for train/test set for forecast combination
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
    
#%% Visualize some prob. forecasts

# step 1: find inverted CDFs
F_inv = [np.array([inverted_cdf([.05, .10, .90, .95] , trainY[zone].values, train_w_list[j][i]) for i in range(500)]) for j in range(N_experts)]

plt.plot(comb_trainY[target_zone][200:300])
for i in [0,2]:
    #plt.fill_between(np.arange(100), F_inv[i][200:300,0], F_inv[i][200:300,-1], alpha = .3, color = 'red')
    
    plt.fill_between(np.arange(100), F_inv[i][200:300,0], F_inv[i][200:300,-1], alpha = .3)
plt.show()

#%% Example: point forecasting (MSE) & convex combination

# 1-1 mapping: decisions are the weighted sum
nobs = train_p_list[0].shape[0]
nlocations = len(y_supp)

# set up the SAA of the target optimization problem
m = gp.Model()
#m.setParam('OutputFlag', 1)
# Decision variables

train_targetY = comb_trainY[target_zone].values.reshape(-1)

lambda_coord = m.addMVar(N_experts, vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1)
p_comb = m.addMVar((nobs, nlocations), vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1)
z = m.addMVar(nobs, vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1)

m.addConstr( lambda_coord.sum() == 1)
# constraints on probability
m.addConstr( p_comb.sum(1) == 1)
m.addConstr( p_comb == sum([lambda_coord[i]*train_p_list[i] for i in range(N_experts)]))

# mapping probability vectors to decisions (replace with trained ML predictor for more complex problems)
m.addConstr( z == p_comb@y_supp)

# Task-loss function
m.setObjective( (train_targetY-z)@(train_targetY-z), gp.GRB.MINIMIZE)
m.optimize()

#%% Combination with diff. opt. layer

# projection step
y_proj = cp.Variable(N_experts)
y_hat = cp.Parameter(N_experts)
proj_problem = cp.Problem(cp.Minimize(0.5*cp.sum_squares(y_proj-y_hat)), [y_proj >= 0, y_proj.sum()==1])

k = len(y_supp)

# GD for differentiable layer
z = cp.Variable(nobs)
lambda_ = cp.Parameter(N_experts)

constraints = [z >= 0, z <=1, z == sum([lambda_[i]*train_p_list[i] for i in range(N_experts)])@y_supp]

objective = cp.Minimize(cp.norm(train_targetY - z, p=2))
problem = cp.Problem(objective, constraints)

layer = CvxpyLayer(problem, parameters=[lambda_], variables=[z,])
l_hat = nn.Parameter(torch.FloatTensor((1/N_experts)*np.ones(N_experts)).requires_grad_())
opt = torch.optim.SGD([l_hat], lr=1e-2)
losses = []

L_t = [to_np(l_hat)]
Projection = True

for i in range(1000):
        
    # Forward pass: solve optimization problem
    zhat, = layer(l_hat)
    np_zhat = to_np(zhat)
    
    #assert(np_zhat.min()>=0)
    assert(np_zhat.max()<=1)
    
    # Estimate model loss
    loss = (zhat - nn.Parameter(torch.FloatTensor(train_targetY))).norm()
    losses.append(to_np(loss))
    

    # gradients and parameter update            
    opt.zero_grad()
    loss.backward()
    opt.step()
      
    if Projection:     
        y_hat.value = to_np(l_hat)
        proj_problem.solve(solver = 'GUROBI')
        # update parameter values
        
        with torch.no_grad():
            l_hat.copy_(torch.FloatTensor(y_proj.value))
    
    L_t.append(to_np(l_hat))

    if i%10==0:
        print(l_hat)
        
        plt.plot(losses)
        plt.show()

#%% Derive test predictions, evaluate models

y_hat_local = [lambda_coord[i].X*test_p_list[i]@y_supp for i in range(N_experts)]
y_hat_comb = sum([lambda_coord[i].X*test_p_list[i] for i in range(N_experts)])@y_supp

for i in range(N_experts):
    print(f'Model {i}:{eval_point_pred(y_hat_local[i], testY[target_zone])[0]}')
print(f'ConvComb:{eval_point_pred(y_hat_comb, testY[target_zone])[0]}')
print(f'Ave:{eval_point_pred(sum(y_hat_local)/N_experts, testY[target_zone])[0]}')

#%%%%%%%%%%%%%%% Newsvendor experiment

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from gurobi_ml import add_predictor_constr
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from gurobi_ml.sklearn import add_decision_tree_regressor_constr, add_random_forest_regressor_constr


# problem parameters
critical_fractile = .5

# optimal solutions to create training set for decision rule
z_opt = []
for j in range(N_experts):
    z_opt.append( np.array([inverted_cdf([critical_fractile] , y_supp, train_p_list[j][i]) for i in range(nobs)]) )

z_opt = np.row_stack(z_opt)
train_p_stacked = np.row_stack((train_p_list))

#%%
#### Train ML predictor to map prob. vectors (p.d.f.) to optimal decisions (linear, tree-based methods)

lr_model = Ridge().fit(train_p_stacked, z_opt)
rf_model = RandomForestRegressor(n_estimators = 10).fit(train_p_stacked, z_opt)
dt_model = DecisionTreeRegressor(max_depth = 4).fit(train_p_stacked, z_opt)

from LinearDecisionTree import *
ldt_model = LinearDecisionTree(type_split = 'quant')
ldt_model.fit(train_p_stacked, z_opt.reshape(-1))

from sklearn.neural_network import MLPRegressor

nn_model = MLPRegressor(hidden_layer_sizes=(20,20,20), activation='relu',).fit(train_p_stacked, z_opt)

#%%

m = gp.Model()

# Decision variables
coef_ = m.addMVar(train_p_stacked.shape[1], vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
inter_ = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
z_pred = m.addMVar(train_p_stacked.shape[0], vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
error = m.addMVar(train_p_stacked.shape[0], vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
loss = m.addMVar(train_p_stacked.shape[0], vtype = gp.GRB.CONTINUOUS, lb = 0)
t = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = 0)

m.addConstr(z_pred == (train_p_stacked@coef_ + inter_))
m.addConstr(error == z_opt.reshape(-1) - z_pred)
m.addConstr(t >= error@error)
m.setObjective(t.sum())
m.optimize()

#%%
t = -200
plt.plot(z_opt[t:])
plt.plot(ldt_model.predict(train_p_stacked)[t:], '--')
plt.plot(nn_model.predict(train_p_stacked)[t:], '-.')
#plt.plot(rf_model.predict(train_p_stacked)[t:], 'd', markersize = 2)
#plt.plot(z_pred.X[t:])
plt.legend(['z_opt', 'LR', 'DT', 'RF'])
plt.show()

#%% 

# Case 1: Convex Combination/ LR decision rule
lambda_cc_lr = tune_combination_newsvendor(train_targetY, train_p_list, lr_model, 
                                           crit_fract = critical_fractile, support = y_supp, bounds = False)
#%%

# Case 2: Convex Combination/ DT decision rule
lambda_cc_dt = tune_combination_newsvendor(train_targetY, train_p_list, nn_model, 
                                           crit_fract = critical_fractile, support = y_supp, verbose = 1)

print(f'Decision rule solution:{lambda_cc_dt}')

#%% Gradient-based approach with CVXPY/ full batch updates
batch_size = 50

z = cp.Variable(batch_size)
error = cp.Variable((batch_size, len(y_supp)))
pinball_loss = cp.Variable((batch_size, len(y_supp)))
lambda_ = cp.Parameter(N_experts)
p_comb = cp.Parameter((batch_size,k))

# inner problem
Z_inner = []
constraints = [z >= 0, z <=1] +[error[i] == y_supp - z[i] for i in range(batch_size)] \
                + [pinball_loss[i] >= critical_fractile*error[i] for i in range(batch_size)] \
                + [pinball_loss[i] >= (critical_fractile-1)*error[i] for i in range(batch_size)] \
#                + [p_comb == sum([lambda_[j]*p_list[j] for j in range(m)])]

#p_comb = sum([lambda_[j]*p_list[j][ix] for j in range(m)])

#objective = cp.Minimize(sum([pinball_loss[i]@p_comb[i] for i in range(batch_size)]))
#problem = cp.Problem(objective, constraints)

L_t = [(1/N_experts)*np.ones(N_experts)]
Loss = []
Projection = True
eta = 1e-2

for i in range(10000):

    ix = np.random.choice(range(nobs), batch_size, replace = False)

    # forward pass
    lambda_.value = L_t[-1]
    
    p_comb_hat = sum([lambda_[j]*train_p_list[j][ix] for j in range(N_experts)])
    
    objective = cp.Minimize(sum([pinball_loss[i]@p_comb_hat[i] for i in range(batch_size)]) )
    problem = cp.Problem(objective, constraints)

        
    problem.solve(solver = 'SCS', requires_grad = True)
    
    # gradient of cost w.r.t. to decision
    z.gradient = np.sign(train_targetY[ix] - z.value)
    z.gradient[z.gradient>0] = critical_fractile
    z.gradient[z.gradient<0] = critical_fractile-1
   
    # gradient of decision w.r.t. to parameter
    problem.backward()    
    Loss.append(objective.value)

    # update parameters (!!! projection step might be required)
    upd_L_t = L_t[-1] + eta*lambda_.gradient
    
    if Projection:
        y_hat.value = upd_L_t
        proj_problem.solve(solver = 'GUROBI')
        L_t.append(y_proj.value)
        
    else:
        L_t.append(upd_L_t)
    
    if i % 10 == 0:
        print(L_t[-1])
        plt.plot(Loss)
        plt.show()
        
#%% Differential opt. layer
batch_size = 50
nobs = len(train_targetY)
k = len(y_supp)

# Variables of inner problem
z = cp.Variable(batch_size)
point_pred = cp.Variable(batch_size)

error = cp.Variable((batch_size,k))
p_comb = cp.Parameter((batch_size,k))
pinball_loss = cp.Variable((batch_size,k))

lambda_ = cp.Parameter(N_experts)
p_list_t = [cp.Parameter((batch_size,k)) for j in range(N_experts)]
p_list_t_aux = [cp.Variable((batch_size,k)) for j in range(N_experts)]

constraints = [z >= 0, z <=1] + [error[i] == y_supp - z[i] for i in range(batch_size)]\
            + [pinball_loss[i] >= critical_fractile*error[i] for i in range(batch_size)]\
            + [pinball_loss[i] >= (critical_fractile-1)*error[i] for i in range(batch_size)]\
            + [p_list_t_aux[j] == p_list_t[j] for j in range(N_experts)]\
            + [p_comb == sum([p_list_t_aux[j]*lambda_[j] for j in range(N_experts)])]\
#            + [point_pred == sum([lambda_[j]*p_list_t_aux[j] for j in range(N_experts)])@y_supp]

#reg_pen = 10*cp.norm(cp.multiply(p_comb, error))
#reg_pen = 10*cp.norm( point_pred )

objective = cp.Minimize( sum([pinball_loss[i]@p_comb[i] for i in range(batch_size)]) ) 
#objective = cp.Minimize( cp.norm(cp.multiply(p_comb, error)) ) 
problem = cp.Problem(objective, constraints)
layer = CvxpyLayer(problem, parameters=[lambda_, p_comb] + p_list_t, variables=[z, pinball_loss, error] + p_list_t_aux)
#layer = CvxpyLayer(problem, parameters=[lambda_, p_comb] + p_list_t, variables=[z, error] + p_list_t_aux)
l_hat = nn.Parameter(torch.FloatTensor((1/N_experts)*np.ones(N_experts)).requires_grad_())

opt = torch.optim.Adam([l_hat], lr=1e-2)
losses = []
L_t = [to_np(l_hat)]
Projection = True

for i in range(10000):
    
    ix = np.random.choice(range(nobs), batch_size, replace = False)
    
    p_list_t_hat = []

    for j in range(N_experts):
        p_list_t_hat.append( nn.Parameter( torch.FloatTensor(train_p_list[j][ix]) ) ) 
        
    p_comb_hat = (sum([p_list_t_hat[j]*l_hat[j] for j in range(N_experts)]))
    
    decisions_hat = layer(l_hat, p_comb_hat, p_list_t_hat[0], p_list_t_hat[1], p_list_t_hat[2])

    zhat = decisions_hat[0]
    #point_pred_hat = decisions_hat[1]
    
    error_hat = zhat - nn.Parameter(torch.FloatTensor([train_targetY[ix]]))
    #!!!!! add pinball loss here
    #loss = error_hat.norm(p = 1)
    #loss = (zhat - nn.Parameter(torch.FloatTensor(train_targetY))).norm()

    loss = (critical_fractile*error_hat[error_hat>0].norm(p=1) + (1-critical_fractile)*error_hat[error_hat<0].norm(p=1))\
#            + 100*error_hat.norm(p=2)
    
    losses.append(to_np(loss))
            
    opt.zero_grad()
    loss.backward()
    opt.step()
    
    if Projection:     
        y_hat.value = to_np(l_hat)
        proj_problem.solve(solver = 'GUROBI')
        # update parameter values
        
        with torch.no_grad():
            l_hat.copy_(torch.FloatTensor(y_proj.value))
    
    L_t.append(to_np(l_hat))

    if i%50==0:
        print(L_t[-1])
        
        plt.plot(losses)
        plt.show()

#%%
# Case 3: Barycentric interpolation/ DT

# Train additional ML predictor to learn mapping from quantile function to p.d.f. (the true is piecewise constant)
target_quantiles = np.arange(0,1.01,.01)

p_inv_list = [np.array([inverted_cdf(target_quantiles, y_supp, w = train_p_list[j][i]) for i in range(nobs)]) for j in range(N_experts)]

dt_model_inv = DecisionTreeRegressor().fit(np.row_stack(p_inv_list), np.row_stack(train_p_list))
lr_model_inv = LinearRegression().fit(np.row_stack(p_inv_list), np.row_stack(train_p_list))

#%%
t = 10

plt.plot(dt_model_inv.predict(np.row_stack(p_inv_list))[t], label = 'DT-apprx')
plt.plot(lr_model_inv.predict(np.row_stack(p_inv_list))[t], label = 'LR-apprx')
plt.plot(np.row_stack(train_p_list)[t])
plt.show()
#%%
#lambda_brc_dt = tune_combination_newsvendor(valid_localY, p_list, dt_model, brc_predictor= dt_model_inv, 
#                                            type_ = 'barycenter', crit_fract = critical_fractile, support = y_supp, verbose = 1)

#%% Testing all methods

# turn weights to distributions, find average distribution
n_test_obs = len(test_w_list[0])
p_ave = sum(test_p_list)/N_experts
p_brc = np.zeros((n_test_obs, nlocations))

# Barycenter with average coordinates
for i in range(n_test_obs):
    temp_p_list = [p[i] for p in test_p_list]

    temp_p_brc, _, _ = wass2_barycenter_1D(N_experts*[y_supp], temp_p_list, lambda_coord = N_experts*[1/N_experts], support = y_supp, p = 2, 
                               prob_dx = .001)
    p_brc[i] = temp_p_brc
#%%
p_cc_lr = sum([lambda_cc_lr[i]*test_p_list[i] for i in range(N_experts)])
p_cc_dt = sum([lambda_cc_dt[i]*test_p_list[i] for i in range(N_experts)])
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
nobs_test = len(testY)

models = [f'Model-{i}' for i in range(N_experts)] + ['CC-Ave', 'BRC-Ave', 'CC-LR', 'CC-DT', 'BRC-DT']

Prescriptions = pd.DataFrame(data = np.zeros((nobs_test, len(models))), columns = models)
for i in range(N_experts):
    Prescriptions[f'Model-{i}'] = np.array([inverted_cdf([critical_fractile], y_supp, test_p_list[i][k]) for k in range(len(testY))]).reshape(-1)

Prescriptions['CC-Ave'] = np.array([inverted_cdf([critical_fractile], y_supp, p_ave[i]) for i in range(nobs_test)]).reshape(-1)
Prescriptions['BRC-Ave'] = np.array([inverted_cdf([critical_fractile], y_supp, p_brc[i]) for i in range(nobs_test)]).reshape(-1)
Prescriptions['CC-LR'] = np.array([inverted_cdf([critical_fractile], y_supp, p_cc_lr[i]) for i in range(nobs_test)]).reshape(-1)
Prescriptions['CC-DT'] = np.array([inverted_cdf([critical_fractile], y_supp, p_cc_dt[i]) for i in range(nobs_test)]).reshape(-1)
#Prescriptions['BRC-DT'] = np.array([inverted_cdf([critical_fractile], y_supp, p_brc_dt[i]) for i in range(len(test_localY))]).reshape(-1)

#%%

for m in models:
    print(f'{m}:{100*newsvendor_loss(Prescriptions[m].values, testY[target_zone], q = critical_fractile).round(4)}')
    
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

