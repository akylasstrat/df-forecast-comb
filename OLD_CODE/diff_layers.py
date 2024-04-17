# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 11:09:25 2023

@author: astratig
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os
#from pathlib import Path
import pickle
import gurobipy as gp
#import pickle
from sklearn.preprocessing import MinMaxScaler

cd = os.path.dirname(__file__)  #Current directory
sys.path.append(cd)
#project_dir=Path(cd).parent.__str__()   #project_directory
plt.rcParams['figure.dpi'] = 600

from sklearn.ensemble import ExtraTreesRegressor
from EnsemblePrescriptiveTree import *
from scipy.stats import norm, gaussian_kde
from scipy import interpolate
from scipy.special import ndtr
from QR_regressor import *
from sklearn.neighbors import KNeighborsRegressor
from EnsemblePrescriptiveTree_OOB import *
from optimization_functions import *
from newsvendor import *
#import ot
from utility_functions import *
from optimal_transport_functions import *
from scipy.special import softmax
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
        target_quantiles = np.arange(0,1,.01)
        prob_inv = [np.array([inverted_cdf(target_quantiles, support, w = p[i]) for i in range(n_obs)]) for p in prob_vectors ]

        # upper/lower bound for inverse correspond to bound on wind production, not quantiles
        p_comb_inv = m.addMVar(prob_inv[0].shape, vtype = gp.GRB.CONTINUOUS, lb = 0)


        ### quantile averaging
        m.addConstr( p_comb.sum(1) == 1)
        m.addConstr( p_comb_inv == sum([prob_inv[i]*lambdas[i] for i in range(n_models)]))

        ### piecewise linear constraint for inverse cdf function
        #for i in range(p1.shape[0]):
        #    m.addGenConstrPWL(y_supp_aux, p_comb_cdf[i], target_quantiles, p_comb_inv[i], "myPWLConstr")    
        #m.addConstr( y_supp_aux == y_supp)
        
        ### mapping from inverse c.d.f. (quantile function) to prob. vector (p.d.f.)
        add_predictor_constr(m, brc_predictor, p_comb_inv, p_comb, epsilon = 1e-6)
        
        # mapping from inverse c.d.f. (quantile function) to prob. vector (p.d.f.)
        #add_predictor_constr(m, dt_model_inv, p_comb_inv, p_comb, epsilon = 1e-6)

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
    params['target'] = 'wind'
    params['start_date'] = '2019-01-01'
    params['split_date'] = '2020-01-01' # Defines train/test split
    params['end_date'] = '2020-05-01'
    params['n_base'] = 60 # number of base series
    
    params['parallel'] = False # If True, then trees are grown in parallel
    params['save'] = False # If True, then saves models and results
    params['train_brc'] = False # If True, then saves models and results
    params['train_trees'] = True # If True, then train models, else tries to load previous runs
    
    # Experimental setup parameters
    params['quality'] = False
    params['problem'] = 'reg_trad' # {mse, newsvendor, cvar, reg_trad}
    params['N_sample'] = [10, 50, 100, 200, 500]
    params['N_assets'] = [5, 10, 20, 50]
    params['iterations'] = 5
    params['critical_fractile'] = 0.2
    params['varying_N'] = False # True: vary between [10, N_max], False: fixed in all
    params['N_max'] = 200
    
    #extra problem parameters
    params['risk_aversion'] = 0.5 # True: vary between [10, N_max], False: fixed in all
    params['epsilon'] = 0.05
    
    return params

#%%

config = params()
hyperparam = tree_params()

case_path = f'{cd}\\smart4res-mult-case-study'
results_path = f'{case_path}\\upd-results'
data_path = f'{case_path}\\data-processed'

target_var = config['target']

# load wind and NWP data for base series
turb_metadata = pd.read_csv(f'{data_path}\\{target_var}_metadata.csv', index_col = 0)
turb_power = pd.read_csv(f'{data_path}\\{target_var}_power_clean_imp_30min.csv', index_col = 0, parse_dates = True, dayfirst = False)
S_mat = pd.read_csv(f'{data_path}\\{target_var}_summation_matrix.csv')
capacity = S_mat.values@turb_metadata['Capacity (kW)']

# dictionary with predictor data for each park
with open(f'{data_path}\\{target_var}_weather_pred_30min.pkl', 'rb') as f:
        weather_dict = pickle.load(f)
turb_nwp = pd.concat(weather_dict, axis = 1)

# dictionary that maps turbine ID to closest grid point
with open(f'{data_path}\\{target_var}_meteo_id.pkl', 'rb') as f:
        meteo_id = pickle.load(f)

#%% Retrieve data for park level

# power for all levels of the hierarchy
power_total = pd.DataFrame(data = (S_mat.values@ turb_power.values.T).T, index = turb_power.index, 
                        columns = ['Top'] + list(turb_metadata['Park ID'].unique()) + list(turb_power.columns))

# scale all levels by capacity
sc_power_total = power_total/capacity

# park metadata
turb_metadata['meteo_id'] = meteo_id.values()
park_metadata = pd.DataFrame(data = turb_metadata['Park ID'].unique(), columns = ['Park ID'])

park_metadata['Lat'] = turb_metadata.groupby(['Park ID']).mean()['Lat'].values
park_metadata['Long'] = turb_metadata.groupby(['Park ID']).mean()['Long'].values
park_metadata['Capacity (kW)'] = turb_metadata.groupby(['Park ID']).sum()['Capacity (kW)'].values.astype(int)
park_metadata['meteo_id'] = turb_metadata.groupby(['Park ID']).mean()['meteo_id'].values.round().astype(int)

# features for parks: the closets NWP grid point from the center
park_weather_dict = {}
for i in range(len(park_metadata['Park ID'])):
    park_meteo_id = park_metadata['meteo_id'][i]
    turb_id = turb_metadata['Turbine ID'][turb_metadata['meteo_id'] == park_meteo_id].iloc[-1]
    park_weather_dict[park_metadata['Park ID'][i]] = weather_dict[str(turb_id)]

park_nwp = pd.concat(park_weather_dict, axis=1)
turb_nwp = pd.concat(weather_dict, axis = 1)
#%% Create supervised learning set

# select targets and features/ features are multilevel dataframe
turb_metadata['Turbine ID'] = turb_metadata['Turbine ID'].astype(str)

Y = sc_power_total[turb_metadata['Turbine ID']]
Pred = turb_nwp.copy()

#%%
target_ids = turb_metadata['Turbine ID']
n_targets = len(target_ids)

# create additional features
for j, p_id in enumerate(target_ids):
    Pred[p_id, 'wspeed'] = (np.sqrt(Pred[p_id]['u100'] ** 2 + Pred[p_id]['v100'] ** 2)).values

    Pred[p_id, 'wdir'] = (180/np.pi)*(np.arctan2(Pred[p_id]['u100'], Pred[p_id]['v100']))
    Pred[p_id, 'wdir'] = np.sin(np.deg2rad(Pred[p_id, 'wdir']))
    Pred[p_id, 'wdir_bin'] = pd.cut(Pred[p_id, 'wdir'], bins = 4, include_lowest = True,
                                labels = [0,1,2,3])
    Pred[p_id, 'wdir_bin'] = Pred[p_id, 'wdir_bin'].astype(int)



feat_col = ['wspeed', 'wdir']

Pred = Pred.reorder_levels([1,0], axis=1)
max_wspeed = Pred['wspeed'].max().max()
Pred['wspeed'] = (Pred['wspeed']/max_wspeed).round(2)

# train/test split
trainY = Y[config['start_date']:config['split_date']].round(2)
testY = Y[config['split_date']:config['end_date']].round(2)
    
trainPred = Pred[feat_col][config['start_date']:config['split_date']].round(2)
testPred = Pred[feat_col][config['split_date']:config['end_date']].round(2)

# re-order multiindex by park level
trainPred = trainPred.reorder_levels([1,0], axis=1)
testPred = testPred.reorder_levels([1,0], axis=1)

#%% Create different probabilistic models for the **same** wind park

target_park_id = '10262'
park_ids = turb_metadata['Turbine ID']

# number of observations to train prob. forecasting model
N_sample = 300
# number of forecasts to combine
N_experts = 2

# sample random parks to serve as experts
sampled_parks = np.random.choice(park_ids, N_experts, replace = False)

# data conditioned on wind speed
step = .01
y_supp = np.arange(0, 1+step, step).round(2)

if feat_col == ['wspeed', 'wdir_bin']:
    x_supp = [np.arange(0, 1+step, step).round(2), np.array([0,1,2,3])]
    xt, xr = np.meshgrid(x_supp[0], x_supp[1])
    x_joint_supp = np.array([xt.ravel(), xr.ravel()]).T
else:
    x_supp = np.arange(0, 1+step, step).round(2)
    x_joint_supp = np.arange(0, 1+step, step).round(2)
    
# Train the prob. forecasting models
train_localY = trainY[target_park_id].values[:N_sample]
train_localX = trainPred[target_park_id][feat_col].values[:N_sample].reshape(-1,1)

valid_localY = trainY[target_park_id].values[N_sample:2*N_sample]
valid_localX = trainPred[target_park_id][feat_col].values[N_sample:2*N_sample].reshape(-1,1)

# Test
test_localX = testPred[target_park_id][feat_col].values.reshape(-1,1)
test_localY = testY[target_park_id].values

# create data sets for the other experts

train_expX = trainPred[sampled_parks][:N_sample]
valid_expX = trainPred[sampled_parks][N_sample:2*N_sample]
test_expX = testPred[sampled_parks]

#%% Train probabilistic forecasting models

prob_models = []

for i, p in enumerate(sampled_parks):
    if i%2 == 0:
        temp_model = EnsemblePrescriptiveTree_OOB(n_estimators = 1, D = 2, max_features = 1, 
                                                  type_split = 'quant' )
    else:
        temp_model = EnsemblePrescriptiveTree_OOB(n_estimators = 100, max_features = 1, 
                                                  type_split = 'quant' )
        
    temp_model.fit(train_expX[p].values, train_localY, x_joint_supp, y_supp, 
                   bootstrap = False, quant = np.arange(.05, 1, .05), 
                   crit_quant = config['critical_fractile'], problem = config['problem'],
                   risk_aversion = config['risk_aversion'], epsilon = config['epsilon'] ) 
 
    
    prob_models.append(temp_model)

# find local weights for meta-training set/ map weights to support locations
w_list = []
p_list = []
for i, p in enumerate(sampled_parks):
    w_list.append(prob_models[i].find_weights(valid_expX[p].values, train_expX[p].values))
    p_list.append(wemp_to_support(w_list[i], train_localY, y_supp))
    
#%%
# find inverted CDFs
F_inv = [np.array([inverted_cdf([.05, .10, .90, .95] , train_localY, w_list[j][i]) for i in range(100)]) for j in range(N_experts)]

# Visualize some probabilistic forecasts
plt.plot(valid_localY[:100])
for i in range(N_experts):
    plt.fill_between(np.arange(100), F_inv[i][:,0], F_inv[i][:,-1], alpha = .3, color = 'blue')
    plt.fill_between(np.arange(100), F_inv[i][:,1], F_inv[i][:,-2], alpha = .3, color = 'blue')
plt.show()

#%% Example: point forecasting (MSE) & convex combination

# 1-1 mapping: decisions are the weighted sum
nobs = p_list[0].shape[0]
nlocations = p_list[0].shape[1]

# set up the SAA of the target optimization problem
m = gp.Model()
#m.setParam('OutputFlag', 1)
# Decision variables

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
m.setObjective( (valid_localY-z)@(valid_localY-z), gp.GRB.MINIMIZE)
m.optimize()

opt_lambda_ = lambda_coord.X
print(f'analytical:{lambda_coord.X}')
print(f'objective value:{m.objval}')
plt.plot(z.X)
plt.show()
#%% Differentiable layers testing

import cvxpy as cp
import tensorflow as tf
from cvxpylayers.tensorflow import CvxpyLayer


# Optimization solution
nobs = len(valid_localY)
m = N_experts
k = len(y_supp)

z = cp.Variable(nobs)
error = cp.Variable(nobs)
lambda_ = cp.Variable(m)


constraints = [z >= 0, z <=1, z == sum([lambda_[i]*p_list[i] for i in range(m)])@y_supp, 
               error == valid_localY - z, lambda_>=0, lambda_.sum() == 1, lambda_ <= 1]

objective = cp.Minimize(0.5*cp.sum_squares(error)/nobs)
problem = cp.Problem(objective, constraints)
problem.solve()
opt_l = lambda_.value
print(f'optimal:{opt_l}')
print(f'objective:{objective.value}')

#%%
# Gradient with full batch updates/ manual implementation with CVXPY
nobs = len(valid_localY)
m = N_experts
k = len(y_supp)

z = cp.Variable(nobs)
lambda_ = cp.Parameter(m)


constraints = [z >= 0, z <=1, z == sum([lambda_[i]*p_list[i] for i in range(m)])@y_supp]

objective = cp.Minimize(0.5*cp.sum_squares(valid_localY - z))
problem = cp.Problem(objective, constraints)

L_t = [[0.5, 0.5]]
Loss = []
eta = 1e-2

# projection step
y_proj = cp.Variable(m)
y_hat = cp.Parameter(m)
proj_problem = cp.Problem(cp.Minimize(0.5*cp.sum_squares(y_proj-y_hat)), [y_proj >= 0, y_proj.sum()==1])

Projection = True
for i in range(1000):
    
    # forward pass
    lambda_.value = L_t[-1]
        
    problem.solve(requires_grad = True)
    
    # gradient of cost w.r.t. to decision
    z.gradient = (valid_localY - z.value)
    
    # gradient of decision w.r.t. to parameter
    problem.backward()    
    Loss.append(objective.value/nobs)
    
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
        
#%% !!!! Example with Tensorflow/CVXPY layers (don't know how to update parameters yet)

nobs = len(valid_localY)
m = N_experts
k = len(y_supp)

z = cp.Variable(nobs)
error = cp.Variable(nobs)
lambda_ = cp.Parameter(m)


constraints = [z >= 0, z <=1, z == sum([lambda_[i]*p_list[i] for i in range(m)])@y_supp, 
               error == valid_localY.reshape(-1) - z ]

objective = cp.Minimize(cp.norm(error, p=2))
problem = cp.Problem(objective, constraints)

cvxpylayer = CvxpyLayer(problem, parameters=[lambda_], variables=[z, error])
lambda_tf = tf.Variable([opt_l])

with tf.GradientTape() as tape:
  # forward pass
  # solve the problem, setting the values of A, b to A_tf, b_tf
  solution = cvxpylayer(lambda_tf)
  summed_solution = tf.math.reduce_sum(solution)
  
# compute the gradient of the summed solution with respect to A, b
grad_lambda = tape.gradient(summed_solution, [lambda_tf])


#%% Pytorch/CVXPY layers example

import cvxpy as cp
import torch
from torch import nn
import torch.nn.functional as F
from cvxpylayers.torch import CvxpyLayer

def to_np(x):
    return x.detach().numpy()


nobs = len(valid_localY)
m = N_experts
k = len(y_supp)

z = cp.Variable(nobs)
lambda_ = cp.Parameter(m)
y_target = cp.Parameter(nobs)

constraints = [z >= 0, z <=1, z == sum([lambda_[i]*p_list[i] for i in range(m)])@y_supp]

objective = cp.Minimize(cp.norm(valid_localY - z, p=2))
problem = cp.Problem(objective, constraints)

layer = CvxpyLayer(problem, parameters=[lambda_], variables=[z,])

l_hat = nn.Parameter(torch.FloatTensor([0.5, 0.5]).requires_grad_())

opt = torch.optim.SGD([l_hat], lr=1e-2)
losses = []

L_t = [to_np(l_hat)]
Projection = True

for i in range(1000):
        
    # Forward pass: solve optimization problem
    zhat, = layer(l_hat)
    np_zhat = to_np(zhat)
    
    assert(np_zhat.min()>=0)
    assert(np_zhat.max()<=1)
    
    # Estimate model loss
    loss = (zhat - nn.Parameter(torch.FloatTensor(valid_localY))).norm()
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


#%%%%%%%%%%%%%%%%%%% Newvendor problem

# CVXPY/ try for single value
critical_fractile = 0.4
z = cp.Variable(1)
error = cp.Variable((len(y_supp)))
pinball_loss = cp.Variable((len(y_supp)))
lambda_init = np.array([0.5, 0.5])
p_comb = sum([lambda_init[j]*p_list[j] for j in range(m)])

# inner problem
Z_inner = []
constraints = [z >= 0, z <=1, error == y_supp - z, pinball_loss >= critical_fractile*error, 
               pinball_loss >= (critical_fractile-1)*error]
for i in range(nobs):

    objective = cp.Minimize(pinball_loss@p_comb[i])
    problem = cp.Problem(objective, constraints)

    problem.solve(solver = 'GUROBI')
    Z_inner.append(z.value[0])

#%% Gradient-based approach with CVXPY/ full batch updates
batch_size = 1

z = cp.Variable(batch_size)
error = cp.Variable((batch_size, len(y_supp)))
pinball_loss = cp.Variable((batch_size, len(y_supp)))
lambda_ = cp.Parameter(m)
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


# projection step
y_proj = cp.Variable(m)
y_hat = cp.Parameter(m)
proj_problem = cp.Problem(cp.Minimize(0.5*cp.sum_squares(y_proj-y_hat)), [y_proj >= 0, y_proj.sum()==1])

L_t = [[0.5, 0.5]]
Loss = []
Projection = True
eta = 1e-2

for i in range(1000):
    
    ix = np.random.choice(range(nobs), batch_size, replace = False)

    # forward pass
    lambda_.value = L_t[-1]
    
    p_comb_hat = sum([lambda_[j]*p_list[j][ix] for j in range(m)])
    
    objective = cp.Minimize(sum([pinball_loss[i]@p_comb_hat[i] for i in range(batch_size)]) )
    problem = cp.Problem(objective, constraints)

        
    problem.solve(solver = 'SCS', requires_grad = True)
    
    # gradient of cost w.r.t. to decision
    z.gradient = np.sign(valid_localY[ix] - z.value)
    z.gradient[z.gradient>0] = critical_fractile
    z.gradient[z.gradient<0] = critical_fractile-1
   
    # gradient of decision w.r.t. to parameter
    problem.backward()    
    Loss.append(objective.value)

    # update parameters (!!! projection step might be required)
    upd_L_t = L_t[-1] - eta*lambda_.gradient
    
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

#%% Solve the same problem with a decision rule approach
from gurobi_ml import add_predictor_constr
from sklearn.tree import DecisionTreeRegressor
from gurobi_ml.sklearn import add_decision_tree_regressor_constr, add_random_forest_regressor_constr

# optimal solutions to create training set for decision rule
z_opt = []
for j in range(N_experts):
    z_opt.append( np.array([inverted_cdf([critical_fractile] , y_supp, p_list[j][i]) for i in range(nobs)]) )

z_opt = np.row_stack(z_opt)
p_stacked = np.row_stack((p_list))

#### Train ML predictor to map prob. vectors (p.d.f.) to optimal decisions (linear, tree-based methods)

dt_model = DecisionTreeRegressor().fit(p_stacked, z_opt )

plt.plot(z_opt[-100:])
plt.plot(dt_model.predict(p_stacked)[-100:], '-.')
plt.legend(['z_opt', 'DT'])
plt.show()

lambda_cc = tune_combination_newsvendor(valid_localY, p_list, dt_model, 
                                           crit_fract = critical_fractile, support = y_supp)

#%% Pytorch example/ stochastic gradient updates

batch_size = 1
nobs = len(valid_localY)
m = N_experts
k = len(y_supp)

# Variables of inner problem
z = cp.Variable(batch_size)
error = cp.Variable((batch_size,k))
p_comb = cp.Parameter((batch_size,k))
pinball_loss = cp.Variable((batch_size,k))

lambda_ = cp.Parameter(m)
p_list_t = [cp.Parameter((batch_size,k)) for j in range(m)]
p_list_t_aux = [cp.Variable((batch_size,k)) for j in range(m)]

constraints = [z >= 0, z <=1] + [error[i] == y_supp - z[i] for i in range(batch_size)]\
            + [pinball_loss[i] >= critical_fractile*error[i] for i in range(batch_size)]\
            + [pinball_loss[i] >= (critical_fractile-1)*error[i] for i in range(batch_size)]\
            + [p_list_t_aux[j] == p_list_t[j] for j in range(m)]\
            + [p_comb == sum([p_list_t_aux[j]*lambda_[j] for j in range(m)])]
    

reg_pen = 10*cp.norm(cp.multiply(p_comb, error))

objective = cp.Minimize( sum([pinball_loss[i]@p_comb[i] for i in range(batch_size)])/batch_size + reg_pen/batch_size) 

problem = cp.Problem(objective, constraints)

layer = CvxpyLayer(problem, parameters=[lambda_, p_comb] + p_list_t, variables=[z, pinball_loss, error] + p_list_t_aux)

l_hat = nn.Parameter(torch.FloatTensor([0.5, 0.5]).requires_grad_())

opt = torch.optim.Adam([l_hat], lr=1e-3)
losses = []
L_t = [to_np(l_hat)]
Projection = True

for i in range(10000):
    
    ix = np.random.choice(range(nobs), batch_size, replace = False)
    
    p_list_t_hat = []

    for j in range(m):
        p_list_t_hat.append( nn.Parameter( torch.FloatTensor([p_list[j][ix]]) ) ) 
        
    p_comb_hat = (sum([p_list_t_hat[j]*l_hat[j] for j in range(m)]))
    
    zhat = layer(l_hat, p_comb_hat, p_list_t_hat[0], p_list_t_hat[1])
    error_hat = nn.Parameter(torch.FloatTensor([valid_localY[ix]])) - zhat[0]
    #!!!!! add pinball loss here
    #loss = error_hat.norm(p = 1)
    loss = (critical_fractile*error_hat[error_hat>0].norm(p=1) + (1-critical_fractile)*error_hat[error_hat<0].norm(p=1))\
    #    + 10*error_hat.norm(p=2)
    
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

    if i%10==0:
        print(L_t[-1])
        
        plt.plot(losses)
        plt.show()
        


#%% PYtorch example
import torch
from torch import nn
import torch.nn.functional as F
from cvxpylayers.torch import CvxpyLayer

nobs = len(valid_localY)
batch_size = nobs
m = N_experts
k = len(y_supp)

# Variables of inner problem
z = cp.Variable(batch_size)
error = cp.Variable((batch_size,k))
p_comb = cp.Parameter((batch_size,k))
pinball_loss = cp.Variable((batch_size,k))

lambda_ = cp.Parameter(m)
#p_list_t = [cp.Parameter((batch_size,k)) for j in range(m)]
#p_list_t_aux = [cp.Variable((batch_size,k)) for j in range(m)]

constraints = [z >= 0, z <=1] + [error[i] == y_supp - z[i] for i in range(batch_size)]\
            + [pinball_loss[i] >= critical_fractile*error[i] for i in range(batch_size)]\
            + [pinball_loss[i] >= (critical_fractile-1)*error[i] for i in range(batch_size)]\
           + [p_comb == sum([p_list[j]*lambda_[j] for j in range(m)])]
#            + [p_list_t_aux[j] == p_list[j] for j in range(m)]\
    
objective = cp.Minimize( sum([ pinball_loss[i]@p_comb[i] for i in range(batch_size)])/batch_size ) 
problem = cp.Problem(objective, constraints)

layer = CvxpyLayer(problem, parameters=[lambda_, p_comb], variables=[z, pinball_loss, error] )

l_hat = nn.Parameter(torch.FloatTensor([0.5, 0.5]).requires_grad_())

opt = torch.optim.SGD([l_hat], lr=1e-2)
losses = []

for i in range(1000):
    
    p_comb_hat = (sum([p_list[j]*l_hat[j] for j in range(m)]))

    zhat = layer(l_hat, p_comb_hat)
    error_hat = nn.Parameter(torch.FloatTensor(valid_localY)) - zhat[0]
    #!!!!! add pinball loss here
    #loss = error_hat.norm(p = 1)
    
    loss = critical_fractile*error_hat[error_hat>0].norm(p=1) + (1-critical_fractile)*error_hat[error_hat<0].norm(p=1)
#        + 10*error_hat.norm(p=2)

    losses.append(to_np(loss))
            
    opt.zero_grad()
    loss.backward()
    opt.step()
    
    if i%10==0:
        print(l_hat)
        
        plt.plot(losses)
        plt.show()

#%% Pytorch example/ Full batch updates

batch_size = 10
nobs = len(valid_localY)
m = N_experts
k = len(y_supp)

# Variables of inner problem
z = cp.Variable(batch_size)
error = cp.Variable((batch_size,k))
pinball_loss = cp.Variable((batch_size,k))
total_loss = cp.Variable((1))

lambda_ = cp.Parameter(m)
p_list_aux = [cp.Parameter((batch_size, k)) for j in range(m)]
p_comb = cp.Parameter((batch_size, k))

constraints =   [z >= 0, z <=1] \
                + [pinball_loss[i] >= critical_fractile*error[i] for i in range(batch_size)] \
                + [pinball_loss[i] >= (critical_fractile - 1)*error[i] for i in range(batch_size)] \
                + [error[i] == y_supp - z[i] for i in range(batch_size)] \
                + [p_comb == sum([p_list_aux[j]*lambda_[j] for j in range(m)])]
    
objective = cp.Minimize( sum([pinball_loss[i]@p_comb[i] for i in range(batch_size)])  )

problem = cp.Problem(objective, constraints)

layer = CvxpyLayer(problem, parameters=[lambda_, p_list_aux[0], p_list_aux[1], p_comb], 
                   variables=[z, error, pinball_loss])
l_hat = nn.Parameter(torch.FloatTensor([0.5, 0.5]).requires_grad_())

opt = torch.optim.Adam([l_hat], lr=1e-3)
losses = []

for i in range(1000):
    
    ix = np.random.choice(range(nobs), batch_size, replace = False)

    # update the rest of the problem parameters given lambda and sampled instances
    p_list_t_hat = [nn.Parameter( torch.FloatTensor([p_list[j][ix]]) ) for j in range(m)]
    for j in range(m):
        p_list_t_hat.append(  ) 

    p_comb_hat = sum(p_list_t_hat)    
    zhat = layer(l_hat, p_list_t_hat[0], p_list_t_hat[1], p_comb_hat)

    adsf    
    error_hat = zhat[0] - nn.Parameter(torch.FloatTensor([valid_localY[ix]]))
    
    loss = error_hat.norm()
    losses.append(to_np(loss))
            
    opt.zero_grad()
    loss.backward()
    opt.step()
    
    if i%10==0:
        plt.plot(losses)
        plt.show()
        
        
#%%



nx, ncon = 2, 10

_G = cp.Parameter((ncon, nx))
_h = cp.Parameter(ncon)
_x = cp.Parameter(nx)
_y = cp.Variable(nx)
obj = cp.Minimize(0.5*cp.sum_squares(_x-_y))
cons = [_G*_y <= _h]
prob = cp.Problem(obj, cons)

layer = CvxpyLayer(prob, parameters=[_G, _h, _x], variables=[_y])

torch.manual_seed(6)
G = torch.FloatTensor(ncon, nx).uniform_(-4, 4)
z0 = torch.full([nx], 0.5)
s0 = torch.full([ncon], 0.5)
h = G.mv(z0)+s0
#plotConstraints(to_np(G), to_np(h))

torch.manual_seed(22)
G_hat = nn.Parameter(torch.FloatTensor(ncon, nx).uniform_(-4, 4).requires_grad_())
h_hat = G_hat.mv(z0)+s0
#plotConstraints(to_np(G), to_np(h), to_np(G_hat), to_np(h_hat))

opt = torch.optim.Adam([G_hat], lr=1e-2)
losses = []

for i in range(2500):
    x = torch.randn(nx)
    y, = layer(G, h, x)
    
    h_hat = G_hat.mv(z0)+s0
    yhat, = layer(G_hat, h_hat, x)
    loss = (yhat-y).norm()
    losses.append(loss)
    
    if i % 50 == 0:
        fig, ax = plotConstraints(to_np(G), to_np(h), to_np(G_hat), to_np(h_hat))
        fig.tight_layout()
        fig.savefig(f'{d}/{i:04d}.png')
        plt.close(fig)
        
    opt.zero_grad()
    loss.backward()
    opt.step()
    
#%%
# Testing
w_test_list = [prob_models[i].find_weights(test_expX[p].values, train_expX[p].values) for i, p in enumerate(sampled_parks)]

#%%
y_hat_1 = lambda_coord[0].X*w_test_list[0]@train_localY
y_hat_2 = lambda_coord[1].X*w_test_list[1]@train_localY
y_hat_comb = sum([lambda_coord[i].X*w_test_list[i] for i in range(N_experts)])@train_localY

print(f'Model 1:{eval_point_pred(y_hat_1, test_localY)[0]}')
print(f'Model 2:{eval_point_pred(y_hat_2, test_localY)[0]}')
print(f'ConvComb:{eval_point_pred(y_hat_comb, test_localY)[0]}')
print(f'Ave:{eval_point_pred((y_hat_1+y_hat_2)/2, test_localY)[0]}')


#%%%%%%%%%%%%%%% Newsvendor experiment


from sklearn.linear_model import LinearRegression, Ridge
from gurobi_ml import add_predictor_constr
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from gurobi_ml.sklearn import add_decision_tree_regressor_constr, add_random_forest_regressor_constr


# problem parameters
critical_fractile = .3

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
dt_model = DecisionTreeRegressor().fit(p_stacked, z_opt )
#%%
plt.plot(z_opt[-100:])
plt.plot(lr_model.predict(p_stacked)[-100:], '--')
plt.plot(dt_model.predict(p_stacked)[-100:], '-.')
plt.plot(rf_model.predict(p_stacked)[-100:], 'd', markersize = 2)
plt.legend(['z_opt', 'LR', 'DT', 'RF'])
plt.show()

#%% 

# Case 1: Convex Combination/ LR decision rule
lambda_cc_lr = tune_combination_newsvendor(valid_localY, p_list, lr_model, 
                                           crit_fract = critical_fractile, support = y_supp)
#%%
# Case 2: Convex Combination/ DT decision rule
lambda_cc_dt = tune_combination_newsvendor(valid_localY, p_list, dt_model, 
                                           crit_fract = critical_fractile, support = y_supp)

#%%
# Case 3: Barycentric interpolation/ DT

# Train additional ML predictor to learn mapping from quantile function to p.d.f. (the true is piecewise constant)
target_quantiles = np.arange(0,1,.01)

p_inv_list = [np.array([inverted_cdf(target_quantiles, y_supp, w = p_list[j][i]) for i in range(nobs)]) for j in range(N_experts)]

dt_model_inv = DecisionTreeRegressor().fit(np.row_stack(p_inv_list), np.row_stack(p_list))

t = -10
plt.plot(dt_model_inv.predict(np.row_stack(p_inv_list))[t])
plt.plot(np.row_stack(p_list)[t])
plt.show()
#%%
lambda_brc_dt = tune_combination_newsvendor(valid_localY, p_list, dt_model, brc_predictor= dt_model_inv, 
                                            type_ = 'barycenter', crit_fract = critical_fractile, support = y_supp)


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

for i in range(n_test_obs):
    temp_p_list = [p[i] for p in p_hat_test_list]

    temp_p_brc, _, _ = wass2_barycenter_1D(N_experts*[y_supp], temp_p_list, lambda_coord = [.5, .5], support = y_supp, p = 2, 
                               prob_dx = .001)
    p_brc[i] = temp_p_brc
#%%
p_cc_lr = sum([lambda_cc_lr[i]*p_hat_test_list[i] for i in range(N_experts)])
p_cc_dt = sum([lambda_cc_dt[i]*p_hat_test_list[i] for i in range(N_experts)])
p_brc_dt = np.zeros((n_test_obs, nlocations))

#p_cc_lr = lambda_cc_lr[0]*p1_hat + lambda_cc_lr[1]*p2_hat
#p_cc_dt = lambda_cc_dt[0]*p1_hat + lambda_cc_dt[1]*p2_hat
#p_brc_dt = np.zeros(p1_hat.shape)

for i in range(n_test_obs):
    
    temp_p_list = [p[i] for p in p_hat_test_list]
    
    temp_p_brc, _, _ = wass2_barycenter_1D(N_experts*[y_supp], temp_p_list, lambda_coord = lambda_brc_dt, support = y_supp, p = 2, 
                               prob_dx = .001)
    p_brc_dt[i] = temp_p_brc
#%%
# turn probability vectors to decisions/ closed-form solution for newsvendor problem

models = ['Model-1', 'Model-2', 'CC-Ave', 'BRC-Ave', 'CC-LR', 'CC-DT', 'BRC-DT']
Prescriptions = pd.DataFrame(data = np.zeros((len(test_localY), len(models))), columns = models)

Prescriptions['Model-1'] = np.array([inverted_cdf([critical_fractile], y_supp, p_hat_test_list[0][i]) for i in range(len(test_localY))]).reshape(-1)
Prescriptions['Model-2'] = np.array([inverted_cdf([critical_fractile], y_supp, p_hat_test_list[1][i]) for i in range(len(test_localY))]).reshape(-1)
Prescriptions['CC-Ave'] = np.array([inverted_cdf([critical_fractile], y_supp, p_ave[i]) for i in range(len(test_localY))]).reshape(-1)
Prescriptions['BRC-Ave'] = np.array([inverted_cdf([critical_fractile], y_supp, p_brc[i]) for i in range(len(test_localY))]).reshape(-1)
Prescriptions['CC-LR'] = np.array([inverted_cdf([critical_fractile], y_supp, p_cc_lr[i]) for i in range(len(test_localY))]).reshape(-1)
Prescriptions['CC-DT'] = np.array([inverted_cdf([critical_fractile], y_supp, p_cc_dt[i]) for i in range(len(test_localY))]).reshape(-1)
Prescriptions['BRC-DT'] = np.array([inverted_cdf([critical_fractile], y_supp, p_brc_dt[i]) for i in range(len(test_localY))]).reshape(-1)

#%%

for m in models:
    print(f'{m}:{100*newsvendor_loss(Prescriptions[m].values, test_localY, q = critical_fractile).round(6)}')
    
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

