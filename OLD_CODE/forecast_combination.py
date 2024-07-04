# -*- coding: utf-8 -*-
"""
Decision-focused forecast combination

@author: akylas.stratigakos@minesparis.psl.eu
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
N_sample = 1000
# number of forecasts to combine
N_experts = 3

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
    print(f'Training model {i}')
    if i%2 == 1:
        temp_model = EnsemblePrescriptiveTree_OOB(n_estimators = 1, D = 2, max_features = 1, 
                                                  type_split = 'quant' )
    else:
        temp_model = EnsemblePrescriptiveTree_OOB(n_estimators = 100, max_features = 1, 
                                                  type_split = 'quant' )
        
    temp_model.fit(train_expX[p].values, train_localY, x_joint_supp, y_supp, 
                   bootstrap = False, quant = np.arange(.01, 1, .01), 
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
#%%
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

#%%
# Testing

# find weights for OOS observations
w_test_list = [prob_models[i].find_weights(test_expX[p].values, train_expX[p].values) for i, p in enumerate(sampled_parks)]

# evaluate all models
y_hat_local = [lambda_coord[i].X*w_test_list[i]@train_localY for i in range(N_experts)]
y_hat_comb = sum([lambda_coord[i].X*w_test_list[i] for i in range(N_experts)])@train_localY

for i in range(N_experts):
    print(f'Model {i}:{eval_point_pred(y_hat_local[i], test_localY)[0]}')
print(f'ConvComb:{eval_point_pred(y_hat_comb, test_localY)[0]}')
print(f'Ave:{eval_point_pred(sum(y_hat_local)/N_experts, test_localY)[0]}')

'''
# Gradient-based solution with differentiable optimization
import cvxpy as cp
import tensorflow as tf
from cvxpylayers.tensorflow import CvxpyLayer

n, m = 2, 3

x = cp.Variable(n)
A = cp.Parameter((m, n))
b = cp.Parameter(m)
constraints = [x >= 0]
objective = cp.Minimize(0.5 * cp.pnorm(A @ x - b, p=1))
problem = cp.Problem(objective, constraints)
assert problem.is_dpp()

cvxpylayer = CvxpyLayer(problem, parameters=[A, b], variables=[x])
A_tf = tf.Variable(tf.random.normal((m, n)))
b_tf = tf.Variable(tf.random.normal((m,)))

with tf.GradientTape() as tape:
  # solve the problem, setting the values of A, b to A_tf, b_tf
  solution, = cvxpylayer(A_tf, b_tf)
  summed_solution = tf.math.reduce_sum(solution)
# compute the gradient of the summed solution with respect to A, b
gradA, gradb = tape.gradient(summed_solution, [A_tf, b_tf])

# Analytical solution

m = N_experts
n = nobs
k = len(y_supp)

# forward pass
lambda_ = cp.Variable((m))
z = cp.Variable(n)
error = cp.Variable(n)

constr_wSAA = [z >= 0, z <= 1, z == sum([p_list[i]*lambda_[i] for i in range(m)])@y_supp, 
               error == z-valid_localY.reshape(-1), lambda_ >=0, lambda_ <=1, lambda_.sum()==1]

objective = cp.Minimize(0.5*cp.square(error).sum())
problem = cp.Problem(objective, constr_wSAA)
problem.solve()

print(f'Analytical solution:{lambda_.value.round(2)}')
print(f'Optimal Obj. value:{objective.value.round(2)}')

#%%
m = N_experts
n = nobs
k = len(y_supp)
#p_list[0] = 1/len(y_supp)*np.ones((n,k))

# forward pass
# Problem 1: forecast combination
lambda_ = cp.Parameter((m))
#p_comb = cp.Variable((n,k))
#constr_FC = [p_comb >=0, p_comb <=1, p_comb.sum(1) == 1, p_comb == sum([p_list[i]*lambda_[i] for i in range(m)])]

# problem 2: find wSAA decisions 
z = cp.Variable(n)
error = cp.Variable(n)

constr_wSAA = [z >= 0, z <= 1, z == sum([p_list[i]*lambda_[i] for i in range(m)])@y_supp]

objective = cp.Minimize(0.5*cp.sum_squares(valid_localY - z))

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
best_Loss = 1e2
for iter_ in range(200):
    print(f'iteration:{iter_}')
    # solve forward pass
    problem.solve(requires_grad = True)
    Loss.append(objective.value*nobs)
    best_Loss = min(Loss)

    # backward pass
    z.gradient = (valid_localY - z.value)
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
    if iter_ > 0:    
        if cp.norm(Loss[-1] - best_Loss, p=1).value <=1e-5: break

plt.plot(L_t)
plt.show()

print(f'Gradient-based solution:{L_t[-1].round(2)}')
print(f'Optimal Obj. value:{objective.value.round(2)}')
#%%

# forward pass
# Problem 1: forecast combination
lambda_ = cp.Parameter((m))
p_comb = cp.Variable((n,k))
#constr_FC = [p_comb >=0, p_comb <=1, p_comb.sum(1) == 1, p_comb == sum([p_list[i]*lambda_[i] for i in range(m)])]

# problem 2: find wSAA decisions 
z = cp.Variable(n)

constr_wSAA = [z >= 0, z <= 1, z == (p_list[0]*lambda_[0] + p_list[1]*lambda_[1])@y_supp]

# outer problem
error = cp.Variable(n)

constr_Outer = [error == valid_localY.reshape(-1) - z]

objective = cp.Minimize(cp.norm(error, p = 2).sum())
problem = cp.Problem(objective, constr_wSAA + constr_Outer)

assert problem.is_dpp()

L_t0 = [0.1, 0.9]

lambda_.value = L_t0
problem.solve(requires_grad=True)

#z.gradient = error.value

problem.backward()
print(f'gradient of z:{lambda_.gradient}')
#%%
import cvxpy as cp
import tensorflow as tf
from cvxpylayers.tensorflow import CvxpyLayer


cvxpylayer = CvxpyLayer(problem, parameters=[lambda_], variables=[z, error])
lambda_tf = tf.Variable(L_t0)

with tf.GradientTape() as tape:
  # solve the problem, setting the values of A, b to A_tf, b_tf
  solution = cvxpylayer(lambda_tf)
  summed_solution = tf.math.reduce_sum(solution)
# compute the gradient of the summed solution with respect to A, b
gradL = tape.gradient(summed_solution, [lambda_tf])

print(gradL)
'''

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
dt_model = DecisionTreeRegressor(max_depth = 5).fit(p_stacked, z_opt)
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

# Case 2: Convex Combination/ DT decision rule
lambda_cc_dt = tune_combination_newsvendor(valid_localY, p_list, dt_model, 
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

