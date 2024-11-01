# -*- coding: utf-8 -*-
"""
Synthetic case study

@author: astratig
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import gurobipy as gp
import torch

cd = os.path.dirname(__file__)  #Current directory
sys.path.append(cd)

from utility_functions import *
# from optimal_transport_functions import *
from torch_layers_functions import *
from torch.utils.data import Dataset, DataLoader
from scipy.stats import norm

# IEEE plot parameters (not sure about mathfont)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 600
plt.rcParams['figure.figsize'] = (3.5, 2) # Height can be changed
plt.rcParams['font.size'] = 7
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams["mathtext.fontset"] = 'dejavuserif'

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
        H_i = 1*np.repeat(y_supp.reshape(1,-1), len(target_y), axis = 0)>=target_y.reshape(-1,1)
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
        
    if problem in ['reg_trad', 'newsvendor']:
        # regularized newsvendor problem
        # solve for multiple test observations/ declares gurobi model once for speed up
        n_test_obs = len(weights)
        Prescriptions = np.zeros((n_test_obs))
        m = gp.Model()            
        m.setParam('OutputFlag', 0)

        # variables
        offer = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'offer')
        deviation = m.addMVar(n_scen, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
        loss = m.addMVar(n_scen, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'aux')
        t = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'aux')
        
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
    
    elif problem =='mse':
        return (target_scen@weights)
 
def GD_params():
    'Gradient Descent hyperparameters'
    nn_params = {}
    nn_params['patience'] = 10
    nn_params['batch_size'] = 512  
    nn_params['num_epochs'] = 1500
    nn_params['learning_rate'] = 5e-2
    nn_params['apply_softmax'] = True
    return nn_params

def params():
    ''' Set up the experiment parameters'''

    params = {}
    params['save'] = False # If True, then saves models and results
    
    # Experimental setup parameters
    params['problem'] = 'newsvendor' # {mse, newsvendor, cvar, reg_trad, pwl}
    params['gamma_list'] = [0, 0.1, 1]    
    params['crit_quant'] = 0.2
    
    return params

#%%
   
config = params()

# Directory path
results_path = f'{cd}\\results\\synthetic_data'
data_path = f'{cd}\\data'

#%% Generate synthetic data

np.random.seed(0)

# experiment parameters
nobs_train = 5000
nobs_test = 5000
nobs = nobs_train + nobs_test
critical_fractile = config['crit_quant']

print(f'Newsvendor problem with critical quantile:{critical_fractile}')
print('Generate synthetic data')
# fixed term to ensure everything is non-negative
# ** Does not affect the results, only to speed up computations with nonnegativity of parameters
bias_term = 15

y_supp = np.arange(-20, 20, 0.1).round(1) + bias_term # support of discrete distribution
n_locs = len(y_supp)
target_quant = np.arange(0.01, 1, 0.01).round(2) # quantiles to evaluate CRPS or pinball loss

alpha_1 = 1.2
alpha_2 = 1.2
alpha_3 = 4.5

beta = -1.3

X0 = np.random.normal(loc = bias_term, scale = 1, size = nobs).round(1)
X1 = np.random.normal(size = nobs, scale = 1).round(1)
X2 = np.random.normal(size = nobs, scale = 1).round(1)
X3 = np.random.normal(size = nobs).round(1)
error = np.random.normal(scale = 1, size = nobs).round(1)

P_1 = (alpha_1*X1 + alpha_2*X2).round(1)

Y = X0 + P_1 + (alpha_3*X3)*((X3) < beta) # + error

Y = Y.round(2)
Y = projection(Y, ub = y_supp.max(), lb = y_supp.min())

Y_train = Y[:nobs_train]
Y_test = Y[nobs_train:]

# # Visualize data
# plt.hist(Y_train, bins = 50)
# plt.show()

### Expert forecasts for both training and test set
# Expert 1: Access to features 1&2
p1_hat = np.zeros((nobs, n_locs))
F1_hat = np.zeros((nobs, n_locs))
Q1_hat = np.zeros((nobs, len(target_quant)))

for i in range(nobs):
    # define probabilistic forecast
    f1_hat_temp = norm(loc = X0[i] + alpha_1*X1[i] + alpha_2*X2[i], scale = 1)
        
    p1_hat[i] = f1_hat_temp.pdf(y_supp)*0.1
    F1_hat[i] = f1_hat_temp.cdf(y_supp)
    Q1_hat[i] = f1_hat_temp.ppf(target_quant)
    

# Expert 2: Access to feature 3/ calibrated probabilistic forecast **only** on the left tail
p2_hat = np.zeros((nobs, n_locs))
F2_hat = np.zeros((nobs, n_locs))
Q2_hat = np.zeros((nobs, len(target_quant)))

for i in range(nobs):

    if X3[i] < beta:           
        f2_hat_temp = norm(loc = X0[i] + (X3[i]*alpha_3), scale = 1)
    else:
        f2_hat_temp = norm(loc = X0[i], scale = 1)

        
    p2_hat[i] = f2_hat_temp.pdf(y_supp)*0.1
    F2_hat[i] = f2_hat_temp.cdf(y_supp)
    Q2_hat[i] = f2_hat_temp.ppf(target_quant)

# evaluate probabilistic predictions
pinball_1 = 100*pinball(Q1_hat[:nobs_train], Y_train, target_quant).round(4)
pinball_2 = 100*pinball(Q2_hat[:nobs_train], Y_train, target_quant).round(4)

# Plot quantile score for sanity check
plt.plot(pinball_1, label = 'Expert 1')
plt.plot(pinball_2, label = 'Expert 2')
plt.title('Prob. Forecast Quality')
plt.xlabel('Quantile')
plt.ylabel('Quantile Score')
plt.show()

print('Average Pinball Loss')
print(f'Expert 1:{pinball_1.mean()}')
print(f'Expert 2:{pinball_2.mean()}')

print('Pinball loss at critical quantile')
print(f'Expert 1:{pinball_1[np.where(target_quant == 0.20)[0]]}')
print(f'Expert 2:{pinball_2[np.where(target_quant == 0.20)[0]]}')

#%% CRPS learning

train_p_list = [p1_hat[:nobs_train], p2_hat[:nobs_train]]
N_experts = 2
lambda_static_dict = {}

lambda_static_dict['Expert1'] = np.array([1,0])
lambda_static_dict['Expert2'] = np.array([0,1])
lambda_static_dict['Ave'] = np.array([0.5, 0.5])

tensor_trainY = torch.FloatTensor(Y_train)
#tensor_train_p = torch.FloatTensor(np.column_stack(([p1_hat, p2_hat])))
tensor_train_p_list = [torch.FloatTensor(train_p_list[i]) for i in range(2)]

# Optimizer hyperparameters
batch_size = 500
learning_rate = 1e-2
num_epochs = 100
patience = 10

train_data_loader = create_data_loader(tensor_train_p_list + [tensor_trainY], batch_size = batch_size, shuffle = False)

#### CRPS Learning, gradient-based approach with torch layer
lpool_crps_model = LinearPoolCRPSLayer(num_inputs=N_experts, support = torch.FloatTensor(y_supp), feasibility_method = 'softmax')
optimizer = torch.optim.SGD(lpool_crps_model.parameters(), lr = 1e-2)
lpool_crps_model.train_model(train_data_loader, train_data_loader, optimizer, epochs = 500, patience = 25)
lambda_crps = lpool_crps_model.get_weights()

print(f'CRPSL weights:{lambda_crps}')

lambda_static_dict['CRPS'] = lambda_crps
#%% Decision-focused learning

# optimization problem parameters
critical_fractile = 0.2
regularization = 0.001 # Small regularization to help convergence of gradient-based algo

# Optimizer hyperparameters
batch_size = 500
learning_rate = 1e-2
num_epochs = 100
patience = 5

train_data_loader = create_data_loader(tensor_train_p_list + [tensor_trainY], batch_size = batch_size, shuffle = False)

# iterate over values of gamma (CRPS regularization)
print('Learn Decision-focused static combinations')
for gamma in [0, 0.1, 1]:
    
    lpool_newsv_model = LinearPoolNewsvendorLayer(num_inputs=N_experts, support = torch.FloatTensor(y_supp),
                                                gamma = gamma, problem = 'reg_trad', critic_fract = critical_fractile, risk_aversion = regularization,
                                                feasibility_method = 'softmax')
    
    optimizer = torch.optim.Adam(lpool_newsv_model.parameters(), lr = 1e-2)
    
    lpool_newsv_model.train_model(train_data_loader, train_data_loader, optimizer, epochs = num_epochs, 
                                      patience = patience, validation = False, relative_tolerance = 1e-5)

    
    lambda_static_dict[f'DF_{gamma}'] = lpool_newsv_model.get_weights()

    # print('Weights')
    # print(lambda_static_dict[f'DF_{gamma}'])

#%% # Inverse Performance-based weights (invW)

# Find optimal in-sample decisions for each expert
trainZopt = np.zeros((nobs_train, N_experts))
testZopt = np.zeros((nobs_test, N_experts))
    
print('Finding optimal decisions in training set')
for j in range(N_experts):
    temp_z_opt = solve_opt_prob(y_supp, train_p_list[j], 'newsvendor', risk_aversion = 0, 
                                crit_quant = critical_fractile)
    trainZopt[:,j] = temp_z_opt

# Find weights for each value of gamma (inf: CRPS minimization)
for g in ([0,0.1,1] + ['inf']):
    
    lambda_tuned_inv, _ = insample_weight_tuning(Y_train, trainZopt, train_p_list, regularization_gamma=g, problem = 'newsvendor',
                                                 crit_quant = critical_fractile, support = y_supp, risk_aversion = 0)
    
    lambda_static_dict[f'invW-{g}'] = lambda_tuned_inv    
    
# Plot learned static weights
for m in list(lambda_static_dict.keys()):
    plt.plot(lambda_static_dict[m], label = m)
plt.legend()
plt.show()

#%% Evaluate results

all_models = lambda_static_dict.keys()
Prescriptions = pd.DataFrame(data = np.zeros((nobs_test, len(all_models))), columns = all_models)

test_p_list = [p1_hat[-nobs_test:], p2_hat[-nobs_test:]]
test_Q_list = [Q1_hat[-nobs_test:], Q2_hat[-nobs_test:]]

# Store pinball loss and Decision cost for task-loss
temp_QS = pd.DataFrame()
temp_QS['risk_aversion'] = 0

temp_Decision_cost = pd.DataFrame()
temp_Decision_cost['Quantile'] = [critical_fractile]
temp_Decision_cost['risk_aversion'] = 0

temp_mean_QS = temp_Decision_cost.copy()

#regularization = 0.01

print('Estimating out-of-sample performance...')

for j, m in enumerate(all_models):
    print(f'Model: {m}')
    # Combine PDFs for each observation
    temp_pdf = sum([lambda_static_dict[m][j]*test_p_list[j] for j in range(N_experts)])            

    temp_prescriptions = solve_opt_prob(y_supp, temp_pdf, 'newsvendor', risk_aversion = 0,
                                        crit_quant = critical_fractile)
       
    Prescriptions[m] = temp_prescriptions
        
    # Estimate task-loss for specific model
    #%
    temp_Decision_cost[m] = 100*task_loss(Prescriptions[m].values, Y_test, 'newsvendor', crit_quant = critical_fractile, risk_aversion = 0)
    
    # Evaluate QS (approximation of CRPS) for each model
    # find quantile forecasts
    temp_q_forecast = np.array([inverted_cdf(target_quant, y_supp, temp_pdf[i]) for i in range(nobs_test)])
    temp_qs = 100*pinball(temp_q_forecast, Y_test, target_quant).round(4)

    temp_QS[m] = [temp_qs]
    
    temp_CDF = temp_pdf.cumsum(1)
    H_i = 1*np.repeat(y_supp.reshape(1,-1), len(Y_test), axis = 0)>=Y_test.reshape(-1,1)
    
    CRPS = np.square(temp_CDF - H_i).sum(1).mean()
    temp_mean_QS[m] = CRPS

    print('Decision Cost')
    print(temp_Decision_cost[m].mean().round(4))
    
    print('CRPS')
    print(temp_mean_QS[m].mean().round(4))

print('Aggregate results')

print('Decision Cost')
print(temp_Decision_cost[all_models].mean().round(4))

print('CRPS')
print(temp_mean_QS[all_models].mean().round(4))

if config['save']:
    #Prescriptions.to_csv(f'{results_path}\\{target_problem}_{critical_fractile}_{target_zone}_Prescriptions.csv')
    lamda_static_df = pd.DataFrame.from_dict(lambda_static_dict)
    lamda_static_df.to_csv(f'{results_path}\\lambda_static_softmax.csv')
    
    temp_Decision_cost.to_csv(f'{results_path}\\synthetic_Decision_cost_softmax.csv')
    temp_mean_QS.to_csv(f'{results_path}\\synthetic_QS_mean_softmax.csv')
    
    
    
    