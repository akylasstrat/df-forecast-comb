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
from optimal_transport_functions import *
from torch_layers_functions import *
from torch.utils.data import Dataset, DataLoader
import torch

from scipy.stats import norm

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
    'Adam optimizer hyperparameters'
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
    
    params['save'] = True # If True, then saves models and results
    
    # Experimental setup parameters
    params['problem'] = 'reg_trad' # {mse, newsvendor, cvar, reg_trad, pwl}
    params['gamma_list'] = [0, 0.1, 1]
        
    params['crit_quant'] = np.arange(0.5, 1, 0.1).round(2)
    params['risk_aversion'] = [0.2]

    return params

#%%
   
config = params()
nn_hparam = nn_params()

results_path = f'{cd}\\results\\synthetic_data'
data_path = f'{cd}\\data'

#%% 

# experiment parameters
nobs_train = 5000
nobs_test = 5000
nobs = nobs_train + nobs_test
# fixed term to ensure everything is non-negative
# ** Does not affect the results, only to speed up computations with nonnegativity of parameters
bias_term = 15

y_supp = np.arange(-15, 7, 0.1).round(1) + bias_term
n_locs = len(y_supp)
target_quant = np.arange(0.01, 1, 0.01).round(2)

#alpha_1 = 1.2
#alpha_2 = 1.2
#alpha_3 = 4

alpha_1 = 1.2
alpha_2 = 1.2
alpha_3 = 4.5

beta = -1.3

X0 = np.random.normal(loc = bias_term, scale = 1, size = nobs).round(1)
X1 = np.random.normal(size = nobs, scale = 1).round(1)
X2 = np.random.normal(size = nobs, scale = 1).round(1)
X3 = np.random.normal(size = nobs).round(1)
error = np.random.normal(scale = 0.25, size = nobs).round(1)

P_1 = (alpha_1*X1 + alpha_2*X2).round(1)

Y = X0 + P_1 + (alpha_3*X3)*((X3) < beta)# + error

Y = Y.round(2)
Y = projection(Y, ub = y_supp.max(), lb = y_supp.min())

Y_train = Y[:nobs_train]
Y_test = Y[nobs_train:]

plt.hist(Y_train, bins = 50)
plt.show()

### Expert forecasts for both training and test set
# Expert 1: Access to features 1&2
p1_hat = np.zeros((nobs, n_locs))
F1_hat = np.zeros((nobs, n_locs))
Q1_hat = np.zeros((nobs, len(target_quant)))

for i in range(nobs):
    # define probabilistic forecast
    f1_hat_temp = norm(loc = X0[i] + alpha_1*X1[i] + alpha_2*X2[i], scale = 0.5)
        
    p1_hat[i] = f1_hat_temp.pdf(y_supp)*0.1
    F1_hat[i] = f1_hat_temp.cdf(y_supp)
    Q1_hat[i] = f1_hat_temp.ppf(target_quant)
    

# Expert 2: Access to feature 3/ calibrated probabilistic forecast **only** on the left tail
p2_hat = np.zeros((nobs, n_locs))
F2_hat = np.zeros((nobs, n_locs))
Q2_hat = np.zeros((nobs, len(target_quant)))

for i in range(nobs):

    if X3[i] < beta:           
        f2_hat_temp = norm(loc = X0[i] + (X3[i]*alpha_3), scale = 0.5)
    else:
        f2_hat_temp = norm(loc = X0[i], scale = 0.5)
        
    p2_hat[i] = f2_hat_temp.pdf(y_supp)*0.1
    F2_hat[i] = f2_hat_temp.cdf(y_supp)
    Q2_hat[i] = f2_hat_temp.ppf(target_quant)

# evaluate probabilistic predictions

pinball_1 = 100*pinball(Q1_hat[:nobs_train], Y_train, target_quant).round(4)
pinball_2 = 100*pinball(Q2_hat[:nobs_train], Y_train, target_quant).round(4)

plt.plot(pinball_1, label = 'Expert 1')
plt.plot(pinball_2, label = 'Expert 2')
plt.show()

print('Average Pinball Loss')
print(f'Expert 1:{pinball_1.mean()}')
print(f'Expert 2:{pinball_2.mean()}')
#%%
for i in range(20):
    plt.plot(y_supp, p1_hat[i], label = 'Expert 1')
    plt.plot(y_supp, p2_hat[i], label = 'Expert 2')
    plt.plot(Y_train[i], 0, 'o')
    plt.legend()
    plt.show()

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

train_data_loader = create_data_loader(tensor_train_p_list + [tensor_trainY], batch_size = batch_size)

from torch_layers_functions import *

#### CRPS minimization/ with torch layer
lpool_crps_model = LinearPoolCRPSLayer(num_inputs=N_experts, support = torch.FloatTensor(y_supp),
                                       apply_softmax = True)
optimizer = torch.optim.Adam(lpool_crps_model.parameters(), lr = learning_rate)
lpool_crps_model.train_model(train_data_loader, optimizer, epochs = num_epochs, patience = patience, 
                             projection = True)


lambda_crps = to_np(torch.nn.functional.softmax(lpool_crps_model.weights))

print(lambda_crps)
#lambda_crps = crps_learning_combination(Y_tail, [p1_hat, p2_hat], support = y_supp, verbose = 1)
lambda_static_dict['CRPS'] = lambda_crps
#%% Decision-focused combination 

# optimization problem parameters
target_problem = config['problem']
critical_fractile = 0.2
regularization = 0.01 # to help convergence of the gradient-descent


# Optimizer hyperparameters
batch_size = 500
learning_rate = 1e-2
num_epochs = 100
patience = 5

for gamma in [0]:
    
    lpool_newsv_model = LinearPoolNewsvendorLayer(num_inputs=N_experts, support = torch.FloatTensor(y_supp),
                                                gamma = gamma, problem = target_problem, critic_fract = critical_fractile, apply_softmax = True, 
                                                risk_aversion = regularization)
    
    optimizer = torch.optim.Adam(lpool_newsv_model.parameters(), lr = learning_rate)
    
    lpool_newsv_model.train_model(train_data_loader, [], optimizer, epochs = num_epochs, patience = patience, projection = False, validation = False, relative_tolerance = 0)

    lambda_static_dict[f'DF_{gamma}'] = to_np(torch.nn.functional.softmax(lpool_newsv_model.weights))

#%%

trainZopt = np.zeros((nobs_train, N_experts))
testZopt = np.zeros((nobs_test, N_experts))
    
print('Finding optimal decisions in training set')
for j in range(N_experts):
    temp_z_opt = solve_opt_prob(y_supp, train_p_list[j], target_problem, risk_aversion = regularization, 
                                crit_quant = critical_fractile)
    trainZopt[:,j] = temp_z_opt
    
# Set weights to in-sample performance
lambda_tuned_inv, _ = insample_weight_tuning(Y_train, trainZopt, problem = target_problem,
                                             crit_quant = critical_fractile, support = y_supp, risk_aversion = regularization)
lambda_static_dict['Insample'] = lambda_tuned_inv    


# Benchmark/ Salva's suggestion/ weighted combination of in-sample optimal (stochastic) decisions
#lambda_ = averaging_decisions(train_targetY, trainZopt, target_problem, crit_fract = critical_fractile, support = y_supp, bounds = False, risk_aversion = risk_aversion)
#lambda_static_dict['SalvaBench'] = lambda_


#%%
for m in list(lambda_static_dict.keys()):
    plt.plot(lambda_static_dict[m], label = m)
plt.legend()
plt.show()

#lambda_static_dict[f'DF_{gamma}'] = np.array([0.39, 0.57])
#lambda_static_dict[f'DF_{gamma}'] = np.array([0.4648, 0.5352])
#lambda_static_dict['Ave'] = np.array([0.5, 0.5])
#%% Evaluate results
#regularization = 0.01
all_models = lambda_static_dict.keys()
Prescriptions = pd.DataFrame(data = np.zeros((nobs_test, len(all_models))), columns = all_models)

test_p_list = [p1_hat[-nobs_test:], p2_hat[-nobs_test:]]
test_Q_list = [Q1_hat[-nobs_test:], Q2_hat[-nobs_test:]]

# Store pinball loss and Decision cost for task-loss
temp_QS = pd.DataFrame()
temp_QS['risk_aversion'] = regularization

temp_Decision_cost = pd.DataFrame()
temp_Decision_cost['Quantile'] = [critical_fractile]
temp_Decision_cost['risk_aversion'] = regularization

temp_mean_QS = temp_Decision_cost.copy()

#regularization = 0.01

print('Estimating out-of-sample performance...')

for j, m in enumerate(all_models):
    print(f'Model: {m}')
    # Combine PDFs for each observation
    temp_pdf = sum([lambda_static_dict[m][j]*test_p_list[j] for j in range(N_experts)])            

    temp_prescriptions = solve_opt_prob(y_supp, temp_pdf, target_problem, risk_aversion = regularization,
                                        crit_quant = critical_fractile)
       
    Prescriptions[m] = temp_prescriptions
        
    # Estimate task-loss for specific model
    #%
    temp_Decision_cost[m] = 100*task_loss(Prescriptions[m].values, Y_test, 
                                      target_problem, crit_quant = critical_fractile, risk_aversion = regularization)
    #%
    
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
print(temp_Decision_cost[all_models].mean().round(4))

print('CRPS')
print(temp_mean_QS[all_models].mean().round(4))

if config['save']:
    #Prescriptions.to_csv(f'{results_path}\\{target_problem}_{critical_fractile}_{target_zone}_Prescriptions.csv')
    lamda_static_df = pd.DataFrame.from_dict(lambda_static_dict)
    lamda_static_df.to_csv(f'{results_path}\\lambda_static.csv')
    
    temp_Decision_cost.to_csv(f'{results_path}\\synthetic_Decision_cost.csv')
    temp_mean_QS.to_csv(f'{results_path}\\synthetic_QS_mean.csv')
    
    
    
    