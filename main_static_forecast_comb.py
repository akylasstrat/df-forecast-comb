# -*- coding: utf-8 -*-
"""
Decision-Focused Forecast Combination/ main script/ static combination only

@author: a.stratigakos
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

def averaging_decisions(target_y, prob_vectors, problem, 
                        crit_fract = 0.5, support = np.arange(0, 1.01, .01).round(2), bounds = False, verbose = 0, 
                        **kwargs):            
    ''' (Salva's benchmark) Solve the stochastic problem in-sample for each observations and each expert, combine the decisions'''

    n_obs = prob_vectors[0].shape[0]
    n_models = len(prob_vectors)
    risk_aversion = kwargs['risk_aversion']

    ### Find optimal decisions under perfect foresight information
    
    print('Solve in-sample stchastic problems...')
    
    z_opt = np.zeros((n_obs, n_models))
    insample_cost = np.zeros(n_models)
    insample_inverse_cost = np.zeros(n_models)
    
    for j in range(n_models):
        if problem == 'newsvendor':
            for i in range(n_obs):
                # Solve stochastic problem, find decision
                z_opt[i,j] = inverted_cdf([crit_fract], support, prob_vectors[j][i])
            # Estimate decision cost (regret)
        elif problem == 'reg_trad':
            temp_z_opt = solve_opt_prob(support, prob_vectors[j], problem, risk_aversion = risk_aversion, 
                                        crit_quant = crit_fract)
            z_opt[:,j] = temp_z_opt
        
        #insample_cost[j] = newsvendor_loss(z_opt[:,j], target_y.reshape(-1), q = crit_fract)
        insample_cost[j] = task_loss(z_opt[:,j], target_y.reshape(-1), problem, crit_quant = crit_fract, 
                                     risk_aversion = risk_aversion)
        insample_inverse_cost[j] = 1/insample_cost[j]
    
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
    
    m.addConstr( z_comb == sum([z_opt[:,j]*lambdas[j] for j in range(n_models)]) )
    
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

def insample_weight_tuning(target_y, prob_vectors, problem = 'newsvendor', 
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
    
    n_obs = prob_vectors[0].shape[0]
    n_models = len(prob_vectors)
    
    risk_aversion = kwargs['risk_aversion']
    crit_quant = kwargs['crit_quant']
    ### Find optimal decisions under perfect foresight information
    
    print('Solve in-sample stochastic problems...')
    
    z_opt = np.zeros((n_obs, n_models))
    insample_cost = np.zeros((n_models))
    insample_inverse_cost = np.zeros((n_models))

    for j in range(n_models):
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
        
        #insample_cost[j] = newsvendor_loss(z_opt[:,j], target_y.reshape(-1), q = crit_fract)
        insample_cost[j] = task_loss(z_opt[:,j], target_y.reshape(-1), problem, risk_aversion = risk_aversion, 
                                     crit_quant = crit_quant)
        insample_inverse_cost[j] = 1/insample_cost[j]

    lambdas_inv = insample_inverse_cost/insample_inverse_cost.sum()
    lambdas_softmax = np.exp(insample_inverse_cost)/sum(np.exp(insample_inverse_cost))

    return lambdas_inv, lambdas_softmax

# Define a custom dataset
class MyDataset(Dataset):
    def __init__(self, *inputs):
        self.inputs = inputs

        # Check that all input tensors have the same length (number of samples)
        self.length = len(inputs[0])
        if not all(len(input_tensor) == self.length for input_tensor in inputs):
            raise ValueError("Input tensors must have the same length.")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return tuple(input_tensor[idx] for input_tensor in self.inputs)

# Define a custom data loader
def create_data_loader(inputs, batch_size, num_workers=0, shuffle=True):
    dataset = MyDataset(*inputs)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle
    )
    return data_loader

class AdaptiveCombNewsvendorLayer(nn.Module):        
    def __init__(self, input_size, hidden_sizes, output_size, 
                 support, gamma, activation=nn.ReLU(), apply_softmax = False):
        super(AdaptiveCombNewsvendorLayer, self).__init__()
        """
        Adaptive forecast combination for newsvendor problem> predicts combination weights given features
        Args:
            input_size, hidden_sizes, output_size: standard arguments for declaring an MLP
            
            output_size: equal to the number of combination weights, i.e., number of experts we want to combine
            
        """
        # Initialize learnable weight parameters
        #self.weights = nn.Parameter(torch.rand(num_inputs), requires_grad=True)
        self.weights = nn.Parameter(torch.FloatTensor((1/output_size)*np.ones(output_size)).requires_grad_())
        self.num_experts = output_size
        self.support = support
        self.gamma = gamma
        self.apply_softmax = apply_softmax
        
        # create sequential MLP model to predict combination weights
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(activation)
                
        self.model = nn.Sequential(*layers)
        if apply_softmax:
            self.model.add_module('softmax', nn.Softmax())
        
        # newsvendor layer
        z = cp.Variable((1))    
        pinball_loss = cp.Variable(len(support))
        error = cp.Variable(len(support))
        w_error = cp.Variable(len(support))
        prob_weights = cp.Parameter(len(support))
        
        newsv_constraints = [z >= 0, z <= 1, error == y_supp - z,
                             pinball_loss >= critical_fractile*(error), 
                             pinball_loss >= (critical_fractile - 1)*(error), 
                             w_error == cp.multiply(prob_weights, error)]

        newsv_objective = cp.Minimize( prob_weights@pinball_loss + self.gamma*cp.norm(w_error)) 
        newsv_problem = cp.Problem(newsv_objective, newsv_constraints)
        self.newsvendor_layer = CvxpyLayer(newsv_problem, parameters=[prob_weights], variables = [z, pinball_loss, error, w_error] )
        
    def forward(self, x, list_inputs):
        """
        Forward pass of the newvendor layer.

        Args:
            x: input tensors/ features
            list_inputs: A list of of input tensors/ probability vectors.

        Returns:
            torch.Tensor: The convex combination of input tensors.
        """

        # Forwatd pass of the MLP to predict the combination weights (use softmax activation)
        weights = self.model(x)

        # Apply the weights element-wise to each input tensor
        weighted_inputs = [weights[k,i] * input_tensor for k in range(weights.shape[0]) for i, input_tensor in enumerate(list_inputs)]

        #print(sum(weighted_inputs).shape)
        
        # Perform the convex combination across input vectors
        combined_vector = sum(weighted_inputs)

        # Pass the combined output to the CVXPY layer
        cvxpy_output = self.newsvendor_layer(combined_vector)
        return cvxpy_output
    
    def train_model(self, train_loader, optimizer, epochs = 20, patience=5, projection = True):
        # define projection problem for backward pass
        lambda_proj = cp.Variable(self.num_experts)
        lambda_hat = cp.Parameter(self.num_experts)
        proj_problem = cp.Problem(cp.Minimize(0.5*cp.sum_squares(lambda_proj-lambda_hat)), [lambda_proj >= 0, lambda_proj.sum()==1])
        
        
        L_t = []
        best_train_loss = float('inf')

        for epoch in range(epochs):
            # activate train functionality
            self.train()
            running_loss = 0.0
            # sample batch
            for batch_data in train_loader:
                
                y_batch = batch_data[-1]
                x_batch = batch_data[-2]
                p_list_batch = batch_data[0:-2]
                
                # clear gradients
                optimizer.zero_grad()
                
                # forward pass: combine forecasts and solve each newsvendor problem
                z_hat = self.forward(x_batch, p_list_batch)[0]
                
                error_hat = (y_batch.reshape(-1,1) - z_hat)
                # estimate regret
                loss = (critical_fractile*error_hat[error_hat>0].norm(p=1) \
                        + (1-critical_fractile)*error_hat[error_hat<0].norm(p=1))\
                        + self.gamma*error_hat.norm()

                # backward pass
                loss.backward()
                optimizer.step()
                
                
                if (Projection)and(self.apply_softmax != True):     
                    lambda_hat.value = to_np(self.weights)
                    proj_problem.solve(solver = 'GUROBI')
                    # update parameter values
                    with torch.no_grad():
                        self.weights.copy_(torch.FloatTensor(lambda_proj.value))
                
                running_loss += loss.item()
            

            L_t.append(to_np(self.weights).copy())
            
            # plot the coefficients for sanity check
            if epoch % 15 ==0:
                plt.plot(L_t)
                plt.show()
                
            average_train_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {average_train_loss:.4f} ")

            if average_train_loss < best_train_loss:
                best_train_loss = average_train_loss
                best_weights = copy.deepcopy(self.state_dict())
                early_stopping_counter = 0
                
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    print("Early stopping triggered.")
                    # recover best weights
                    self.load_state_dict(best_weights)
                    return
                
    def predict_weights(self, x):
        'Forecast combination weights, inference only'
        with torch.no_grad():            
            return self.model(x).detach().numpy()
        
class LinearPoolNewsvendorLayer(nn.Module):        
    def __init__(self, num_inputs, support, 
                 gamma, apply_softmax = False, critic_fract = 0.5, regularizer = 'crps', risk_aversion = 0):
        super(LinearPoolNewsvendorLayer, self).__init__()

        # Initialize learnable weight parameters
        #self.weights = nn.Parameter(torch.rand(num_inputs), requires_grad=True)
        self.weights = nn.Parameter(torch.FloatTensor((1/num_inputs)*np.ones(num_inputs)).requires_grad_())
        self.num_inputs = num_inputs
        self.support = support
        self.risk_aversion = risk_aversion
        self.gamma = gamma
        self.apply_softmax = apply_softmax
        self.regularizer = regularizer
        self.crit_fract = critic_fract
        
        n_locations = len(self.support)
        # newsvendor layer (for i-th observation)
        z = cp.Variable((1))    
        pinball_loss = cp.Variable(n_locations)
        error = cp.Variable(n_locations)
        prob_weights = cp.Parameter(n_locations)
            
        newsv_constraints = [z >= 0, z <= 1, error == self.support - z,
                             pinball_loss >= self.crit_fract*(error), 
                             pinball_loss >= (self.crit_fract - 1)*(error),]
        
        newsv_cost = (1-self.risk_aversion)*prob_weights@pinball_loss
        
        # define aux variable
        w_error = cp.multiply(prob_weights, error)
        l2_regularization = self.risk_aversion*cp.norm(w_error)

        objective_funct = cp.Minimize( newsv_cost + l2_regularization ) 
        
        newsv_problem = cp.Problem(objective_funct, newsv_constraints)
        self.newsvendor_layer = CvxpyLayer(newsv_problem, parameters=[prob_weights],
                                           variables = [z, pinball_loss, error] )
        
    def forward(self, list_inputs):
        """
        Forward pass of the newvendor layer.

        Args:
            list_inputs: A list of of input tensors/ PDFs.

        Returns:
            torch.Tensor: The convex combination of input tensors/ combination of PDFs.
        """
        # Ensure that the weights are in the range [0, 1] using softmax activation
        if self.apply_softmax:
            weights = torch.nn.functional.softmax(self.weights, dim = 0)
        else:
            weights = self.weights

        # Apply the weights element-wise to each input tensor
        weighted_inputs = [weights[i] * input_tensor for i, input_tensor in enumerate(list_inputs)]

        # Perform the convex combination across input vectors
        combined_pdf = sum(weighted_inputs)

        # Pass the combined output to the CVXPY layer
        cvxpy_output = self.newsvendor_layer(combined_pdf)
        return combined_pdf, cvxpy_output
    
    def train_model(self, train_loader, optimizer, epochs = 20, patience=5, projection = True):
        # define projection problem for backward pass

        if (projection)and(self.apply_softmax != True):     
            lambda_proj = cp.Variable(self.num_inputs)
            lambda_hat = cp.Parameter(self.num_inputs)
            proj_problem = cp.Problem(cp.Minimize(0.5*cp.sum_squares(lambda_proj-lambda_hat)), [lambda_proj >= 0, lambda_proj.sum()==1])
        
        
        L_t = []
        best_train_loss = float('inf')
        early_stopping_counter = 0
        best_weights = copy.deepcopy(self.state_dict())

        for epoch in range(epochs):
            # activate train functionality
            self.train()
            running_loss = 0.0
            # sample batch data
            for batch_data in train_loader:
                
                y_batch = batch_data[-1]
                
                # clear gradients
                optimizer.zero_grad()
                
                # forward pass: combine forecasts and solve each newsvendor problem
                output_hat = self.forward(batch_data[:-1])

                pdf_comb_hat = output_hat[0]
                cdf_comb_hat = pdf_comb_hat.cumsum(1)
                
                z_hat = output_hat[1][0]
                
                # estimate aggregate pinball loss and CRPS (for realization of uncertainty)
                error_hat = (y_batch.reshape(-1,1) - z_hat)

                crps_i = sum([torch.square( cdf_comb_hat[i] - 1*(self.support >= y_batch[i]) ).sum() for i in range(len(y_batch))])
                pinball_loss = (self.crit_fract*error_hat[error_hat>0].norm(p=1) + (1-self.crit_fract)*error_hat[error_hat<0].norm(p=1))
                l2_loss = error_hat.norm(p=2)
                
                # Total regret (scale CRPS for better trade-off control)
                loss = (1-self.risk_aversion)*pinball_loss + self.risk_aversion*l2_loss \
                    + self.gamma*crps_i/len(self.support)
                
                # estimate regret
                #loss = (critical_fractile*error_hat[error_hat>0].norm(p=1) \
                #        + (1-critical_fractile)*error_hat[error_hat<0].norm(p=1))\
                #        + self.gamma*()

                # backward pass
                loss.backward()
                optimizer.step()
                
                # Apply projection
                if (projection)and(self.apply_softmax != True):     
                    lambda_hat.value = to_np(self.weights)
                    proj_problem.solve(solver = 'GUROBI')
                    # update parameter values
                    with torch.no_grad():
                        self.weights.copy_(torch.FloatTensor(lambda_proj.value))
                
                running_loss += loss.item()
            

            L_t.append(to_np(self.weights).copy())
            
            # plot the coefficients for sanity check
            #if epoch % 15 ==0:
            #    plt.plot(L_t)
            #    plt.show()
                
            average_train_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {average_train_loss:.4f} ")

            if average_train_loss < best_train_loss:
                best_train_loss = average_train_loss
                best_weights = copy.deepcopy(self.state_dict())
                early_stopping_counter = 0
                
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    print("Early stopping triggered.")
                    # recover best weights
                    self.load_state_dict(best_weights)
                    return
                          
class LinearPoolCRPSLayer(nn.Module):        
    def __init__(self, num_inputs, support, apply_softmax = False):
        super(LinearPoolCRPSLayer, self).__init__()

        # Initialize learnable weight parameters
        #self.weights = nn.Parameter(torch.rand(num_inputs), requires_grad=True)
        self.weights = nn.Parameter(torch.FloatTensor((1/num_inputs)*np.ones(num_inputs)).requires_grad_())
        self.num_inputs = num_inputs
        self.support = support
        self.apply_softmax = apply_softmax
        
    def forward(self, list_inputs):
        """
        Forward pass of the linear pool minimizing CRPS.

        Args:
            list_inputs: A list of of input tensors (discrete PDFs).

        Returns:
            torch.Tensor: The convex combination of input tensors.
        """
        # Ensure that the weights are in the range [0, 1] using sigmoid activation
        #weights = torch.nn.functional.softmax(self.weights)

        # Ensure that the weights are in the range [0, 1] using sigmoid activation
        if self.apply_softmax:
            weights = torch.nn.functional.softmax(self.weights, dim = 0)
        else:
            weights = self.weights
        
        # Apply the weights element-wise to each input tensor !!!! CDFs
        weighted_inputs = [weights[i] * input_tensor.cumsum(1) for i, input_tensor in enumerate(list_inputs)]

        # Perform the convex combination across input vectors
        combined_CDF = sum(weighted_inputs)

        return combined_CDF
    
    def train_model(self, train_loader, optimizer, epochs = 20, patience=5, projection = True):
        # define projection problem for backward pass
        lambda_proj = cp.Variable(self.num_inputs)
        lambda_hat = cp.Parameter(self.num_inputs)
        proj_problem = cp.Problem(cp.Minimize(0.5*cp.sum_squares(lambda_proj-lambda_hat)), [lambda_proj >= 0, lambda_proj.sum()==1])
        
        
        L_t = []
        best_train_loss = float('inf')

        for epoch in range(epochs):
            # activate train functionality
            self.train()
            running_loss = 0.0
            # sample batch
            for batch_data in train_loader:
                
                y_batch = batch_data[-1]
                
                #cdf_batch = [batch_data[i] for i in range(self.num_inputs)]

                # clear gradients
                optimizer.zero_grad()
                
                # forward pass: combine forecasts and solve each newsvendor problem
                comb_CDF = self.forward(batch_data[:-1])
                
                # estimate CRPS (heavyside function)
                loss_i = [torch.square( comb_CDF[i] - 1*(self.support >= y_batch[i]) ).sum() for i in range(len(y_batch))]
                loss = sum(loss_i)/len(loss_i)

                # Decomposition (see Online learning with the Continuous Ranked Probability Score for ensemble forecasting) 
                #divergence_i = [(weights[j]*torch.norm(self.support - y_batch[i] )) for ]
                
                #loss_i = [weights[j]*torch.abs(self.support >= y_batch[i]) ).sum() for i in range(len(y_batch))]
                #loss = sum(loss_i)/len(loss_i)
                
                # backward pass
                loss.backward()
                optimizer.step()                
                
                if (projection)and(self.apply_softmax != True):     
                    lambda_hat.value = to_np(self.weights)
                    proj_problem.solve(solver = 'GUROBI')
                    # update parameter values
                    with torch.no_grad():
                        self.weights.copy_(torch.FloatTensor(lambda_proj.value))
                
                running_loss += loss.item()
            

            L_t.append(to_np(self.weights).copy())
            average_train_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {average_train_loss:.4f} ")

            if average_train_loss < best_train_loss:
                best_train_loss = average_train_loss
                best_weights = copy.deepcopy(self.state_dict())
                early_stopping_counter = 0
                
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    print("Early stopping triggered.")
                    # recover best weights
                    self.load_state_dict(best_weights)
                    return

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
                Prescriptions[row] = offer.X
                
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


def params():
    ''' Set up the experiment parameters'''

    params = {}

    params['start_date'] = '2012-01-01'
    params['split_date'] = '2013-01-01' # Defines train/test split
    params['end_date'] = '2013-12-30'
    
    params['save'] = True # If True, then saves models and results
    params['train_brc'] = False # If True, then saves models and results
    
    # Experimental setup parameters
    params['problem'] = 'reg_trad' # {mse, newsvendor, cvar, reg_trad}
    params['N_experts'] = 9
    params['iterations'] = 5
    params['target_zones'] = ['Z1', 'Z2', 'Z3', 'Z4', 'Z5',
                              'Z6', 'Z7', 'Z8', 'Z9', 'Z10']
    
    
    params['crit_quant'] = [0.2]
    params['risk_aversion'] = [0.5]
    
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
train_forecast_model = True
initial_training = True

Decision_cost = pd.read_csv(f'{cd}\\results\\{target_problem}_static_linearpool_Decision_cost.csv', index_col = 0)
QS_df = pd.read_csv(f'{cd}\\results\\{target_problem}_static_linear_pool_QS.csv', index_col = 0)
row_counter = len(Decision_cost)

if target_problem == 'newsvendor':
    config['risk_aversion'] = [0]
    tuple_list = [tup for tup in itertools.product(config['target_zones'], config['crit_quant'])]
elif target_problem == 'reg_trad':
    tuple_list = [tup for tup in itertools.product(config['target_zones'], config['crit_quant'], 
                                                   config['risk_aversion'])]

#%%
for tup in tuple_list[row_counter:]:

    target_zone = tup[0]    
    critical_fractile = tup[1]
    risk_aversion = tup[2]
    
    if row_counter == 0:
        train_forecast_model = True
    elif (row_counter != 0) and (target_zone == tuple_list[row_counter-1][0]):    
        train_forecast_model = False
    elif (row_counter != 0) and (target_zone != tuple_list[row_counter-1][0]):
        train_forecast_model = True
        
    if initial_training:
        train_forecast_model = True
        initial_training = False
        
    all_zones = [f'Z{i}' for i in range(1,11)]
    np.random.seed(row_counter)
    
    print(f'Quantile:{critical_fractile}, zone:{target_zone}')
    #target_zone = config['target_zone']
    #expert_zones = ['Z2', 'Z4', 'Z8', 'Z9']
    
    #target_zone = np.random.choice(all_zones)
    # number of forecasts to combine
    N_experts = config['N_experts']
    
    # number of observations to train prob. forecasting model
    N_sample = len(aggr_wind_df)//4
    
    step = .01
    y_supp = np.arange(0, 1+step, step).round(2)
    nlocations = len(y_supp)
    pred_col = ['wspeed10', 'wdir10_rad', 'wspeed100', 'wdir100_rad']
    
    if train_forecast_model:
        
        expert_zones = all_zones.copy()
        expert_zones.remove(target_zone)
        expert_zones = expert_zones[:config['N_experts']]
        #expert_zones = list(np.random.choice(expert_zones, config['N_experts'], replace = False))
        #expert_zones = list(np.random.choice(expert_zones, config['N_experts'], replace = False))
    
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
    
        #% Train experts, i.e., probabilistic forecasting models in adjacent locations    
        prob_models = []
        for i, zone in enumerate(expert_zones[:N_experts]):
            print(f'Training model {i}')
            
            #temp_model = EnsemblePrescriptiveTree_OOB(n_estimators = 10, max_features = 1, type_split = 'quant')
            #temp_model.fit(trainX_exp[zone][pred_col].values.round(2), trainY[zone].values, y_supp, y_supp, bootstrap = False, quant = np.arange(.01, 1, .01), problem = 'mse') 
         
            temp_model = EnsemblePrescriptiveTree(n_estimators = 30, max_features = len(pred_col), type_split = 'quant' )
            temp_model.fit(trainX_exp[zone][pred_col].values, trainY[zone].values, quant = np.arange(.01, 1, .01), problem = 'mse') 
            prob_models.append(temp_model)
        
        #% Generate predictions for training/test set for forecast combination
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
            
        #% Visualize some prob. forecasts
        #%
        # step 1: find inverted CDFs
        F_inv = [np.array([inverted_cdf([.05, .10, .90, .95] , trainY[zone].values, train_w_list[j][i]) for i in range(500)]) for j in range(N_experts)]
        
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
    
    if target_problem == 'mse':
        
        #### Solve to optimality with a single level problem
        # 1-1 mapping: decisions are the weighted sum
        
        # set up the SAA of the target optimization problem
        m = gp.Model()
        #m.setParam('OutputFlag', 1)
        # Decision variables
        
        lambda_coord = m.addMVar(N_experts, vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1)
        p_comb = m.addMVar((n_obs, nlocations), vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1)
        z = m.addMVar(n_obs, vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1)
    
        m.addConstr( lambda_coord.sum() == 1)
        # constraints on probability
        m.addConstr( p_comb.sum(1) == 1)
        m.addConstr( p_comb == sum([lambda_coord[i]*train_p_list[i] for i in range(N_experts)]))
        
        # mapping probability vectors to decisions (replace with trained ML predictor for more complex problems)
        m.addConstr( z == p_comb@y_supp)
        
        # Task-loss function
        m.setObjective( (train_targetY-z)@(train_targetY-z), gp.GRB.MINIMIZE)
        m.optimize()
    
        #### Solve with Diff. Opt. Layer (should converge to the same solution as above)
        k = len(y_supp)
        print(np.linalg.norm((train_targetY-z.X)))
        
        #%
        ################## Pytorch example    
        patience = 25
        batch_size = 100
        num_epochs = 100
        
        train_loader = create_data_loader(tensor_train_p_list + [tensor_trainY], batch_size = batch_size)
        #convex_layer = ConvexCombinationLayer(num_inputs=N_experts, support = torch.FloatTensor(y_supp))
        
        
        # Forward pass through the layer with 3 inputs
        optimizer = torch.optim.SGD(convex_layer.parameters(), lr=1e-3)
        Projection = True
        L_t = []
        for epoch in range(num_epochs):
            # activate train functionality
            convex_layer.train()
            running_loss = 0.0
            # sample batch
            for batch_data in train_loader:
                
                y_batch = batch_data[-1]
                
                # clear gradients
                optimizer.zero_grad()
                # forward pass
                z_hat = convex_layer(batch_data[0], batch_data[1], batch_data[2])
                
                # loss evaluation
                loss = (z_hat - y_batch).norm()
                
                # backward pass
                loss.backward()
                optimizer.step()
                
            # Projection step (!!!! enforce no gradient update)
            #with torch.no_grad():
            #    convex_layer.weights.copy_ = nn.Softmax(dim=-1)(convex_layer.weights)
                
            if Projection:     
                y_hat.value = to_np(convex_layer.weights)
                proj_problem.solve(solver = 'GUROBI')
                # update parameter values
                with torch.no_grad():
                    convex_layer.weights.copy_(torch.FloatTensor(y_proj.value))
                        
                running_loss += loss.item()
            
    
            L_t.append(to_np(convex_layer.weights).copy())
            
            if epoch % 15 ==0:
                plt.plot(L_t)
                plt.show()
            average_train_loss = running_loss / len(train_loader)
            
            with torch.no_grad():            
                y_prediction = convex_layer(tensor_train_p_list[0], tensor_train_p_list[1], tensor_train_p_list[2])
    
            
            print( ( tensor_trainY - y_prediction ).norm() )
            
            print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {average_train_loss:.4f} ")
        
        #%
        z = cp.Variable(1)
        lambda_ = cp.Parameter(N_experts)
        x_i = cp.Parameter((1, tensor_train_p_list.shape[1]))
        x_i_aux = cp.Variable((1, tensor_train_p_list.shape[1]))
        
        constraints = [z >= 0, z <=1, z == sum([lambda_[i]*x_i_aux[:,i*k:(i+1)*k] for i in range(N_experts)])@y_supp, 
                       x_i == x_i_aux]
        
        # analytical solution
        objective = cp.Minimize(0)
        problem = cp.Problem(objective, constraints)
        
        Comb_Layer = CvxpyLayer(problem, parameters=[lambda_, x_i], variables=[z, x_i_aux])
        
        l_hat = nn.Parameter(torch.FloatTensor((1/N_experts)*np.ones(N_experts)).requires_grad_())
    
        optimizer = torch.optim.Adam([l_hat], lr=1e-3)
    
        for epoch in range(num_epochs):
            # activate train functionality
            Comb_Layer.train()
            running_loss = 0.0
            # sample batch
            for inputs, labels in train_loader:
                
                # clear gradients
                optimizer.zero_grad()
                # forward pass
                outputs = Comb_Layer(l_hat, inputs)
                z_hat = outputs[0]
                
                # loss evaluation
                loss = (labels-z_hat).norm()
    
                # backward pass
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
    
    
            
        #%
        ################## GD for differentiable layer
        # As currently written, does full batch gradient updates/ probably eats a lot of memory
        
        z = cp.Variable(n_obs)
        lambda_ = cp.Parameter(N_experts)
        constraints = [z >= 0, z <=1, z == sum([lambda_[i]*train_p_list[i] for i in range(N_experts)])@y_supp]
        
        objective = cp.Minimize(cp.norm(train_targetY - z, p=2))
        problem = cp.Problem(objective, constraints)
        
        layer = CvxpyLayer(problem, parameters=[lambda_], variables=[z,])
        l_hat = nn.Parameter(torch.FloatTensor((1/N_experts)*np.ones(N_experts)).requires_grad_())
        opt = torch.optim.SGD([l_hat], lr=1e-3)
        losses = []
        
        L_t = [to_np(l_hat)]
        Projection = False
        
        layer.train()
        for i in range(num_epochs):
                
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
    
        # Derive test predictions, evaluate models
    
        y_hat_local = [lambda_coord[i].X*test_p_list[i]@y_supp for i in range(N_experts)]
        y_hat_comb = sum([lambda_coord[i].X*test_p_list[i] for i in range(N_experts)])@y_supp
        
        for i in range(N_experts):
            print(f'Model {i}:{eval_point_pred(y_hat_local[i], testY[target_zone])[0]}')
        print(f'ConvComb:{eval_point_pred(y_hat_comb, testY[target_zone])[0]}')
        print(f'Ave:{eval_point_pred(sum(y_hat_local)/N_experts, testY[target_zone])[0]}')

#% Newsvendor experiment
#%%
    if target_problem in ['newsvendor', 'reg_trad']:
                
        ###########% Static forecast combinations
        
        lambda_cc_dict = {}
        
        for i in range(N_experts):
            temp_ind = np.zeros(N_experts)
            temp_ind[i] = 1
            lambda_cc_dict[f'Model-{i}'] = temp_ind
            
        lambda_cc_dict['Ave'] = (1/N_experts)*np.ones(N_experts)
        
        # Set weights to in-sample performance
        lambda_tuned_inv, _ = insample_weight_tuning(train_targetY, train_p_list, problem = target_problem,
                                                     crit_quant = critical_fractile, 
                                                     support = y_supp, risk_aversion = risk_aversion)
        #%%
        # Benchmark/ Salva's suggestion/ weighted combination of in-sample optimal (stochastic) decisions
        lambda_ = averaging_decisions(train_targetY, train_p_list, target_problem, crit_fract = critical_fractile,
                                      support = y_supp, bounds = False, risk_aversion = risk_aversion)

        lambda_cc_dict['Insample'] = lambda_tuned_inv    
        lambda_cc_dict['SalvaBench'] = lambda_
        #%%        
        #% PyTorch layers
                
        patience = 15
        batch_size = 2000
        num_epochs = 1000
        learning_rate = 5e-2
        apply_softmax = True
        
        train_loader = create_data_loader(tensor_train_p_list + [tensor_trainY], batch_size = batch_size)
                
        #### CRPS minimization/ with torch layer
        lpool_crps_model = LinearPoolCRPSLayer(num_inputs=N_experts, support = torch.FloatTensor(y_supp),
                                               apply_softmax = True)
        optimizer = torch.optim.Adam(lpool_crps_model.parameters(), lr = learning_rate)
        lpool_crps_model.train_model(train_loader, optimizer, epochs = batch_size, patience = patience, 
                                     projection = True)
        if apply_softmax:
            lambda_cc_dict['CRPS'] = to_np(torch.nn.functional.softmax(lpool_crps_model.weights))
        else:
            lambda_cc_dict['CRPS'] = to_np(lpool_crps_model.weights)

        #%%
        ##### Decision-focused combination for different values of gamma
        
        for gamma in [0, 0.1, 1, 10]:
            
            lpool_newsv_model = LinearPoolNewsvendorLayer(num_inputs=N_experts, support = torch.FloatTensor(y_supp), 
                                                        gamma = gamma, critic_fract = critical_fractile, risk_aversion = risk_aversion,
                                                        apply_softmax = True, regularizer=None)
            
            optimizer = torch.optim.Adam(lpool_newsv_model.parameters(), lr = learning_rate)
            
            lpool_newsv_model.train_model(train_loader, optimizer, epochs = num_epochs, 
                                              patience = patience, projection = False)
            if apply_softmax:
                lambda_cc_dict[f'DF_{gamma}'] = to_np(torch.nn.functional.softmax(lpool_newsv_model.weights))
            else:
                lambda_cc_dict[f'DF_{gamma}'] = to_np(lpool_newsv_model.weights)
                
        print(lambda_cc_dict)
        #%%
        for m in list(lambda_cc_dict.keys())[N_experts:]:
            plt.plot(lambda_cc_dict[m], label = m)
        plt.legend()
        plt.show()
        #%%
        ### Adaptive combination model
        '''
        train_adaptive_loader = create_data_loader(tensor_train_p_list + [tensor_trainX, tensor_trainY], batch_size = batch_size)
        
        lr_adapt_Newsv_Comb_Model = AdaptiveCombNewsvendorLayer(input_size = tensor_trainX.shape[1], hidden_sizes = [], output_size = N_experts, 
                                                                support = torch.FloatTensor(y_supp), gamma = 0, apply_softmax = True)
        
        optimizer = torch.optim.SGD(lr_adapt_Newsv_Comb_Model.parameters(), lr = learning_rate)
        lr_adapt_Newsv_Comb_Model.train_model(train_adaptive_loader, optimizer, epochs = 200, patience = patience, projection = False)
        
        
        mlp_adapt_Newsv_Comb_Model = AdaptiveCombNewsvendorLayer(input_size = tensor_trainX.shape[1], hidden_sizes = [20, 20], output_size = N_experts, 
                                                                support = torch.FloatTensor(y_supp), gamma = gamma, apply_softmax = True)
        
        optimizer = torch.optim.SGD(mlp_adapt_Newsv_Comb_Model.parameters(), lr = learning_rate)
        
        mlp_adapt_Newsv_Comb_Model.train_model(train_adaptive_loader, optimizer, epochs = 200, patience = patience, projection = False)
        '''
        ##### Static Forecast Combinations
        
        # turn probability vectors to decisions/ closed-form solution for newsvendor problem    
        models = list(lambda_cc_dict.keys())
        
        Prescriptions = pd.DataFrame(data = np.zeros((n_test_obs, len(models))), columns = models)

        for j, m in enumerate(models):
            
            temp_pdf = sum([lambda_cc_dict[m][j]*test_p_list[j] for j in range(N_experts)])            
            if target_problem == 'newsvendor':
                Prescriptions[m] = np.array([inverted_cdf([critical_fractile], y_supp, temp_pdf[i]) for i in range(n_test_obs)]).reshape(-1)
            elif target_problem == 'reg_trad':
                
                temp_prescriptions = solve_opt_prob(y_supp, temp_pdf, target_problem, risk_aversion = risk_aversion, 
                                                    crit_quant = critical_fractile)
                
                Prescriptions[m] = temp_prescriptions
                
            #!!!!!! Add estimation of quantile loss or CRPS over the whole grid
        #%%
        temp_QS = pd.DataFrame()
        temp_QS['Target'] = [target_zone]
        temp_QS['risk_aversion'] = risk_aversion

        temp_Decision_cost = pd.DataFrame()
        temp_Decision_cost['Quantile'] = [critical_fractile]
        temp_Decision_cost['risk_aversion'] = risk_aversion
        temp_Decision_cost['Target'] = target_zone

        for m in models:
            print(f'{m}:{100*newsvendor_loss(Prescriptions[m].values, testY[target_zone], q = critical_fractile).round(4)}')
            
            temp_Decision_cost[m] = 100*newsvendor_loss(Prescriptions[m].values, testY[target_zone], q = critical_fractile).round(4)

        # Evalaute QS (approximates CRPS)
        target_quant = np.arange(0.1, 1, 0.1).round(2)
        
        for j,m in enumerate(models):
            temp_pdf = sum([lambda_cc_dict[m][j]*test_p_list[j] for j in range(N_experts)])            
            # find quantile forecasts
            temp_q_forecast = np.array([inverted_cdf(target_quant, y_supp, temp_pdf[i]) for i in range(n_test_obs)])
            
            temp_qs = 100*pinball(temp_q_forecast, testY[target_zone].values, target_quant).round(4)
            
            temp_QS[m] = [temp_qs]
            
            if m in ['Ave', 'SalvaBench', 'CRPS', 'DF_0.1', 'DF_1', 'DF_10']:
                plt.plot(temp_qs, label = m)
        plt.legend()
        plt.ylabel('Pinball loss')
        plt.xticks(np.arange(len(target_quant)), target_quant)
        plt.xlabel('Quantile')
        plt.show()
        

        if row_counter == 0: 
            Decision_cost = temp_Decision_cost.copy()
        else:
            Decision_cost = pd.concat([Decision_cost, temp_Decision_cost], ignore_index = True)
            
        if row_counter == 0: 
            QS_df = temp_QS.copy()
        else:
            QS_df = pd.concat([QS_df, temp_QS], ignore_index = True)        
            
        Decision_cost.to_csv(f'{cd}\\results\\{target_problem}_static_linearpool_Decision_cost.csv')
        QS_df.to_csv(f'{cd}\\results\\{target_problem}_static_linear_pool_QS.csv')
        
        row_counter += 1
        
        ### Adaptive Forecast Combinations
        
        #lambda_adapt_lr = lr_adapt_Newsv_Comb_Model.predict_weights(tensor_testX)    
        #lambda_adapt_mlp = mlp_adapt_Newsv_Comb_Model.predict_weights(tensor_testX)
        
        #p_adapt_cc_lr = np.array([sum([lambda_adapt_lr[i,j]*test_p_list[j][i] for j in range(N_experts)]) for i in range(n_test_obs)])    
        #p_adapt_cc_mlp = np.array([sum([lambda_adapt_mlp[i,j]*test_p_list[j][i] for j in range(N_experts)]) for i in range(n_test_obs)])
        
        
        #Prescriptions['BRC-DT'] = np.array([inverted_cdf([critical_fractile], y_supp, p_brc_dt[i]) for i in range(len(test_localY))]).reshape(-1)
            
        
