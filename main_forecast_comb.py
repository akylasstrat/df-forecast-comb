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

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from gurobi_ml import add_predictor_constr
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor

from gurobi_ml.sklearn import add_decision_tree_regressor_constr, add_random_forest_regressor_constr

cd = os.path.dirname(__file__)  #Current directory
sys.path.append(cd)
#project_dir=Path(cd).parent.__str__()   #project_directory

from EnsemblePrescriptiveTree import *
from EnsemblePrescriptiveTree_OOB import *
from optimization_functions import *

from LinearDecisionTree import *
from sklearn.neural_network import MLPRegressor


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


class ConvexCombinationLayer(nn.Module):
    def __init__(self, num_inputs, support):
        super(ConvexCombinationLayer, self).__init__()

        # Initialize learnable weight parameters
        #self.weights = nn.Parameter(torch.rand(num_inputs), requires_grad=True)
        self.weights = nn.Parameter(torch.FloatTensor((1/num_inputs)*np.ones(num_inputs)).requires_grad_())
        self.support = support
        
    def forward(self, *inputs):
        """
        Forward pass of the convex combination layer.

        Args:
            *inputs: Variable number of input tensors.

        Returns:
            torch.Tensor: The convex combination of input tensors.
        """
        # Ensure that the weights are in the range [0, 1] using sigmoid activation
        #weights = torch.nn.functional.softmax(self.weights)
        weights = self.weights

        # Apply the weights element-wise to each input tensor
        weighted_inputs = [weights[i] * input_tensor for i, input_tensor in enumerate(inputs)]

        # Perform the convex combination across input vectors
        combined_output = sum(weighted_inputs)
        
        prediction = combined_output@self.support
        
        return prediction

class CombinationLayer(nn.Module):
    def __init__(self, num_inputs):
        super(CombinationLayer, self).__init__()

        # Initialize learnable weight parameters
        #self.weights = nn.Parameter(torch.rand(num_inputs), requires_grad=True)
        self.weights = nn.Parameter(torch.FloatTensor((1/num_inputs)*np.ones(num_inputs)).requires_grad_())
        
    def forward(self, *inputs):
        """
        Forward pass of the convex combination layer.

        Args:
            *inputs: Variable number of input tensors.

        Returns:
            torch.Tensor: The convex combination of input tensors.
        """
        # Ensure that the weights are in the range [0, 1] using sigmoid activation
        #weights = torch.nn.functional.softmax(self.weights)
        weights = self.weights
        # Apply the weights element-wise to each input tensor
        weighted_inputs = [weights[i] * input_tensor for i, input_tensor in enumerate(inputs)]

        # Perform the convex combination across input vectors
        combined_output = sum(weighted_inputs)
        
        return combined_output
    
class NewsvendorLayer(nn.Module):        
    def __init__(self, support, gamma, apply_softmax = False):
        super(NewsvendorLayer, self).__init__()

        # Initialize learnable weight parameters
        #self.weights = nn.Parameter(torch.rand(num_inputs), requires_grad=True)
        #self.weights = nn.Parameter(torch.FloatTensor((1/num_inputs)*np.ones(num_inputs)).requires_grad_())
        self.support = support
        self.gamma = gamma
        self.apply_softmax = apply_softmax
        
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
        
    def forward(self, x):
        """
        Forward pass of the newvendor layer.

        Args:
            x: Probability vector.

        Returns:
            solves newsvendor problem, returns all variables of cvxpy layer.
        """

        # Pass the combined output to the CVXPY layer
        cvxpy_output = self.newsvendor_layer(x)
        return cvxpy_output

class CustomSequentialLayer(nn.Module):
    def __init__(self, num_probs, prob_size, cvxpy_params):
        super(CustomSequentialLayer, self).__init__()

        # Define the parameters for the first layer (learnable weights)
        self.weights = nn.Parameter(torch.rand(num_probs), requires_grad=True)

        # Define the second layer as a CVXPY layer
        self.cvxpy_layer = CvxpyLayer(
            cp.Problem(**cvxpy_params),  # Define the CVXPY problem
            parameters=[],
            variables=[],
            modules=[]  # No additional modules needed for the CVXPY layer
        )

    def forward(self, input_probs):
        """
        Forward pass of the custom sequential layer.

        Args:
            input_probs (list of torch.Tensor): List of input probability vectors.

        Returns:
            torch.Tensor: Output of the sequential layer after both layers.
        """
        # Apply the sigmoid activation to the learned weights
        weights = torch.sigmoid(self.weights)

        # Element-wise weighted sum of input probability vectors
        combined_output = sum(weights[i] * input_probs[i] for i in range(len(input_probs)))

        # Pass the combined output to the CVXPY layer
        cvxpy_output = self.cvxpy_layer([combined_output])

        return cvxpy_output
    
class CombNewsvendorLayer(nn.Module):        
    def __init__(self, num_inputs, support, gamma, apply_softmax = False):
        super(CombNewsvendorLayer, self).__init__()

        # Initialize learnable weight parameters
        #self.weights = nn.Parameter(torch.rand(num_inputs), requires_grad=True)
        self.weights = nn.Parameter(torch.FloatTensor((1/num_inputs)*np.ones(num_inputs)).requires_grad_())
        self.support = support
        self.gamma = gamma
        self.apply_softmax = apply_softmax
        
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
        
    def forward(self, *inputs):
        """
        Forward pass of the newvendor layer.

        Args:
            *inputs: Variable number of input tensors.

        Returns:
            torch.Tensor: The convex combination of input tensors.
        """
        # Ensure that the weights are in the range [0, 1] using sigmoid activation
        #weights = torch.nn.functional.softmax(self.weights)

        # Ensure that the weights are in the range [0, 1] using sigmoid activation
        if self.apply_softmax:
            weights = torch.nn.functional.softmax(self.weights)
        else:
            weights = self.weights

        # Apply the weights element-wise to each input tensor
        weighted_inputs = [weights[i] * input_tensor for i, input_tensor in enumerate(inputs)]

        # Perform the convex combination across input vectors
        combined_vector = sum(weighted_inputs)

        # Pass the combined output to the CVXPY layer
        cvxpy_output = self.newsvendor_layer(combined_vector)
        return cvxpy_output
    
def params():
    ''' Set up the experiment parameters'''

    params = {}
    # Either pv or wind
    params['target_zone'] = 'Z1'
        
    params['start_date'] = '2012-01-01'
    params['split_date'] = '2013-01-01' # Defines train/test split
    params['end_date'] = '2013-12-30'
    
    params['save'] = False # If True, then saves models and results
    params['train_brc'] = False # If True, then saves models and results
    
    # Experimental setup parameters
    params['problem'] = 'newsvendor' # {mse, newsvendor, cvar, reg_trad}
    params['N_sample'] = [10, 50, 100, 200, 500]
    params['N_experts'] = 3
    params['iterations'] = 5
    
    params['critical_fractile'] = 0.6
    
    # approaches to map data to decisions
    # LR: linear regression, DecComb: combination of perfect-foresight decisions (both maintain convexity)
    # DT: decision tree, NN: neural network (both results in MIPs)
    params['decision_rules'] = ['LR', 'DecComb'] 

    return params
    
#%%
config = params()
hyperparam = tree_params()

results_path = f'{cd}\\results'
data_path = f'{cd}\\data'

aggr_wind_df = pd.read_csv(f'{data_path}\\GEFCom2014-processed.csv', index_col = 0, header = [0,1])
#%%

#!!!! Add randomization based on the iteration counter here
all_zones = [f'Z{i}' for i in range(1,11)]

#target_zone = config['target_zone']
#expert_zones = ['Z2', 'Z4', 'Z8', 'Z9']

target_zone = np.random.choice(all_zones)
expert_zones = all_zones.copy()
expert_zones.remove(target_zone)
expert_zones = list(np.random.choice(expert_zones, config['N_experts'], replace = False))

pred_col = ['wspeed10', 'wdir10_rad', 'wspeed100', 'wdir100_rad']
#%%
# number of forecasts to combine
N_experts = config['N_experts']

# number of observations to train prob. forecasting model
N_sample = len(aggr_wind_df)//4

step = .01
y_supp = np.arange(0, 1+step, step).round(2)
nlocations = len(y_supp)

#%%
### Create train/test sets for all series

trainY = aggr_wind_df.xs('POWER', axis=1, level=1)[[target_zone] + expert_zones][config['start_date']:config['split_date']][:N_sample].round(2)
comb_trainY = aggr_wind_df.xs('POWER', axis=1, level=1)[[target_zone] + expert_zones][config['start_date']:config['split_date']][N_sample:].round(2)
testY = aggr_wind_df.xs('POWER', axis=1, level=1)[[target_zone] + expert_zones][config['split_date']:].round(2)

# feature data for target location/zone

local_features_df = aggr_wind_df[target_zone].copy()
local_features_df['wspeed10_sq'] = np.power(local_features_df['wspeed10'],2)
local_features_df['wspeed10_cb'] = np.power(local_features_df['wspeed10'],3)

local_features_df[['diurnal_1', 'diurnal_2', 'diurnal_3', 'diurnal_4']] = aggr_wind_df[['diurnal_1', 'diurnal_2', 'diurnal_3', 'diurnal_4']]

local_feat = ['wspeed10', 'wspeed10_sq', 'wspeed10_cb', 'wdir10_rad', 'diurnal_1', 'diurnal_2', 'diurnal_3', 'diurnal_4']
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

#%% Train experts, i.e., probabilistic forecasting models in adjacent locations

# data conditioned on wind speed

prob_models = []

for i, zone in enumerate(expert_zones[:N_experts]):
    print(f'Training model {i}')
    
    #temp_model = EnsemblePrescriptiveTree_OOB(n_estimators = 10, max_features = 1, type_split = 'quant')
    #temp_model.fit(trainX_exp[zone][pred_col].values.round(2), trainY[zone].values, y_supp, y_supp, bootstrap = False, quant = np.arange(.01, 1, .01), problem = 'mse') 
 
    temp_model = EnsemblePrescriptiveTree(n_estimators = 30, max_features = len(pred_col), type_split = 'quant' )
    temp_model.fit(trainX_exp[zone][pred_col].values, trainY[zone].values, quant = np.arange(.01, 1, .01), problem = 'mse') 
    prob_models.append(temp_model)

#%% Generate predictions for train/test set for forecast combination
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
#%%
# step 1: find inverted CDFs
F_inv = [np.array([inverted_cdf([.05, .10, .90, .95] , trainY[zone].values, train_w_list[j][i]) for i in range(500)]) for j in range(N_experts)]

plt.plot(comb_trainY[target_zone][200:300])
for i in [0,2]:
    #plt.fill_between(np.arange(100), F_inv[i][200:300,0], F_inv[i][200:300,-1], alpha = .3, color = 'red')
    
    plt.fill_between(np.arange(100), F_inv[i][200:300,0], F_inv[i][200:300,-1], alpha = .3)
plt.show()

#%% Example: point forecasting (MSE) & convex combination

target_problem = 'mse'
critical_fractile = config['critical_fractile']

# projection step (used for gradient-based methods)
y_proj = cp.Variable(N_experts)
y_hat = cp.Parameter(N_experts)
proj_problem = cp.Problem(cp.Minimize(0.5*cp.sum_squares(y_proj-y_hat)), [y_proj >= 0, y_proj.sum()==1])

train_targetY = comb_trainY[target_zone].values.reshape(-1)

# Supervised learning set for torch
tensor_trainY = torch.FloatTensor(train_targetY)
tensor_train_p = torch.FloatTensor(np.column_stack((train_p_list)))
tensor_train_p_list = [torch.FloatTensor(train_p_list[i]) for i in range(N_experts)]

train_data = torch.utils.data.TensorDataset(tensor_train_p_list[0], tensor_train_p_list[1], tensor_train_p_list[2], tensor_trainY)


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
    
    #%%
    ################## Pytorch example    
    patience = 25
    batch_size = 100
    num_epochs = 100
    
    train_loader = create_data_loader(tensor_train_p_list + [tensor_trainY], batch_size = batch_size)
    convex_layer = ConvexCombinationLayer(num_inputs=N_experts, support = torch.FloatTensor(y_supp))
    
    
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
    
    #%%
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


        
    #%%
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

#%%%%%%%%%%%%%%% Newsvendor experiment

if target_problem == 'newsvendor':
    ### Train decision rules to mapping probability vectors to decisions
    
    # optimal solutions to create training set for decision rule
    z_opt = []
    for j in range(N_experts):
        z_opt.append( np.array([inverted_cdf([critical_fractile] , y_supp, train_p_list[j][i]) for i in range(n_obs)]) )
    
    z_opt = np.row_stack(z_opt)
    train_p_stacked = np.row_stack((train_p_list))

    #### Train ML predictor to map prob. vectors (p.d.f.) to optimal decisions (linear, tree-based methods)
    
    lr_model = Ridge().fit(train_p_stacked, z_opt)
    rf_model = RandomForestRegressor(n_estimators = 10).fit(train_p_stacked, z_opt)
    dt_model = DecisionTreeRegressor().fit(train_p_stacked, z_opt)
    
    # linear decision tree
    ldt_model = LinearDecisionTree(type_split = 'quant')
    ldt_model.fit(train_p_stacked, z_opt.reshape(-1))
    # neural net
    nn_model = MLPRegressor(hidden_layer_sizes=(20,20,20), activation='relu',).fit(train_p_stacked, z_opt)
    
    # benchmark rule (see Juanmi's suggestion)
    z0_opt = y_supp

    # visualize decision rules
    t = -200
    plt.plot(z_opt[t:], label = 'z_opt')
    plt.plot(ldt_model.predict(train_p_stacked)[t:], '--', label = 'LR')
    plt.plot(nn_model.predict(train_p_stacked)[t:], '-.', label = 'NN')
    plt.plot(train_p_stacked[t:]@z0_opt, 'd', markersize = 2, label = 'Combination')
    #plt.plot(z_pred.X[t:])
    plt.legend()
    plt.show()

    #%% 
    lambda_cc_dict = {}
    for DR in config['decision_rules']:
        
        if DR == 'LR':
            # Case 1: Convex Combination/ LR decision rule
            lambda_ = tune_combination_newsvendor(train_targetY, train_p_list, lr_model, 
                                                       crit_fract = critical_fractile, support = y_supp, bounds = False)
            
        elif DR == 'DT':
            # Case 2: Convex Combination/ DT decision rule
            lambda_ = tune_combination_newsvendor(train_targetY, train_p_list, nn_model, 
                                                       crit_fract = critical_fractile, support = y_supp, verbose = 1)
        elif DR == 'DecComb':
            
            # Benchmark/ Juanmi's suggestion/ weighted combination / for newsvendor is just the expected value
            lambda_ = wsum_tune_combination_newsvendor(train_targetY, train_p_list, 
                                                           crit_fract = critical_fractile, support = y_supp, bounds = False)
        lambda_cc_dict[DR] = lambda_


    # Adaptive decision rules
    coef_, inter_ = adapt_combination_newsvendor(train_targetY, comd_trainX_local.values, train_p_list, lr_model, 
                                               crit_fract = critical_fractile, support = y_supp, bounds = False)

#%% PyTorch example
    
    gamma = 0.1
    
    patience = 25
    batch_size = 100
    num_epochs = 25
    
    train_loader = create_data_loader(tensor_train_p_list + [tensor_trainY], batch_size = batch_size)

    combination_layer = CombinationLayer(num_inputs=N_experts)
    
    # newsvendor layer
    z = cp.Variable((1))    
    pinball_loss = cp.Variable(len(y_supp))
    
    prob_weights = cp.Parameter(len(y_supp))
    
    newsv_constraints = [z >= 0, z <= 1, 
                         pinball_loss >= critical_fractile*(y_supp-z), 
                         pinball_loss >= (critical_fractile - 1)*(y_supp-z)]

    newsv_objective = cp.Minimize( prob_weights@pinball_loss ) 
    newsv_problem = cp.Problem(newsv_objective, newsv_constraints)
    newsvendor_layer = CvxpyLayer(newsv_problem, parameters=[prob_weights], variables = [z, pinball_loss] )


    # Forward pass through the layer with 3 inputs
    #Newsvendor_Comb_Model = nn.Sequential(CombinationLayer(num_inputs=N_experts), 
    #                                      NewsvendorLayer(support = y_supp, gamma = gamma))
    
    Newsvendor_Comb_Model = CombNewsvendorLayer(num_inputs=N_experts, support = torch.FloatTensor(y_supp), gamma = gamma, 
                                                apply_softmax = False)
    
    optimizer = torch.optim.SGD(Newsvendor_Comb_Model.parameters(), lr = 1e-3)
    Projection = True

    L_t = []
    for epoch in range(num_epochs):
        # activate train functionality
        Newsvendor_Comb_Model.train()
        running_loss = 0.0
        # sample batch
        for batch_data in train_loader:
            
            y_batch = batch_data[-1]
            
            # clear gradients
            optimizer.zero_grad()
            
            # forward pass: combine forecasts and solve each newsvendor problem
            z_hat = Newsvendor_Comb_Model(batch_data[0], batch_data[1], batch_data[2])[0]
            
            error_hat = (y_batch.reshape(-1,1) - z_hat)
            loss = (critical_fractile*error_hat[error_hat>0].norm(p=1) \
                    + (1-critical_fractile)*error_hat[error_hat<0].norm(p=1))\
                    + gamma*error_hat.norm()

            # backward pass
            loss.backward()
            optimizer.step()
                        
        # Projection step (!!!! enforce no gradient update)
            #with torch.no_grad():
            #    Newsvendor_Comb_Model.weights.copy_ = torch.nn.functional.softmax(Newsvendor_Comb_Model.weights)
            
            
            if Projection:     
                y_hat.value = to_np(Newsvendor_Comb_Model.weights)
                proj_problem.solve(solver = 'GUROBI')
                # update parameter values
                with torch.no_grad():
                    Newsvendor_Comb_Model.weights.copy_(torch.FloatTensor(y_proj.value))
            
            running_loss += loss.item()
        

        L_t.append(to_np(Newsvendor_Comb_Model.weights).copy())
        
        if epoch % 15 ==0:
            plt.plot(L_t)
            plt.show()
        average_train_loss = running_loss / len(train_loader)
        
        with torch.no_grad():            
            y_prediction = convex_layer(tensor_train_p_list[0], tensor_train_p_list[1], tensor_train_p_list[2])

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {average_train_loss:.4f} ")


#%% Gradient-based approach with CVXPY/ full batch updates
    batch_size = 50
    k = N_experts
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
    #L_t = [[.4, .4, .2]]
    
    Loss = []
    Projection = True
    eta = 1e-3
    
    for i in range(10000):
    
        ix = np.random.choice(range(n_obs), batch_size, replace = False)
    
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
        
#%% Differential Opt. Layer/ Pytorch/ Full stochastic solution for inner problem
    batch_size = 10
    nobs = len(train_targetY)
    k = len(y_supp)

    ###### Full stochastic opt. layer
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
                + [point_pred == sum([lambda_[j]*p_list_t_aux[j] for j in range(N_experts)])@y_supp]
    #reg_pen = 10*cp.norm(cp.multiply(p_comb, error))
    #reg_pen = 10*cp.norm( point_pred )
    objective = cp.Minimize( sum([pinball_loss[i]@p_comb[i] for i in range(batch_size)]) ) 

    problem = cp.Problem(objective, constraints)
    layer = CvxpyLayer(problem, parameters=[lambda_, p_comb] + p_list_t, variables=[z, point_pred, pinball_loss, error] + p_list_t_aux)
    
    l_hat = nn.Parameter(torch.FloatTensor((1/N_experts)*np.ones(N_experts)).requires_grad_())

    opt = torch.optim.Adam([l_hat], lr=5e-1)
    losses = []
    L_t = [to_np(l_hat).copy()]
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
        
        error_hat = nn.Parameter(torch.FloatTensor([train_targetY[ix]])) - zhat
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
        
        L_t.append(to_np(l_hat).copy())
    
        if i%50==0:
            print(L_t[-1])
            
            plt.plot(losses)
            plt.show()
            
#%% Differential Opt. Layer/ Pytorch/ **Deterministic** solution for inner problem
    batch_size = 500
    nobs = len(train_targetY)
    k = len(y_supp)

    ###### Full stochastic opt. layer
    # Variables of inner problem
    z = cp.Variable(batch_size)    
    error = cp.Variable((batch_size))
    pinball_loss = cp.Variable((batch_size))
    
    lambda_ = cp.Parameter(N_experts)
    point_pred_exp_t = cp.Parameter((batch_size, N_experts))
    point_pred_aux = cp.Variable((batch_size, N_experts))
    
    comb_point_pred = cp.Variable(batch_size)
    #target_t = cp.Parameter(batch_size)
    
    #p_list_t = [cp.Parameter((batch_size,k)) for j in range(N_experts)]
    #p_list_t_aux = [cp.Variable((batch_size,k)) for j in range(N_experts)]
    
    constraints = [z >= 0, z <=1] + [error == comb_point_pred - z]\
                + [pinball_loss >= critical_fractile*error]\
                + [pinball_loss >= (critical_fractile-1)*error]\
                + [point_pred_aux == point_pred_exp_t]\
                + [comb_point_pred == point_pred_aux@lambda_]\
                #+ [z == comb_point_pred]
                #+ [p_list_t_aux[j] == p_list_t[j] for j in range(N_experts)]\

    objective = cp.Minimize( sum(pinball_loss) ) 
    problem = cp.Problem(objective, constraints)
    
    layer = CvxpyLayer(problem, parameters=[lambda_, point_pred_exp_t], variables=[z, comb_point_pred, point_pred_aux, pinball_loss, error] )
    
    l_hat = nn.Parameter(torch.FloatTensor((1/N_experts)*np.ones(N_experts)).requires_grad_())

    opt = torch.optim.Adam([l_hat], lr=5e-2)
    losses = []
    L_t = [to_np(l_hat).copy()]
    Projection = True
    
    for i in range(50000):
        
        ix = np.random.choice(range(nobs), batch_size, replace = False)
        
        p_list_t_hat = []
        temp_point_pred = []
        for j in range(N_experts):
            p_list_t_hat.append( nn.Parameter( torch.FloatTensor(train_p_list[j][ix]) ) ) 
            temp_point_pred.append(train_p_list[j][ix]@y_supp)

        point_pred_exp_t_hat = nn.Parameter(torch.FloatTensor( np.array(temp_point_pred).T ))
        target_t_hat = nn.Parameter(torch.FloatTensor([train_targetY[ix]]))
        
        ### Forward pass: solve the problem        
        decisions_hat = layer(l_hat, point_pred_exp_t_hat)
        zhat = decisions_hat[0]
        #point_pred_hat = decisions_hat[1]
        
        error_hat = nn.Parameter(torch.FloatTensor([train_targetY[ix]])) - zhat
        
        ### Estimate regret on the master problem
        loss = (critical_fractile*error_hat[error_hat>0].norm(p=1) + (1-critical_fractile)*error_hat[error_hat<0].norm(p=1))\
#            + error_hat.norm(p=2)
        losses.append(to_np(loss))
        
        ### Find gradients and update parameters                
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        ### Projection to probability simplex
        if Projection:     
            y_hat.value = to_np(l_hat)
            proj_problem.solve(solver = 'GUROBI')
            # update parameter values
            
            with torch.no_grad():
                l_hat.copy_(torch.FloatTensor(y_proj.value))
        
        L_t.append(to_np(l_hat).copy())
        
        if i%50==0:
            print(L_t[-1])
            
            plt.plot(L_t)
            plt.show()
    
    # find best iteration
    best_ind = np.where(np.array(losses) == min(losses))[0][0]
    
    #%% Barycentric interpolation: learn mapping from inverse c.d.f. to c.d.f. (stepwise function analytically)
    
    # Train additional ML predictor to learn mapping from quantile function to p.d.f. (the true is piecewise constant)
    target_quantiles = np.arange(0,1.01,.01)
    
    p_inv_list = [np.array([inverted_cdf(target_quantiles, y_supp, w = train_p_list[j][i]) for i in range(nobs)]) for j in range(N_experts)]
    
    dt_model_inv = DecisionTreeRegressor().fit(np.row_stack(p_inv_list), np.row_stack(train_p_list))
    lr_model_inv = LinearRegression().fit(np.row_stack(p_inv_list), np.row_stack(train_p_list))
    
    t = 10
    
    plt.plot(dt_model_inv.predict(np.row_stack(p_inv_list))[t], label = 'DT-apprx')
    plt.plot(lr_model_inv.predict(np.row_stack(p_inv_list))[t], label = 'LR-apprx')
    plt.plot(np.row_stack(train_p_list)[t])
    plt.show()

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
                                   prob_dx = .01)
        p_brc[i] = temp_p_brc
    #%%
    p_cc_lr = sum([lambda_cc_dict['LR'][i]*test_p_list[i] for i in range(N_experts)])
    p_cc_base = sum([lambda_cc_dict['DecComb'][i]*test_p_list[i] for i in range(N_experts)])
    p_cc_dt = sum([L_t[best_ind][i]*test_p_list[i] for i in range(N_experts)])
    p_brc_dt = np.zeros((n_test_obs, nlocations))
    
    # Adaptive convex combination
    lambda_adapt = testX_local.values@coef_.T + inter_
    p_adapt_cc_lr = []
    # project to simplex
    for i in range(n_test_obs):
        if (lambda_adapt[i]<0).any() or (lambda_adapt[i]>1).any() or (lambda_adapt[i].sum()!=1):
            y_hat.value = lambda_adapt[i]
            proj_problem.solve(solver = 'GUROBI')
            lambda_adapt[i] = y_proj.value
    
        p_adapt_cc_lr.append( sum([lambda_adapt[i,j]*test_p_list[j][i] for j in range(N_experts)]) )
    p_adapt_cc_lr = np.array(p_adapt_cc_lr)

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
    
    models = [f'Model-{i}' for i in range(N_experts)] + ['CC-Ave', 'BRC-Ave', 'CC-DecComb', 'CC-LR', 'CC-LR-adapt', 'CC-DT', 'BRC-DT']
    
    Prescriptions = pd.DataFrame(data = np.zeros((nobs_test, len(models))), columns = models)
    for i in range(N_experts):
        Prescriptions[f'Model-{i}'] = np.array([inverted_cdf([critical_fractile], y_supp, test_p_list[i][k]) for k in range(len(testY))]).reshape(-1)
    
    Prescriptions['CC-DecComb'] = np.array([inverted_cdf([critical_fractile], y_supp, p_cc_base[i]) for i in range(nobs_test)]).reshape(-1)
    Prescriptions['CC-Ave'] = np.array([inverted_cdf([critical_fractile], y_supp, p_ave[i]) for i in range(nobs_test)]).reshape(-1)
    Prescriptions['BRC-Ave'] = np.array([inverted_cdf([critical_fractile], y_supp, p_brc[i]) for i in range(nobs_test)]).reshape(-1)
    Prescriptions['CC-LR'] = np.array([inverted_cdf([critical_fractile], y_supp, p_cc_lr[i]) for i in range(nobs_test)]).reshape(-1)
    Prescriptions['CC-LR-adapt'] = np.array([inverted_cdf([critical_fractile], y_supp, p_adapt_cc_lr[i]) for i in range(nobs_test)]).reshape(-1)
    Prescriptions['CC-DT'] = np.array([inverted_cdf([critical_fractile], y_supp, p_cc_dt[i]) for i in range(nobs_test)]).reshape(-1)
    #Prescriptions['BRC-DT'] = np.array([inverted_cdf([critical_fractile], y_supp, p_brc_dt[i]) for i in range(len(test_localY))]).reshape(-1)
        
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

