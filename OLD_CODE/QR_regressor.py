# -*- coding: utf-8 -*-
"""
Generic quantile regression

@author: a.stratigakos
"""

#Import Libraries
import numpy as np
import gurobipy as gp
import time
import scipy.sparse as sp

class QR_regressor(object):
  '''This function initializes the GPT.
  
  Paremeters:
      D: maximum depth of the tree (should include pruning??)
      Nmin: minimum number of observations at each leaf
      type_split: regular or random splits for the ExtraTree algorithm (perhaps passed as hyperparameter in the forest)
      cost_complexity: Should be included for a simple tree
      spo_weight: Parameter that controls the trade-off between prediction and prescription, to be included
      max_features: Maximum number of features to consider at each split (used for ensembles). If False, then all features are used
      **kwargs: keyword arguments to solve the optimization problem prescribed

      '''
  def __init__(self, quantile = 0.5, alpha = 0, fit_intercept = True):
      
    self.quantile = quantile
    self.alpha = alpha
    self.fit_intercept = fit_intercept
    
  def fit(self, X, Y, sample_weights = None, verbose = -1):

    n_train_obs = len(Y)
    n_feat = X.shape[1]
    
    try:
        if sample_weights == None:
            sample_weights = (1/n_train_obs)*np.ones(n_train_obs)
    except:
        sample_weights = sample_weights
    mask = sample_weights>0
    # loss quantile and robustness budget
    target_quant = self.quantile
    alpha = self.alpha
    
    m = gp.Model()
    if verbose == -1:
        m.setParam('OutputFlag', 0)
    else:
        m.setParam('OutputFlag', 1)
        
    print('Setting up GUROBI model...')
    
    # Variables
    fitted = m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'fitted')
    bias = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'bias')
    loss = m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'loss')
    w_loss = m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'weighted loss')

    # Linear Decision Rules: different set of coefficients for each group
    coef = m.addMVar(n_feat, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'LDR')
    aux = m.addMVar(n_feat, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'loss')
    
    # Dual Constraints
    m.addConstr( fitted == X@coef + np.ones((n_train_obs,1))@bias)
    m.addConstr( fitted[mask] >= 0)
    m.addConstr( fitted[mask] <= 1)
    
    m.addConstr( loss >= target_quant*(Y.reshape(-1) - fitted))
    m.addConstr( loss >= (target_quant-1)*(Y.reshape(-1) - fitted))
    m.addConstr( w_loss == np.diag(sample_weights)@loss)

    # l1 penalty
    m.addConstr( aux >= coef)
    m.addConstr( aux >= -coef)

    # Objective
    m.setObjective( w_loss.sum() + alpha*aux.sum(), gp.GRB.MINIMIZE)
    
    print('Solving the problem...')
    
    m.optimize()
    self.coef_ = coef.X
    self.bias_ = bias.X
    self.cpu_time = m.Runtime

    return 
    
  def predict(self, X):
    predictions = X@self.coef_ + self.bias_
    return np.array(predictions)