# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 21:59:00 2022

@author: akylas.stratigakos
"""

import gurobipy as gp
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

def comb_opt_problem(loc_Y, glob_Y, lambda_, opt_quant = .5, loc_weights = None, glob_weights = None,
                train_metric = 'mse', test_metric = 'mse'):
    
    ''' SAA of newsvendor problem
        Y: pooled data and indicator on whether observation belongs to local data set
        lambda_: the weight of each dataset (or objective)
        weights: sample weights for prescription (not used in training)
        opt_quant: critical fractile of the newsvendor problem
        uniform_fit: regular fitting, without re-weighting samples
        '''
    
    #W_diag = sp.diags(weights)            
    
    if test_metric == 'mse':
        pred = lambda_*(loc_weights@loc_Y[:,0]) + (1-lambda_)*(glob_weights@glob_Y[:,0])
        return [], pred    
    else:    
        m = gp.Model()
        m.setParam('OutputFlag', 0)
        # Decision variables
        offer = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1, name = 'E_offer')
        loc_cost = m.addMVar(len(loc_Y), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'cost')       
        glob_cost = m.addMVar(len(glob_Y), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'cost')       

        aux_loc = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'cost')       
        aux_glob = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'cost')       
        
        # linearize maxima terms  
        for i in range(len(loc_Y)):
            m.addConstr( loc_cost[i] >= opt_quant*(loc_Y[i]-offer))
            m.addConstr( loc_cost[i] >= (opt_quant-1)*(loc_Y[i]-offer))

        for i in range(len(glob_Y)):
            m.addConstr( glob_cost[i] >= opt_quant*(glob_Y[i]-offer))
            m.addConstr( glob_cost[i] >= (opt_quant-1)*(glob_Y[i]-offer))

        m.addConstr( aux_loc >= loc_weights@loc_cost)
        m.addConstr( aux_glob >= glob_weights@glob_cost)

        # weight cost by sample weight obtained from the random forest
        m.setObjective(  lambda_*aux_loc + (1-lambda_)*aux_glob, gp.GRB.MINIMIZE)
        m.optimize()
        return m.objVal, offer.X[0]
