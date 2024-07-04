# -*- coding: utf-8 -*-
"""
Opt function for newsvendor problem

@author: a.stratigakos
"""

import gurobipy as gp
import numpy as np
import scipy.sparse as sp

def opt_problem(Y, lambda_, opt_quant, weights = None, prescribe = False,
                train_metric = 'mse', test_metric = 'mse'):
    
    ''' SAA of newsvendor problem
        Y: pooled data and indicator on whether observation belongs to local data set
        lambda_: the weight of each dataset (or objective)
        weights: sample weights for prescription (not used in training)
        opt_quant: critical fractile of the newsvendor problem
        '''
        
    pooled_Y = Y[:,0]
    nobs = len(Y)
    target_ind = Y[:,1]
    target_mask = target_ind == 1
    
    target_Y = pooled_Y[target_mask]    
    remain_Y = pooled_Y[~target_mask]    
    
    if type(weights) != np.ndarray:
        if weights == None:
            weights = np.ones(len(pooled_Y))/len(pooled_Y)
            
    #W_diag = sp.diags(weights)    
        
    if prescribe == False:
        if train_metric == 'mse':
            if (target_ind == 1).any() and (target_ind == 0).any():
                
                pred = lambda_*target_Y.mean() + (1-lambda_)*remain_Y.mean()
                pred_error = lambda_*np.square((target_Y - pred)).sum() + (1-lambda_)*np.square((remain_Y - pred)).sum()

            elif (target_ind == 1).any():     
                pred = target_Y.mean()
                pred_error = lambda_*np.square((target_Y - pred)).sum()
            else:
                pred = remain_Y.mean()
                pred_error = (1-lambda_)*np.square((remain_Y - pred)).sum()

            pred = lambda_*target_Y.mean() + (1-lambda_)*remain_Y.mean()
            pred_error = lambda_*np.square((target_Y - pred)).sum() + (1-lambda_)*np.square((remain_Y - pred)).sum()

            return pred_error, pred

        else:
            
            m = gp.Model()
            m.setParam('OutputFlag', 0)
            # Decision variables
            offer = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1, name = 'E_offer')
            target_d = m.addMVar(len(target_Y), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'Deviation')        
            remain_d = m.addMVar(len(remain_Y), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'Deviation')        

            target_u = m.addMVar(len(target_Y), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'aux')        
            remain_u = m.addMVar(len(remain_Y), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'aux')        
                    
            # deviation            
            m.addConstr( target_d == target_Y - np.ones((len(target_Y),1))@offer)
            m.addConstr( remain_d == remain_Y - np.ones((len(remain_Y),1))@offer)

            # linearize maxima terms  
            m.addConstr( target_u >= opt_quant*(target_d))
            m.addConstr( target_u >= (opt_quant-1)*(target_d))

            m.addConstr( remain_u >= opt_quant*(remain_d))
            m.addConstr( remain_u >= (opt_quant-1)*(remain_d))
    
            m.setObjective( lambda_*target_u.sum() + (1-lambda_)*remain_u.sum(), gp.GRB.MINIMIZE)
            m.optimize()
            return m.objVal, offer.X[0]
            
    else:
        if test_metric == 'mse':
            scale_factor = lambda_*weights[target_mask].sum() + (1-lambda_)*weights[~target_mask].sum()
            pred = ( lambda_*(weights[target_mask]@pooled_Y[target_mask]) \
                + (1-lambda_)*(weights[~target_mask]@pooled_Y[~target_mask]) ) /scale_factor
            objval = lambda_*np.square((pooled_Y[target_mask] - pred)).sum() \
                + (1-lambda_)*np.square((pooled_Y[~target_mask] - pred)).sum()
            return objval, pred
        
        else:
            m = gp.Model()
            m.setParam('OutputFlag', 0)
            # Decision variables
            offer = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1, name = 'E_offer')
            total_cost = m.addMVar(nobs, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'cost')       
            deviation = m.addMVar(nobs, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'cost')       
            
            # deviation            
            m.addConstr( deviation == pooled_Y - np.ones((nobs,1))@offer)
    
            # linearize maxima terms  
            for i in range(len(pooled_Y)):
                if target_ind[i] == 1:
                    m.addConstr( total_cost[i] >= lambda_*opt_quant*deviation[i])
                    m.addConstr( total_cost[i] >= lambda_*(opt_quant-1)*deviation[i])
                else:                
                    m.addConstr( total_cost[i] >= (1-lambda_)*opt_quant*deviation[i])
                    m.addConstr( total_cost[i] >= (1-lambda_)*(opt_quant-1)*deviation[i])
    
            # weight cost by sample weight obtained from the random forest
            m.setObjective( weights@total_cost, gp.GRB.MINIMIZE)
            m.optimize()
            return m.objVal, offer.X[0]
