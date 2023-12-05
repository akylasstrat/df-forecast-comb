# -*- coding: utf-8 -*-
"""
Opt function for newsvendor problem

@author: a.stratigakos
"""

import gurobipy as gp
import numpy as np
import scipy.sparse as sp

def opt_problem(Y, weights = None, prescribe = False,
                prescriptive_train = False, problem = 'mse', crit_quant = 0.5, risk_aversion = 0.5, epsilon = 0.5):
    
    ''' SAA of newsvendor problem
        Y: pooled data and indicator on whether observation belongs to local data set
        weights: sample weights for prescription (not used in training)
        '''
    Y = Y.reshape(-1)
    nobs = len(Y)
    
    if type(weights) != np.ndarray:
        if weights == None:
            weights = np.ones(len(Y))/len(Y)

    #W_diag = sp.diags(weights)    
    if prescribe == False:
        if prescriptive_train == False:
            pred = Y.mean()
            pred_error = np.square((Y - pred)).sum() 
            return pred_error, pred
    
        else:
            # train prescriptive tree to minimize the true cost function (not used here)
            if problem == 'newsvendor':
                # solve the newsvendor
                pred = np.quantile(Y, [crit_quant], method = 'inverted_cdf')
                pinball_loss = np.maximum(crit_quant*(Y - pred), (crit_quant-1)*(Y - pred)).sum()                
                return pinball_loss, pred

            
    elif prescribe == True:
        if problem == 'mse':
            pred = weights@Y
            objval = np.square((Y - pred)).sum()

            return objval, pred
            
        elif problem == 'newsvendor':
            sort_Y = np.sort(Y.copy())
            arg_sort_g = np.argsort(Y.copy())    
            pred = sort_Y[np.where(weights[arg_sort_g].cumsum()>=crit_quant)][0]
            pinball_loss = np.maximum(crit_quant*(Y - pred), (crit_quant-1)*(Y - pred)).sum()                
            return pinball_loss, pred
        
        elif problem == 'cvar':
            
            # CVaR tail probability
            e = epsilon
            # risk level/ objective
            k = risk_aversion
            
            m = gp.Model()
            m.setParam('OutputFlag', 0)
            
            #CVaR: auxiliary parameters
            # Multi-temporal: Minimize average Daily costs
            ### Variables
            offer = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1)
            deviation = m.addMVar(nobs, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
            loss = m.addMVar(nobs, vtype = gp.GRB.CONTINUOUS, lb = 0)
            profit = m.addMVar(nobs, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)

            ### CVaR variables (follows Georghiou, Kuhn, et al.)
            beta = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name='VaR')
            zeta = m.addMVar(nobs, vtype = gp.GRB.CONTINUOUS, lb = 0)  # Aux
            cvar = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
             
            m.addConstr( deviation == Y - offer)
            
            m.addConstr( loss >= (crit_quant)*deviation)
            m.addConstr( loss >= (crit_quant-1)*deviation)

            m.addConstr( profit == Y*27 - loss )            
            m.addConstr( zeta >= beta - profit)

            #m.addConstr( zeta >=  -beta + loss )
            #m.addConstr( cvar == beta + (1/e)*(zeta@weights))
            
            m.addConstr( cvar == beta - (1/(1-e))*(zeta.sum()) )            
            m.setObjective( (1-k)*(profit@weights) + k*(cvar), gp.GRB.MAXIMIZE )
            
            #m.setObjective( (1-k)*loss@weights + k*cvar, gp.GRB.MINIMIZE)
            m.optimize()
                
            return m.objVal, offer.X[0]
                
        elif problem == 'reg_trad':
            
            # risk level/ objective
            k = risk_aversion
            
            m = gp.Model()
            m.setParam('OutputFlag', 0)
            
            #CVaR: auxiliary parameters
            # Multi-temporal: Minimize average Daily costs
            ### Variables
            offer = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1)
            deviation = m.addMVar(nobs, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
            loss = m.addMVar(nobs, vtype = gp.GRB.CONTINUOUS, lb = 0)
                         
            m.addConstr( deviation == Y - offer)
            
            m.addConstr( loss >= (crit_quant)*deviation)
            m.addConstr( loss >= (crit_quant-1)*deviation)
            
            m.setObjective( (1-k)*loss@weights + k*(deviation*deviation)@weights, gp.GRB.MINIMIZE)
            m.optimize()
                
            return m.objVal, offer.X[0]     
        
        else:
            print('Problem type not found')