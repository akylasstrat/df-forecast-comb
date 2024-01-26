# -*- coding: utf-8 -*-
"""
Utility functions

@author: akylas.stratigakos
"""
import numpy as np
import gurobipy as gp
import matplotlib.pyplot as plt

# IEEE plot parameters (not sure about mathfont)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 600
plt.rcParams['figure.figsize'] = (3.5, 2) # Height can be changed
plt.rcParams['font.size'] = 7
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams["mathtext.fontset"] = 'dejavuserif'

def cart_find_weights(trainX, testX, cart_model):
    ''' Find weights for a sklearn forest model '''    
    Leaf_nodes = cart_model.apply(trainX).reshape(-1,1) # nObs*nTrees: a_ij shows the leaf node for observation i in tree j
    Index = cart_model.apply(testX).reshape(-1,1) # Leaf node for test set
    nTrees = 1
    Weights = np.zeros(( len(testX), len(trainX) ))
    #print(Weights.shape)
    #Estimate sample weights
    print('Retrieving weights...')
    for i in range(len(testX)):
        #New query point
        x0 = Index[i:i+1,:]
        #Find observations in terminal nodes/leaves (all trees)
        obs = 1*(x0.repeat(len(trainX), axis = 0) == Leaf_nodes)
        #Cardinality of leaves
        cardinality = np.sum(obs, axis = 0).reshape(-1,1).T.repeat(len(trainX), axis = 0)
        #Update weights
        Weights[i,:] = (obs/cardinality).sum(axis = 1)/nTrees

    return Weights

def forest_find_weights(trainX, testX, forest_model):
    ''' Find weights for a sklearn forest model '''    
    Leaf_nodes = forest_model.apply(trainX) # nObs*nTrees: a_ij shows the leaf node for observation i in tree j
    Index = forest_model.apply(testX) # Leaf node for test set
    nTrees = forest_model.n_estimators
    Weights = np.zeros(( len(testX), len(trainX) ))
    #print(Weights.shape)
    #Estimate sample weights
    print('Retrieving weights...')
    for i in range(len(testX)):
        #New query point
        x0 = Index[i:i+1, :]
        #Find observations in terminal nodes/leaves (all trees)
        obs = 1*(x0.repeat(len(trainX), axis = 0) == Leaf_nodes)
        #Cardinality of leaves
        cardinality = np.sum(obs, axis = 0).reshape(-1,1).T.repeat(len(trainX), axis = 0)
        #Update weights
        Weights[i,:] = (obs/cardinality).sum(axis = 1)/nTrees

    return Weights


def wemp_to_support(weights, Y, support_locations):
    'Map weighted empirical to discrete support locations (for speed-ups)'
    if weights.ndim == 1:
        w = weights.reshape(1,-1)
    else:
        w = weights
    Prob_supp = np.zeros((len(w), len(support_locations)))

    # effective set
    for i in range(w.shape[0]):
        w_i = w[i]
        mask = (w_i>0).reshape(-1)
        for j,y in enumerate(Y[mask]):        
            ind = np.where(y == support_locations)[0][0]
            Prob_supp[i,ind] = Prob_supp[i,ind] + w_i.reshape(-1)[mask][j]
    return Prob_supp

def find_weights(tree_model, train_feat, test_feat):
    ''' Function that returns the local weights of a tree-based model'''
    weights = np.zeros((len(test_feat), len(train_feat)))
    Leaf_nodes = tree_model.apply(train_feat)
    Index = tree_model.apply(test_feat) # Leaf node for test set
    nTrees = tree_model.n_estimators
    print('Retrieving weights...')
    for i in range(len(test_feat)):
        #New query point
        x0 = Index[i:i+1, :]
        
        #Find observations in terminal nodes/leaves (all trees)
        obs = 1*(x0.repeat(len(train_feat), axis = 0) == Leaf_nodes)
        #Cardinality of leaves
        cardinality = np.sum(obs, 0).reshape(-1,1).T.repeat(len(train_feat), 0)
        #Update weights
        weights[i,:] = ((obs/cardinality).sum(axis = 1)/nTrees)
    return weights

def mse(pred, actual):                
    return np.square(pred.reshape(-1) - actual.reshape(-1)).mean()

def newsvendor_loss(pred, actual, q = 0.5):                
    return np.maximum(q*(actual - pred), (q-1)*(actual - pred)).mean()


def reg_trad_loss(pred, actual, q = 0.5, risk_aversion = 0.5):   
    pred_copy = pred.copy().reshape(-1)
    actual_copy = actual.copy().reshape(-1)
    deviation = actual_copy - pred_copy
    pinball_loss = np.maximum(q*deviation, (q-1)*deviation)    
    
    return (1-risk_aversion)*pinball_loss.mean() + risk_aversion*(deviation@deviation)/len(deviation)

def newsvendor_cvar_loss(pred, actual, q = 0.5, risk_aversion = 0.5, e = 0.05):   
    pred_copy = pred.copy().reshape(-1)
    actual_copy = actual.copy().reshape(-1)
    
    pinball_loss = np.maximum(q*(actual_copy - pred_copy), (q-1)*(actual_copy - pred_copy))    
    cvar_mask = pinball_loss>=np.quantile(pinball_loss, 1-e)
    
    return (1-risk_aversion)*pinball_loss.mean() + risk_aversion*pinball_loss[cvar_mask].mean()


def pinball(prediction, target, quantiles):
    ''' Evaluates Probabilistic Forecasts, outputs average Pinball Loss for specified quantiles'''
    num_quant = len(quantiles)
    
    quantiles = np.array(quantiles)
    target_copy = target.copy().reshape(-1,1)
    pred_copy = prediction.copy()
    
    pinball_loss = np.maximum( (np.tile(target_copy, (1,num_quant)) - pred_copy)*quantiles, 
                              (pred_copy - np.tile(target_copy , (1,num_quant) ))*(1-quantiles))

    return pinball_loss.mean(0)

def eval_point_pred(predictions, actual, digits = 3, per_node = True):
    ''' Returns point forecast metrics: RMSE, MAE '''
    assert predictions.shape == actual.shape, "Shape missmatch"
    
    #mape = np.mean(abs( (predictions-actual)/actual) )
    if per_node:
        rmse = np.sqrt( np.square( predictions-actual).mean(axis=0) )
        mae = abs(predictions-actual).mean(axis=0)
    else:
        rmse = np.sqrt( np.square( predictions-actual).mean() )
        mae = abs(predictions-actual).mean()

    if digits is None:
        return rmse, mae
    else: 
        return rmse.round(digits), mae.round(digits)

def scaled_rmse(predictions, actual, S, digits = None, per_node = True):
    ''' Scaled NRMSE, see Di Modica, et al. '''
    assert predictions.shape == actual.shape, "Shape missmatch"
    nrmse = np.sqrt( np.square( (predictions-actual)/S.sum(axis=1) ).mean(axis=0) )
    return nrmse

def projection(predictions, ub = 1, lb = 0):
    predictions[predictions<lb] = lb
    predictions[predictions>ub] = ub
    return predictions

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
    ''' Weights SAA:
        scenarios: support/ fixed locations
        weights: the learned probabilities'''
    
    risk_aversion = kwargs['risk_aversion']
    e = kwargs['epsilon']
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
        
        '''
        m = gp.Model()
        m.setParam('OutputFlag', 0)

        # target variable
        offer = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'offer')
        loss = m.addMVar(n_scen, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'aux')
                        
        ### CVaR variables (follows Georghiou, Kuhn, et al.)
        beta = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name='VaR')
        zeta = m.addMVar(n_scen, vtype = gp.GRB.CONTINUOUS, lb = 0)  # Aux
        cvar = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
        
        m.addConstr(loss >= crit_quant*(target_scen - offer) )
        m.addConstr(loss >= (crit_quant-1)*(target_scen - offer) )

        # cvar constraints
        m.addConstr( zeta >=  -beta + loss )
        m.addConstr( cvar == beta + (1/e)*(zeta@weights))
        
        m.setObjective( (1-risk_aversion)*(weights@loss) + risk_aversion*cvar, gp.GRB.MINIMIZE)
        m.optimize()
        return offer.X
    '''
    elif problem == 'reg_trad':

        if weights.ndim == 1:
            # solve for a single instance
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
            # solve for multiple test observations/ declares gurobi model once for speed
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

def saa(scenarios, weights, quant):
    ''' Weights SAA:
        scenarios: support/ fixed locations
        weights: the learned probabilities'''
    n_scen = len(scenarios)
    m = gp.Model()
    m.setParam('OutputFlag', 0)

    # target variable
    x = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'offer')
    u = m.addMVar(n_scen, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'aux')
    
    m.addConstr(u >= quant*(scenarios-np.ones((n_scen, 1))@x))
    m.addConstr(u >= (quant-1)*(scenarios-np.ones((n_scen, 1))@x))
    
    m.setObjective( weights@u)
    m.optimize()

    return x.X

def find_weights(tree_model, train_feat, test_feat):
    ''' Function that returns the local weights of a tree-based model'''
    weights = np.zeros((len(test_feat), len(train_feat)))
    Leaf_nodes = tree_model.apply(train_feat)
    Index = tree_model.apply(test_feat) # Leaf node for test set
    nTrees = tree_model.n_estimators
    print('Retrieving weights...')
    for i in range(len(test_feat)):
        #New query point
        x0 = Index[i:i+1, :]
        
        #Find observations in terminal nodes/leaves (all trees)
        obs = 1*(x0.repeat(len(train_feat), axis = 0) == Leaf_nodes)
        #Cardinality of leaves
        cardinality = np.sum(obs, 0).reshape(-1,1).T.repeat(len(train_feat), 0)
        #Update weights
        weights[i,:] = ((obs/cardinality).sum(axis = 1)/nTrees)
    return weights

