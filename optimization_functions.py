# -*- coding: utf-8 -*-
"""
Optimization for optimal transport, univariate & multivariate case
    - Barycenter & Trimmed barycenter
    - Prescriptive barycenter
    - Barycenter-local data convex combination based on prescriptive divergence
    - Bilevel formulation of prescriptive barycenter

@author: akylas.stratigakos@minesparis.psl
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import gurobipy as gp
cd = os.path.dirname(__file__)  #Current directory
sys.path.append(cd)
    

##################
# Functions for univariate barycenter

def w_barycenter(emp_data, weights = None, support = np.arange(0, 1, .01), alpha = 1, p = 1):
    ''' Wasserstein barycenter barycenter from empirical distributions/ LP formulation in GUROBI
        emp_data: list of empirical 1d datasets,
        weights: weight of each distribution. If None, sets uniform weight
        support: finite support
        alpha: 1-% of probability trimmings'''

    if weights == None:
        weights = 1/len(emp_data)*np.ones(len(emp_data))
            
    n_supp = len(support)
    x_supp = support
    
    # create LP model in gurobi
    m = gp.Model()
    m.setParam('OutputFlag', 1)

    # find cost matrices, declare transporation matrices (one per each distribution)
    S = []
    gamma = []
    for i, emp_distr in enumerate(emp_data):
        cost_mat = np.array([np.power(np.linalg.norm(x_supp[i].reshape(-1,1) - emp_distr.reshape(-1,1), axis=1), p) for i in range(n_supp)])
        cost_mat/=cost_mat.sum()
        S.append(cost_mat)
        gamma.append(m.addMVar((n_supp, len(emp_distr)), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'OT matrix'))
    
        print(S[i].shape)
    # weights to be optimized
    a_wass = m.addMVar((n_supp), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'OT matrix')

    dist_W = []
    # Iterate over target distributions
    if alpha == 1:    
        for d, emp_distr in enumerate(emp_data):
            n_obs = len(emp_distr)
            # estimate distance
            m.addConstrs( gamma[d][i]@np.ones(n_obs) == a_wass[i] for i in range(n_supp))
            m.addConstrs( gamma[d][:,i]@np.ones(n_supp) == 1/n_obs for i in range(n_obs))
            
            dist_W.append(weights[d]*sum(gamma[d][i]@S[d][i] for i in range(n_supp)))    
    else:
        # barycenter of trimmed distributions
        for d, emp_distr in enumerate(emp_data):
            n_obs = len(emp_distr)
            # estimate distance
            m.addConstrs( gamma[d][i]@np.ones(n_obs) == a_wass[i] for i in range(n_supp))
            m.addConstrs( gamma[d][:,i]@np.ones(n_supp) <= (1/n_obs)/alpha for i in range(n_obs))
            m.addConstr( gamma[d].sum() == 1)
            
            dist_W.append(weights[d]*sum(gamma[d][i]@S[d][i] for i in range(n_supp)))    
        
    #m.addConstr(a_wass.sum()==1)
    m.setObjective( sum(dist_W) )
    m.optimize()

    print(m.ObjVal)
    
    return a_wass.X

def w_blvl_brc(emp_data, target_data, weights = None, lambda_=1, support = np.arange(0, 1, .01), 
                alpha = 1, p = 1, quantile = .5, reformulation = 'decision rule'):
    ''' Bilevel formulation of wassersteing barycenter with prescriptive cot
        **Not sure it works properly**
        emp_data: list of empirical 1d datasets,
        weights: weight of each distribution. If None, sets uniform weight
        support: finite support
        alpha: probability trimmings'''
    
    # check if any distribution has zero length and drop it 
    emp_clean_data = []
    for d in emp_data:
        if len(d) > 0:
            emp_clean_data.append(d)
    
    if weights == None:
        weights = 1/len(emp_clean_data)*np.ones(len(emp_clean_data))

    
    n_dist = len(emp_clean_data) # number of data sets        
    n_supp = len(support)
    x_supp = support
    
    # create LP model in gurobi
    m = gp.Model()
    m.setParam('OutputFlag', 0)

    # find cost matrices, declare transporation matrices (one per each distribution)
    S = []
    gamma = []
    prescr_cost = 0
    prescr_dist = []
    
    # weights to be optimized
    a_wass = m.addMVar((n_supp), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'OT matrix')

    for i, emp_distr in enumerate(emp_clean_data):
        n_obs = len(emp_distr)
        # estimate distance
        dist_mat = np.array([np.power(np.linalg.norm(x_supp[i].reshape(-1,1) - emp_distr.reshape(-1,1), axis=1), p) for i in range(n_supp)])
        #dist_mat/=dist_mat.sum()
        
        total_cost = dist_mat 
        S.append(total_cost)
        gamma.append(m.addMVar((n_supp, len(emp_distr)), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'OT matrix'))

    # Estimate prescriptive cost on target distribution
    # decision variable from derived from the barycenter
    z = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'OT matrix')
    xi = m.addMVar(len(target_data), vtype = gp.GRB.CONTINUOUS, lb = 0)
    prescr_cost = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = 0)
    
    m.addConstr(xi >= quantile*(target_data - np.ones((len(target_data),1))@z))
    m.addConstr(xi >= (quantile-1)*(target_data - np.ones((len(target_data),1))@z))
    m.addConstr(prescr_cost == xi.sum())
    #m.addConstr(np.ones((len(target_data),1)) @prescr_cost >= xi)
    
    
    if reformulation == 'decision rule':
        # decision rule for z/ coupling of the two problems
        m.addConstr(z == a_wass@x_supp)
    
        # prescriptive cost (from learned distribution)
        # !!!! SAA solution is known; the goal is to derive distribution that min. distance (MSE)
        # for the resulting distribution; the learned solution z = arg min SAA problem for 
        # the **learned** distribution    
    elif reformulation == 'analytical':
        ### analytical solution for the lower level arg min problem
        # indicator variable for each quantile
        q_ind = m.addMVar(len(x_supp), vtype = gp.GRB.BINARY, lb = 0)

        #col_ = np.where(a_blvl_brc.cumsum() >= q)[0][0]
        #print('Solution found: ' + str(supp[col_]))                    
        #a_prescr_list[k][j,:] = a_prescr_weights
        
    elif reformulation == 'KKT':
        ### KKT conditions for the SAA problem
        # additional variables
        residual = m.addMVar(len(x_supp), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'aux')

        zeta_i = m.addMVar(len(x_supp), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'aux')
        mu = m.addMVar(len(x_supp), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'dual-1')
        heta = m.addMVar(len(x_supp), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'dual-2')


        m.addConstr( residual == x_supp - np.ones((len(x_supp),1))@z )
    
        # primal feasibility
        m.addConstrs( zeta_i[i] >= quantile*( x_supp[i] - z) for i in range(len(x_supp)))
        m.addConstrs( zeta_i[i] >= (quantile-1)*( x_supp[i] - z) for i in range(len(x_supp)))
        # stationarity
        m.addConstr( quantile*heta.sum() + (quantile-1)*mu.sum() == 0)
        m.addConstr( a_wass == mu + heta)

        # compl. slackness
        u_dn = m.addMVar(len(x_supp), vtype = gp.GRB.BINARY)
        u_up = m.addMVar(len(x_supp), vtype = gp.GRB.BINARY)
        big_M = 1e4
        for i in range(len(x_supp)):
            
            m.addConstr( heta[i] <= u_up[i]*big_M)
            m.addConstr( zeta_i[i] - quantile*( x_supp[i] - z) <= (1-u_up[i]) *big_M )
    
            m.addConstr( mu[i] <= u_dn[i]*big_M)
            m.addConstr( zeta_i[i] - (quantile-1)*( x_supp[i] - z) <= (1-u_dn[i]) *big_M )        
    
    # Wasserstein distances cost
    dist_W = []
    # Iterate over target distributions
    if alpha == 1:    
        for d, emp_distr in enumerate(emp_clean_data):
            n_obs = len(emp_distr)
            # estimate distance
            m.addConstrs( gamma[d][i]@np.ones(n_obs) == a_wass[i] for i in range(n_supp))
            m.addConstrs( gamma[d][:,i]@np.ones(n_supp) == 1/n_obs for i in range(n_obs))
            
            dist_W.append(weights[d]*sum(gamma[d][i]@S[d][i] for i in range(n_supp)))    
    else:
        # barycenter of trimmed distributions
        for d, emp_distr in enumerate(emp_clean_data):
            n_obs = len(emp_distr)
            # estimate distance
            m.addConstrs( gamma[d][i]@np.ones(n_obs) == a_wass[i] for i in range(n_supp))
            m.addConstrs( gamma[d][:,i]@np.ones(n_supp) <= (1/n_obs)/alpha for i in range(n_obs))
            m.addConstr( gamma[d].sum() == 1)
            
            dist_W.append(weights[d]*sum(gamma[d][i]@S[d][i] for i in range(n_supp)))    
        
    m.addConstr(a_wass.sum()==1)

    m.setObjective( lambda_*sum(dist_W) + (1-lambda_)*prescr_cost)
    m.optimize()
    
    return a_wass.X, z.X

def univ_prescr_brc(emp_data, target_data, weights = None, lambda_= 1, 
                    support = np.arange(0, 1, .01), alpha = 1, p = 1, quantile = .5, d_idx = None):
    ''' Convex combination of W barycenter and local data, based on prescriptive divergence
        emp_data: list of empirical 1d datasets,
        weights: weight of each distribution. If None, sets uniform weight
        support: finite support
        alpha: probability trimmings
        lamba_: hyperparameter to control the trade-off'''
    # check if any distribution has zero length and drop it 
    emp_clean_data = []
    for d in emp_data:
            if len(d) > 0:
                emp_clean_data.append(d)

    if weights == None:
        weights = 1/len(emp_clean_data)*np.ones(len(emp_clean_data))

    
    n_dist = len(emp_clean_data) # number of data sets       
    x_supp = support
    n_supp = len(x_supp)
        
    # create LP model in gurobi
    m = gp.Model()
    m.setParam('OutputFlag', 0)

    # target variable: weights for each location
    a_wass = m.addMVar((n_supp), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'location weights')
    prescr_cost = m.addMVar((n_supp, len(target_data)), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'location weights')

    # find cost matrices, declare transporation matrices (one per each distribution)
    S = []
    gamma = []
    
    # pairwise distance for cost matrix
    for i, emp_distr in enumerate(emp_clean_data):
        n_obs = len(emp_distr)

        # estimate distance
        dist_mat = [np.power(np.linalg.norm(x_supp[i].reshape(-1,1) - emp_distr.reshape(-1,1), axis=1), p) for i in range(n_supp)]
        dist_mat = np.array(dist_mat)
        #dist_mat/=dist_mat.sum()        
        
        S.append(dist_mat)
        gamma.append(m.addMVar((n_supp, len(emp_distr)), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'OT matrix'))
    # pairwise *expected* prescriptive cost for the fixed locations    
    upper_constr = None
    if upper_constr == None:
        empirical_loss = 0
    else:
        # solution for each empirical scenario
        emp_sol = target_data.copy()
        emp_sol[target_data>= upper_constr] = upper_constr

        empirical_loss = np.maximum(quantile*(target_data - emp_sol), (quantile-1)*(target_data - emp_sol))
        
    for i, x_i in enumerate(x_supp):
        if upper_constr == None:
            newsvend_loss = np.maximum(quantile*(target_data - x_i), (quantile-1)*(target_data - x_i)).reshape(-1,1)
        else:
            brc_sol = x_i
            if brc_sol>=upper_constr:
                brc_sol = upper_constr
            brc_distr_loss = np.maximum(quantile*(target_data - brc_sol), (quantile-1)*(target_data - brc_sol))
            newsvend_loss = (brc_distr_loss - empirical_loss).reshape(-1,1)

        assert(newsvend_loss.all()>=0)
        # (normalization is optional)
        #m.addConstr(prescr_cost[i,:] ==  newsvend_loss@a_wass[i])
        m.addConstr(prescr_cost[i,:] == np.diag(newsvend_loss.reshape(-1))@gamma[d_idx][i])
                
    # sum of Wasserstein distances
    dist_W = []
    # Iterate over target distributions
    if alpha == 1:    
        for d, emp_distr in enumerate(emp_clean_data):
            n_obs = len(emp_distr)
            # estimate distance
            m.addConstrs( gamma[d][i]@np.ones(n_obs) == a_wass[i] for i in range(n_supp))
            m.addConstrs( gamma[d][:,i]@np.ones(n_supp) == 1/n_obs for i in range(n_obs))
            
            dist_W.append(weights[d]*sum(gamma[d][i]@S[d][i] for i in range(n_supp)))    
    else:
        # barycenter of trimmed distributions
        for d, emp_distr in enumerate(emp_clean_data):
            n_obs = len(emp_distr)
            # estimate distance
            m.addConstrs( gamma[d][i]@np.ones(n_obs) == a_wass[i] for i in range(n_supp))
            m.addConstrs( gamma[d][:,i]@np.ones(n_supp) <= (1/n_obs)/alpha for i in range(n_obs))
            m.addConstr( gamma[d].sum() == 1)
            
            dist_W.append(weights[d]*sum(gamma[d][i]@S[d][i] for i in range(n_supp)))    
        
    m.addConstr(a_wass.sum()==1)

    m.setObjective( lambda_*sum(dist_W) + (1-lambda_)*prescr_cost.sum())
    m.optimize()
        
    return a_wass.X

def prescr_div_brc(emp_data, weights = None, lambda_= 1, 
                    support = np.arange(0, 1, .01), alpha = 1, p = 1, quantile = .5):
    ''' Barycenter where the cost matrix is the convex combination of distance and prescriptive divergence
        emp_data: list of empirical 1d datasets,
        weights: weight of each distribution. If None, sets uniform weight
        support: finite support
        alpha: probability trimmings'''
    # check if any distribution has zero length and drop it 
    emp_clean_data = []
    for d in emp_data:
            if len(d) > 0:
                emp_clean_data.append(d)

    if weights == None:
        weights = 1/len(emp_clean_data)*np.ones(len(emp_clean_data))

    
    x_supp = support
    n_supp = len(x_supp)
        
    # create LP model in gurobi
    m = gp.Model()
    m.setParam('OutputFlag', 0)

    # target variable: weights for each location
    a_wass = m.addMVar((n_supp), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'location weights')

    # transportation plans for each distribution
    gamma = [(m.addMVar((n_supp, len(emp_distr)), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'OT matrix')) for emp_distr in emp_clean_data]
    
    # Estimate cost matrices
    S = []

    # loop over distributions

    for i, emp_distr in enumerate(emp_clean_data):
        n_obs = len(emp_distr)

        # estimate norm distance
        dist_mat = [np.power(np.linalg.norm(x_supp[i].reshape(-1,1) - emp_distr.reshape(-1,1), axis=1), p) for i in range(n_supp)]
        dist_mat = np.array(dist_mat)
        #dist_mat/=dist_mat.sum()        

        # estimate prescriptive divergence     
        prescr_cost_mat = [np.maximum(quantile*(emp_distr - x_supp[i]), +(quantile-1)*(emp_distr - x_supp[i])) for i in range(n_supp)]          
        prescr_cost_mat = np.array(prescr_cost_mat  )

        total_cost = lambda_*dist_mat + (1-lambda_)*prescr_cost_mat
        S.append(total_cost)
                
    # sum of Wasserstein distances
    dist_W = []
    # Iterate over target distributions
    if alpha == 1:    
        for d, emp_distr in enumerate(emp_clean_data):
            n_obs = len(emp_distr)
            # estimate distance
            m.addConstrs( gamma[d][i]@np.ones(n_obs) == a_wass[i] for i in range(n_supp))
            m.addConstrs( gamma[d][:,i]@np.ones(n_supp) == 1/n_obs for i in range(n_obs))
            
            dist_W.append(weights[d]*sum(gamma[d][i]@S[d][i] for i in range(n_supp)))    
    else:
        # barycenter of trimmed distributions
        for d, emp_distr in enumerate(emp_clean_data):
            n_obs = len(emp_distr)
            # estimate distance
            m.addConstrs( gamma[d][i]@np.ones(n_obs) == a_wass[i] for i in range(n_supp))
            m.addConstrs( gamma[d][:,i]@np.ones(n_supp) <= (1/n_obs)/alpha for i in range(n_obs))
            m.addConstr( gamma[d].sum() == 1)
            
            dist_W.append(weights[d]*sum(gamma[d][i]@S[d][i] for i in range(n_supp)))    
        
    m.addConstr(a_wass.sum()==1)

    m.setObjective( sum(dist_W))
    m.optimize()
        
    return a_wass.X


##################
# Functions to estimate barycenter of multivariate/joint distributions


def w__multobj_brc(emp_data, z_saa, weights = None, lambda_=1, 
                   support = np.arange(0, 1, .01), 
                   alpha = 1, p = 1, quantile = .5):
    ''' Estimates the w1 or w2 barycenter from empirical distributions with exact LP/GUROBI
        emp_data: list of empirical 1d datasets,
        weights: weight of each distribution. If None, sets uniform weight
        support: finite support
        alpha: probability trimmings'''

    n_dist = len(emp_data) # number of data sets
    if weights == None:
        weights = 1/len(emp_data)*np.ones(len(emp_data))
        
    
    n_supp = len(support)
    x_supp = support
    
    # create LP model in gurobi
    m = gp.Model()
    m.setParam('OutputFlag', 0)

    # find cost matrices, declare transporation matrices (one per each distribution)
    S = []
    gamma = []
    prescr_cost = 0
    prescr_dist = []
    
    z = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'OT matrix')
    xi = [m.addMVar(len(d), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'OT matrix') for d in emp_data]

    for i, emp_distr in enumerate(emp_data):
        n_obs = len(emp_distr)

        # estimate distance
        dist_mat = np.array([np.power(np.linalg.norm(x_supp[i].reshape(-1,1) - emp_distr.reshape(-1,1), axis=1, ord=p), p) for i in range(n_supp)])
        dist_mat/=dist_mat.sum()
        # Estimate prescriptive cost (for observations)
        #m.addConstr( xi[i] >= quantile*( emp_distr - np.ones((n_obs,1))@z) )
        #m.addConstr( xi[i] >= -(1-quantile)*( emp_distr - np.ones((n_obs,1))@z) )
        #prescr_cost += xi[i].sum()

        # Estimate prescriptive divergence
        # for each support point -> z_opt = x_i -> find distance from
        # actual observations over all samples
        #for xi in x_supp:
        #    prescr_cost += np.sum(np.maximum(quantile*( emp_distr - xi), (quantile-1)*( emp_distr - xi)))
        prescr_dist = [(np.maximum(quantile*( emp_distr - x_supp[i]), (quantile-1)*( emp_distr - x_supp[i]))) for i in range(n_supp)]
        prescr_dist = np.array(prescr_dist)
        
        total_cost = lambda_*dist_mat + (1-lambda_)*prescr_dist
        S.append(total_cost)
        gamma.append(m.addMVar((n_supp, len(emp_distr)), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'OT matrix'))
    
    # weights to be optimized
    a_wass = m.addMVar((n_supp), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'OT matrix')
    
    # SAA empirical cost *given* the saa solution
    saa_cost = 0
    for i, emp_distr in enumerate(emp_data):
        saa_cost += np.sum(np.maximum(quantile*(emp_distr - z_saa), (quantile-1)*(emp_distr - z_saa)))


    dist_W = []
    # Iterate over target distributions
    if alpha == 1:    
        for d, emp_distr in enumerate(emp_data):
            n_obs = len(emp_distr)
            # estimate distance
            m.addConstrs( gamma[d][i]@np.ones(n_obs) == a_wass[i] for i in range(n_supp))
            m.addConstrs( gamma[d][:,i]@np.ones(n_supp) == 1/n_obs for i in range(n_obs))
            
            dist_W.append(weights[d]*sum(gamma[d][i]@S[d][i] for i in range(n_supp)))    
    else:
        # barycenter of trimmed distributions
        for d, emp_distr in enumerate(emp_data):
            n_obs = len(emp_distr)
            # estimate distance
            m.addConstrs( gamma[d][i]@np.ones(n_obs) == a_wass[i] for i in range(n_supp))
            m.addConstrs( gamma[d][:,i]@np.ones(n_supp) <= (1/n_obs)/alpha for i in range(n_obs))
            m.addConstr( gamma[d].sum() == 1)
            
            dist_W.append(weights[d]*sum(gamma[d][i]@S[d][i] for i in range(n_supp)))    
        
    m.addConstr(a_wass.sum()==1)

    m.setObjective( sum(dist_W) )
    m.optimize()
    
    return a_wass.X, z.X

def w_joint_barycenter(emp_data, weights = None, p = 2, support_locations = None, euler_step = .05, alpha = 1, verbose = 0):
    ''' 
    Wasserstein barycenter of multivariate distributions
        emp_data: list of empirical datasets (same number of variables for all)
        weights: weight of each distribution. If None, then sets uniform weight
        p: distance norm
        support_locations: the fixed locations of the barycenter. If None, then sets in range [0,1] with step == euler_step
        alpha: probability trimmings for partial mass transportation        
    '''

    n_dist = len(emp_data) # number of data sets
    weights = None # weight of each data set

    if weights == None:
        weights = 1/len(emp_data)*np.ones(len(emp_data))

    n_var = emp_data[0].shape[1] # number of variables per data set (same)

    # the support location of each variable
    if support_locations == 'union':
        pooled_data = np.concatenate(emp_data)
        x_joint_supp = np.unique(pooled_data.round(2), axis=0).T
    else:
        if support_locations == None:
            # perform euler discretazation
            x_supp = [np.arange(0, 1, euler_step) for var_ in range(n_var)] 
        else:
            x_supp = support_locations
    
        # meshgrid of the cartesian product
        xt, xr = np.meshgrid(x_supp[0], x_supp[1])
        x_joint_supp = np.array([xt.ravel(), xr.ravel()])
        
    # number of location points
    n_supp_locs = x_joint_supp.shape[1]

    ### construct LP model
    m = gp.Model()
    m.setParam('OutputFlag', verbose)

    # loop over data sets, estimate cost matrices, declare variables
    S = []  # list of cost matrices
    gamma = []  # list of transportation matrices

    # loop over data sets
    for j, emp_distr in enumerate(emp_data):
        cost_mat = []
        print('Data set: ', j)
        # loop over locations and estimate sq eucl distance
        for i in range(n_supp_locs):
            temp_vec = x_joint_supp[:,i] 
            # vector distance
            eucl_dist = np.linalg.norm(temp_vec - emp_distr, axis=1)
            cost_mat.append(np.power(eucl_dist, p))
            
            # newsvendor loss
            #eucl_dist = np.maximum(.2*(temp_vec[0] - emp_distr[:,0]),
            #                   -.8*(temp_vec[0] - emp_distr[:,0]))
            #cost_mat.append(eucl_dist)

        # store cost matrix
        cost_mat = np.array(cost_mat)
        cost_mat /= cost_mat.sum()
        S.append(cost_mat)
        gamma.append(m.addMVar((n_supp_locs, len(emp_distr)), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'OT matrix'))


    # target variable: weights for each location
    a_wass = m.addMVar((n_supp_locs), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'OT matrix')

    dist_W = []
    # Iterate over target distributions
    if alpha == 1:
        for d, emp_distr in enumerate(emp_data):
            print('Data set: ', d)
            n_obs = len(emp_distr)
            # estimate distance
            m.addConstrs( gamma[d][i]@np.ones(n_obs) == a_wass[i] for i in range(n_supp_locs))
            m.addConstrs( gamma[d][:,i]@np.ones(n_supp_locs) == 1/n_obs for i in range(n_obs))
            
            dist_W.append(weights[d]*sum(gamma[d][i]@S[d][i] for i in range(n_supp_locs)))    
    else:
        for d, emp_distr in enumerate(emp_data):
            print('Data set: ', d)
            n_obs = len(emp_distr)
            # estimate distance
            m.addConstrs( gamma[d][i]@np.ones(n_obs) == a_wass[i] for i in range(n_supp_locs))
            m.addConstrs( gamma[d][:,i]@np.ones(n_supp_locs) <= (1/n_obs)/alpha for i in range(n_obs))
            m.addConstr( gamma[d].sum() == 1 )
            
            dist_W.append(weights[d]*sum(gamma[d][i]@S[d][i] for i in range(n_supp_locs)))    

    m.addConstr(a_wass.sum()==1)
    m.setObjective( sum(dist_W) )
    m.optimize()
    try:
        return a_wass.X, x_joint_supp, x_supp
    except:
        return a_wass.X, x_joint_supp, []
    
def prescr_global_brc(emp_data, quantile = .5, lambda_ = 1, weights = None, p = 2, 
                        support_locations = None, euler_step = .05, alpha = 1, verbose = 0, target_data = None, 
                        d_idx = None):
    ''' 
    Barycenter with prescriptive divergence/multivariate. Minimizes decision cost distance for **every** distribution.
        emp_data: list of empirical datasets (same number of variables for all)
        lambda_: trade-off in cost matrix
        quantile: for the newsvendor problem
        weights: weight of each distribution. If None, then sets uniform weight
        p: distance norm
        support_locations: the fixed locations of the barycenter. If None, then sets in range [0,1] with step == euler_step
        alpha: probability trimmings for partial mass transportation        
    '''

    n_dist = len(emp_data) # number of data sets
    weights = None # weight of each data set

    if weights == None:
        weights = 1/len(emp_data)*np.ones(len(emp_data))

    n_var = emp_data[0].shape[1] # number of variables per data set (same)

    # the support location of each variable
    if support_locations == 'union':
        pooled_data = np.concatenate(emp_data)
        x_joint_supp = np.unique(pooled_data.round(2), axis=0).T
        contextual_supp = x_joint_supp[1]
    else:
        if support_locations == None:
            # perform euler discretazation
            x_supp = [np.arange(0, 1 + euler_step, euler_step) for var_ in range(n_var)] 
        else:
            x_supp = support_locations
    
        # meshgrid of the cartesian product
        xt, xr = np.meshgrid(x_supp[0], x_supp[1])
        x_joint_supp = np.array([xt.ravel(), xr.ravel()])
    
        contextual_supp = x_supp[1]
    # find bin for obs of contextual inf
    #binning = np.array([np.where(x_i < contextual_supp)[0][0] for emp_distr in emp_data for x_i in emp_distr[:,1] ])
    #contextual_supp_freq = np.zeros(len(contextual_supp))
    #for j in range(len(contextual_supp_freq)):
    #    contextual_supp_freq[j] = (binning==j).sum()/len(binning)
    
    # number of location points
    n_supp_locs = x_joint_supp.shape[1]

    ### construct LP model
    m = gp.Model()
    m.setParam('OutputFlag', verbose)

    # loop over data sets, estimate cost matrices, declare variables
    S = []  # list of cost matrices
    gamma = []  # list of transportation matrices
    prescr_S = []  # list of *prescriptive* cost matrices
    
    # target variable: weights for each location
    a_wass = m.addMVar((n_supp_locs), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'location weights')
        
    # loop over data sets
    for j, emp_distr in enumerate(emp_data):
        n_obs = len(emp_distr)
        gamma.append(m.addMVar((n_supp_locs, len(emp_distr)), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'OT matrix'))
        prescr_S.append(m.addMVar((n_supp_locs, len(emp_distr)), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'OT matrix'))

        distance_cost_mat = []
        prescr_cost_mat = []
        print('Data set: ', j)
        
        # loop over locations and estimate sq eucl distance
        for i in range(n_supp_locs):
            temp_vec = x_joint_supp[:,i] 

            # estimate pairwise distance and raise to power of p
            pair_dist = np.linalg.norm(temp_vec - emp_distr, axis=1, ord = 2)
            pair_dist_p = np.power(pair_dist, p)

            distance_cost_mat.append(pair_dist_p)
            
            
            # newsvendor loss for each observed sample (the solution of the prob. is trivial)
            newsvend_loss = np.maximum(quantile*(emp_distr[:,0] - temp_vec[0]), +(quantile-1)*(emp_distr[:,0] - temp_vec[0]))\
                + np.sqrt(np.square(temp_vec[1] - emp_distr[:,1]))
                       
            prescr_cost_mat.append(newsvend_loss)
            
        # distance cost matrix: normalize and store 
        distance_cost_mat = np.array(distance_cost_mat)
        #distance_cost_mat /= distance_cost_mat.sum()
        
        prescr_cost_mat = np.array(prescr_cost_mat)
        #prescr_cost_mat /= prescr_cost_mat.sum()
        
        total_cost = lambda_*distance_cost_mat + (1-lambda_)*prescr_cost_mat
        S.append(total_cost)
        
    dist_W = []
    # Iterate over target distributions
    if alpha == 1:
        for d, emp_distr in enumerate(emp_data):
            print('Data set: ', d)
            n_obs = len(emp_distr)
            # estimate distance
            m.addConstrs( gamma[d][i]@np.ones(n_obs) == a_wass[i] for i in range(n_supp_locs))
            m.addConstrs( gamma[d][:,i]@np.ones(n_supp_locs) == 1/n_obs for i in range(n_obs))
            dist_W.append(weights[d]*sum(gamma[d][i]@S[d][i] for i in range(n_supp_locs)))    
                    
    else:
        for d, emp_distr in enumerate(emp_data):
            print('Data set: ', d)
            n_obs = len(emp_distr)
            # estimate distance
            m.addConstrs( gamma[d][i]@np.ones(n_obs) == a_wass[i] for i in range(n_supp_locs))
            m.addConstrs( gamma[d][:,i]@np.ones(n_supp_locs) <= (1/n_obs)/alpha for i in range(n_obs))
            m.addConstr( gamma[d].sum() == 1 )
                        
            dist_W.append(weights[d]*sum(gamma[d][i]@S[d][i] for i in range(n_supp_locs)))    
    
    m.addConstr(a_wass.sum()==1)
    #m.setObjective(lambda_*sum(dist_W) + (1-lambda_)*prescr_cost.sum())
    m.setObjective(sum(dist_W) )
    
    m.optimize()
    try:
        return a_wass.X, x_joint_supp, x_supp
    except:
        return a_wass.X, x_joint_supp, []
    
def prescr_local_brc(emp_data, quantile = .5, lambda_ = 1, weights = None, p = 2, 
                        support_locations = None, euler_step = .05, alpha = 1, verbose = 0, target_data = None, 
                        d_idx = None):
    ''' 
    Barycenter with prescriptive divergence/multivariate. Minimizes decision cost distance of the **target** distribution.
        emp_data: list of empirical datasets (same number of variables for all)
        lambda_: trade-off in cost matrix
        quantile: for the newsvendor problem
        weights: weight of each distribution. If None, then sets uniform weight
        p: distance norm
        support_locations: the fixed locations of the barycenter. If None, then sets in range [0,1] with step == euler_step
        alpha: probability trimmings for partial mass transportation        
        d_idx: index of target distribution in list emp_data
    '''

    weights = None # weight of each data set

    if weights == None:
        weights = 1/len(emp_data)*np.ones(len(emp_data))

    n_var = emp_data[0].shape[1] # number of variables per data set (same)

    # the support location of each variable
    if support_locations == 'union':
        pooled_data = np.concatenate(emp_data)
        x_joint_supp = np.unique(pooled_data.round(2), axis=0).T
        contextual_supp = x_joint_supp[1]
    else:
        if support_locations == None:
            # perform euler discretazation
            x_supp = [np.arange(0, 1 + euler_step, euler_step) for var_ in range(n_var)] 
        else:
            x_supp = support_locations
    
        # meshgrid of the cartesian product
        xt, xr = np.meshgrid(x_supp[0], x_supp[1])
        x_joint_supp = np.array([xt.ravel(), xr.ravel()])
    
    # find bin for obs of contextual inf
    # number of location points
    n_supp_locs = x_joint_supp.shape[1]

    ### construct LP model
    m = gp.Model()
    m.setParam('OutputFlag', verbose)

    # loop over data sets, estimate cost matrices, declare variables
    S = []  # list of cost matrices
    gamma = []  # list of transportation matrices
    prescr_S = []  # list of *prescriptive* cost matrices
    
    # target variable: weights for each location
    a_wass = m.addMVar((n_supp_locs), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'location weights')
        
    # loop over data sets
    for j, emp_distr in enumerate(emp_data):
        n_obs = len(emp_distr)
        gamma.append(m.addMVar((n_supp_locs, len(emp_distr)), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'OT matrix'))
        prescr_S.append(m.addMVar((n_supp_locs, len(emp_distr)), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'OT matrix'))

        distance_cost_mat = []
        print('Data set: ', j)
        
        # loop over locations and estimate sq eucl distance
        for i in range(n_supp_locs):
            temp_vec = x_joint_supp[:,i] 

            # estimate pairwise distance and raise to power of p
            pair_dist = np.linalg.norm(temp_vec - emp_distr, axis=1, ord = 2)
            pair_dist_p = np.power(pair_dist, p)

            distance_cost_mat.append(pair_dist_p)
        # distance cost matrix: normalize and store 
        distance_cost_mat = np.array(distance_cost_mat)
        distance_cost_mat /= distance_cost_mat.sum()
                
        S.append(distance_cost_mat)
    
    ############# Prescriptive cost of specific data set
    prescr_cost = m.addMVar((n_supp_locs, len(target_data)), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'location weights')
    for i in range(n_supp_locs):
        temp_vec = x_joint_supp[:,i]

        newsvend_loss = np.maximum(quantile*(target_data[:,0] - temp_vec[0]), +(quantile-1)*(target_data[:,0] - temp_vec[0]))\
            + np.sqrt(np.square(temp_vec[1] - target_data[:,1]))
        #newsvend_loss = newsvend_loss.reshape(-1,1)
        m.addConstr(prescr_cost[i,:] == np.diag(newsvend_loss.reshape(-1))@gamma[d_idx][i])
        #m.addConstr(prescr_cost[i] ==  newsvend_loss@a_wass[i] )
    ############
    
    dist_W = []
    # Iterate over target distributions
    if alpha == 1:
        for d, emp_distr in enumerate(emp_data):
            print('Data set: ', d)
            n_obs = len(emp_distr)
            # estimate distance
            m.addConstrs( gamma[d][i]@np.ones(n_obs) == a_wass[i] for i in range(n_supp_locs))
            m.addConstrs( gamma[d][:,i]@np.ones(n_supp_locs) == 1/n_obs for i in range(n_obs))
            dist_W.append(weights[d]*sum(gamma[d][i]@S[d][i] for i in range(n_supp_locs)))    
                    
    else:
        for d, emp_distr in enumerate(emp_data):
            print('Data set: ', d)
            n_obs = len(emp_distr)
            # estimate distance
            m.addConstrs( gamma[d][i]@np.ones(n_obs) == a_wass[i] for i in range(n_supp_locs))
            m.addConstrs( gamma[d][:,i]@np.ones(n_supp_locs) <= (1/n_obs)/alpha for i in range(n_obs))
            m.addConstr( gamma[d].sum() == 1 )
                        
            dist_W.append(weights[d]*sum(gamma[d][i]@S[d][i] for i in range(n_supp_locs)))    
        
    m.addConstr(a_wass.sum()==1)
    m.setObjective(lambda_*sum(dist_W) + (1-lambda_)*prescr_cost.sum())    
    m.optimize()
    try:
        return a_wass.X, x_joint_supp, x_supp
    except:
        return a_wass.X, x_joint_supp, []
    