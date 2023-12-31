# -*- coding: utf-8 -*-
"""
Functions related to optimal transport

@author: akylas.stratigakos
"""
import numpy as np
import gurobipy as gp
import matplotlib.pyplot as plt

############### Auxiliary functions 

def discrete_cdf(target_vals, x_observations, w = []):
    ''' CDF from weighted discrete distribution
        target_vals: array to evaluate, 
        x_observations: discrete observations, 
        w: observation weights, if == [], then set to uniform, 
        
        Returns the probability of x_observations <= target_vals'''
        
    if len(w) == 0:
        # dirac distribution 
        w = np.ones(len(x_observations))*(1/len(x_observations))
    
    ind_sort = np.argsort(x_observations)
    x_sort = x_observations.copy()[ind_sort]
    w_sort = w[ind_sort]
    
    prob_vals = np.zeros(len(target_vals))
    for i, t in enumerate(target_vals):
        if t <= x_sort.min():
            prob_vals[i] = 0
        else:
            # estimate cdf: F(t) = Prob(X<=t)
            prob_vals[i] = w_sort.cumsum()[np.where(x_sort<=t)[0][-1]]

    return prob_vals

def inverted_cdf(target_probs, x_observations, w = []):
    ''' Inverted CDF from weighted discrete distribution
        target_probs: array with values to evaluate probabilities/quantiles/percentiles, 
        x_observations: discrete observations, 
        w: observation weights, if == [], then set to uniform, 
        
        Returns the probability of x_observations <= target_vals'''
        
    if len(w) == 0:
        # dirac distribution 
        w = np.ones(len(x_observations))*(1/len(x_observations))
    
    ind_sort = np.argsort(x_observations.reshape(-1))
    x_sort = x_observations.copy()[ind_sort]
    w_sort = w[ind_sort]
    
    x_vals = np.zeros(len(target_probs))
    for i, prob in enumerate(target_probs):        
        if prob == 0:
            # return min value
            x_vals[i] = x_observations.min()
        elif (prob == 1)or((w_sort.cumsum() < prob).all()):
            # return max value
            x_vals[i] = x_observations.max()
        else:
            # inv cdf: Q(prob) = inf{x: F(x) >= prob}                
            q1 = x_sort[np.where(w_sort.cumsum() >= prob)[0][0]]
            x_vals[i] = q1
    return x_vals

############################## Main functions 

def w_barycenter_LP(emp_locs, loc_prob, w_coordinates = [], support = np.arange(0, 1+.01, .01), p = 2, return_plans = False):
    ''' Wasserstein barycenter barycenter from empirical distributions/ LP formulation in GUROBI
        emp_locs: list of arrays with empirical locations, 1D datasets
        loc_prob: list of arrays with location probabilities in emp_locs
        w_coordinates: barycentric coordinates. If None, sets uniform weight
        support: fixed support locations for the barycenter
        p: p-Wasserstein distance
        return_plans: if True, also return transportation plans
        
        Returns a probability vector for locations at support'''

    if len(w_coordinates)==0:
        w_coordinates = 1/len(emp_locs)*np.ones(len(emp_locs))
            
    n_supp = len(support)
    x_supp = support
    
    # create LP model in gurobi
    m = gp.Model()
    m.setParam('OutputFlag', 0)

    # find cost matrices, declare transporation matrices (one per each distribution)
    S = []
    gamma = []
    for i, emp_distr in enumerate(emp_locs):

        cost_mat = np.array([np.power(np.linalg.norm(x_supp[j].reshape(-1,1) - emp_distr.reshape(-1,1), axis=1), p) for j in range(n_supp)])
        #cost_mat/=cost_mat.sum()

        S.append(cost_mat)
        gamma.append(m.addMVar((n_supp, len(emp_distr)), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'OT matrix'))
        
    # weights to be optimized
    a_wass = m.addMVar((n_supp), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'OT matrix')
    dist_W = []
    # Iterate over target distributions
    for d, emp_distr in enumerate(emp_locs):
        n_obs = len(emp_distr)
        # estimate distance
        m.addConstr( gamma[d]@np.ones(n_obs) == a_wass )
        m.addConstr( gamma[d].T@np.ones(n_supp) == loc_prob[d])

        dist_W.append(w_coordinates[d]*sum(gamma[d][i]@S[d][i] for i in range(n_supp)))    
        
    #m.addConstr(a_wass.sum()==1)
    m.setObjective( sum(dist_W) )
    m.optimize()

    print(m.ObjVal)
    if return_plans:
        return a_wass.X, [g.X for g in gamma]
    else:
        return a_wass.X

def optimal_barycenter_weights(expert_probs, empirical_data, support_locations, p = 2, prob_dx = .01):
    ''' Find the optimal weights (barycentric coordinates) to aggregate a number of experts (prob. forecasts)
        Args
            - expert_probs: the historical probabilistic forecasts, list of numpys
            - observed_probs: empirical probability of each location, true data
            - support_locations: for simplicity, we consider the discrete case
            - p: p-Wasserstein metric
            - prob_dx: step to approximate the quantile function'''

    n_obs = len(empirical_data)
    target_quantiles = np.arange(0, 1+prob_dx, prob_dx)
    n_experts = len(expert_probs)
    
    ### turn PDFs to quantile functions
    emprical_q_funct = inverted_cdf(target_quantiles, empirical_data)
    
    # for each expert, turn historical predictions to quantile functions
    Q_hat = []
    for s in range(n_experts):
        q_hat = []
        for i in range(n_obs):
            temp_q_hat = inverted_cdf(target_quantiles, support_locations, w = expert_probs[s][i])
            q_hat.append(temp_q_hat)
        q_hat = np.array(q_hat)
        Q_hat.append(q_hat)
    
    
    ### Find weights lambda that minimize the wasserstein distance in the training set from the emprical inverse c.d.f.
    
    m = gp.Model()
    m.setParam('OutputFlag', 1)
    
    lambda_ = m.addMVar((n_experts), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'barycentric coordinates')    
    Q_barycenter = m.addMVar( (n_obs, len(target_quantiles)), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'barycenter')    
    wass_dist_sq_i = m.addMVar( n_obs, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'barycenter')    

    m.addConstr( lambda_.sum() == 1)
    
    m.addConstrs( Q_barycenter[i,:] == sum([lambda_[j]*Q_hat[j][i] for j in range(n_experts)]) for i in range(n_obs) )
    m.addConstrs( wass_dist_sq_i[i] >=  (Q_barycenter[i] - emprical_q_funct)@(Q_barycenter[i] - emprical_q_funct) for i in range(n_obs))
    
    m.setObjective( sum(wass_dist_sq_i) )
    m.optimize()
    
    return lambda_.X

def wass_distance_LP(x1,x2, w1=[], w2=[], p = 1):
    ''' p-Wasserstein distance between x1, x2. Exact solution with Gurobi. 
        Returns the distance and the transportation plan.'''

    n1 = len(x1)
    n2 = len(x2)
    
    if len(w1)==0:
        w1 = np.ones(n1)*(1/n1)
    if len(w2)==0:
        w2 = np.ones(n2)*(1/n2)
            
    m = gp.Model()
    m.setParam('OutputFlag', 0)
        
    S = np.array([ np.linalg.norm(x1[i].reshape(-1,1) - x2.reshape(-1,1), axis=1, ord = 2)  for i in range(n1)])    
    S = np.power(S, p)
    
    gamma = m.addMVar((n1, n2), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'OT matrix')        
    m.addConstr( gamma@np.ones(n2) == w1)
    m.addConstr( gamma.T@np.ones(n1) == w2)
    
    m.addConstr( gamma.sum() == 1 )
    
    m.setObjective( sum(gamma[i]@S[i] for i in range(n1)) )
    m.optimize()
    
    w_distance = np.power(m.ObjVal, 1/p)
    
    return w_distance, gamma.X

def wass2_barycenter_1D(locs, locs_prob, support = np.arange(0, 1+.01, .01).round(2), lambda_coord = [], p = 2, prob_dx = .001):
    ''' Closed-form solution for the 2-Wasserstein of 1D discrete distributions.
        Note that this applies only to the 2-wasserstein metric.
        locs: list with locations
        locs_probs: weights of each location
        lambda_coord: barycentric coordinates, if [], then set to uniform
        prob_dx: discretization step to integrate over probability interval'''

    quantiles = np.arange(0, 1+prob_dx, prob_dx)
    n_assets = len(locs)
    if len(lambda_coord) == 0:
        lambda_coord = (1/n_assets)*np.ones(n_assets)
    else:
        lambda_coord = np.array(lambda_coord).reshape(-1)
    # for each distribution, evaluate the inverse CDF
    q_funct = []
    for i, distr in enumerate(locs):
        weights_temp = locs_prob[i]

        # evaluate inverted c.d.f.
        q_function_temp = inverted_cdf(quantiles, distr, weights_temp)
        
        q_funct.append(q_function_temp)
        
    q_funct = np.array(q_funct)
    # average inverse CDF
    Q_target = (lambda_coord@q_funct)
    
    # estimate CDF at support locations from average inv CDF
    CDF = np.zeros(len(support))
    for i, x0 in enumerate(support):
        if x0 >= Q_target.max():
            CDF[i] = 1
        elif x0 < Q_target.min():
            CDF[i] = 0
        else:
            # !!!!!! Note sure if <= or < ??
            q_ind = np.where(Q_target <= x0)[0][-1]            
            CDF[i] = quantiles[q_ind]
        
    barycenter = np.diff(CDF)
    barycenter = np.insert(barycenter, 0, CDF[0])
    
    return barycenter, CDF, Q_target


def wass_distance_1D(x1, x2, w1 = [], w2 = [], p = 1, prob_dx = .001):
    ''' Closed-form solution for the p-Wasserstein of 1D discrete distributions
        x1,x2: np.arrays, discrete locations
        w1,w2: weights, if None: use uniform weights
        prob_dx: discretization step to estimate to evaluate the quantile functions'''
    
    quantiles = np.arange(0, 1+prob_dx, prob_dx)

    if (len(w1) == 0)*(len(w2) == 0):
        x1_sort = np.sort(x1.copy().reshape(-1))
        x2_sort = np.sort(x2.copy().reshape(-1))

        #w1 = np.ones(len(x1))*1/len(x1)            
        #w2 = np.ones(len(x2))*1/len(x2)            

        #q1 = inverted_cdf(quantiles, x1, w1)
        #q2 = inverted_cdf(quantiles, x2, w2)
        
        #dM = np.array([ (np.linalg.norm(q1.reshape(-1,1) - q2.reshape(-1,1), axis=1))  for i in range(len(q1))])
        #dM = np.power(dM, p)
        
        # !!!!! Not sure it works correct for p = 2        
        # Samples from uniform distribution
        #dM = np.array([ (np.linalg.norm(x1_sort.reshape(-1,1) - x2_sort.reshape(-1,1), axis=1))  for i in range(len(x1))])
        
        dM = np.linalg.norm(x1_sort.reshape(-1,1) - x2_sort.reshape(-1,1), axis=1)
        dM = np.power(dM, p)
                
        return np.power(dM.mean(), 1/p)

    else:
        # !!!!! Works correct, slower than the above
        if len(w1) == 0:
            w1 = np.ones(len(x1))*1/len(x1)            
        if len(w2) == 0:
            w2 = np.ones(len(x2))*1/len(x2)            
            
        # create quantile functions        
        #w1_sort = w1[np.argsort(x1.reshape(-1))]
        #w2_sort = w2[np.argsort(x2.reshape(-1))]
        
        #q1 = np.array([ x1_sort[np.where(w1_sort.cumsum()>=q)[0][0]] for q in quantiles])
        #q2 = np.array([ x2_sort[np.where(w2_sort.cumsum()>=q)[0][0]] for q in quantiles])
        
        q1 = inverted_cdf(quantiles, x1, w1)
        q2 = inverted_cdf(quantiles, x2, w2)
        
        #dM = np.array([ (np.linalg.norm(q1[i].reshape(-1,1) - q2.reshape(-1,1), axis=1))  for i in range(len(q1))])
        dM = np.linalg.norm(q1.reshape(-1,1) - q2.reshape(-1,1), axis=1)
        dM = np.power(dM, p)

        return np.power(dM.mean(), 1/p)
    

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
    