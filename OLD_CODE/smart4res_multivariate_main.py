# -*- coding: utf-8 -*-
"""
Data pooling smart4res

@author: akylas.stratigakos@minesparisl.psl.eu
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os
#from pathlib import Path
import pickle
import gurobipy as gp
#import pickle
from sklearn.preprocessing import MinMaxScaler

cd = os.path.dirname(__file__)  #Current directory
sys.path.append(cd)
#project_dir=Path(cd).parent.__str__()   #project_directory
plt.rcParams['figure.dpi'] = 600

from sklearn.ensemble import ExtraTreesRegressor
from EnsemblePrescriptiveTree import *
from scipy.stats import norm, gaussian_kde
from scipy import interpolate
from scipy.special import ndtr
from QR_regressor import *
from sklearn.neighbors import KNeighborsRegressor
from EnsemblePrescriptiveTree_OOB import *
from optimization_functions import *
from newsvendor import *
#import ot
from utility_functions import *
from optimal_transport_functions import *
from scipy.special import softmax

# IEEE plot parameters (not sure about mathfont)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 600
plt.rcParams['figure.figsize'] = (3.5, 2) # Height can be changed
plt.rcParams['font.size'] = 7
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams["mathtext.fontset"] = 'dejavuserif'


def create_aggr_predictors(order, S, w_dict, config, meteo_id):
    ''' Create predictors (weather) for specific series in the hierarchy
        For higher level series, concatenate feature data of the lower levels '''
    feat_list = []        
    current_meteo_grid = []
    # Loop over base nodes
    for park_id in S.columns:
        # Check if park_id belongs to node
        if S[park_id][order] == 0:
            continue
        elif S[park_id][order] == 1:
            # Check for duplicate meteo stations
            if meteo_id[park_id] in current_meteo_grid:
                continue
            else:
                feat_list.append(w_dict[park_id])
                current_meteo_grid.append(meteo_id[park_id])
    pred_df = pd.concat(feat_list, axis=1)
    
    if config['target'] == 'pv':
        pred_df['Hour']  = pred_df.index.hour
        pred_df['Month']  = pred_df.index.month
            
    return pred_df
    
def create_corrupt_data(Y, num_nodes, percentage, base_nodes_ind, same_obs = True):
    ''' Returns a corrupt data set Y and the respective indicator matrix
     Indicator matrix: If 1, then observation is clean, if 0, then it's corrupt (missing/error)'''
       
    if num_nodes == len(base_nodes_ind):
        c_node_ind = base_nodes_ind
    else:
        c_node_ind = np.random.choice(base_nodes_ind, size = num_nodes, replace = False)
        
    corrupt_Y = Y.copy()
    indicator_mat = np.ones(Y.shape)
    if same_obs:
        # Same timestamps missing in all nodes
        c_obs_ind = np.random.choice(np.arange(0, len(Y)), size = int(len(Y)*percentage), replace = False)

        for n in c_node_ind: 
            corrupt_Y[c_obs_ind, n] = 0            
            indicator_mat[c_obs_ind, n] = 0

    else:        
        
        # Different timestamps per node
        for n in c_node_ind: 
            c_obs_ind = np.random.choice(np.arange(0, len(Y)), 
                                         size = int(len(Y)*percentage), replace = False)
            corrupt_Y[c_obs_ind, n] = 0            
            indicator_mat[c_obs_ind, n] = 0

    return corrupt_Y, indicator_mat

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

def tree_params():
    ''' Hyperparameters for tree algorithms'''
    params = {}
    params['n_estimators'] = 50
    params['n_min'] = 2
    params['max_features'] = 1
    return params

def params():
    ''' Set up the experiment parameters'''

    params = {}
    # Either pv or wind
    params['target'] = 'wind'
    params['start_date'] = '2019-01-01'
    params['split_date'] = '2020-01-01' # Defines train/test split
    params['end_date'] = '2020-05-01'
    params['n_base'] = 60 # number of base series
    
    params['parallel'] = False # If True, then trees are grown in parallel
    params['save'] = True # If True, then saves models and results
    params['train_brc'] = True # If True, then saves models and results
    params['train_trees'] = True # If True, then train models, else tries to load previous runs
    
    # Experiment parameters
    if params['target'] == 'wind':
        params['n_corrupt_nodes'] = [5, 10, 20] # number of nodes with erroneous measurements
    elif params['target'] == 'pv':
        params['n_corrupt_nodes'] = [2, 5, 10] # number of base series
    params['percentage'] = [.05, .10, .20, .50]  # percentage of corrupted datapoints
    
    # Experimental setup parameters
    params['quality'] = False
    params['problem'] = 'newsvendor' # {mse, newsvendor, cvar, reg_trad}
    params['N_sample'] = [10, 50, 100, 200, 500]
    params['N_assets'] = [5, 10, 20, 50]
    params['iterations'] = 5
    params['critical_fractile'] = 0.2
    params['varying_N'] = False # True: vary between [10, N_max], False: fixed in all
    params['N_max'] = 200
    
    #extra problem parameters
    params['risk_aversion'] = 0.5 # True: vary between [10, N_max], False: fixed in all
    params['epsilon'] = 0.05
    
    return params

#%%

config = params()
hyperparam = tree_params()

case_path = f'{cd}\\smart4res-mult-case-study'
results_path = f'{case_path}\\results-samepark'
data_path = f'{case_path}\\data-processed'

target_var = config['target']
#%%
# load wind and NWP data for base series
turb_metadata = pd.read_csv(f'{data_path}\\{target_var}_metadata.csv', index_col = 0)
turb_power = pd.read_csv(f'{data_path}\\{target_var}_power_clean_imp_30min.csv', index_col = 0, parse_dates = True, dayfirst = False)
S_mat = pd.read_csv(f'{data_path}\\{target_var}_summation_matrix.csv')
capacity = S_mat.values@turb_metadata['Capacity (kW)']

# dictionary with predictor data for each park
with open(f'{data_path}\\{target_var}_weather_pred_30min.pkl', 'rb') as f:
        weather_dict = pickle.load(f)
turb_nwp = pd.concat(weather_dict, axis = 1)

# dictionary that maps turbine ID to closest grid point
with open(f'{data_path}\\{target_var}_meteo_id.pkl', 'rb') as f:
        meteo_id = pickle.load(f)

#%% Retrieve data for park level

# power for all levels of the hierarchy
power_total = pd.DataFrame(data = (S_mat.values@ turb_power.values.T).T, index = turb_power.index, 
                        columns = ['Top'] + list(turb_metadata['Park ID'].unique()) + list(turb_power.columns))

# scale all levels by capacity
sc_power_total = power_total/capacity

# park metadata
turb_metadata['meteo_id'] = meteo_id.values()
park_metadata = pd.DataFrame(data = turb_metadata['Park ID'].unique(), columns = ['Park ID'])

park_metadata['Lat'] = turb_metadata.groupby(['Park ID']).mean()['Lat'].values
park_metadata['Long'] = turb_metadata.groupby(['Park ID']).mean()['Long'].values
park_metadata['Capacity (kW)'] = turb_metadata.groupby(['Park ID']).sum()['Capacity (kW)'].values.astype(int)
park_metadata['meteo_id'] = turb_metadata.groupby(['Park ID']).mean()['meteo_id'].values.round().astype(int)

# features for parks: the closets NWP grid point from the center
park_weather_dict = {}
for i in range(len(park_metadata['Park ID'])):
    park_meteo_id = park_metadata['meteo_id'][i]
    turb_id = turb_metadata['Turbine ID'][turb_metadata['meteo_id'] == park_meteo_id].iloc[-1]
    park_weather_dict[park_metadata['Park ID'][i]] = weather_dict[str(turb_id)]

park_nwp = pd.concat(park_weather_dict, axis=1)
turb_nwp = pd.concat(weather_dict, axis = 1)
#%% Create supervised learning set

# select targets and features/ features are multilevel dataframe
turb_metadata['Turbine ID'] = turb_metadata['Turbine ID'].astype(str)

Y = sc_power_total[turb_metadata['Turbine ID']]
Pred = turb_nwp.copy()

#%%
target_ids = turb_metadata['Turbine ID']
n_targets = len(target_ids)

# create additional features
for j, p_id in enumerate(target_ids):
    Pred[p_id, 'wspeed'] = (np.sqrt(Pred[p_id]['u100'] ** 2 + Pred[p_id]['v100'] ** 2)).values

    Pred[p_id, 'wdir'] = (180/np.pi)*(np.arctan2(Pred[p_id]['u100'], Pred[p_id]['v100']))
    Pred[p_id, 'wdir'] = np.sin(np.deg2rad(Pred[p_id, 'wdir']))
    Pred[p_id, 'wdir_bin'] = pd.cut(Pred[p_id, 'wdir'], bins = 4, include_lowest = True,
                                labels = [0,1,2,3])
    Pred[p_id, 'wdir_bin'] = Pred[p_id, 'wdir_bin'].astype(int)



feat_col = ['wspeed']

Pred = Pred.reorder_levels([1,0], axis=1)
max_wspeed = Pred['wspeed'].max().max()
Pred['wspeed'] = (Pred['wspeed']/max_wspeed).round(2)

# train/test split
trainY = Y[config['start_date']:config['split_date']].round(2)
testY = Y[config['split_date']:config['end_date']].round(2)
    
trainPred = Pred[feat_col][config['start_date']:config['split_date']].round(2)
testPred = Pred[feat_col][config['split_date']:config['end_date']].round(2)

# re-order multiindex by park level
trainPred = trainPred.reorder_levels([1,0], axis=1)
testPred = testPred.reorder_levels([1,0], axis=1)

#%% Problem parameters

# sample joint data for each part

if config['varying_N']:
    N_max = config['N_max']
    N_sample = [config['N_max']]
else:
    N_sample = config['N_sample']

N_assets = config['N_assets']
iterations = config['iterations']
problem = config['problem']

if config['problem'] in ['newsvendor', 'cvar', 'reg_trad']:
    problem_name = str(config['problem']) + '_' + str(np.round(config['critical_fractile'], decimals = 2))
elif config['problem'] == 'mse':
    problem_name = config['problem']

if config['varying_N']:
    if config['quality']:
        output_file_name = f'{problem_name}_quality_randomN_results.csv'
    else:
        output_file_name = f'{problem_name}_randomN_results.csv'
else:        
    if config['quality']:
        output_file_name = f'{problem_name}_quality_results.csv'
    else:
        output_file_name = f'{problem_name}_results.csv'
                    
Parks_ids = turb_metadata['Turbine ID'].values

# data conditioned on wind speed
step = .01
y_supp = np.arange(0, 1+step, step).round(2)

if feat_col == ['wspeed', 'wdir_bin']:
    x_supp = [np.arange(0, 1+step, step).round(2), np.array([0,1,2,3])]
    xt, xr = np.meshgrid(x_supp[0], x_supp[1])
    x_joint_supp = np.array([xt.ravel(), xr.ravel()]).T
else:
    x_supp = np.arange(0, 1+step, step).round(2)
    x_joint_supp = np.arange(0, 1+step, step).round(2)
    
#opt_quant = np.arange(.1, 1, .1).round(2)
#quantiles = opt_quant.copy()

#quantiles = [0.80]
critical_fractile = config['critical_fractile']
risk_aversion = config['risk_aversion']
epsilon = config['epsilon']

model_cols = ['Local','Pooled', 'WBC','optWBC', 'L2-BRC', 'Interpol', 'ConvComb']
cols =  model_cols + ['N', 'Quantile', 'Park_id']
Results = pd.DataFrame(data = [], columns = cols)

total_count = 0

#%%
# loop over sample sizes
#Results = pd.read_csv(f'{case_path}\\results\\{output_file_name}', index_col = 0)
#Results = Results.query('N_assets<20')
#total_count = Results.index[-1]

#%%

for n_assets in N_assets:
    #np.random.seed(0)
    # choose parks
    #plt.scatter(park_metadata.loc[:,'Long'], park_metadata.loc[:,'Lat'])
    #plt.scatter(park_metadata.loc[target_parks, 'Long'], park_metadata.loc[target_parks, 'Lat'], marker = 'd')
    #plt.xlabel('Longitude')
    #plt.ylabel('Latitude')
    #plt.show()

    # sample parks
    np.random.seed(total_count)
    target_parks = np.random.choice(Parks_ids, n_assets, replace = False)
    
    for n, n_sample in enumerate(N_sample):
        for iter_ in range(iterations):
            
            #if (n_sample>=200)and(iter_>0)and(problem=='newsvendor'): continue
            
            
            ###### Sample local data sets
            np.random.seed(total_count)
            
            if config['varying_N']:
                # random number of observations per each problem
                sample_sizes = np.random.randint(10, N_max, len(target_parks))
                #sample_sizes = np.random.normal(100, 25, len(target_parks))
                sample_sizes = projection(sample_sizes, ub = N_max, lb = 10).round(0).astype(int)

            else:
                # fixed random of observations across all problems
                sample_sizes = np.random.randint(n_sample, n_sample+1, len(target_parks))
            
            ##### create local/global data sets
            obs_idx = [ np.sort(np.random.randint(0, trainY.shape[0], size)) for size in sample_sizes]
            emp_joint_data = [ np.column_stack((trainY[park].values[obs_idx[i]], (trainPred[park][feat_col].values[obs_idx[i]])))
                              for i, park in enumerate(target_parks) ]

            train_localY = [ trainY[park].values[obs_idx[i]].reshape(-1,1) for i, park in enumerate(target_parks) ]
            train_localX = [ trainPred[park][feat_col].values[obs_idx[i]] for i, park in enumerate(target_parks) ]
            
            test_localY = [ testY[park].values.reshape(-1,1) for i, park in enumerate(target_parks) ]
            test_localX = [ testPred[park][feat_col].values for i, park in enumerate(target_parks) ]
                
            
            joint_data_dict = {}
            for i, park in enumerate(target_parks):
                joint_data_dict[park] = emp_joint_data[i]
            
            
            ##### Naive pooled data set
            pooled_data = np.concatenate(emp_joint_data, axis=0)
            
            train_poolX = np.concatenate(train_localX)
            train_poolY = np.concatenate(train_localY)
            
            #X_pooled = pooled_data[:,1:]
            #Y_pooled = np.column_stack((pooled_data[:,0], np.ones(len(pooled_data)) ))
            
            plt.scatter(train_poolX[:,0], train_poolY)
            plt.title('Pooled data')
            plt.show()
            
            
            ##########################
            ##### train global and local models
            
            if (config['train_trees']):
                print('fit global tree')
                global_tree = EnsemblePrescriptiveTree_OOB(n_estimators = hyperparam['n_estimators'], Nmin = hyperparam['n_min'], 
                                                        max_features = 1, type_split = 'quant' )
                global_tree.fit(train_poolX, train_poolY, x_joint_supp, y_supp, bootstrap = True, quant = np.arange(.05 ,1, .05),
                                crit_quant = config['critical_fractile'], problem = config['problem'],
                                risk_aversion = config['risk_aversion'], epsilon = config['epsilon'] ) 
            
                # loop over all the target parks
                local_models = {}
                
                for s, p_id in enumerate(target_parks):
                    print('Park: '+p_id)
                    print('fit local model')
                    
                    if (config['quality'] == True)and(s<=len(target_parks)/2):        
                        local_tree = EnsemblePrescriptiveTree_OOB(n_estimators = 1, D = 2, Nmin = hyperparam['n_min'], 
                                                                max_features = 1, type_split = 'quant' )
                    else:
                        local_tree = EnsemblePrescriptiveTree_OOB(n_estimators = hyperparam['n_estimators'], Nmin = hyperparam['n_min'], 
                                                                max_features = 1, type_split = 'quant' )
                        
                    local_tree.fit(train_localX[s], train_localY[s], x_joint_supp, y_supp, bootstrap = True, quant = np.arange(.05, 1, .05), 
                                   crit_quant = config['critical_fractile'], problem = config['problem'],
                                   risk_aversion = config['risk_aversion'], epsilon = config['epsilon'] ) 
            
                    local_models[p_id] = local_tree
            
                #if (config['save']) and (iter_==0):
                #    pickle.dump(global_tree, open(f'{results_path}\\global_tree_{n_sample}_{iter_}.sav', 'wb'))
                #    pickle.dump(local_models, open(f'{results_path}\\local_models_{n_sample}_{iter_}.sav', 'wb'))
                
            else:
                
                with open(f'{results_path}\\global_tree_{n_sample}_{iter_}.sav', 'rb') as handle: global_tree = pickle.load(handle)
                with open(f'{results_path}\\local_models_{n_sample}_{iter_}.sav', 'rb') as handle: local_models = pickle.load(handle)
                
                # change the problem definition for each tree
                global_tree.decision_kwargs['problem'] = config['problem']
                for park in target_parks:
                    local_models[park].decision_kwargs['problem'] = config['problem']
            
            ##############################
            '''
            #### Estimate OOB and cross-validation errors
            
            OOB_errors = np.zeros(len(target_parks))          
            Sfold_error = np.zeros(len(target_parks))            
            corrected_OOB_errors = np.zeros(len(target_parks))
            
            for s, p_id in enumerate(target_parks):
                print('Park: '+p_id)                
                
                #fold_X = np.concatenate([train_localX[l] for l in range(len(target_parks)) if l!=s])
                #fold_Y = np.concatenate([train_localY[l] for l in range(len(target_parks)) if l!=s])

                #fold_Prediction = local_models[p_id].predict_constr(fold_X, train_localX[s], train_localY[s]).reshape(-1,1)                
                #in_sample_Pred = local_models[p_id].predict_constr(train_localX[s], train_localX[s], train_localY[s]).reshape(-1,1)
 
                sfold_Predictions = [local_models[p_id].predict_constr(train_localX[l], train_localX[s], train_localY[s]).reshape(-1,1) for l in range(len(target_parks))]
                
                if problem == 'mse':
                    s_local_loss =  [np.square(train_localY[l] - sfold_Predictions[l]).mean() for l in range(len(target_parks)) if l!=s]
                    in_sample_loss = np.square(train_localY[s] - sfold_Predictions[s]).mean()                                    
                else:                
                    s_local_loss =  [newsvendor_loss(train_localY[l], sfold_Predictions[l], q = critical_fractile) for l in range(len(target_parks)) if l!=s]
                    in_sample_loss = newsvendor_loss(train_localY[s], sfold_Predictions[s], q = critical_fractile)
                
                Sfold_error[s] = np.array(s_local_loss).mean()
                OOB_errors[s] = local_models[p_id].estimate_OOB_score(train_localX[s], train_localY[s])                
                corrected_OOB_errors[s] = .632*OOB_errors[s] + .368*in_sample_loss
                Sfold_error[s] = ((n_assets-1)/n_assets)*Sfold_error[s] + (1/n_assets)*OOB_errors[s]

            plt.plot(Sfold_error, label='Sfold')
            plt.plot(OOB_errors, label='OBB')
            plt.plot(corrected_OOB_errors, label='.632 OOB')
            plt.legend()
            plt.show()
            '''
            
            ############## Estimate Barycenters conditioned on feature data
            if config['train_brc'] or (config['varying_N'] == True):
                
                # Barycenters of conditional marginals (for a given context realization)
                w_brc = np.zeros((len(x_joint_supp), len(y_supp)))
                opt_w_brc = np.zeros((len(x_joint_supp), len(y_supp)))
                l2_brc = np.zeros((len(x_joint_supp), len(y_supp)))
                
                for row, context in enumerate(x_joint_supp):
                    context = context.reshape(1,-1)
                    print(f'Context = {context}')
                    # find conditional marginal for each park
                    Prob_supp = [np.zeros(len(y_supp)) for p in local_models.keys()]
            
                    for s, p_id in enumerate(target_parks):

                        #w = find_weights(local_models[park], train_localX[s], context) 
                        w = local_models[p_id].find_weights(context.reshape(1,-1), train_localX[s]) 
                
                        # effective set
                        mask = (w>0).reshape(-1)
                        for j,y in enumerate(train_localY[s][mask]):
                                
                            ind = np.where(y == y_supp)[0][0]
                            Prob_supp[s][ind] = Prob_supp[s][ind] + w.reshape(-1)[mask][j]
                        
                    # estimate barycenter of conditional marginals
                    locs = n_assets*[y_supp]
                    
                    # LP solution
                    #a_w_brc, gamma = w_barycenter_LP(locs, Prob_supp, w_coordinates = [], support = y_supp, p = 1, 
                    #                              return_plans = True)
                    
                    
                    # coordinates based on the number of observations
                    obs_coord = sample_sizes/sample_sizes.sum()
                    a_w_brc, _, _ = wass2_barycenter_1D(locs, Prob_supp, lambda_coord = obs_coord, support = y_supp, p = 2, 
                                                   prob_dx = .001)
                    
                    w_brc[row] = a_w_brc

                    '''
                    # barycenter with optimized coordinates
                    bar_coord = (1/Sfold_error)/(1/Sfold_error).sum()
                    #bar_coord = softmax(1/Sfold_error)
                    
                    a_w_brc, _, _ = wass2_barycenter_1D(locs, Prob_supp, lambda_coord = bar_coord, support = y_supp, p = 2, 
                                                   prob_dx = .001)
                    
                    opt_w_brc[row] = a_w_brc
                    '''
                    
                    if row == 0:
                        plt.plot(obs_coord, label='sample coord')
                        #plt.plot(bar_coord, label='validation coord')
                        plt.show()
                    # l2 barycenter
                    ave_Prob = 0
                    for p, park in enumerate(local_models.keys()):            
                        ave_Prob += Prob_supp[p]
                    ave_Prob/=len(local_models.keys())
                    
                    l2_brc[row] = ave_Prob

                if (config['save']) and (config['varying_N']==False): 
                    pickle.dump(w_brc, open(f'{results_path}\\w_brc_{n_assets}_{n_sample}_{iter_}.sav', 'wb'))
                    pickle.dump(l2_brc, open(f'{results_path}\\l2_brc_{n_assets}_{n_sample}_{iter_}.sav', 'wb'))
            else:
                if (n_assets==50)and(n_sample==500):
                    with open(f'{results_path}\\w_brc_{n_assets}_{n_sample}_0.sav', 'rb') as handle: w_brc = pickle.load(handle)
                    with open(f'{results_path}\\l2_brc_{n_assets}_{n_sample}_0.sav', 'rb') as handle: l2_brc = pickle.load(handle)
                else:                        
                    with open(f'{results_path}\\w_brc_{n_assets}_{n_sample}_{iter_}.sav', 'rb') as handle: w_brc = pickle.load(handle)
                    with open(f'{results_path}\\l2_brc_{n_assets}_{n_sample}_{iter_}.sav', 'rb') as handle: l2_brc = pickle.load(handle)
            
            #w_brc_prescriptions = np.zeros((len(x_joint_supp), 1))
            #l2_ave_prescriptions = np.zeros((len(x_joint_supp), 1))
            #if problem == 'mse':
            #    for c, context in enumerate(x_joint_supp):
                    #!!!! to solve optimization problem here
            #        w_brc_prescriptions[c] = w_brc[c]@y_supp
            #        l2_ave_prescriptions[c] = l2_brc[c]@y_supp
            
            #%%
            
            ###############################################
            
            #### Hyperparameter tuning for interpolation
            print('Find optimal hyperparameter for interpolation')
            lambda_grid = np.arange(0, 1+.1, .1).round(2)
                            
            for s, target_park_id in enumerate(target_parks):
                
                print(target_park_id)
                
                idx = np.where(target_park_id == target_parks)[0][0]
                target_data = emp_joint_data[idx]
                    
                #### OOB combination and out-of-sample testing                                            

                # find combination weight
                oob_comb_loss = local_models[target_park_id].interpolation_tuning(train_localX[s], train_localY[s], l2_brc, lambda_= lambda_grid, 
                                                                                  type_interpolation = 'average')
                
                
                oob_comb_loss = local_models[target_park_id].interpolation_tuning(train_localX[s], train_localY[s], w_brc, lambda_= lambda_grid, 
                                                                                  type_interpolation = 'barycenter')
                
                # update saved version
                #if config['save']:
                #    pickle.dump(local_models, open(f'{results_path}\\local_models_{n_sample}_{iter_}.sav', 'wb'))
                
                if s%10==0:
                    plt.plot(lambda_grid, oob_comb_loss)
                    plt.ylabel('OOB Error')                
                    plt.xlabel('$\lambda$')                
                    plt.show()
                            
            ###############################################
            #%%#### Out-of-sample estimation    
            print('Out-of-sample testing')
            
            # To speed-up: pre-compute prescriptions for each model and for each support location of the contextual information

            # Step 1: Find probabilities for contextual information (discretized for speed)
            global_weights = global_tree.find_weights(x_supp.reshape(-1,1), train_poolX)
            
            local_weights = []
            conv_comb_weights = []
            interpol_weights = []
            for s, park in enumerate(target_parks):
                
                # weights for s-th park
                temp_local_weights = local_models[park].find_weights(x_supp.reshape(-1,1), train_localX[s])
                
                # map local weights of s-th park to support location
                local_weights.append(wemp_to_support(temp_local_weights, train_localY[s], y_supp))

                conv_comb_weights.append(local_models[park].best_lambda_conv_comb*local_weights[s] +\
                                    (1-local_models[park].best_lambda_conv_comb)*l2_brc)
                
                            
                # interpolated barycenter
                inter_wbc = []
                lamba_coord = [local_models[park].best_lambda_interpol, 1-local_models[park].best_lambda_interpol]
                
                if local_models[park].best_lambda_interpol == 0:
                    inter_wbc = w_brc
                    interpol_weights.append(w_brc)
                elif local_models[park].best_lambda_interpol == 1:
                    inter_wbc = local_weights[s]
                else:
                    for row, context in enumerate(x_joint_supp):
                        temp_inter_wbc,_,_ = wass2_barycenter_1D(2*[y_supp], [local_weights[s][row], w_brc[row]], 
                                                                 lambda_coord=[lamba_coord], prob_dx = .01)
                        inter_wbc.append(temp_inter_wbc)

                    inter_wbc = np.array(inter_wbc)
                interpol_weights.append(inter_wbc)

            #%%
            # Step 2: pre-compute prescriptions for each support location
            # model_cols = ['Local','Pooled', 'W-BRC', 'L2-BRC', 'Combined']
                
            
            Global_xloc_prescriptions = np.zeros(x_joint_supp.shape)
            WBC_xloc_prescriptions = np.zeros(x_joint_supp.shape)
            opt_WBC_xloc_prescriptions = np.zeros(x_joint_supp.shape)
            l2BC_xloc_prescriptions = np.zeros(x_joint_supp.shape)
            Local_xloc_prescriptions = [np.zeros(x_joint_supp.shape) for i in range(len(target_parks))]                
            Comb_xloc_prescriptions = [np.zeros(x_joint_supp.shape) for i in range(len(target_parks))]                
            Interpol_xloc_prescriptions = [np.zeros(x_joint_supp.shape) for i in range(len(target_parks))]                
            
            
            if problem == 'mse':
                Global_xloc_prescriptions = global_weights@train_poolY
                WBC_xloc_prescriptions = (w_brc@y_supp).reshape(-1,1)
                #opt_WBC_xloc_prescriptions = (opt_w_brc@y_supp).reshape(-1,1)

                l2BC_xloc_prescriptions = (l2_brc@y_supp).reshape(-1,1)
                Local_xloc_prescriptions = []                
                Comb_xloc_prescriptions = []                
                Interpol_xloc_prescriptions = []                
                
                for s, park in enumerate(target_parks):
                    # !!!! weights are mapped to support locations
                    Local_xloc_prescriptions.append(local_weights[s]@y_supp)
                    Comb_xloc_prescriptions.append(conv_comb_weights[s]@y_supp)
                    Interpol_xloc_prescriptions.append(interpol_weights[s]@y_supp)

            elif problem == 'newsvendor':
                
                sort_pooled_Y = np.sort(train_poolY.copy(), axis = 0)                
                argsort_g = np.argsort(train_poolY, axis = 0).reshape(-1)

                for row, context in enumerate(x_joint_supp):                    
                    Global_xloc_prescriptions[row] = sort_pooled_Y[np.where(global_weights[row][argsort_g].cumsum()>=critical_fractile)][0]
                    WBC_xloc_prescriptions[row] = y_supp[np.where(w_brc[row].cumsum()>=critical_fractile)][0]
                    #opt_WBC_xloc_prescriptions[row] = y_supp[np.where(opt_w_brc[row].cumsum()>=critical_fractile)][0]
                    l2BC_xloc_prescriptions[row] = y_supp[np.where(l2_brc[row].cumsum()>=critical_fractile)][0]
                    
                for s, park in enumerate(target_parks):
                    sort_local_Y = np.sort(train_localY[s].copy(), axis = 0)                
                    argsort_g = np.argsort(train_localY[s], axis = 0).reshape(-1)

                    for row, context in enumerate(x_joint_supp):
                        #Local_xloc_prescriptions[s][row] = sort_local_Y[np.where(local_weights[s][row][argsort_g].cumsum()>=critical_fractile)][0]
                        Local_xloc_prescriptions[s][row] = y_supp[np.where(local_weights[s][row].cumsum()>=critical_fractile)][0]
                        Comb_xloc_prescriptions[s][row] = y_supp[np.where(conv_comb_weights[s][row].cumsum()>=critical_fractile)][0]
                        Interpol_xloc_prescriptions[s][row] = y_supp[np.where(interpol_weights[s][row].cumsum()>=critical_fractile)][0]

            elif problem in ['cvar', 'reg_trad']:
                    
                Global_xloc_prescriptions = solve_opt_prob(train_poolY, global_weights, config['problem'], model_once = True, crit_quant = critical_fractile, 
                                                                 risk_aversion = risk_aversion, epsilon = epsilon)
                WBC_xloc_prescriptions = solve_opt_prob(y_supp, w_brc, config['problem'], model_once = True, crit_quant = critical_fractile, risk_aversion = risk_aversion, epsilon = epsilon) 
                l2BC_xloc_prescriptions = solve_opt_prob(y_supp, l2_brc, config['problem'], model_once = True, crit_quant = critical_fractile, risk_aversion = risk_aversion, epsilon = epsilon) 
                    
                for s, park in enumerate(target_parks):
                    # !!!! weights mapped to support locations for speed-up
                    Local_xloc_prescriptions[s] = solve_opt_prob(y_supp, local_weights[s], config['problem'], model_once = True, crit_quant = critical_fractile, risk_aversion = risk_aversion, epsilon = epsilon)
                    Comb_xloc_prescriptions[s] = solve_opt_prob(y_supp, conv_comb_weights[s], config['problem'], model_once = True, crit_quant = critical_fractile, risk_aversion = risk_aversion, epsilon = epsilon)
                    Interpol_xloc_prescriptions[s] = solve_opt_prob(y_supp, interpol_weights[s], config['problem'], model_once = True, crit_quant = critical_fractile, risk_aversion = risk_aversion, epsilon = epsilon)
            
            ########
            # Out-of-sample testing
            for s, target_park_id in enumerate(target_parks):

                print(target_park_id)
                #temp_Presciptions = pd.DataFrame(data=np.zeros((len(test_localX[s], len(model_cols)))), columns=model_cols)
                                
                idx = np.where(target_park_id == target_parks)[0][0]
                target_data = emp_joint_data[idx]

                # find indexes of test observations
                indexes = []
                if x_joint_supp.ndim==1:
                    indexes = [np.where(x0 == x_joint_supp)[0][0] for x0 in test_localX[s]]
                else:
                    indexes = [np.where(x0 == x_joint_supp.all(1))[0][0] for x0 in test_localX[s]]

                # find prescriptions                    
                global_pred = Global_xloc_prescriptions[indexes].reshape(-1,1)
                local_pred = Local_xloc_prescriptions[s][indexes].reshape(-1,1)
                w_brc_pred = WBC_xloc_prescriptions[indexes].reshape(-1,1)
                comb_pred = Comb_xloc_prescriptions[s][indexes].reshape(-1,1)
                interpol_pred = Interpol_xloc_prescriptions[s][indexes].reshape(-1,1)
                
                l2_brc_pred = l2BC_xloc_prescriptions[indexes].reshape(-1,1)            
                opt_w_brc_pred = opt_WBC_xloc_prescriptions[indexes].reshape(-1,1)

                temp_results = pd.DataFrame()


                temp_results['N'] = [n_sample]
                temp_results['iteration'] = [iter_]
                temp_results['N_assets'] = [n_assets]
                temp_results['Park_id'] = target_park_id

                temp_results['Pooled'] = task_loss(global_pred, test_localY[s], config['problem'], crit_quant = critical_fractile, 
                                                   risk_aversion = risk_aversion, epsilon = epsilon)
                temp_results['Local'] = task_loss(local_pred, test_localY[s], config['problem'], crit_quant = critical_fractile, 
                                                   risk_aversion = risk_aversion, epsilon = epsilon)
                temp_results['WBC'] = task_loss(w_brc_pred, test_localY[s], config['problem'], crit_quant = critical_fractile, 
                                                   risk_aversion = risk_aversion, epsilon = epsilon)
                temp_results['optWBC'] = task_loss(opt_w_brc_pred, test_localY[s], config['problem'], crit_quant = critical_fractile, 
                                                   risk_aversion = risk_aversion, epsilon = epsilon)
                temp_results['L2-BRC'] = task_loss(l2_brc_pred, test_localY[s], config['problem'], crit_quant = critical_fractile, 
                                                   risk_aversion = risk_aversion, epsilon = epsilon)
                temp_results['ConvComb'] = task_loss(comb_pred, test_localY[s], config['problem'], crit_quant = critical_fractile, 
                                                   risk_aversion = risk_aversion, epsilon = epsilon)
                temp_results['Interpol'] = task_loss(interpol_pred, test_localY[s], config['problem'], crit_quant = critical_fractile, 
                                                   risk_aversion = risk_aversion, epsilon = epsilon)
                
                Results = pd.concat([Results, temp_results], axis=0, ignore_index=True)
                
                '''
                if problem == 'mse':
                    
                    # out-of-sample predictions for local, brc, and combined approach            
                    #comb_pred, local_pred, w_brc_pred = local_models[target_park_id].comb_predict(test_localX[s], train_localX[s], 
                    #                                                                 train_localY[s], w_brc)
                    
                    temp_results['N'] = [n_sample]
                    temp_results['iteration'] = [iter_]
                    temp_results['N_assets'] = [n_assets]
                    temp_results['Park_id'] = target_park_id
                                
                    temp_results['Pooled'] = np.square(test_localY[s] - global_pred).mean()
                    temp_results['Local'] = np.square(test_localY[s] - local_pred).mean()
                    temp_results['WBC'] = np.square(test_localY[s] - w_brc_pred).mean()
                    temp_results['optWBC'] = np.square(test_localY[s] - opt_w_brc_pred).mean()
                    
                    temp_results['L2-BRC'] = np.square(test_localY[s] - l2_brc_pred).mean()
                    temp_results['ConvComb'] = np.square(test_localY[s] - comb_pred).mean()
                    temp_results['Interpol'] = np.square(test_localY[s] - interpol_pred).mean()

                    Results = pd.concat([Results, temp_results], axis=0, ignore_index=True)

                
                elif problem == 'newsvendor':
                    
                    
                    # out-of-sample predictions for local, brc, and combined approach            
                    #comb_pred, local_pred, w_brc_pred = local_models[target_park_id].comb_predict(test_localX[s], train_localX[s], 
                    #                                                                 train_localY[s], w_brc, crit_quant = q)
        
                    temp_results['N'] = [n_sample]
                    temp_results['N_assets'] = [n_assets]
                    temp_results['iteration'] = [iter_]
                    temp_results['Quantile'] = [critical_fractile]
                    temp_results['Park_id'] = target_park_id
                                
                    temp_results['Pooled'] = newsvendor_loss(global_pred, test_localY[s], q = critical_fractile)
                    temp_results['Local'] = newsvendor_loss(local_pred, test_localY[s], q = critical_fractile)
                    temp_results['WBC'] = newsvendor_loss(w_brc_pred, test_localY[s], q = critical_fractile)
                    temp_results['optWBC'] = newsvendor_loss(opt_w_brc_pred, test_localY[s], q = critical_fractile)

                    temp_results['L2-BRC'] = newsvendor_loss(l2_brc_pred, test_localY[s], q = critical_fractile)
                    temp_results['ConvComb'] = newsvendor_loss(comb_pred, test_localY[s], q = critical_fractile)
                    temp_results['Interpol'] = newsvendor_loss(interpol_pred, test_localY[s], q = critical_fractile)
                    
                    Results = pd.concat([Results, temp_results], axis=0, ignore_index=True)

                elif problem == 'cvar':
                    # out-of-sample predictions for local, brc, and combined approach            
                    #comb_pred, local_pred, w_brc_pred = local_models[target_park_id].comb_predict(test_localX[s], train_localX[s], 
                    #                                                                 train_localY[s], w_brc, crit_quant = q)
        
                    temp_results['N'] = [n_sample]
                    temp_results['N_assets'] = [n_assets]
                    temp_results['iteration'] = [iter_]
                    temp_results['Quantile'] = [critical_fractile]
                    temp_results['Park_id'] = target_park_id
                                
                    temp_results['Pooled'] = newsvendor_cvar_loss(global_pred, test_localY[s], q = critical_fractile)
                    temp_results['Local'] = newsvendor_cvar_loss(local_pred, test_localY[s], q = critical_fractile)
                    temp_results['WBC'] = newsvendor_cvar_loss(w_brc_pred, test_localY[s], q = critical_fractile)
                    temp_results['optWBC'] = newsvendor_cvar_loss(opt_w_brc_pred, test_localY[s], q = critical_fractile)

                    temp_results['L2-BRC'] = newsvendor_cvar_loss(l2_brc_pred, test_localY[s], q = critical_fractile)
                    temp_results['ConvComb'] = newsvendor_cvar_loss(comb_pred, test_localY[s], q = critical_fractile)
                    temp_results['Interpol'] = newsvendor_cvar_loss(interpol_pred, test_localY[s], q = critical_fractile)
                    
                    Results = pd.concat([Results, temp_results], axis=0, ignore_index=True)
                    
                elif problem == 'reg_trad':
                    # out-of-sample predictions for local, brc, and combined approach            
                    #comb_pred, local_pred, w_brc_pred = local_models[target_park_id].comb_predict(test_localX[s], train_localX[s], 
                    #                                                                 train_localY[s], w_brc, crit_quant = q)
        
                    temp_results['N'] = [n_sample]
                    temp_results['N_assets'] = [n_assets]
                    temp_results['iteration'] = [iter_]
                    temp_results['Quantile'] = [critical_fractile]
                    temp_results['Park_id'] = target_park_id
                                
                    temp_results['Pooled'] = reg_trad_loss(global_pred, test_localY[s], q = critical_fractile)
                    temp_results['Local'] = reg_trad_loss(local_pred, test_localY[s], q = critical_fractile)
                    temp_results['WBC'] = reg_trad_loss(w_brc_pred, test_localY[s], q = critical_fractile)
                    temp_results['optWBC'] = reg_trad_loss(opt_w_brc_pred, test_localY[s], q = critical_fractile)

                    temp_results['L2-BRC'] = reg_trad_loss(l2_brc_pred, test_localY[s], q = critical_fractile)
                    temp_results['ConvComb'] = reg_trad_loss(comb_pred, test_localY[s], q = critical_fractile)
                    temp_results['Interpol'] = reg_trad_loss(interpol_pred, test_localY[s], q = critical_fractile)
                    
                    Results = pd.concat([Results, temp_results], axis=0, ignore_index=True)
                '''
                    
                fig, ax = plt.subplots(constrained_layout = True)
                
                #Results.query(f'N_assets=={n_assets} and iteration=={iter_}')[model_cols].plot(ax=ax)
                plt.plot(Results.query(f'N_assets=={n_assets} and iteration=={iter_}')[['Local', 'Pooled', 'WBC',
                                                                                         'L2-BRC', 'Interpol', 'ConvComb']].values)
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                plt.annotate(f'$N={n_sample}$', (.2, .8),
                             xycoords = 'figure fraction', bbox=props)
                plt.ylabel('MSE')        
                plt.legend(['Local', 'Pooled', 'WBC', 'L2-BRC', 'Interpol', 'ConvComb'])
                plt.show()
                
                print(Results.groupby('N')[['Local', 'Pooled', 'WBC', 'L2-BRC', 'Interpol', 'ConvComb']].mean())
                
                total_count += 1

            if config['save']:
                Results.to_csv(f'{results_path}\\{output_file_name}')

        #fig, ax = plt.subplots(constrained_layout = True)    
        #Results.query(f'iteration=={iter_} and N_assets=={n_assets}').groupby(['N'])[['Local', 'Pooled', 'WBC', 'L2-BRC', 'Interpol', 'ConvComb']].mean().plot()
        #plt.show()

if config['save']:
    Results.to_csv(f'{results_path}\\{output_file_name}')

#%%
marker = ['o', 'd', 's', '+', '2', '3', '4']
m_plot = model_cols
if problem == 'mse':
    fig, ax  = plt.subplots()
    for i, m in enumerate(m_plot):
        Results.groupby(['N'])[[m]].mean().plot(ax = ax, marker = marker[i], label = m)
    plt.ylabel('mse')
    plt.legend()
    plt.xticks(N_sample, N_sample)
    #if config['save']: plt.savefig(f'{results_path}\\{problem}_aggregate.pdf')
    plt.show()

elif problem == 'newsvendor':
    fig, ax  = plt.subplots(nrows = 3, sharex = True, figsize = (3.5, 4))
    for j, q in enumerate([.2, .6, .8]):
        for i, m in enumerate(m_plot):
            Results[Results['Quantile']==q].groupby(['N', 'Quantile'])[m].mean().plot(ax = ax[j], marker = marker[i], label = m)
        ax[j].set_title('Q: '+str(q))
    plt.ylabel('Pinball loss')
    plt.legend()
    #if config['save']: plt.savefig(f'{results_path}\\{problem}_aggregate.pdf') 
    plt.show()

