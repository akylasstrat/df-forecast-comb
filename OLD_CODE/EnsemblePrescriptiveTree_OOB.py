# -*- coding: utf-8 -*-
"""
This function trains an ensemble of Greedy Decision Trees trees that minimize decision cost.
The ensmble is based on the Extremely Randomized Trees (ExtraTrees) algorithm.

References: 
- Geurts, P., Ernst, D. and Wehenkel, L., 2006. Extremely randomized trees. Machine learning, 63(1), pp.3-42.
- Bertsimas, D. and Kallus, N., 2020. From predictive to prescriptive analytics. Management Science, 66(3), pp.1025-1044.
- Bertsimas, Dimitris, and Jack Dunn. "Optimal classification trees." Machine Learning 106.7 (2017): 1039-1082.  
- Dunn, J.W., 2018. Optimal trees for prediction and prescription (Doctoral dissertation, Massachusetts Institute of Technology).
- Elmachtoub, Adam N., and Paul Grigas. "Smart" predict, then optimize"." arXiv preprint arXiv:1710.08005 (2017).
@author: a.stratigakos
"""

#Import Libraries
import numpy as np
import gurobipy as gp
from math import sqrt
from GreedyPrescriptiveTree import GreedyPrescriptiveTree
from opt_problem import *
import time
import pandas as pd
import matplotlib.pyplot as plt
from optimal_transport_functions import *

#from forecast_opt_problem import *
#from decision_solver import *
from joblib import Parallel, delayed
    
class EnsemblePrescriptiveTree_OOB(object):
  '''Initialize object.
  
  Paremeters:
      n_estimators: number of trees to grow
      D: maximum depth of the tree (Inf as default)
      Nmin: minimum number of observations at each leaf
      type_split: random splits as default (if regular, then this is the RandomForest algorithm, almost)
      spo_weight: Parameter that controls the trade-off between prediction and prescription, to be included
      max_features: Number of features considered for each node split
      
      **kwargs: keyword arguments to solve the optimization problem prescribed (to be included)

      '''
  def __init__(self, n_estimators = 10, D = 'Inf' , Nmin = 5, 
               max_features = 'auto', type_split = 'random'):
      
    self.n_estimators = n_estimators
    if D == 'Inf':
        self.D = np.inf
    else:
        self.D = D
    self.Nmin = Nmin
    self.type_split = type_split
    self.max_features = max_features
    
    
  def fit(self, X, Y, x_support, y_support, quant = np.arange(.1, 1, .1), bootstrap = True, 
          oob_score = True, parallel = False, boot_prob = 'uniform', n_jobs = -1, cpu_time = False, brc = True, 
          problem = 'mse', prescriptive_train = False, bin_features = False, **kwargs):
    ''' Function for training the tree ensemble.
    Requires a separate function that solves the inner optimization problem.
    - quant: quantiles to evaluate continuous features (only used if type_split=='regular')
    - parallel: grows trees in parallel
    - n_jobs: only used for parallel training
    - cpu_time: if True, then returns cpu_time for each tree. If parallel==True, returns cpu_time for the ensemble (not sure how to interpret)
    '''       
    self.bin_features = bin_features
    self.xbins = x_support

    self.decision_kwargs = kwargs
    
    # main: {problem, critical fractile}
    self.decision_kwargs['problem'] = problem
    self.decision_kwargs['prescriptive_train'] = prescriptive_train

    num_features = X.shape[1]    #Number of features
    index_nodes = [np.arange(len(Y))]
    self.trees = []
    self.cpu_time = []
    self.ind_per_tree = []
    self.x_support = x_support
    self.y_support = y_support
    self.bootstrap = bootstrap

    total_obs = len(Y)
    
    if parallel == False:
        for i in range(self.n_estimators):
            if i%50==0:
                print('Ensemble Tree: ',i)
            if cpu_time: start_time = time.process_time()
             
            if bootstrap:
                # sample with replacement from target data
                if isinstance(boot_prob,(list,np.ndarray)):
                    sample_ind = np.random.choice(np.arange(total_obs), size = total_obs, replace=True, p = boot_prob)
                else:
                    sample_ind = np.random.choice(np.arange(total_obs), size = total_obs, replace=True)
                    #sample_ind = np.random.choice(np.arange(total_obs), size = int(.66*total_obs), replace=False)
                    
                #sample_ind = np.unique(np.sort(sample_ind))
                self.ind_per_tree.append(sample_ind)                
                tree_Y = Y[sample_ind]
                tree_X = X[sample_ind]
            else:
                tree_Y = Y
                tree_X = X
            #Select subset of predictors
            #col = np.random.choice(range(num_features), p_select, replace = False)
            #temp_X = X[:,col]
            #Train decision tree        
            new_tree = GreedyPrescriptiveTree(D = self.D, Nmin = self.Nmin, 
                                              type_split = self.type_split, max_features = self.max_features)
            new_tree.fit(tree_X, tree_Y, self.xbins, bin_features = self.bin_features, **self.decision_kwargs)
            if cpu_time: self.cpu_time.append(time.process_time()-start_time)
            #Map tree features to actual columns from original dataset
            #new_tree.feature = [col[f] if f>=0 else f for f in new_tree.feature]
            #Store result
            self.trees.append(new_tree)
    else:
        if cpu_time: start_time = time.process_time()
        def fit_tree(X, Y, self):
            new_tree = GreedyPrescriptiveTree(D=self.D, Nmin=self.Nmin,
                                            type_split=self.type_split, max_features=self.max_features)
            new_tree.fit(X, Y, **self.decision_kwargs)
            return new_tree
            
        self.trees = Parallel(n_jobs = n_jobs, verbose=10)(delayed(fit_tree)(X, Y, self)for i in range(self.n_estimators))
        if cpu_time: self.cpu_time.append(time.process_time()-start_time)

    raw_importances = np.array([self.trees[i].feat_importance/self.trees[i].feat_importance.sum() for i in range(self.n_estimators)] )
    
    self.feat_importance_mean = raw_importances.mean(axis = 0)
    self.feat_importance_std = raw_importances.std(axis = 0)

  def apply(self, X):
     ''' Function that returns the Leaf id for each point. Similar to sklearn's implementation
     '''
     Leaf_id = np.zeros((X.shape[0], self.n_estimators))
     for j, tree in enumerate(self.trees):
         for i in range(X.shape[0]): 
             x0 = X[i:i+1,:]
             node = 0
             while ((tree.children_left[node] != -1) and (tree.children_right[node] != -1)):
                 if x0[:, tree.feature[node]] < tree.threshold[node]:
                     node = tree.children_left[node]
                     #print('Left')
                 elif x0[:,tree.feature[node]] >= tree.threshold[node]:
                    node = tree.children_right[node]
                    #print('Right')
                 #print('New Node: ', node)
             Leaf_id[i,j] = tree.Node_id[node]
     return Leaf_id
 
  def find_weights(self, testX, trainX):
     ''' Return local weights'''
     
     #Step 1: Estimate weights for weighted SAA
     Leaf_nodes = self.apply(trainX) # nObs*nTrees: a_ij shows the leaf node for observation i in tree j
     Index = self.apply(testX) # Leaf node for test set
     nTrees = self.n_estimators
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

  def find_OOB_weights(self, x0, boot_trainX, oob_trees):
     ''' Return local weights, only for trees that *did not* use testX for training
         boot_trainX: bootstrapped training sample'''
     
     #Step 1: Estimate weights for weighted SAA
     
     Leaf_nodes = self.apply(boot_trainX)[:,oob_trees] # nObs*nTrees: a_ij shows the leaf node for observation i in tree j
     x0_leaf_ind = self.apply(x0)[:,oob_trees] # Leaf node for test set
     
     nTrees = len(oob_trees)

     #Find observations in terminal nodes/leaves (all trees)
     obs = 1*(x0_leaf_ind.repeat(len(boot_trainX), axis = 0) == Leaf_nodes)
     #Cardinality of leaves
     cardinality = np.sum(obs, axis = 0).reshape(-1,1).T.repeat(len(boot_trainX), axis = 0)

     #Update weights
     oob_Weights = (obs/cardinality).sum(axis = 1)/nTrees

     return oob_Weights
 
  def estimate_OOB_score(self, trainX, trainY, parallel = False, return_estimates = False):
     ''' Estimate Prescriptive OOB cost for a tree-based ensemble trained with bootstrapping
         testX == trainX in this case'''

     # set "test" set features
     testX = trainX.copy()

     #target_mask = trainY[:,1]==1
     # consider only data from target distribution
     #target_X = trainX[target_mask]
     #target_Y = trainY[target_mask]
     
     # If data are pandas, transform to numpys
     targetX = trainX.copy()
     targetY = trainY.copy()
     
     if isinstance(targetX, pd.DataFrame) or isinstance(targetX, pd.core.series.Series):
         targetX = targetX.copy().values        
     if isinstance(targetY, pd.DataFrame) or isinstance(targetY, pd.core.series.Series):
         targetY = targetY.copy().values
     targetY = targetY.reshape(-1,1)
     
     # Step 1: For each observation of the training set, find trees grown **without** using during training 
     List_oob_trees = []
     for i, y_i in enumerate(trainY):
         # check if observation belongs to target data set
         tree_ind = []
         for j in range(self.n_estimators):
             if i not in self.ind_per_tree[j]:
                tree_ind.append(j)
         List_oob_trees.append(tree_ind)
     
     '''
     # Step 2: Estimate weights for weighted SAA
     Leaf_nodes = self.apply(trainX) # nObs*nTrees: a_ij shows the leaf node for observation i in tree j
     Index = self.apply(testX) # Leaf node for test set
     nTrees = self.n_estimators
     
     Weights = np.zeros(( len(targetX), len(trainX) ))
     #print(Weights.shape)
     #Estimate sample weights
     print('Retrieving weights...')
     for i in range(len(testX)):
         #if trainY[i,1] == 0:
         #    continue
         # trees that did *not* use x0 to train
         temp_trees = List_oob_trees[i]
         if temp_trees == []: continue
     
         # New query point
         x0 = Index[i:i+1, temp_trees]

         # Find observations in terminal nodes/leaves
         # **only trees that did not use sample i for training**
         #!!!! does it include the i-th training observation???
         
         obs = 1*(x0.repeat(len(trainX), axis = 0) == Leaf_nodes[:,temp_trees])
         #Cardinality of leaves
         cardinality = np.sum(obs, axis = 0).reshape(-1,1).T.repeat(len(trainX), axis = 0)
         #Update weights
         Weights[i,:] = (obs/cardinality).sum(axis = 1)/len(temp_trees)
     
     print('Optimizing Prescriptions...')
     #Check that weigths are correct (numerical issues)
     #assert( all(Weights.sum(1) >= 1-1e-4))
     #assert( all(Weights.sum(1) <= 1+1e-4))
     # loop over trees
     
     '''

     '''
     for t, tree in enumerate(self.trees):
         # find observations not used in training
         ind = [i for i in range(len(trainY)) if (i not in self.ind_per_tree[t]) and (trainY[i,1]==1)]
         oob_Y = trainY[ind,0]
         oob_X = trainX[ind]
         oob_prescriptions = tree.predict(oob_X)
         oob_error = np.square(oob_Y - oob_prescriptions.reshape(-1)).mean()
         OOB_score += oob_error
     OOB_score/=self.n_estimators
     '''
     
     #### Estimate OOB score:
     # !!!! each tree has its own training set due to bootstrapping, make sure to update it
     oob_prescriptions = np.zeros((targetY.shape))
     OOB_score = 0
     
     mask_oob_trees = [] # to check whether an observation is not be included in any tree/ has been used in all of the trees
     
     # sort observations for analytical solution in newsvendor
     sort_Y = np.sort(trainY.copy())                
     argsort_g = np.argsort(trainY.copy())
     
     # Step 3: loop over training observations
     for i in range(len(testX)):
         x0 = testX[i:i+1,:]

         if len(List_oob_trees[i]) == 0:
             mask_oob_trees.append(False) # ignore this observation at validation
             continue
         else:
             mask_oob_trees.append(True) # include this observation at validation

         if i%1000 == 0:
             print('Observation ', i) 
         
         # !!!! Step 4: Estimate OOB prescription: 
         # For each observation create a surrogate training set with the union of training observations for each bootstrap sample
         # !!!! the actual observation should not be included 
         # Alternative way to do this: find prescription for each OOB tree and average them
         # -------------------
         oob_trees = List_oob_trees[i]
         temp_obs_ind = np.unique(np.array([self.ind_per_tree[t] for t in oob_trees]))
         boot_temp_Y = trainY[temp_obs_ind]
         boot_temp_trainX = trainX[temp_obs_ind]
         oob_weights = self.find_OOB_weights(x0, boot_temp_trainX, oob_trees)
        
         _, tree_oob_prescr = opt_problem(boot_temp_Y, weights = oob_weights, 
                                            prescribe = True, **self.decision_kwargs)

         oob_prescriptions[i] = tree_oob_prescr    
         # -------------------
         
         # !!!!!! (not sure about this step)
         '''
         if self.bootstrap and self.test_problem == 'mse':
             temp_pred = np.array([self.trees[t].predict(x0) for t in List_oob_trees[i]])         
             oob_prescriptions[i] = temp_pred.mean()
             
             #oob_prescriptions[i] = Weights[i]@trainY
             
         #elif self.bootstrap and self.test_problem == 'newsvendor':             
             #oob_prescriptions[i] = sort_Y[np.where(Weights[i][argsort_g].cumsum()>crit_quant)][0]             
             #temp_pred = np.array([self.trees[t].predict_constr(testX[i:i+1], trainX, trainY) for t in List_of_trees[i]])                      
             #oob_prescriptions[i] = temp_pred.mean()
             
         else:            
             #mask = Weights[i]>0
             #_, temp_prescription = opt_problem(trainY[mask], weights = Weights[i][mask], prescribe = True, **self.decision_kwargs)
             #oob_prescriptions[i] = temp_prescription  
             
             temp_prescr = []
             
             # loop over each OOB tree, find the prescription
             for tree_ind in List_oob_trees[i]:                 
                temp_obs_ind = np.unique(self.ind_per_tree[tree_ind])
                card = len(temp_obs_ind)
                # find bootstrap training set
                boot_temp_Y = trainY[temp_obs_ind]
                boot_temp_trainX = trainX[temp_obs_ind]
                
                # Predict from single prescriptive tree (!!!! alternatively: find weights here, use opt_problem)
                tree_oob_prescr = self.trees[tree_ind].predict_constr(x0, boot_temp_trainX, boot_temp_Y,
                                                                      self.decision_kwargs) 
                
                #_, temp_oob_prescr = opt_problem(trainY[temp_indices], 
                #weights = np.ones(len(temp_indices))*1/card, prescribe = True, **self.decision_kwargs)

                temp_prescr.append(tree_oob_prescr)
             oob_prescriptions[i] = np.array(temp_prescr).mean()
 
         '''
         #else:
     mask_oob_trees = np.array(mask_oob_trees)
     
     '''
     # estimate OOB score given loss function
     if self.test_problem == 'mse':
         OOB_score = mse(oob_prescriptions[mask_oob_trees], targetY[mask_oob_trees])

     elif self.test_problem == 'newsvendor':
         OOB_score = newsvendor_loss(oob_prescriptions[mask_oob_trees], targetY[mask_oob_trees], q = crit_quant)
     
     #!!!!!! change this to the actual optimization problem later on
     elif self.test_problem == 'cvar':
         OOB_score = newsvendor_cvar_loss(oob_prescriptions[mask_oob_trees], targetY[mask_oob_trees], q = crit_quant)
     elif self.test_problem == 'reg_trad':
         xx
     '''
     # evaluate task loss
     OOB_score = task_loss(oob_prescriptions[mask_oob_trees], targetY[mask_oob_trees], self.decision_kwargs['problem'], 
                           self.decision_kwargs)

     self.OOB_score = OOB_score
     self.oob_loc_prescriptions = oob_prescriptions
     
     if return_estimates:
         return OOB_score, oob_prescriptions
     else:
         return OOB_score

  def convex_combination_tuning(self, localX, localY, anchor_distr, lambda_ = np.arange(0, 1.1, .1).round(2),
                                max_oob_samples = 200):
      ''' Hyperparameter tuning for convex combination between local and anchor distribution, based on OOB error.
          for simplicity, we iterate over all the fixed support points of contextual information      
          Inputs: 
              localX, localY: local features and target variable
              anchor_distr: nD array where n = number of features + 1 
              lambda_: grid of values for convex combination 
              crit_quant: critical quantile (optional, used for newsvendor only)'''
         
      self.anchor_distr = anchor_distr
      self.lambda_ = lambda_
      
      targetX = localX.copy()
      targetY = localY.copy()

      if isinstance(localX, pd.DataFrame) or isinstance(localX, pd.core.series.Series):
          targetX = localX.copy().values        
      if isinstance(localY, pd.DataFrame) or isinstance(localY, pd.core.series.Series):
          targetY = localY.copy().values
      
      # turn vectors to arrays
      if targetY.ndim == 1: targetY = targetY.reshape(-1,1)
      if targetX.ndim == 1: targetX = targetX.reshape(-1,1)
      
      local_trainX = targetX.copy()
      local_trainY = targetY.copy()
     
      
      ##############
      ### Choice 1: estimate local/global decisions, find combination of decisions
      '''
      # Estimate oob score and local prescriptions
      self.estimate_OOB_score(localX, localY, crit_quant = crit_quant, return_estimates = True)
      loc_prescriptions = self.oob_loc_prescriptions.reshape(-1,1)
      oob_error_loc = self.OOB_score
      print('OOB-Local')
      print(oob_error_loc)
            
      # find global prescriptions with barycenter
      w_brc_pred = np.zeros(targetY.shape)
      for row, x0 in enumerate(targetX):                        
          id_ = np.where(x0 == self.x_support)[0][0]           
          if self.test_problem == 'mse':       
              w_brc_pred[row] = barycenter[id_]@self.y_support
          #elif self.test_problem == 'newsvendor':
          #    w_brc_pred[row] = self.y_support[np.where(barycenter[id_].cumsum()>crit_quant)][0]
          else:
              temp_weights = barycenter[id_]
              _, temp_prescription = opt_problem(self.y_support, weights = temp_weights, prescribe = True, 
                                                 **self.decision_kwargs)
              w_brc_pred[row] = temp_prescription
      #plt.plot(w_brc_pred, '--')
      #plt.plot(loc_prescriptions, '-.')
      #plt.plot(targetY, color = 'black')
      #plt.show()

      # Find combinations of decisions, evaluate lambda
      for l in self.lambda_:
          comb_oob_prescriptions = l*loc_prescriptions + (1-l)*w_brc_pred
          #!!!! estimate prescriptive cost here
          if self.test_problem == 'mse':
              oob_comb = mse(comb_oob_prescriptions, targetY)
          elif self.test_problem == 'newsvendor':
              oob_comb = newsvendor_loss(comb_oob_prescriptions, targetY, q = crit_quant)
              
          OOB_loss.append(oob_comb)
      
      self.best_lambda_ = self.lambda_[np.argmin(np.array(OOB_loss))]
      '''
      ##############
      ### Choice 2: create convex combination of distributions, solve a separate problem for each lambda

      print('Estimate predictive OOB cost for all lambda')
      
      OOB_loss = []      
      # for each training observation, find trees that *did not* use it for training
      List_oob_trees = []
      for i, x_i in enumerate(local_trainX):
          # check if observation belongs to target data set
          tree_ind = []
          for j in range(self.n_estimators):
              if i not in self.ind_per_tree[j]:
                  tree_ind.append(j)
          List_oob_trees.append(tree_ind)
      
      num_oob_samples = min(len(targetY), max_oob_samples)
      comb_oob_prescriptions = np.zeros((num_oob_samples, len(self.lambda_) ))
      
      # !!!!! Need to esnure that you don't count observations not included in any trained tree
      mask_oob_trees = [] # to check whether an observation might not be included in any tree
      
      # iterate over training observations
      for i in range(num_oob_samples):      
          
          x_i = local_trainX[i:i+1]

          # find weights for both local approach and anchor distribution 
            
          ### Local: create bootstrapped training sample
          oob_trees = List_oob_trees[i]
          
          if len(oob_trees) == 0:
              mask_oob_trees.append(False) # ignore this observation at validation
              continue
          else:
              mask_oob_trees.append(True) # include this observation at validation

          temp_obs_ind = np.unique(np.array([self.ind_per_tree[t] for t in oob_trees]))
          boot_temp_Y = local_trainY[temp_obs_ind]
          boot_temp_trainX = local_trainX[temp_obs_ind]
              
          # OOB weights for local training set
          oob_weights = self.find_OOB_weights(x_i, boot_temp_trainX, oob_trees)
                            
          ### Barycenter: weights for barycenter
          if self.x_support.ndim == 1:
              id_ = np.where(x_i[0] == self.x_support)[0][0]           
          else:
              id_ = np.where((x_i == self.x_support).all(1))[0][0]           
                  
          brc_weights = self.anchor_distr[id_]

          # iterate over lambdas, find prescriptions
          for col, l in enumerate(self.lambda_):
              ###################
              #!!!!!!!!!!!!!!!!! here you should estimate an update that interpolates between local and anchor
              
              # concat weights and Y observations, use lambda for convex combination
              f_weights = np.concatenate([l*oob_weights, (1-l)*brc_weights])
        
              f_y_target = np.row_stack((boot_temp_Y, self.y_support.reshape(-1,1)))
              mask = f_weights>0

              _, tree_oob_prescr = opt_problem(f_y_target[mask], weights = f_weights[mask], 
                                         prescribe = True, **self.decision_kwargs)


              comb_oob_prescriptions[i, col] = tree_oob_prescr    
        
      target_Y_oob = targetY[:max_oob_samples]
      for col, l in enumerate(self.lambda_):
          '''
          if self.test_problem == 'mse':
              oob_comb = mse(comb_oob_prescriptions[mask_oob_trees, col:col+1], target_Y_oob[mask_oob_trees])
          elif self.test_problem == 'newsvendor':
              oob_comb = newsvendor_loss(comb_oob_prescriptions[mask_oob_trees, col:col+1], target_Y_oob[mask_oob_trees], q = crit_quant)
          elif self.test_problem == 'cvar':
              oob_comb = newsvendor_cvar_loss(comb_oob_prescriptions[mask_oob_trees, col:col+1], target_Y_oob[mask_oob_trees], q = crit_quant)
          elif self.test_problem == 'reg_trad':
              oob_comb = reg_trad_loss(comb_oob_prescriptions[mask_oob_trees, col:col+1], target_Y_oob[mask_oob_trees], q = crit_quant)
         '''
          oob_comb = task_loss(comb_oob_prescriptions[mask_oob_trees, col:col+1], target_Y_oob[mask_oob_trees],
                               self.decision_kwargs['problem'], crit_quant = self.decision_kwargs['crit_quant'], 
                                                                  risk_aversion = self.decision_kwargs['risk_aversion'], epsilon = self.decision_kwargs['epsilon'])
          OOB_loss.append(oob_comb)
      
      self.best_lambda_conv_comb = self.lambda_[np.argmin(np.array(OOB_loss))]
      
      return OOB_loss

  def interpolation_tuning(self, localX, localY, anchor_distr, lambda_ = np.arange(0, 1.1, .1).round(2),
                               max_oob_samples = 100, type_interpolation = 'barycenter'):
      
      ''' Hyperparameter tuning for interpolation using 2-Wasserstein barycenter between local and anchor distribution, based on OOB error.
          Inputs: 
              localX, localY: local features and target variable
              anchor_distr: nD array where n = number of features + 1 
              lambda_: grid of values for convex combination 
              crit_quant: critical quantile (optional, used for newsvendor only)'''
         
      self.anchor_distr = anchor_distr
      self.lambda_ = lambda_
      
      targetX = localX.copy()
      targetY = localY.copy()

      if isinstance(localX, pd.DataFrame) or isinstance(localX, pd.core.series.Series):
          targetX = localX.copy().values        
      if isinstance(localY, pd.DataFrame) or isinstance(localY, pd.core.series.Series):
          targetY = localY.copy().values
      
      # turn vectors to arrays
      if targetY.ndim == 1: targetY = targetY.reshape(-1,1)
      if targetX.ndim == 1: targetX = targetX.reshape(-1,1)
      
      local_trainX = targetX.copy()
      local_trainY = targetY.copy()
     
      ##############
      print('Estimate predictive OOB cost for all lambda')
      
      OOB_loss = []      
      # for each training observation, find trees that *did not* use it for training
      List_oob_trees = []
      for i, x_i in enumerate(local_trainX):
          # check if observation belongs to target data set
          tree_ind = []
          for j in range(self.n_estimators):
              if i not in self.ind_per_tree[j]:
                  tree_ind.append(j)
          List_oob_trees.append(tree_ind)
      
      num_oob_samples = min(len(targetY), max_oob_samples)
      comb_oob_prescriptions = np.zeros((num_oob_samples, len(self.lambda_) ))
      
      # !!!!! Need to esnure that you don't count observations not included in any trained tree
      mask_oob_trees = [] # to check whether an observation might not be included in any tree
      

      # find weights for all lambda iterations and all OOB observations
      print('Finding weights for all OOB observations and all lambdas...')
      
      # list of len==num_oob_samples, each entry is an array with the weights for all lambdas of the i-th observation

      OOB_weights = []
      for i in range(num_oob_samples):      

          # list of weights for all gamma for i-th observation
          OOB_weights_i_lambda = np.zeros((len(self.lambda_), len(self.y_support)))

          x_i = local_trainX[i:i+1]

          # find weights for both local approach and anchor distribution 
            
          ### Local: create bootstrapped training sample
          oob_trees = List_oob_trees[i]
          
          if len(oob_trees) == 0:
              mask_oob_trees.append(False) # ignore this observation at validation
              continue
          else:
              mask_oob_trees.append(True) # include this observation at validation

          temp_obs_ind = np.unique(np.array([self.ind_per_tree[t] for t in oob_trees]))
          boot_temp_Y = local_trainY[temp_obs_ind]
          boot_temp_trainX = local_trainX[temp_obs_ind]
              
          # OOB weights for local training set
          oob_weights = self.find_OOB_weights(x_i, boot_temp_trainX, oob_trees)
                            
          ### Barycenter: weights for barycenter
          if self.x_support.ndim == 1:
              id_ = np.where(x_i[0] == self.x_support)[0][0]           
          else:
              id_ = np.where((x_i == self.x_support).all(1))[0][0]           
          
          # map OOB weights onto the discrete support locations
          oob_weights_supp = wemp_to_support(oob_weights, boot_temp_Y, self.y_support).reshape(-1)
          brc_weights = self.anchor_distr[id_]

          for row, l in enumerate(self.lambda_):
              # estimate barycentric interpolation
              if  l == 1:
                  inter_weights = oob_weights_supp
              elif l == 0:
                  inter_weights = brc_weights
              elif (l>0)and(l<1)and(type_interpolation == 'average'):
                  # convex combination
                  inter_weights = l*oob_weights_supp + (1-l)*brc_weights
              elif (l>0)and(l<1)and(type_interpolation == 'barycenter'):      
                  inter_weights,_,_ = wass2_barycenter_1D(2*[self.y_support.reshape(-1,1)], 
                                                  [oob_weights_supp, brc_weights], lambda_coord=[l, 1-l])

              # update array of weights
              OOB_weights_i_lambda[row] = inter_weights
              
          # update list of weights for all OBB observations
          OOB_weights.append(OOB_weights_i_lambda)
          
      # Generate prescriptions for all OOB observations
      print('Generating OOB prescriptions...')
      
      # !!!! Stack everything, declare problem once, reshape (much faster to solve for multiple instances)
      stacked_OOB_weights = np.row_stack(OOB_weights)
      stacked_OOB_prescriptions = solve_opt_prob(self.y_support, stacked_OOB_weights, 
                                          self.decision_kwargs['problem'], 
                                          model_once = True, crit_quant = self.decision_kwargs['crit_quant'], 
                                          risk_aversion = self.decision_kwargs['risk_aversion'], epsilon = self.decision_kwargs['epsilon'])          
  
      #OOB_prescriptions_all = = solve_opt_prob(self.y_support, OOB_weights, 
      #                                              self.decision_kwargs['problem'], model_once = True, crit_quant = self.decision_kwargs['crit_quant'], 
      #                                              risk_aversion = self.decision_kwargs['risk_aversion'], epsilon = self.decision_kwargs['epsilon'])          
      
      comb_oob_prescriptions = stacked_OOB_prescriptions.reshape(-1,len(self.lambda_))

      '''
      for i in range(num_oob_samples):      
          # prescriptions for i-th observations, over all values of lambda
          temp_prescriptions = solve_opt_prob(self.y_support, OOB_weights[i], 
                                              self.decision_kwargs['problem'], 
                                              model_once = True, crit_quant = self.decision_kwargs['crit_quant'], 
                                              risk_aversion = self.decision_kwargs['risk_aversion'], epsilon = self.decision_kwargs['epsilon'])          
          comb_oob_prescriptions[i,:] = temp_prescriptions
      '''
      
      '''
      ###################################
      # iterate over training observations
      for i in range(num_oob_samples):      
          x_i = local_trainX[i:i+1]

          # find weights for both local approach and anchor distribution 
            
          ### Local: create bootstrapped training sample
          oob_trees = List_oob_trees[i]
          
          if len(oob_trees) == 0:
              mask_oob_trees.append(False) # ignore this observation at validation
              continue
          else:
              mask_oob_trees.append(True) # include this observation at validation

          temp_obs_ind = np.unique(np.array([self.ind_per_tree[t] for t in oob_trees]))
          boot_temp_Y = local_trainY[temp_obs_ind]
          boot_temp_trainX = local_trainX[temp_obs_ind]
              
          # OOB weights for local training set
          oob_weights = self.find_OOB_weights(x_i, boot_temp_trainX, oob_trees)
                            
          ### Barycenter: weights for barycenter
          if self.x_support.ndim == 1:
              id_ = np.where(x_i[0] == self.x_support)[0][0]           
          else:
              id_ = np.where((x_i == self.x_support).all(1))[0][0]           
                  
          brc_weights = self.anchor_distr[id_]

          # iterate over lambdas, find prescriptions
          for col, l in enumerate(self.lambda_):
              if l == 1:
                  mask = oob_weights>0
                  _, tree_oob_prescr = opt_problem(boot_temp_Y[mask], weights = oob_weights[mask], 
                                             prescribe = True, **self.decision_kwargs)

              elif l == 0:
                  mask = brc_weights>0
                  _, tree_oob_prescr = opt_problem(self.y_support.reshape(-1,1)[mask], weights = brc_weights[mask], 
                                             prescribe = True, **self.decision_kwargs)
              else:
                  
                  interpol_weights,_,_ = wass2_barycenter_1D([boot_temp_Y, self.y_support.reshape(-1,1)], 
                                                         [oob_weights, brc_weights], lambda_coord=[l, 1-l])
                  
                  mask = interpol_weights>0
                  _, tree_oob_prescr = opt_problem(self.y_support.reshape(-1,1)[mask], weights = interpol_weights[mask], prescribe = True, **self.decision_kwargs)


              comb_oob_prescriptions[i, col] = tree_oob_prescr    
      
      '''
      
      # evaluate task-loss for all lambda
      target_Y_oob = targetY[:max_oob_samples]
      
      for col, l in enumerate(self.lambda_):
          oob_comb = task_loss(comb_oob_prescriptions[mask_oob_trees, col:col+1], target_Y_oob[mask_oob_trees],
                               self.decision_kwargs['problem'], crit_quant = self.decision_kwargs['crit_quant'], 
                               epsilon = self.decision_kwargs['epsilon'], risk_aversion = self.decision_kwargs['risk_aversion'])
             
          OOB_loss.append(oob_comb)
      if type_interpolation == 'barycenter':
          self.best_lambda_interpol = self.lambda_[np.argmin(np.array(OOB_loss))]
      elif type_interpolation == 'average':
          self.best_lambda_conv_comb = self.lambda_[np.argmin(np.array(OOB_loss))]

      return OOB_loss
  
  def comb_predict(self, testX, trainX, trainY, barycenter, parallel = False):
     ''' Generate combined predictions from local method and anchor distribution'''
     
     #Step 1: Estimate weights for weighted SAA
     Leaf_nodes = self.apply(trainX) # nObs*nTrees: a_ij shows the leaf node for observation i in tree j
     Index = self.apply(testX) # Leaf node for test set
     nTrees = self.n_estimators
     Weights = np.zeros(( len(testX), len(trainX) ))
     #print(Weights.shape)
     #Estimate sample weights
     print('Retrieving weights...')
     obs_card = np.zeros(( len(testX), len(trainX) ))
     for i in range(len(testX)):
         #New query point
         x0 = Index[i:i+1, :]
         
         #Find observations in terminal nodes/leaves (all trees)
         obs = 1*(x0.repeat(len(trainX), axis = 0) == Leaf_nodes)
         #Cardinality of leaves
         cardinality = np.sum(obs, 0).reshape(-1,1).T.repeat(len(trainX), 0)
         #Update weights
         Weights[i,:] = (obs/cardinality).sum(axis = 1)/nTrees
         obs_card[i,:] = (obs).sum(axis = 1)

     print('Optimizing Prescriptions...')
     #Check that weights are correct (numerical issues)
     assert( all(Weights.sum(axis = 1) >= 1-10e-4))
     assert( all(Weights.sum(axis = 1) <= 1+10e-4))

     # Find local prescriptions
     
     '''
     local_Prediction = []
        
     for i in range(len(testX)):
         if i%1000 == 0:
             print('Observation ', i) 
         if self.bootstrap:
             temp_prescription = np.array([t.predict(testX[i:i+1]).reshape(-1) for t in self.trees])         
             temp_prescription = temp_prescription.mean()
         else:
             mask = Weights[i]>0
             _, temp_prescription = opt_problem(trainY[mask], weights = Weights[i][mask], prescribe = True, **self.decision_kwargs)
             #print(temp_prescription)
             #temp_Y = trainY[mask,0].copy()
             #temp_w = Weights[i][mask].copy()
             #cdf = temp_w[np.argsort(temp_Y)].cumsum()
             #print(np.sort(temp_Y)[np.where(cdf>=0.7)[0][0]])
        
         local_Prediction.append(temp_prescription)  
     '''
     local_Prescr = []
     brc_Prescr = []
     comb_Prescr = []
                            
     if self.test_problem == 'mse':
         local_Prescr = (Weights@trainY).reshape(-1)
         if trainY.ndim == 1:
             concat_Y = np.concatenate((trainY, self.y_support))
         else:
             concat_Y = np.concatenate((trainY, self.y_support.reshape(-1,1)))

         # Find global predictions w barycenter
         for row, x0 in enumerate(testX):   
             if self.x_support.ndim==1:
                 id_ = np.where(x0 == self.x_support)[0][0]       
             else:
                 id_ = np.where((x0 == self.x_support).all(1))[0][0]
                 
             temp_prescription = barycenter[id_]@self.y_support
    
             brc_Prescr.append(temp_prescription)  
             
             concat_W = np.concatenate((self.best_lambda_*Weights[row], (1-self.best_lambda_)*barycenter[id_]))
             comb_Prescr.append(concat_W@concat_Y) 
             
     elif self.test_problem == 'newsvendor':       
         # Find global predictions w barycenter
         sort_Y = np.sort(trainY.copy().reshape(-1), axis=0)                
         argsort_g = np.argsort(trainY.copy().reshape(-1), axis=0)

         if trainY.ndim == 1:
             concat_Y = np.concatenate((trainY, self.y_support))
         else:
             concat_Y = np.concatenate((trainY, self.y_support.reshape(-1,1)))
         sort_concat_Y = np.sort(concat_Y.copy().reshape(-1), axis=0)
         concat_argsort_g = np.argsort(concat_Y.copy().reshape(-1), axis=0)
         
         for row, x0 in enumerate(testX):                        
             # local prescription
             local_Prescr.append(sort_Y[np.where(Weights[row][argsort_g].cumsum()>=self.decision_kwargs['crit_quant'])][0])              

             # global prescription
             if self.x_support.ndim==1:
                 id_ = np.where(x0 == self.x_support)[0][0]       
             else:
                 id_ = np.where((x0 == self.x_support).all(1))[0][0]
             
             brc_Prescr.append(self.y_support[np.where(barycenter[id_].cumsum() >=self.decision_kwargs['crit_quant'])][0]) 
             
             concat_W = np.concatenate((self.best_lambda_*Weights[row], (1-self.best_lambda_)*barycenter[id_]))
             comb_Prescr.append(sort_concat_Y[np.where(concat_W[concat_argsort_g].cumsum() >= self.decision_kwargs['crit_quant'])][0]) 
             
     else: 
         # solve full optimization problem
         if trainY.ndim == 1:
             concat_Y = np.concatenate((trainY, self.y_support))
         else:
             concat_Y = np.concatenate((trainY, self.y_support.reshape(-1,1)))
             
         # Iterate over test set
         for row, x0 in enumerate(testX):                        
            # local prescription
            mask = Weights[row]>0
            _, temp_prescription = opt_problem(trainY[mask], weights = Weights[row][mask], prescribe = True, **self.decision_kwargs)
            local_Prescr.append(temp_prescription)  

            # global prescription with barycenter
            if self.x_support.ndim==1:
                id_ = np.where(x0 == self.x_support)[0][0]       
            else:
                id_ = np.where((x0 == self.x_support).all(1))[0][0]
                
            _, temp_prescription = opt_problem(self.y_support, weights = barycenter[id_], prescribe = True, **self.decision_kwargs)    
            brc_Prescr.append(temp_prescription)  
            
            # combined prescription
            f_weights = np.concatenate([self.best_lambda_*Weights[row], 
                                        (1-self.best_lambda_)*barycenter[id_]])
            mask = f_weights>0

            _, temp_comb_prescr = opt_problem(concat_Y[mask], weights = f_weights[mask], 
                                       prescribe = True, **self.decision_kwargs)


            comb_Prescr.append(temp_comb_prescr)    
            
     local_Prescr = np.array(local_Prescr)
     brc_Prescr = np.array(brc_Prescr)    
     comb_Prescr = np.array(comb_Prescr)
     #comb_Prescr = self.best_lambda_*local_Prescr + (1-self.best_lambda_)*brc_Prescr                  
                      
     return comb_Prescr.reshape(-1,1), local_Prescr.reshape(-1,1), brc_Prescr.reshape(-1,1)
  
  def predict_constr(self, testX, trainX, trainY, parallel = False):
     ''' Generate predictive prescriptions
         testX: test feature observations
         trainX: training feature observations
         trainY: training target observations
         '''
     
     #Step 1: Estimate weights for weighted SAA
     Leaf_nodes = self.apply(trainX) # nObs*nTrees: a_ij shows the leaf node for observation i in tree j
     Index = self.apply(testX) # Leaf node for test set
     nTrees = self.n_estimators
     
     Weights = np.zeros(( len(testX), len(trainX) ))
     #print(Weights.shape)
     #Estimate sample weights
     print('Retrieving weights...')
     obs_card = np.zeros(( len(testX), len(trainX) ))
     for i in range(len(testX)):
         #New query point
         x0 = Index[i:i+1, :]
         
         #Find observations in terminal nodes/leaves (all trees)
         obs = 1*(x0.repeat(len(trainX), axis = 0) == Leaf_nodes)
         #Cardinality of leaves
         cardinality = np.sum(obs, 0).reshape(-1,1).T.repeat(len(trainX), 0)
         #Update weights
         Weights[i,:] = (obs/cardinality).sum(axis = 1)/nTrees
         obs_card[i,:] = (obs).sum(axis = 1)
         
     
     print('Optimizing Prescriptions...')
     #Check that weights are correct (numerical issues)
     assert( all(Weights.sum(axis = 1) >= 1-10e-4))
     assert( all(Weights.sum(axis = 1) <= 1+10e-4))

     Prescription = []#np.zeros((testX.shape[0],1))
             
     for i in range(len(testX)):
         if i%1000 == 0: print('Observation ', i) 
         #if self.bootstrap:
         #    temp_prescription = np.array([t.predict(testX[i:i+1]).reshape(-1) for t in self.trees])         
         #    temp_prescription = temp_prescription.mean()
         #else:
         mask = Weights[i]>0
         _, temp_prescription = opt_problem(trainY[mask], weights = Weights[i][mask], prescribe = True, **self.decision_kwargs)
                 #print(temp_prescription)
                 #temp_Y = trainY[mask,0].copy()
                 #temp_w = Weights[i][mask].copy()
                 #cdf = temp_w[np.argsort(temp_Y)].cumsum()
                 #print(np.sort(temp_Y)[np.where(cdf>=0.7)[0][0]])
             
         Prescription.append(temp_prescription)  
         
     return np.array(Prescription)
 
  def cost_oriented_forecast(self, testX, trainX, trainY, parallel = False):
     ''' Generate Cost-/Value-Oriented Forecasts'''
     
     #Step 1: Estimate weights for weighted SAA
     Leaf_nodes = self.apply(trainX) # nObs*nTrees: a_ij shows the leaf node for observation i in tree j
     Index = self.apply(testX) # Leaf node for test set
     nTrees = self.n_estimators
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
         
     #Cost-oriented forecasts
     Point_Prediction = Weights@trainY
     
     return Point_Prediction
    


############## Auxiliary functions to evaluate task loss 

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
        return (1-kwargs['risk_aversion'])*pinball_loss.mean() + kwargs['risk_aversion']*square_loss.mean()


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

# aux functions
'''
def mse(pred, actual):                
    return np.square(pred.reshape(-1) - actual.reshape(-1)).mean()

def reg_trad_loss(pred, actual, q = 0.5, risk_aversion = 0.5):   
    pred_copy = pred.copy().reshape(-1)
    actual_copy = actual.copy().reshape(-1)
    deviation = actual_copy - pred_copy
    pinball_loss = np.maximum(q*deviation, (q-1)*deviation)    
    
    return (1-risk_aversion)*pinball_loss.mean() + risk_aversion*(deviation@deviation)

def newsvendor_loss(pred, actual, q = 0.5):   
    if actual.ndim != pred.ndim:
        return np.maximum(q*(actual.reshape(-1) - pred.reshape(-1)), (q-1)*(actual.reshape(-1) - pred.reshape(-1))).mean()
    else:          
        return np.maximum(q*(actual - pred), (q-1)*(actual - pred)).mean()


def newsvendor_cvar_loss(pred, actual, q = 0.5, risk_aversion = 0.5, e = 0.05):   
    pred_copy = pred.copy().reshape(-1)
    actual_copy = actual.copy().reshape(-1)
    
    pinball_loss = np.maximum(q*(actual_copy - pred_copy), (q-1)*(actual_copy - pred_copy))    
    cvar_mask = pinball_loss>=np.quantile(pinball_loss, 1-e)
    
    return (1-risk_aversion)*pinball_loss.mean() + risk_aversion*pinball_loss[cvar_mask].mean()
'''
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
            deviation = m.addMVar(n_scen, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'offer')
            loss = m.addMVar(n_scen, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'aux')
            
            # constraints
            m.addConstr(deviation == (target_scen - offer) )            
            m.addConstr(loss >= crit_quant*(target_scen - offer) )
            m.addConstr(loss >= (crit_quant-1)*(target_scen - offer) )
            
            for row in range(len(weights)):
                m.setObjective( weights[row]@((1-risk_aversion)*(loss) 
                               + risk_aversion*(deviation*deviation)) , gp.GRB.MINIMIZE)

                m.optimize()
                Prescriptions[row] = offer.X
                
            return Prescriptions
    
    elif problem =='mse':
        return (weights@target_scen)
    
    elif problem == 'newsvendor':
        
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
            
            m.setObjective( weights@loss, gp.GRB.MINIMIZE)
            m.optimize()
                
            return offer.X
        
        else:
            
            sort_target_scen = np.sort(target_scen.copy().reshape(-1), axis=0)                
            argsort_g = np.argsort(target_scen.copy().reshape(-1), axis=0)

            n_test_obs = len(weights)
            Prescriptions = np.zeros((n_test_obs))
            for row in range(len(weights)):
                Prescriptions[row] = (sort_target_scen[np.where(weights[row][argsort_g].cumsum() >= crit_quant)][0])              
                
            return Prescriptions