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
from math import sqrt
from GreedyPrescriptiveTree import GreedyPrescriptiveTree
from opt_problem import *
import time

#from forecast_opt_problem import *
#from decision_solver import *
from joblib import Parallel, delayed

import matplotlib.pyplot as plt

class EnsemblePrescriptiveTree(object):
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
  def __init__(self, n_estimators = 10, D = 'Inf' , Nmin = 5, max_features = 'auto', type_split = 'random'):
      
    self.n_estimators = n_estimators
    if D == 'Inf':
        self.D = np.inf
    else:
        self.D = D
    self.Nmin = Nmin
    self.type_split = type_split
    self.max_features = max_features
    
    
  def fit(self, X, Y, quant = np.arange(.1, 1, .1), parallel = False, n_jobs = -1, cpu_time = False, **kwargs):
    ''' Function for training the tree ensemble.
    Requires a separate function that solves the inner optimization problem.
    - quant: quantiles to evaluate continuous features (only used if type_split=='regular')
    - parallel: grows trees in parallel
    - n_jobs: only used for parallel training
    - cpu_time: if True, then returns cpu_time for each tree. If parallel==True, returns cpu_time for the ensemble (not sure how to interpret)
    '''       
    self.decision_kwargs = kwargs
    num_features = X.shape[1]    #Number of features
    index_nodes = [np.arange(len(Y))]
    self.trees = []
    self.cpu_time = []

    if parallel == False:
        for i in range(self.n_estimators):
            print('Ensemble Tree: ',i+1)
            if cpu_time: start_time = time.process_time()

            #Select subset of predictors
            #col = np.random.choice(range(num_features), p_select, replace = False)
            #temp_X = X[:,col]
            #Train decision tree        
            new_tree = GreedyPrescriptiveTree(D = self.D, Nmin = self.Nmin, 
                                              type_split = self.type_split, max_features = self.max_features)
            new_tree.fit(X, Y, **self.decision_kwargs)
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

  def predict_constr(self, testX, trainX, trainY, parallel = False):
     ''' Generate predictive prescriptions'''
     
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
         
     print('Optimizing Prescriptions...')
     #Check that weigths are correct (numerical issues)
     assert( all(Weights.sum(axis = 1) >= 1-10e-4))
     assert( all(Weights.sum(axis = 1) <= 1+10e-4))
     Prescription = []#np.zeros((testX.shape[0],1))
     for i in range(len(testX)):
         if i%250 == 0:
             print('Observation ', i) 
         mask = Weights[i]>0
         _, temp_prescription = opt_problem(trainY[mask], weights = Weights[i][mask], prescribe = True, **self.decision_kwargs)
         #Prescription[i] = temp_prescription     
         Prescription.append(temp_prescription)  
         
     return Prescription
 
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
    
    
    
    