# -*- coding: utf-8 -*-
"""
Greedy Prescriptive Trees (GPT): Create a tree structure that maps auxiliary data to decisions.
The goal is to create weighted SAA that combines explanatory data from different sources of uncertainty, 
e.g. both energy and price quantities. The tree is grown in a greedy fashion with binary splits., 
following the CART algorithm

References: 
- Bertsimas, Dimitris, and Jack Dunn. "Optimal classification trees." Machine Learning 106.7 (2017): 1039-1082.  
- Dunn, J.W., 2018. Optimal trees for prediction and prescription (Doctoral dissertation, Massachusetts Institute of Technology).
- Elmachtoub, Adam N., and Paul Grigas. "Smart" predict, then optimize"." arXiv preprint arXiv:1710.08005 (2017).
@author: a.stratigakos
"""
import matplotlib.pyplot as plt

#Import Libraries
import numpy as np
from opt_problem import *

#from decision_solver import *
from math import sqrt

class GreedyPrescriptiveTree(object):
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
  def __init__(self, D = 4, Nmin = 5, spo_weight = 1.0, max_features = 'auto', type_split = 'regular'):
      
    self.D = D
    self.Nmin = Nmin
    self.type_split = type_split
    self.max_features = max_features
    
        
  def fit(self, X, Y, quant = np.arange(.1, 1, .1), **kwargs):
    ''' Function to train the Tree.
    Requires a separate function that solves the inner optimization problem, can be used with any optimization tool.
    '''       
    num_features = X.shape[1]    #Number of features
    index_nodes = [np.arange(len(Y))]
    n = len(Y)
    #Initialize tree structure
    #self.bin_features = bin_features
    #self.xbins = xbins.reshape(-1,1)
    self.Node_id = [0]
    self.Depth_id = [0]
    self.parent_node  = [None]
    self.children_left = [-1]
    self.children_right = [-1]
    self.feature = [-1]
    self.threshold = [-1]
    self.decision_kwargs = kwargs
    self.nobs_per_node = [n]
    self.feat_importance = np.zeros(num_features)
    self.sub_Error = []
    self.feat_type = [type(X[0,j]) for j in range(X.shape[1])]
    node_id_counter = 0
    
    # find candidate parallel splits from data
    #if self.bin_features == True:
    #    cand_binary_splits = self.xbins
    
    sorted_X = np.sort(X.copy(), 0)
    
    if self.type_split == 'quant':
        cand_binary_splits = np.quantile(X, quant.round(3),0).round(2)
    elif self.type_split == 'regular':
        cand_binary_splits = (sorted_X[:-1] + sorted_X[1:])/2 


    for node in self.Node_id:
        if self.Depth_id[node] >= self.D:
            #Fix as leaf node
            self.children_left.append(-1)
            self.children_right.append(-1)
            continue
        
        ''' Select subset of features (used for ensembles/ forests/ etc.)'''
        if self.max_features == 'auto':
            #Include all features
            feat_selected = np.arange(num_features)
        else:
            # sample features at each node (without replacement)
            if self.max_features == 'sqrt':    
                p_select = int(sqrt(num_features))
            else:
                p_select = int(self.max_features)
            #Sample features
            feat_selected = np.random.choice(range(num_features), p_select, replace = False)

        #Keep only data that falls within this node
        #print('Node ', node)
        #Update only with data that falls in subtree
        sub_X = X[index_nodes[node]].copy()
        sub_Y = Y[index_nodes[node]].copy()
        
        
        '''Candidate splitting points: either check all (very slow), check quantiles (good enough)
        or if the ExtraTree algorithm is used, pick split at random'''
        #sorted_X = np.sort(sub_X.copy(), 0)
        #if self.type_split == 'regular':
        #    cand_binary_splits = (sorted_X[:-1] + sorted_X[1:])/2 
            #Candidate_Splits = np.quantile(sub_X[:,feat_selected], quant, axis = 0)
        #elif self.type_split == 'random':    
            #Candidate_Splits = np.array([ np.random.choice(sub_X[:,feat]) for feat in feat_selected ]).reshape(1,-1)
        #    Candidate_Splits = np.array([ np.random.uniform(low=sub_X[:,feat].min(), high = sub_X[:,feat].max()) for feat in feat_selected ]).reshape(1,-1)
                
        #print('Feat selected ', feat_selected)
        #Initial solution for node without split
        if node == 0:
            #Initialize list that holds previous subtree errors    
            er, pred = opt_problem(sub_Y, weights = None, **self.decision_kwargs)
            self.sub_Error = [er]
            self.Node_Prediction = [pred]
        #print(opt_problem(sub_Y.reshape(-1), **self.decision_kwargs))
        #print(sub_Error[node])
            
        #Initialize placeholder for subtree error
        Best_Error = self.sub_Error[node]
        #Check for splitting node (separate function)
        solution_count = 0
        apply_split = False
        #print('Root node cost:', Best_Error)
        for j, cand_feat in enumerate(feat_selected):
            #print('Col: ', j)
            
            #If leaf has constant values, skip the evaluation, else select candidate split points
            # Needs to check whether is categorical or quantitative feature
            
            '''
            #If feature is categorical
            if self.feat_type[cand_feat] == str:
                discrete_values_set = set(sub_X[:,cand_feat])
                #print('Set values = ', discrete_values_set)
                
                #if not list(discrete_values_set):
                    #print("List is empty")
                    #print('Node ', node)
                    #print('nodes ', index_nodes[node-1])
                    
                if len(discrete_values_set) == 1:
                    continue
                else:
                    Candidate_Splits = np.random.choice(np.array(list(discrete_values_set)))  #Randomly select categorical variable
                    #print('Splitpoint= ',Candidate_Splits)
            else:
                if all(np.diff(sub_X[:, cand_feat])==0):
                    #print('Feature: ', cand_feat)
                    #print(np.diff(sub_X[:, cand_feat]))
                    continue
                else:
                    if self.type_split == 'regular':
                        Candidate_Splits = (sorted_X[:-1,cand_feat] + sorted_X[1:,cand_feat])/2 
                    elif self.type_split == 'random':    
                        Candidate_Splits = [np.random.uniform(low=sub_X[:,cand_feat].min(), high = sub_X[:,cand_feat].max())]
                    elif self.type_split == 'quant':
                        Candidate_Splits = np.quantile(sub_X[:,cand_feat], quant)
            '''
                        
            for i, cand_split in enumerate(cand_binary_splits[:,cand_feat]):    
            #for i, cand_split in enumerate(Candidate_Splits):    
                if self.feat_type[cand_feat] == str:                    
                    mask_left = sub_X[:,cand_feat] == cand_split
                    mask_right = sub_X[:,cand_feat] != cand_split
                else:
                    mask_left = sub_X[:,cand_feat] < cand_split
                    mask_right = sub_X[:,cand_feat] >= cand_split

                # Check leaf cardinality
                if (sum(mask_left)<self.Nmin) or (sum(mask_right)<self.Nmin):
                    continue
                
                #Evaluate error for each split
                left_tree_Error, Pred_left = opt_problem(sub_Y[mask_left], weights = None, **self.decision_kwargs)
                right_tree_Error, Pred_right = opt_problem(sub_Y[mask_right], weights = None, **self.decision_kwargs)
                #print('Candidate Split: ', left_tree_Error+right_tree_Error)
                #Update split
                if (left_tree_Error + right_tree_Error) < Best_Error:
                    
                    solution_count = solution_count + 1
                    apply_split = True
                    
                    best_left_error = left_tree_Error
                    best_left_Prediction = Pred_left
                    
                    best_right_error = right_tree_Error
                    best_right_Prediction = Pred_right
                    
                    #Placeholder for current minimum error
                    Best_Error = left_tree_Error + right_tree_Error
                    self.feature[node] = cand_feat
                    self.threshold[node] = cand_split
                    
        #If split is applied, update tree structure            
        if apply_split == True:
            
            self.parent_node.extend(2*[node])
            
            self.sub_Error.append(best_left_error)
            self.sub_Error.append(best_right_error)
            
            self.Node_Prediction.append(best_left_Prediction)
            self.Node_Prediction.append(best_right_Prediction)
            
            self.Node_id.extend([node_id_counter + 1, node_id_counter + 2])
            self.Depth_id.extend(2*[self.Depth_id[node]+1])
            
            if self.feat_type[ self.feature[node] ] == str:
                index_left = index_nodes[node][sub_X[:,self.feature[node]] == self.threshold[node]]
                index_right = index_nodes[node][sub_X[:,self.feature[node]] != self.threshold[node]]
            else:                
                index_left = index_nodes[node][sub_X[:,self.feature[node]] < self.threshold[node]]
                index_right = index_nodes[node][sub_X[:,self.feature[node]] >= self.threshold[node]]
                
            assert (len(index_left) + len(index_right)) == len(index_nodes[node])
            
            index_nodes.append(index_left)
            index_nodes.append(index_right)
            
            self.feature.extend(2*[-1])
            self.threshold.extend(2*[-1])
            self.nobs_per_node.extend([len(index_left), len(index_right)])

            if node==0:
                self.children_left[node] = node_id_counter+1
                self.children_right[node] = node_id_counter+2
            else:
                self.children_left.append(node_id_counter+1)
                self.children_right.append(node_id_counter+2)
            node_id_counter = node_id_counter + 2
            
            'Feature importance, internal estimation'
            w_imp = self.nobs_per_node[node]/n
            w_left = len(index_left)/self.nobs_per_node[node]
            w_right = len(index_right)/self.nobs_per_node[node]
            
            self.feat_importance[self.feature[node]] += w_imp*(self.sub_Error[node]\
                                     - best_left_error - best_right_error)
            
        else:
            #Fix as leaf node
            self.children_left.append(-1)
            self.children_right.append(-1)
                        
  def apply(self, X):
     ''' Function that returns the Leaf id for each point. Similar to sklearn's implementation.
     '''
     Leaf_id = np.zeros((X.shape[0],1))
     for i in range(X.shape[0]): 
         #New query point
         x0 = X[i:i+1,:]
         #Start from root node
         node = 0
         #Go downwards until reach a Leaf Node
         while ((self.children_left[node] != -1) and (self.children_right[node] != -1)):
             if x0[:, self.feature[node]] < self.threshold[node]:
                 node = self.children_left[node]
                 #print('Left')
             elif x0[:,self.feature[node]] >= self.threshold[node]:
                node = self.children_right[node]
                #print('Right')
             #print('New Node: ', node)
         Leaf_id[i] = self.Node_id[node]
     return Leaf_id
  
  def predict(self, X):
     ''' Function to predict using a trained tree. Trees are fully compiled, i.e., 
     leaves correspond to predictive prescriptions 
     '''
     Predictions = []
     for i in range(X.shape[0]): 
         #New query point
         x0 = X[i:i+1,:]
         #Start from root node
         node = 0
         #Go down the tree until a Leaf node is reached
         while ((self.children_left[node] != -1) and (self.children_right[node] != -1)):
             if x0[:, self.feature[node]] < self.threshold[node]:
                 node = self.children_left[node]
                 #print('Left')
             elif x0[:,self.feature[node]] >= self.threshold[node]:
                node = self.children_right[node]
                #print('Right')
             #print('New Node: ', node)
         Predictions.append(self.Node_Prediction[node])
     return np.array(Predictions)

  def predict_constr(self, testX, trainX, trainY, parallel = False, **kwargs):
     ''' Generate predictive prescriptions'''
     #Step 1: Estimate weights for weighted SAA
     Leaf_nodes = self.apply(trainX) # nObs*nTrees: a_ij shows the leaf node for observation i in tree j
     Index = self.apply(testX) # Leaf node for test set
     nTrees = 1
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
     assert( all(Weights.sum(axis = 1) >= 1-1e-4))
     assert( all(Weights.sum(axis = 1) <= 1+1e-4))
     Prescription = []#np.zeros((testX.shape[0],1))
        
     for i in range(len(testX)):
         if i%1000 == 0:
             print('Observation ', i) 
         mask = Weights[i]>0
         _, temp_prescription = opt_problem(trainY[mask], weights = Weights[i][mask], prescribe = True, **self.decision_kwargs)
        
         Prescription.append(temp_prescription)  
        
     return np.array(Prescription)