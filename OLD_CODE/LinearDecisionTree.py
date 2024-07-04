# -*- coding: utf-8 -*-
"""
Linear Decision Tree: Greedy decision trees with linear regression output.
Tree is grown following the CART algorithm

@author: akylas.stratigakos@mines-paristech.fr
"""


#import matplotlib.pyplot as plt

#Import Libraries
import numpy as np
#from opt_problem import *
#from decision_solver import *
#from math import sqrt

class LinearDecisionTree(object):
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
  def __init__(self, D = 4, Nmin = 5, max_features = 'auto', 
               type_split = 'regular', policy = 'linear'):
      
    self.D = D
    self.Nmin = Nmin
    self.type_split = type_split
    self.max_features = max_features
    self.policy = policy
        
  def fit(self, X, Y, quant = np.arange(.1, 1, .1), **kwargs):
    ''' Function to train the Tree.
    Requires a separate function that solves the inner optimization problem, can be used with any optimization tool.
    '''       
    num_features = X.shape[1]    #Number of features
    index_nodes = [np.arange(len(Y))]
    n = len(Y)
    #Initialize tree structure
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
    self.coef_ = []
    node_id_counter = 0
    
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
            #!!!!! Do I sample with or without replacement here?
            if self.max_features == 'sqrt':    
                p_select = int(np.sqrt(num_features))
            else:
                p_select = int(self.max_features)
            #Sample features
            feat_selected = np.random.choice(range(num_features), p_select, replace = False)

        #Keep only data that falls within this node
        #print('Node ', node)
        #Update only with data that falls in subtree
        sub_X = X[index_nodes[node]].copy()
        sub_Y = Y[index_nodes[node]].copy()
        
                
        #print('Feat selected ', feat_selected)
        #Initial solution for node without split
        if node == 0:
            
            #Initialize list that holds previous subtree errors    
            if self.policy == 'constant':
                y_hat = Y.mean()
                loss_initial = np.square(Y-y_hat).sum()
                
                self.sub_Error = [loss_initial]
                self.Node_Prediction = [y_hat]

            elif self.policy == 'linear':
                
                coef_ = np.linalg.lstsq(X, Y, rcond=None)
                y_hat = X@coef_[0]
                loss_initial = np.square(Y-y_hat).sum()
                
                self.sub_Error = [loss_initial]
                self.Node_Prediction = [y_hat]
                self.coef_ = [coef_[0]]
                
        #print(opt_problem(sub_Y.reshape(-1), **self.decision_kwargs))
        #print(sub_Error[node])
            
        #Initialize placeholder for subtree error
        Best_Error = self.sub_Error[node]
        #Check for splitting node (separate function)
        solution_count = 0
        apply_split = False
        #print('Root node cost:', Best_Error)
        for j, cand_feat in enumerate(feat_selected):

            for i, cand_split in enumerate(cand_binary_splits[:,cand_feat]):    

                #if self.feat_type[cand_feat] == str:                    
                #    mask_left = sub_X[:,cand_feat] == cand_split
                #    mask_right = sub_X[:,cand_feat] != cand_split
                #else:
                #    mask_left = sub_X[:,cand_feat] < cand_split
                #    mask_right = sub_X[:,cand_feat] >= cand_split

                mask_left = sub_X[:,cand_feat] < cand_split
                mask_right = sub_X[:,cand_feat] >= cand_split

                # Check leaf cardinality
                if (sum(mask_left)<self.Nmin) or (sum(mask_right)<self.Nmin):
                    continue
                
                #!!!! Estimation should be recursive here
                
                if self.policy == 'constant':
                    left_yhat = sub_Y[mask_left].mean()
                    right_yhat = sub_Y[mask_right].mean()
                    
                elif self.policy == 'linear':
                    left_coef = np.linalg.lstsq(sub_X[mask_left], sub_Y[mask_left], rcond=None)
                    right_coef = np.linalg.lstsq(sub_X[mask_right], sub_Y[mask_right], rcond=None)
                    
                    left_yhat = sub_X[mask_left]@left_coef[0]
                    right_yhat = sub_X[mask_right]@right_coef[0]
                
                left_tree_Error = np.square(sub_Y[mask_left]-left_yhat).sum()
                right_tree_Error = np.square(sub_Y[mask_right]-right_yhat).sum()
                
                #print('Candidate Split: ', left_tree_Error+right_tree_Error)
                #Update split

                if (left_tree_Error + right_tree_Error) < Best_Error:
                    
                    solution_count = solution_count + 1
                    apply_split = True
                    
                    best_left_error = left_tree_Error
                    best_left_Prediction = left_yhat
                    
                    best_right_error = right_tree_Error
                    best_right_Prediction = right_yhat
                    
                    #Placeholder for current minimum error
                    Best_Error = left_tree_Error + right_tree_Error
                    self.feature[node] = cand_feat
                    self.threshold[node] = cand_split
                    
                    if self.policy == 'linear':
                        best_left_coef_ = left_coef[0]
                        best_right_coef_ = right_coef[0]
                        
                        
        #If split is applied, update tree structure            
        if apply_split == True:
            
            self.parent_node.extend(2*[node])
            
            self.sub_Error.append(best_left_error)
            self.sub_Error.append(best_right_error)
            
            self.Node_Prediction.append(best_left_Prediction)
            self.Node_Prediction.append(best_right_Prediction)
            
            if self.policy == 'linear':
                self.coef_.append(best_left_coef_)
                self.coef_.append(best_right_coef_)

                
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
         if self.policy == 'constant':
             Predictions.append(self.Node_Prediction[node])
         elif self.policy == 'linear':
             Predictions.append(x0@ self.coef_[node])
             
     return np.array(Predictions)
 
