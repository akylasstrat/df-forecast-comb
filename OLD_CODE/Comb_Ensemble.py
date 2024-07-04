# -*- coding: utf-8 -*-
"""
Generates the convex combination of the two ensemble forests

@author: akylas.stratigakos@minesparis.psl.eu
"""

#Import Libraries
import numpy as np
from math import sqrt
from GreedyPrescriptiveTree import GreedyPrescriptiveTree
from opt_problem import *
import time
from comb_opt_problem import *

#from forecast_opt_problem import *
#from decision_solver import *
from joblib import Parallel, delayed

import matplotlib.pyplot as plt

class Comb_Ensemble(object):
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
  def __init__(self, loc_tree, glob_tree, lambda_):      
    self.loc_tree = loc_tree
    self.glob_tree = glob_tree
    self.lambda_ = lambda_
         
  def comb_oob_error(self, localX, localY, globalX, globalY):
      
      oob_loc, loc_prescr = self.loc_tree.estimate_OOB_score(localX, localY, return_estimates = True)
      oob_glob, glob_prescr = self.glob_tree.estimate_OOB_score(globalX, globalY, return_estimates = True)
      print('OOB-Local')
      print(oob_loc)
      print('OOB-Global')
      print(oob_glob)
      print(np.square(localY[:,0]-glob_prescr).mean())
      
      OOB_loss = []
      for l in self.lambda_:
          comb_prescr = l*loc_prescr + (1-l)*glob_prescr
          oob_comb = np.square(localY[:,0]-comb_prescr).mean()
          OOB_loss.append(oob_comb)
      
      self.best_lambda_ = self.lambda_[np.argmin(np.array(OOB_loss))]
      
      return OOB_loss
  
  def comb_oob_error_brc(self, localX, localY, barycenter):
      ''' Input: local non-parametric model plus anchor distribution (barycenter)
          Estimates OOB error of combination
          for simplicity, we iterate over all the fixed support points of contextual information'''
          
      
      oob_loc, loc_prescr = self.loc_tree.estimate_OOB_score(localX, localY, return_estimates = True)
      oob_glob, glob_prescr = self.glob_tree.estimate_OOB_score(globalX, globalY, return_estimates = True)
      print('OOB-Local')
      print(oob_loc)
      print('OOB-Global')
      print(oob_glob)
      print(np.square(localY[:,0]-glob_prescr).mean())
      
      OOB_loss = []
      for l in self.lambda_:
          comb_prescr = l*loc_prescr + (1-l)*glob_prescr
          oob_comb = np.square(localY[:,0]-comb_prescr).mean()
          OOB_loss.append(oob_comb)
      
      self.best_lambda_ = self.lambda_[np.argmin(np.array(OOB_loss))]
      
      return OOB_loss
    
  def predict(self, test_localX, train_localX, train_localY, 
              train_globalX, train_globalY):
      
      loc_w = self.loc_tree.find_weights(test_localX, train_localX)      
      glob_w = self.glob_tree.find_weights(test_localX, train_globalX)
      
      plt.plot(glob_w)
      plt.show()
      
      Prescription = []#np.zeros((testX.shape[0],1))
           
      for i in range(len(test_localX)):
          if i%1000 == 0:
              print('Observation ', i) 
          loc_mask = loc_w[i]>0
          glob_mask = glob_w[i]>0
          _, temp_prescription = comb_opt_problem(train_localY[loc_mask], train_globalY[glob_mask],
                                                self.best_lambda_, opt_quant = .5, 
                                                loc_weights = loc_w[i][loc_mask], glob_weights = glob_w[i][glob_mask],
                                                train_metric = 'mse', test_metric = 'mse')

          Prescription.append(temp_prescription)  
   
      return Prescription
