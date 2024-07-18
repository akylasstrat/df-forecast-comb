# -*- coding: utf-8 -*-
"""
Torch custom layers and helper functions

@author: a.stratigakos
"""

import cvxpy as cp
import torch
from torch import nn
from cvxpylayers.torch import CvxpyLayer
from torch.utils.data import Dataset, DataLoader
import numpy as np
import copy
import time

def to_np(x):
    return x.detach().numpy()

# Define a custom dataset
class MyDataset(Dataset):
    def __init__(self, *inputs):
        self.inputs = inputs

        # Check that all input tensors have the same length (number of samples)
        self.length = len(inputs[0])
        if not all(len(input_tensor) == self.length for input_tensor in inputs):
            raise ValueError("Input tensors must have the same length.")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return tuple(input_tensor[idx] for input_tensor in self.inputs)

# Define a custom data loader
def create_data_loader(inputs, batch_size, num_workers=0, shuffle=False):
    dataset = MyDataset(*inputs)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle
    )
    return data_loader

# Closed-form projection on the probability simplex

def simplex_projection_func(torch_model, w_init):
    """
    Projection to unit simplex, closed-form solution
    Ref: Wang, Weiran, and Miguel A. Carreira-Perpinán. "Projection onto the probability simplex: An efficient algorithm with a simple proof, and an application." arXiv preprint arXiv:1309.1541 (2013).
    """

    u_sorted, indices = torch.sort(w_init, descending = True)
    j_ind = torch.arange(1, torch_model.num_inputs + 1)
    rho = (u_sorted + (1/j_ind)*(1-torch.cumsum(u_sorted, dim = 0)) > 0).sum().detach().numpy()
    dual_mu = 1/rho*(1-u_sorted[:rho].sum())
    
    w_proj = torch.maximum(w_init + dual_mu, torch.zeros_like(w_init))
    return w_proj

class AdaptiveLinearPoolDecisions(nn.Module):        
    def __init__(self, input_size, hidden_sizes, output_size, 
                 support, problem = 'reg_trad', activation=nn.ReLU(), apply_softmax = True, critic_fract = 0.5, regularizer = 'crps', risk_aversion = 0):
        super(AdaptiveLinearPoolDecisions, self).__init__()
        """
        Adaptive combination of decisions (Salva's benchmark)
        Args:
            input_size, hidden_sizes, output_size: standard arguments for declaring an MLP
            
            output_size: equal to the number of combination weights, i.e., number of experts we want to combine
            
        """
        # Initialize learnable weight parameters
        #self.weights = nn.Parameter(torch.rand(num_inputs), requires_grad=True)
        self.weights = nn.Parameter(torch.FloatTensor((1/output_size)*np.ones(output_size)).requires_grad_())
        self.num_experts = output_size
        self.support = support
        self.risk_aversion = risk_aversion
        self.apply_softmax = apply_softmax
        self.regularizer = regularizer
        self.crit_fract = critic_fract
        self.problem = problem
        
        if (self.problem == 'newsvendor'): self.risk_aversion = 0

        # create sequential MLP model to predict combination weights
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(activation)
                
        self.model = nn.Sequential(*layers)
        if self.apply_softmax:
            self.model.add_module('softmax', nn.Softmax())
                            
    def forward(self, x, z):
        """
        Forward pass of the newvendor layer.

        Args:
            x: input tensors/ features
            z: input tensors/ historical optimal decisions for each PDF vector

        Returns:
            torch.Tensor: The convex combination of input tensors.
        """

        # Forwatd pass of the MLP to predict the combination weights (use softmax activation)
        weights = self.model(x)

        # Apply the weights element-wise to each input tensor        
        #weighted_z = [weights[:,j] * z[:,j] for j in range(self.num_experts)]        
        weighted_z = weights*z
        
        # Convex combination of decisions
        z_comb = weighted_z.sum(1).reshape(-1,1)
        return z_comb

    def train_model(self, train_loader, val_loader, optimizer, epochs = 20, patience=5, validation = True, 
                    relative_tolerance = 0):
        """
        Run gradient-descent algorithm for model training
        """
        
        # Initialize training loss and placeholder variables        
        print('Initialize loss...')
        
        best_train_loss = self.epoch_train(train_loader)
        # best_train_loss = float('inf')
        best_val_loss = float('inf')
        early_stopping_counter = 0
        best_weights = copy.deepcopy(self.state_dict())
                
        for epoch in range(epochs):
                        
            if validation == False:
                # only check performance on training data
                average_train_loss = self.epoch_train(train_loader, optimizer)
                print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {average_train_loss:.4f} ")
    
                if (average_train_loss < best_train_loss) and ( (best_train_loss-average_train_loss)/best_train_loss > relative_tolerance):
                    best_train_loss = average_train_loss
                    best_weights = copy.deepcopy(self.state_dict())
                    early_stopping_counter = 0
                    
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= patience:
                        print("Early stopping triggered.")
                        # recover best weights
                        self.load_state_dict(best_weights)
                        return
                    
            elif validation == True:
                
                average_train_loss = self.epoch_train(train_loader, optimizer)
                val_loss = self.epoch_train(val_loader)

                print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {average_train_loss:.4f} - Val Loss: {val_loss:.4f}")
    
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = copy.deepcopy(self.state_dict())
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= patience:
                        print("Early stopping triggered.")
                        # recover best weights
                        self.load_state_dict(best_weights)
                        break
                
                
    def epoch_train(self, data_loader, opt=None):
        """Standard training/evaluation epoch over the dataset"""
        # evaluate loss criterion/ used for estimating validation loss
        total_loss = 0.0

        for batch_data in data_loader:

            y_batch = batch_data[-1]
            x_batch = batch_data[-2]
            z_opt_batch = batch_data[0]
                                
            # forward pass: combine forecasts and solve each newsvendor problem
            z_comb_hat = self.forward(x_batch, z_opt_batch)
            error_hat = (y_batch.reshape(-1,1) - z_comb_hat)
            
            # Estimate Regret and CRPS
            sql2_loss_i = torch.square(error_hat)
            
            if (self.problem == 'reg_trad') or (self.problem == 'newsvendor'):
                pinball_loss_i = torch.max(self.crit_fract*error_hat, (self.crit_fract-1)*error_hat)

                # Total regret (scale CRPS for better trade-off control)
                loss_i = (1-self.risk_aversion)*pinball_loss_i + self.risk_aversion*sql2_loss_i                        

            loss = torch.mean(loss_i)
            
            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()
                                
            total_loss += loss.item() * y_batch.shape[0]
            
        return total_loss / len(data_loader.dataset)
                
    def predict_weights(self, x):
        'Forecast combination weights, inference only'
        with torch.no_grad():            
            return self.model(x).detach().numpy()
        
class AdaptiveLinearPoolNewsvendorLayer(nn.Module):        
    def __init__(self, input_size, hidden_sizes, output_size, 
                 support, gamma, problem = 'reg_trad', activation=nn.ReLU(), apply_softmax = True, critic_fract = 0.5, regularizer = 'crps', risk_aversion = 0):
        super(AdaptiveLinearPoolNewsvendorLayer, self).__init__()
        """
        Adaptive forecast combination for newsvendor problem> predicts combination weights given features
        Args:
            input_size, hidden_sizes, output_size: standard arguments for declaring an MLP
            
            output_size: equal to the number of combination weights, i.e., number of experts we want to combine
            
        """
        # Initialize learnable weight parameters
        #self.weights = nn.Parameter(torch.rand(num_inputs), requires_grad=True)
        self.weights = nn.Parameter(torch.FloatTensor((1/output_size)*np.ones(output_size)).requires_grad_())
        self.num_experts = output_size
        self.support = support
        self.gamma = gamma
        self.risk_aversion = risk_aversion
        self.apply_softmax = apply_softmax
        self.regularizer = regularizer
        self.crit_fract = critic_fract
        self.problem = problem

        if (self.problem == 'newsvendor'): self.risk_aversion = 0
        
        # create sequential MLP model to predict combination weights
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(activation)
                
        self.model = nn.Sequential(*layers)
        if self.apply_softmax:
            self.model.add_module('softmax', nn.Softmax())
                    
        n_locations = len(self.support)

        if (self.problem == 'reg_trad') or (self.problem == 'newsvendor'):
            # newsvendor layer (for i-th observation)
            z = cp.Variable((1), nonneg = True)    
            #pinball_loss = cp.Variable(n_locations)
            error = cp.Variable(n_locations)
            prob_weights = cp.Parameter(n_locations, nonneg = True)
            sqrt_prob_weights = cp.Parameter(n_locations, nonneg = True)

            newsv_constraints = [error == self.support - z]
            #                     pinball_loss >= self.crit_fract*(error), 
            #                     pinball_loss >= (self.crit_fract - 1)*(error)]
            pinball_loss_expr = cp.maximum(self.crit_fract*(error), (self.crit_fract - 1)*(error))
            newsv_cost = prob_weights@pinball_loss_expr
            
            # define aux variable
            #w_error = cp.multiply(prob_weights, error)
            #sq_error = cp.power(error, 2)
            w_error = cp.multiply(sqrt_prob_weights, error)
            l2_regularization = cp.sum_squares(w_error)
    
            #l2_regularization = (prob_weights@sq_error)
    
            objective_funct = cp.Minimize( 2*(1-self.risk_aversion)*newsv_cost + 2*(self.risk_aversion)*l2_regularization ) 
            
            newsv_problem = cp.Problem(objective_funct, newsv_constraints)
            self.newsvendor_layer = CvxpyLayer(newsv_problem, parameters=[prob_weights, sqrt_prob_weights],
                                               variables = [z, error] )
        elif self.problem == 'pwl':
            # newsvendor layer (for i-th observation)
            z = cp.Variable((1))    
            pwl_loss = cp.Variable(n_locations)
            error = cp.Variable(n_locations)
            prob_weights = cp.Parameter(n_locations)
            sqrt_prob_weights = cp.Parameter(n_locations)

            newsv_constraints = [z >= 0, z <= 1, error == self.support - z,
                                 pwl_loss >= self.crit_fract*(error),
                                 pwl_loss >= -0.5*(error),
                                 pwl_loss >= (1-self.crit_fract)*(error + 0.1)]
            
            newsv_cost = prob_weights@pwl_loss
            
            w_error = cp.multiply(sqrt_prob_weights, error)
            l2_regularization = cp.sum_squares(w_error)

            objective_funct = cp.Minimize( newsv_cost + (self.risk_aversion)*l2_regularization)             
            newsv_problem = cp.Problem(objective_funct, newsv_constraints)
            
            self.newsvendor_layer = CvxpyLayer(newsv_problem, parameters=[prob_weights, sqrt_prob_weights],
                                               variables = [z, pwl_loss, error] )
    
    def simplex_projection(self, w_init):
        """
        Projection to unit simplex, closed-form solution
        Ref: Wang, Weiran, and Miguel A. Carreira-Perpinán. "Projection onto the probability simplex: An efficient algorithm with a simple proof, and an application." arXiv preprint arXiv:1309.1541 (2013).
        """

        u_sorted, indices = torch.sort(w_init, descending = True)
        j_ind = torch.arange(1, self.num_inputs + 1)
        rho = (u_sorted + (1/j_ind)*(1-torch.cumsum(u_sorted, dim = 0)) > 0).sum().detach().numpy()
        dual_mu = 1/rho*(1-u_sorted[:rho].sum())
        
        w_proj = torch.maximum(w_init + dual_mu, torch.zeros_like(w_init))
        return w_proj
    
    def forward(self, x, list_inputs):
        """
        Forward pass of the newvendor layer.

        Args:
            x: input tensors/ features
            list_inputs: A list of of input tensors/ probability vectors.

        Returns:
            torch.Tensor: The convex combination of input tensors.
        """

        # Forwatd pass of the MLP to predict the combination weights (use softmax activation)
        weights = self.model(x)

        # Apply the weights element-wise to each input tensor        
        weighted_inputs = [torch.tile(weights[:,i:i+1], (1, input_tensor.shape[1])) * input_tensor for i, input_tensor in enumerate(list_inputs)]
        
        # Perform the convex combination across input vectors
        combined_pdf = sum(weighted_inputs)
        
        # Pass the combined output to the CVXPY layer
        if (self.problem == 'reg_trad') or (self.problem == 'newsvendor') or (self.problem == 'pwl'):
            cvxpy_output = self.newsvendor_layer(combined_pdf, torch.sqrt(combined_pdf + 1e-4))

        return combined_pdf, cvxpy_output


    def train_model(self, train_loader, val_loader, optimizer, epochs = 20, patience=5, validation = True, 
                    relative_tolerance = 0):
        """
        Run gradient-descent algorithm for model training
        """
        
        # Initialize training loss and placeholder variables        
        print('Initialize loss...')
        
        best_train_loss = self.epoch_train(train_loader)
        # best_train_loss = float('inf')
        best_val_loss = float('inf')
        early_stopping_counter = 0
        best_weights = copy.deepcopy(self.state_dict())
                
        for epoch in range(epochs):
                        
            if validation == False:
                # only check performance on training data
                average_train_loss = self.epoch_train(train_loader, optimizer)
                print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {average_train_loss:.4f} ")
    
                if (average_train_loss < best_train_loss) and ( (best_train_loss-average_train_loss)/best_train_loss > relative_tolerance):
                    best_train_loss = average_train_loss
                    best_weights = copy.deepcopy(self.state_dict())
                    early_stopping_counter = 0
                    
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= patience:
                        print("Early stopping triggered.")
                        # recover best weights
                        self.load_state_dict(best_weights)
                        return
                    
            elif validation == True:
                
                average_train_loss = self.epoch_train(train_loader, optimizer)
                val_loss = self.epoch_train(val_loader)

                print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {average_train_loss:.4f} - Val Loss: {val_loss:.4f}")
    
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = copy.deepcopy(self.state_dict())
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= patience:
                        print("Early stopping triggered.")
                        # recover best weights
                        self.load_state_dict(best_weights)
                        break
                
                
    def epoch_train(self, data_loader, opt=None):
        """Standard training/evaluation epoch over the dataset"""
        # evaluate loss criterion/ used for estimating validation loss
        total_loss = 0.0

        for batch_data in data_loader:

            y_batch = batch_data[-1]
            x_batch = batch_data[-2]
            p_list_batch = batch_data[0:-2]
                                
            # forward pass: combine forecasts and solve each newsvendor problem
            output_hat = self.forward(x_batch, p_list_batch)
                
            pdf_comb_hat = output_hat[0]
            cdf_comb_hat = pdf_comb_hat.cumsum(1)
                
            z_hat = output_hat[1][0]

            error_hat = (y_batch.reshape(-1,1) - z_hat)
            
            # Estimate Regret and CRPS
            crps_i = torch.sum(torch.square( cdf_comb_hat - 1*(self.support >= y_batch.reshape(-1,1))), 1).reshape(-1,1)
            sql2_loss_i = torch.square(error_hat)
            
            if (self.problem == 'reg_trad') or (self.problem == 'newsvendor'):
                pinball_loss_i = torch.max(self.crit_fract*error_hat, (self.crit_fract-1)*error_hat)

                # Total regret (scale CRPS for better trade-off control)
                loss_i = (1-self.risk_aversion)*pinball_loss_i + self.risk_aversion*sql2_loss_i \
                    + self.gamma*crps_i
                        
            loss = torch.mean(loss_i)
            
            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()
                                
                # if self.projection_simplex == True:
                #     w_proj = self.simplex_projection(self.weights.clone())
                #     with torch.no_grad():
                #         self.weights.copy_(torch.FloatTensor(w_proj))
                    
            total_loss += loss.item() * y_batch.shape[0]
            
        return total_loss / len(data_loader.dataset)
                    
    def predict_weights(self, x):
        'Forecast combination weights, inference only'
        with torch.no_grad():            
            return self.model(x).detach().numpy()
        
class LinearPoolNewsvendorLayer(nn.Module):        
    def __init__(self, num_inputs, support, gamma, problem = 'reg_trad', projection_simplex = True, 
                 critic_fract = 0.5, regularizer = 'crps', risk_aversion = 0):
        super(LinearPoolNewsvendorLayer, self).__init__()

        # Initialize learnable weight parameters
        #self.weights = nn.Parameter(torch.rand(num_inputs), requires_grad=True)
        self.weights = nn.Parameter(torch.FloatTensor((1/num_inputs)*np.ones(num_inputs)).requires_grad_())
        self.num_inputs = num_inputs
        self.support = support
        self.risk_aversion = risk_aversion
        self.gamma = gamma
        # self.apply_softmax = apply_softmax
        self.projection_simplex = projection_simplex
        self.regularizer = regularizer
        self.crit_fract = critic_fract
        self.problem = problem
        if (self.problem == 'newsvendor'): 
            self.risk_aversion = 0

        n_locations = len(self.support)
        
        if (self.problem == 'reg_trad') or (self.problem == 'newsvendor'):
            # newsvendor layer (for i-th observation)
            z = cp.Variable((1), nonneg = True)    
            #pinball_loss = cp.Variable(n_locations, nonneg = True)
            error = cp.Variable(n_locations)
            prob_weights = cp.Parameter(n_locations, nonneg = True)
            sqrt_prob_weights = cp.Parameter(n_locations, nonneg = True)
            
            newsv_constraints = []
            newsv_constraints += [error == self.support - z, z <= self.support.max()]
            #newsv_constraints += [pinball_loss >= self.crit_fract*(error), pinball_loss >= (self.crit_fract - 1)*(error)]
            
            pinball_loss_expr = cp.maximum(self.crit_fract*(error), (self.crit_fract - 1)*(error))
            newsv_cost = prob_weights@pinball_loss_expr
            
            # define aux variable
            #w_error = cp.multiply(prob_weights, error)
            #sq_error = cp.power(error, 2)
            w_error = cp.multiply(sqrt_prob_weights, error)
            l2_regularization = cp.sum_squares(w_error)
            
            #l2_regularization = (prob_weights@sq_error)
            objective_funct = cp.Minimize( 2*(1-self.risk_aversion)*newsv_cost + 2*(self.risk_aversion)*l2_regularization ) 
            
            newsv_problem = cp.Problem(objective_funct, newsv_constraints)
            self.newsvendor_layer = CvxpyLayer(newsv_problem, parameters=[prob_weights, sqrt_prob_weights],
                                               variables = [z, error] )
        elif self.problem == 'pwl':
            # newsvendor layer (for i-th observation)
            z = cp.Variable((1))    
            pwl_loss = cp.Variable(n_locations)
            error = cp.Variable(n_locations)
            prob_weights = cp.Parameter(n_locations)
            sqrt_prob_weights = cp.Parameter(n_locations)

            newsv_constraints = [z >= 0, z <= 1, error == self.support - z,
                                 pwl_loss >= self.crit_fract*(error),
                                 pwl_loss >= -0.5*(error),
                                 pwl_loss >= (1-self.crit_fract)*(error + 0.1)]
            
            w_error = cp.multiply(sqrt_prob_weights, error)
            l2_regularization = cp.sum_squares(w_error)

            objective_funct = cp.Minimize( prob_weights@pwl_loss + self.risk_aversion*l2_regularization)             
            newsv_problem = cp.Problem(objective_funct, newsv_constraints)
            
            self.newsvendor_layer = CvxpyLayer(newsv_problem, parameters=[prob_weights, sqrt_prob_weights],
                                               variables = [z, pwl_loss, error] )


    def forward(self, list_inputs):
        """
        Forward pass of the newvendor layer.

        Args:
            list_inputs: A list of of input tensors/ PDFs.

        Returns:
            torch.Tensor: The convex combination of input tensors/ combination of PDFs.
        """
        # Ensure that the weights are in the range [0, 1] using softmax activation
        # if self.projection_simplex:
        #     weights = self.simplex_projection(self.weights)            
        # elif self.apply_softmax:
        #     weights = torch.nn.functional.softmax(self.weights, dim = 0)
        # else:
        #     weights = self.weights

        # Apply the weights element-wise to each input tensor
        weighted_inputs = [self.weights[i] * input_tensor for i, input_tensor in enumerate(list_inputs)]

        # Perform the convex combination across input vectors
        combined_pdf = sum(weighted_inputs)

        # Pass the combined output to the CVXPY layer
        if self.problem in ['reg_trad', 'newsvendor', 'pwl']:
            cvxpy_output = self.newsvendor_layer(combined_pdf, torch.sqrt(combined_pdf + 1e-4))
            
        return combined_pdf, cvxpy_output
    
    def simplex_projection(self, w_init):
        """
        Projection to unit simplex, closed-form solution
        Ref: Wang, Weiran, and Miguel A. Carreira-Perpinán. "Projection onto the probability simplex: An efficient algorithm with a simple proof, and an application." arXiv preprint arXiv:1309.1541 (2013).
        """

        u_sorted, indices = torch.sort(w_init, descending = True)
        j_ind = torch.arange(1, self.num_inputs + 1)
        rho = (u_sorted + (1/j_ind)*(1-torch.cumsum(u_sorted, dim = 0)) > 0).sum().detach().numpy()
        dual_mu = 1/rho*(1-u_sorted[:rho].sum())
        
        w_proj = torch.maximum(w_init + dual_mu, torch.zeros_like(w_init))
        return w_proj
        
    def train_model(self, train_loader, val_loader, optimizer, epochs = 20, patience=5, validation = False, 
                    relative_tolerance = 0):
        """
        Run gradient-descent algorithm for model training
        """
        
        # Initialize training loss and placeholder variables        
        print('Initialize loss...')
        
        best_train_loss = self.epoch_train(train_loader)
        # best_train_loss = float('inf')
        best_val_loss = float('inf')
        early_stopping_counter = 0
        best_weights = copy.deepcopy(self.state_dict())
                
        for epoch in range(epochs):
            
            print('Current weights')
            print(self.weights)
            
            
            if validation == False:
                # only check performance on training data
                average_train_loss = self.epoch_train(train_loader, optimizer)
                print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {average_train_loss:.4f} ")
    
                if (average_train_loss < best_train_loss) and ( (best_train_loss-average_train_loss)/best_train_loss > relative_tolerance):
                    best_train_loss = average_train_loss
                    best_weights = copy.deepcopy(self.state_dict())
                    early_stopping_counter = 0
                    
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= patience:
                        print("Early stopping triggered.")
                        # recover best weights
                        self.load_state_dict(best_weights)
                        return
                    
            elif validation == True:
                
                average_train_loss = self.epoch_train(train_loader, optimizer)
                val_loss = self.epoch_train(val_loader)

                print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {average_train_loss:.4f} - Val Loss: {val_loss:.4f}")
    
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = copy.deepcopy(self.state_dict())
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= patience:
                        print("Early stopping triggered.")
                        # recover best weights
                        self.load_state_dict(best_weights)
                        break
                
                
    def epoch_train(self, data_loader, opt=None):
        """Standard training/evaluation epoch over the dataset"""
        # evaluate loss criterion/ used for estimating validation loss
        total_loss = 0.0

        for batch_data in data_loader:
            y_batch = batch_data[-1]

            # forward pass: combine forecasts and solve each newsvendor problem
            output_hat = self.forward(batch_data[:-1])

            pdf_comb_hat = output_hat[0]
            cdf_comb_hat = pdf_comb_hat.cumsum(1)
            
            z_hat = output_hat[1][0]
                
            # estimate aggregate pinball loss and CRPS (for realization of uncertainty)
            error_hat = (y_batch.reshape(-1,1) - z_hat)
            
            # crps_i = sum([torch.square( cdf_comb_hat[i] - 1*(self.support >= y_batch[i]) ).sum() for i in range(len(y_batch))])
            crps_i = torch.sum(torch.square( cdf_comb_hat - 1*(self.support >= y_batch.reshape(-1,1))), 1).reshape(-1,1)

            sql2_loss_i = torch.square(error_hat)
            
            if (self.problem == 'reg_trad') or (self.problem == 'newsvendor'):
                pinball_loss_i = torch.max(self.crit_fract*error_hat, (self.crit_fract-1)*error_hat)

                # Total regret (scale CRPS for better trade-off control)
                loss_i = (1-self.risk_aversion)*pinball_loss_i + self.risk_aversion*sql2_loss_i \
                    + self.gamma*crps_i
            
            # print(crps_i.shape)
            # print(sql2_loss_i.shape)
            # print(pinball_loss_i.shape)
            # print(loss_i.shape)
            
            # print(crps_i.mean())
            # print(sql2_loss_i.mean())
            # print(pinball_loss_i.mean())
            # print(loss_i.mean())
            
            # print(crps_i.sum())
            # print(sql2_loss_i.sum())
            # print(pinball_loss_i.sum())
            # print(loss_i.sum())
            
            loss = torch.mean(loss_i)
            
            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                # print('before projection')
                # print(self.weights)
                
                if self.projection_simplex == True:
                    w_proj = self.simplex_projection(self.weights.clone())
                    with torch.no_grad():
                        self.weights.copy_(torch.FloatTensor(w_proj))
                    
                    # print('after projection')
                    # print(self.weights)

            total_loss += loss.item() * y_batch.shape[0]
            
        return total_loss / len(data_loader.dataset)

class LinearPoolSchedLayer(nn.Module):        
    def __init__(self, num_inputs, support, grid, 
                 gamma, regularization = 0, apply_softmax = False, initial_weights = None, clearing_type = 'det', include_network = False, 
                 add_network = False):
        super(LinearPoolSchedLayer, self).__init__()

        # Initialize learnable weight parameters
        #self.weights = nn.Parameter(torch.rand(num_inputs), requires_grad=True)
        try:
            if initial_weights == None:         
                self.weights = nn.Parameter(torch.FloatTensor((1/num_inputs)*np.ones(num_inputs)).requires_grad_())
        except:
            self.weights = nn.Parameter(torch.FloatTensor(initial_weights).requires_grad_())
        print(self.weights)
        self.num_inputs = num_inputs
        self.clearing_type = clearing_type
        self.grid = grid
        self.regularization = regularization
        self.support = support
        self.include_network = include_network
        #self.risk_aversion = risk_aversion
        self.gamma = gamma
        self.apply_softmax = apply_softmax
        self.add_network = add_network
        
        n_locations = len(self.support)
        

        #### Stochastic scheduling layer        
        prob_weights = cp.Parameter(n_locations, nonneg = True)
        #sqrt_prob_weights = cp.Parameter(n_locations)
        #error = cp.Variable(n_locations)
        #y_hat = cp.Variable(1)

        if self.clearing_type == 'stoch':
            ###### DA variables    
            p_DA = cp.Variable((grid['n_unit']), nonneg = True)
            #slack_DA = cp.Variable(1)
            cost_DA = cp.Variable(1, nonneg = True)
    
            ###### RT variables
            r_up = cp.Variable((grid['n_unit'], n_locations), nonneg = True)
            r_down = cp.Variable((grid['n_unit'], n_locations), nonneg = True)
    
            #g_shed = cp.Variable((1, n_locations), nonneg = True)
            #l_shed = cp.Variable((1, n_locations), nonneg = True)
            cost_RT = cp.Variable(n_locations)
            
            ## DA constraints
            DA_constraints = [p_DA <= grid['Pmax'].reshape(-1), 
                              cost_DA == grid['Cost']@p_DA]            
                              #p_DA.sum() + grid['w_capacity']*(prob_weights@self.support) == grid['Pd'].sum(), 

            ## RT constraints            
            #RT_constraints = [l_shed >= 0, g_shed >= 0, r_down >= 0, r_up >= 0]
            RT_constraints = []
            
            for s in range(n_locations):
                # Upper bounds

                RT_constraints += [r_up[:,s] <= grid['Pmax'].reshape(-1) - p_DA]            
                RT_constraints += [r_down[:,s] <= p_DA]            

                # Ramping limits
                #RT_constraints += [r_up[:,s] <= grid['R_u_max'].reshape(-1), r_down[:,s] <= grid['R_d_max'].reshape(-1)]            
                #RT_constraints += [g_shed[:,s] <= p_DA]            
                # balancing
                RT_constraints += [p_DA.sum() + r_up[:,s].sum() - r_down[:,s]
                                   + grid['w_capacity']*self.support[s] >= grid['Pd'].sum()]
                
                #RT_constraints += [cost_RT[s] ==  (grid['C_up'])@r_up[:,s] + (- grid['C_down'])@r_down[:,s]]
                                           
            RT_cost_expr = cp.sum([ prob_weights[s]*(grid['C_up'])@r_up[:,s] + (- grid['C_down'])@r_down[:,s] for s in range(n_locations)])
            objective_funct = cp.Minimize( cost_DA +  RT_cost_expr) 
            #l2_regularization = (prob_weights@sq_error)
            
            ### Alternatively: only RT re-dispatch costs
            #RT_cost_expr = cp.sum([ prob_weights[s]*(grid['C_up'] - grid['Cost'])@r_up[:,s] + (grid['Cost'] - grid['C_down'])@r_down[:,s] for s in range(n_locations)])            
            #objective_funct = cp.Minimize( RT_cost_expr) 
            
            sched_problem = cp.Problem(objective_funct, DA_constraints + RT_constraints)
             
            self.sched_layer = CvxpyLayer(sched_problem, parameters=[prob_weights],
                                               variables = [p_DA, r_up, r_down, cost_DA] )
        
        elif self.clearing_type == 'det':
            ###### DA variables    
            p_DA = cp.Variable((grid['n_unit']), nonneg = True)
            slack_DA = cp.Variable(1, nonneg = True)
            cost_DA = cp.Variable(1, nonneg = True)
                    
            #DA_constraints = [p_DA >= 0, p_DA <= grid['Pmax'].reshape(-1), slack_DA >= 0, 
            #                  p_DA.sum() + grid['w_capacity']*(prob_weights@self.support) + slack_DA.sum() == grid['Pd'].sum(), 
            #                  cost_DA == grid['Cost']@p_DA + grid['VOLL']*slack_DA.sum()]

            DA_constraints = [p_DA <= grid['Pmax'].reshape(-1), 
                              p_DA.sum() + grid['w_capacity']*(prob_weights@self.support) + slack_DA.sum() >= grid['Pd'].sum(), 
                              cost_DA == grid['Cost']@p_DA + grid['VOLL']*slack_DA.sum()]
            
            
            #Reg_constraints = [error == self.support - y_hat, y_hat == prob_weights@self.support]            
            #w_error = cp.multiply(sqrt_prob_weights, error)
            #l2_regularization = cp.sum_squares(w_error)
    
            objective_funct = cp.Minimize( cost_DA ) 
            sched_problem = cp.Problem(objective_funct, DA_constraints )
             
            self.sched_layer = CvxpyLayer(sched_problem, parameters=[prob_weights],
                                               variables = [p_DA, slack_DA, cost_DA])
        

        #### RT layer: takes as input parameter the dispatch decisions, optimizes the real-time dispatch
        tolerance = .1
        
        p_gen = cp.Parameter(grid['n_unit'], nonneg = True)
        w_actual = cp.Parameter(1, nonneg = True)
                
        ###### RT variables
        r_up = cp.Variable((grid['n_unit']), nonneg = True)
        r_down = cp.Variable((grid['n_unit']), nonneg = True)
        
        g_shed = cp.Variable(1, nonneg = True)
        l_shed = cp.Variable(1, nonneg = True)
        cost_RT = cp.Variable(1, nonneg = True)
        
        RT_sched_constraints = []

        RT_sched_constraints += [r_up <= grid['Pmax'].reshape(-1) - p_gen]            
        RT_sched_constraints += [r_down <= p_gen]            
        
        # Ramping constraints
        #RT_sched_constraints += [r_up <= grid['R_u_max'].reshape(-1), r_down <= grid['R_d_max'].reshape(-1)]            
            
        
        #m.addConstrs(node_inj[:,i] == (node_G@p_G_i[:,i] + node_L@slack_i[:,i] -node_L@curt_i[:,i] 
        #                               - node_L@node_load_i[:,i]) for i in range(n_samples))    
        #m.addConstrs(PTDF@node_inj[:,i] <= grid['Line_Capacity'].reshape(-1) for i in range(n_samples))
        #m.addConstrs(PTDF@node_inj[:,i] >= -grid['Line_Capacity'].reshape(-1) for i in range(n_samples))
        
        # balancing
        RT_sched_constraints += [p_gen.sum() + r_up.sum() - r_down.sum() -g_shed.sum() + grid['w_capacity']*w_actual + l_shed.sum() == grid['Pd'].sum()]
        RT_sched_constraints += [cost_RT == (grid['C_up'])@r_up + ( - grid['C_down'])@r_down +grid['VOLL']*(g_shed.sum() + l_shed.sum()) ]
            
        objective_funct = cp.Minimize( cost_RT ) 
        rt_problem = cp.Problem(objective_funct, RT_sched_constraints)
         
        self.rt_layer = CvxpyLayer(rt_problem, parameters=[p_gen, w_actual],
                                           variables = [r_up, r_down, g_shed, l_shed, cost_RT] )

    def forward(self, list_inputs):
        """
        Forward pass of the newvendor layer.

        Args:
            list_inputs: A list of of input tensors/ PDFs.

        Returns:
            torch.Tensor: The convex combination of input tensors/ combination of PDFs.
        """
        # Ensure that the weights are in the range [0, 1] using softmax activation
        if self.apply_softmax:
            weights = torch.nn.functional.softmax(self.weights, dim = 0)
        else:
            weights = self.weights

        # Apply the weights element-wise to each input tensor
        weighted_inputs = [weights[i] * input_tensor for i, input_tensor in enumerate(list_inputs)]

        # Perform the convex combination across input vectors
        combined_pdf = sum(weighted_inputs)

        # Pass the combined output to the CVXPY layer
        cvxpy_output = self.sched_layer(combined_pdf, solver_args={'max_iters':20_000})

        return combined_pdf, cvxpy_output
    
    def train_model(self, train_loader, val_loader, optimizer, epochs = 20, patience=5, projection = True, validation = False, 
                    relative_tolerance = 0):
        
        # define projection problem for backward pass
        if (projection)and(self.apply_softmax != True):     
            lambda_proj = cp.Variable(self.num_inputs)
            lambda_hat = cp.Parameter(self.num_inputs)
            proj_problem = cp.Problem(cp.Minimize(0.5*cp.sum_squares(lambda_proj-lambda_hat)), [lambda_proj >= 0, lambda_proj.sum()==1])
        
        
        #L_t = []
        best_train_loss = 1e7
        best_val_loss = 1e7 
        early_stopping_counter = 0
        best_weights = copy.deepcopy(self.state_dict())

        Pmax_tensor = torch.FloatTensor(self.grid['Pmax'].reshape(-1))
        Pmin_tensor = torch.FloatTensor(self.grid['Pmin'].reshape(-1))
        
        # estimate initial loss
        print('Estimate Initial Loss...')
        initial_train_loss = 0
        with torch.no_grad():
            for batch_data in train_loader:
                y_batch = batch_data[-1]
                # clear gradients
                optimizer.zero_grad()
                output_hat = self.forward(batch_data[:-1])
                
                pdf_comb_hat = output_hat[0]
                cdf_comb_hat = pdf_comb_hat.cumsum(1)
                
                decisions_hat = output_hat[1][0]                
                p_hat = decisions_hat[0]
                
                # Project p_hat to feasible set
                p_hat_proj = torch.maximum(torch.minimum(p_hat, Pmax_tensor), Pmin_tensor)
                cost_DA_hat = decisions_hat[-1]

                # solve RT layer, find redispatch cost                
                rt_output = self.rt_layer(p_hat_proj, y_batch.reshape(-1,1), solver_args={'max_iters':50_000})                
                # CRPS of combination
                crps_i = sum([torch.square( cdf_comb_hat[i] - 1*(self.support >= y_batch[i]) ).sum() for i in range(len(y_batch))])
                loss = cost_DA_hat.mean() + rt_output[-1].mean() + self.gamma*crps_i
 
                initial_train_loss += loss.item()
                
        initial_train_loss = initial_train_loss/len(train_loader)
        best_train_loss = initial_train_loss
        print(f'Initial Estimate: {best_train_loss}')
        for epoch in range(epochs):
            # activate train functionality
            self.train()
            running_loss = 0.0                

            # sample batch data
            for batch_data in train_loader:
                y_batch = batch_data[-1]
                
                # clear gradients
                optimizer.zero_grad()

                # forward pass: combine forecasts and each stochastic ED problem
                #start = time.time()
                #start = time.time()
                output_hat = self.forward(batch_data[:-1])
                
                #end = time.time()
                #print('Forward pass', end - start)

                
                #end = time.time()
                #print('Forward pass time', end - start)


                pdf_comb_hat = output_hat[0]
                cdf_comb_hat = pdf_comb_hat.cumsum(1)
                
                decisions_hat = output_hat[1][0]
                
                p_hat = decisions_hat[0]
                #y_hat = decisions_hat[1]
                
                # Project p_hat to feasible set
                p_hat_proj = torch.maximum(torch.minimum(p_hat, Pmax_tensor), Pmin_tensor)
                #print(p_hat)
                #print(p_hat_proj)
                cost_DA_hat = decisions_hat[-1]

                # total loss
                #print(cost_DA_hat.mean())
                #print(p_hat)
                
                # solve RT layer, find redispatch cost
                # forward pass: combine forecasts and each stochastic ED problem
                

                rt_output = self.rt_layer(p_hat_proj, y_batch.reshape(-1,1), solver_args={'max_iters':50_000})                

                # CRPS of combination
                # forward pass: combine forecasts and each stochastic ED problem
                crps_i = sum([torch.square( cdf_comb_hat[i] - 1*(self.support >= y_batch[i]) ).sum() for i in range(len(y_batch))])

                
                # L2 regularization
                #error_hat = (y_batch.reshape(-1,1) - y_hat)
                #print(error_hat.shape)
                #sql2_loss = self.grid['w_capacity']*torch.square(error_hat).sum()
                
                loss = cost_DA_hat.mean() + rt_output[-1].mean() + self.gamma*crps_i
                #loss = rt_output[-1].mean() + self.gamma*crps_i
                
                #loss = cost_DA_hat.mean()
                    
                # backward pass
                # forward pass: combine forecasts and each stochastic ED problem
                loss.backward()
                optimizer.step()


                
                # Apply projection
                if (projection)and(self.apply_softmax != True):     
                    lambda_hat.value = to_np(self.weights)
                    proj_problem.solve(solver = 'GUROBI')
                    # update parameter values
                    with torch.no_grad():
                        self.weights.copy_(torch.FloatTensor(lambda_proj.value))

                running_loss += loss.item()

                
            if epoch%1 == 0:
                print(torch.nn.functional.softmax(self.weights))
                
            average_train_loss = running_loss / len(train_loader)
                                        
            if validation == True:
                # evaluate performance on stand-out validation set
                val_loss = self.evaluate(val_loader)
                print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {average_train_loss:.4f} - Val Loss: {val_loss:.4f}")
                
                if (val_loss < best_val_loss) and ( (best_val_loss-val_loss)/best_val_loss > relative_tolerance):
                    best_val_loss = val_loss
                    best_weights = copy.deepcopy(self.state_dict())
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= patience:
                        print("Early stopping triggered.")
                        # recover best weights
                        self.load_state_dict(best_weights)
                        return
            else:
                # only evaluate on training data set
                print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {average_train_loss:.4f} ")
    
                if (average_train_loss < best_train_loss) and ( (best_train_loss-average_train_loss)/best_train_loss > relative_tolerance):
                    best_train_loss = average_train_loss
                    best_weights = copy.deepcopy(self.state_dict())
                    early_stopping_counter = 0
                    
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= patience:
                        print("Early stopping triggered.")
                        # recover best weights
                        self.load_state_dict(best_weights)
                        return
        
        print('Reached epoch limit.')
        self.load_state_dict(best_weights)
        return

    def evaluate(self, data_loader):
        # evaluate loss criterion/ used for estimating validation loss
        self.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch_data in data_loader:
                y_batch = batch_data[-1]

                # forward pass: combine forecasts and each stochastic ED problem
                output_hat = self.forward(batch_data[:-1])

                pdf_comb_hat = output_hat[0]
                cdf_comb_hat = pdf_comb_hat.cumsum(1)
                
                decisions_hat = output_hat[1][0]
                
                p_hat = decisions_hat[0]
                
                # solve RT layer, find redispatch cost
                rt_output = self.rt_layer(p_hat, y_batch.reshape(-1,1), solver_args={'max_iters':50000})                

                # CRPS of combination
                crps_i = sum([torch.square( cdf_comb_hat[i] - 1*(self.support >= y_batch[i]) ).sum() for i in range(len(y_batch))])

                # total loss
                loss = rt_output[-1].mean() + self.gamma*crps_i

                total_loss += loss.item()
                
        average_loss = total_loss / len(data_loader)
        return average_loss
    
class LinearPoolCRPSLayer(nn.Module):        
    def __init__(self, num_inputs, support, projection_simplex = True):
        super(LinearPoolCRPSLayer, self).__init__()

        # Initialize learnable weight parameters
        #self.weights = nn.Parameter(torch.rand(num_inputs), requires_grad=True)
        self.weights = nn.Parameter(torch.FloatTensor((1/num_inputs)*np.ones(num_inputs)).requires_grad_())
        self.num_inputs = num_inputs
        self.support = support
        self.projection_simplex = projection_simplex
    
    def simplex_projection(self, w_init):
        """
        Projection to unit simplex, closed-form solution
        Ref: Wang, Weiran, and Miguel A. Carreira-Perpinán. "Projection onto the probability simplex: An efficient algorithm with a simple proof, and an application." arXiv preprint arXiv:1309.1541 (2013).
        """

        u_sorted, indices = torch.sort(w_init, descending = True)
        j_ind = torch.arange(1, self.num_inputs + 1)
        rho = (u_sorted + (1/j_ind)*(1-torch.cumsum(u_sorted, dim = 0)) > 0).sum().detach().numpy()
        dual_mu = 1/rho*(1-u_sorted[:rho].sum())
        
        w_proj = torch.maximum(w_init + dual_mu, torch.zeros_like(w_init))
        return w_proj
    
    def forward(self, list_inputs):
        """
        Forward pass of the linear pool minimizing CRPS.

        Args:
            list_inputs: A list of of input tensors (discrete PDFs).

        Returns:
            torch.Tensor: The convex combination of input tensors.
        """
        
        # Apply the weights element-wise to each input tensor !!!! CDFs
        weighted_inputs = [self.weights[i] * input_tensor.cumsum(1) for i, input_tensor in enumerate(list_inputs)]

        # Perform the convex combination across input vectors
        combined_CDF = sum(weighted_inputs)

        return combined_CDF
    
    def train_model(self, train_loader, val_loader, optimizer, epochs = 20, patience=5, validation = False, 
                    relative_tolerance = 0):
        """
        Run gradient-descent algorithm for model training
        """
        
        # Initialize training loss and placeholder variables        
        print('Initialize loss...')
        
        best_train_loss = self.epoch_train(train_loader)
        # best_train_loss = float('inf')
        best_val_loss = float('inf')
        early_stopping_counter = 0
        best_weights = copy.deepcopy(self.state_dict())
                
        for epoch in range(epochs):
            
            print('Current weights')
            print(self.weights)
            
            
            if validation == False:
                # only check performance on training data
                average_train_loss = self.epoch_train(train_loader, optimizer)
                print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {average_train_loss:.4f} ")
    
                if (average_train_loss < best_train_loss) and ( (best_train_loss-average_train_loss)/best_train_loss > relative_tolerance):
                    best_train_loss = average_train_loss
                    best_weights = copy.deepcopy(self.state_dict())
                    early_stopping_counter = 0
                    
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= patience:
                        print("Early stopping triggered.")
                        # recover best weights
                        self.load_state_dict(best_weights)
                        return
                    
            elif validation == True:
                
                average_train_loss = self.epoch_train(train_loader, optimizer)
                val_loss = self.epoch_train(val_loader)

                print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {average_train_loss:.4f} - Val Loss: {val_loss:.4f}")
    
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = copy.deepcopy(self.state_dict())
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= patience:
                        print("Early stopping triggered.")
                        # recover best weights
                        self.load_state_dict(best_weights)
                        break
                
                
    def epoch_train(self, data_loader, opt=None):
        """Standard training/evaluation epoch over the dataset"""
        # evaluate loss criterion/ used for estimating validation loss
        total_loss = 0.0

        for batch_data in data_loader:
            y_batch = batch_data[-1]

            # forward pass: combine forecasts and solve each newsvendor problem
            cdf_comb_hat = self.forward(batch_data[:-1])

            crps_i = torch.sum(torch.square( cdf_comb_hat - 1*(self.support >= y_batch.reshape(-1,1))), 1).reshape(-1,1)

            loss_i = crps_i
            loss = torch.mean(loss_i)
            
            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                if self.projection_simplex == True:
                    w_proj = self.simplex_projection(self.weights.clone())
                    with torch.no_grad():
                        self.weights.copy_(torch.FloatTensor(w_proj))
                    
            total_loss += loss.item() * y_batch.shape[0]
            
        return total_loss / len(data_loader.dataset)
    
class AdaptiveLinearPoolCRPSLayer(nn.Module):        
    def __init__(self, input_size, hidden_sizes, output_size, support, activation=nn.ReLU(), apply_softmax = True):
        super(AdaptiveLinearPoolCRPSLayer, self).__init__()
        """
        Adaptive forecast combination, predicts weights for linear pool
        Args:
            input_size, hidden_sizes, output_size: standard arguments for declaring an MLP
            
            output_size: equal to the number of combination weights, i.e., number of experts we want to combine
            
        """
        # Initialize learnable weight parameters
        self.weights = nn.Parameter(torch.FloatTensor((1/output_size)*np.ones(output_size)).requires_grad_())
        self.num_features = input_size
        self.num_experts = output_size
        self.support = support
        self.apply_softmax = apply_softmax
            
        # create sequential MLP model to predict combination weights
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(activation)
                    
        self.model = nn.Sequential(*layers)
        if self.apply_softmax:
            self.model.add_module('softmax', nn.Softmax())
                          
    def forward(self, x, list_inputs):
        """
        Forward pass of linear pool.

        Args:
            x: input tensors/ features
            list_inputs: A list of of input tensors/ probability vectors.

        Returns:
            torch.Tensor: The convex combination of input tensors.
        """

        # Forwatd pass of the MLP to predict the combination weights (use softmax activation)
        weights = self.model(x)

        # if (self.projection_simplex == True):
        #     w_proj = self.simplex_projection(self.weights.clone())
        #     with torch.no_grad():
        #         self.weights.copy_(torch.FloatTensor(w_proj))

        # Apply the weights element-wise to each input tensor !!!! CDFs
        
        #weighted_inputs = [weights[k,i] * input_tensor for k in range(weights.shape[0]) for i, input_tensor in enumerate(list_inputs)]
       # weighted_inputs = [weights[:,i] * input_tensor[:,k] for k in range(len(self.support)) 
       #                    for i, input_tensor in enumerate(list_inputs)]

        weighted_inputs = [torch.tile(weights[:,i:i+1], (1, input_tensor.shape[1])) * input_tensor for i, input_tensor in enumerate(list_inputs)]
        
        # Perform the convex combination across input vectors
        
        combined_PDF = sum(weighted_inputs)
        
        return combined_PDF
    
    def train_model(self, train_loader, val_loader, optimizer, epochs = 20, patience=5, validation = True, 
                    relative_tolerance = 0):
        """
        Run gradient-descent algorithm for model training
        """
        
        # Initialize training loss and placeholder variables        
        print('Initialize loss...')
        
        best_train_loss = self.epoch_train(train_loader)
        # best_train_loss = float('inf')
        best_val_loss = float('inf')
        early_stopping_counter = 0
        best_weights = copy.deepcopy(self.state_dict())
                
        for epoch in range(epochs):
                        
            if validation == False:
                # only check performance on training data
                average_train_loss = self.epoch_train(train_loader, optimizer)
                print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {average_train_loss:.4f} ")
    
                if (average_train_loss < best_train_loss) and ( (best_train_loss-average_train_loss)/best_train_loss > relative_tolerance):
                    best_train_loss = average_train_loss
                    best_weights = copy.deepcopy(self.state_dict())
                    early_stopping_counter = 0
                    
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= patience:
                        print("Early stopping triggered.")
                        # recover best weights
                        self.load_state_dict(best_weights)
                        return
                    
            elif validation == True:
                
                average_train_loss = self.epoch_train(train_loader, optimizer)
                val_loss = self.epoch_train(val_loader)

                print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {average_train_loss:.4f} - Val Loss: {val_loss:.4f}")
    
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = copy.deepcopy(self.state_dict())
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= patience:
                        print("Early stopping triggered.")
                        # recover best weights
                        self.load_state_dict(best_weights)
                        break
                
                
    def epoch_train(self, data_loader, opt=None):
        """Standard training/evaluation epoch over the dataset"""
        # evaluate loss criterion/ used for estimating validation loss
        total_loss = 0.0

        for batch_data in data_loader:

            y_batch = batch_data[-1]
            x_batch = batch_data[-2]
            p_list_batch = batch_data[0:-2]
                        
            # forward pass: predict weights and combine forecasts
            comb_PDF = self.forward(x_batch, p_list_batch)
            cdf_comb_hat = comb_PDF.cumsum(1)

            crps_i = torch.sum(torch.square( cdf_comb_hat - 1*(self.support >= y_batch.reshape(-1,1))), 1).reshape(-1,1)

            loss_i = crps_i
            loss = torch.mean(loss_i)
            
            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()
                                    
            total_loss += loss.item() * y_batch.shape[0]
            
        return total_loss / len(data_loader.dataset)
    
    def predict_weights(self, x):
        'Forecast combination weights, inference only'
        with torch.no_grad():            
            return self.model(x).detach().numpy()

    # def train_model(self, train_loader, val_loader, 
    #                 optimizer, epochs = 20, patience=5, projection = False):
        
    #     if (projection)and(self.apply_softmax != True):     
    #         lambda_proj = cp.Variable(self.num_inputs)
    #         lambda_hat = cp.Parameter(self.num_inputs)
    #         proj_problem = cp.Problem(cp.Minimize(0.5*cp.sum_squares(lambda_proj-lambda_hat)), [lambda_proj >= 0, lambda_proj.sum()==1])
        
    #     L_t = []
    #     best_train_loss = float('inf')
    #     best_val_loss = float('inf')
    #     early_stopping_counter = 0
    #     best_weights = copy.deepcopy(self.state_dict())

    #     for epoch in range(epochs):
    #         # activate train functionality
    #         self.train()
    #         running_loss = 0.0
    #         # sample batch data
    #         for batch_data in train_loader:
                
    #             y_batch = batch_data[-1]
    #             x_batch = batch_data[-2]
    #             p_list_batch = batch_data[0:-2]
                
    #             # clear gradients
    #             optimizer.zero_grad()
                
    #             # forward pass: predict weights and combine forecasts
    #             comb_PDF = self.forward(x_batch, p_list_batch)
    #             comb_CDF = comb_PDF.cumsum(1)
                
    #             # estimate CRPS (heavyside function)
    #             loss_i = [torch.square( comb_CDF[i] - 1*(self.support >= y_batch[i]) ).sum() for i in range(len(y_batch))]
    #             loss = sum(loss_i)/len(loss_i)
                
    #             # backward pass
    #             loss.backward()
    #             optimizer.step()                
                
    #             # Apply projection
    #             if (projection)and(self.apply_softmax != True):     
    #                 lambda_hat.value = to_np(self.weights)
    #                 proj_problem.solve(solver = 'GUROBI')
    #                 # update parameter values
    #                 with torch.no_grad():
    #                     self.weights.copy_(torch.FloatTensor(lambda_proj.value))
                
    #             running_loss += loss.item()
            

    #         L_t.append(to_np(self.weights).copy())
                
    #         average_train_loss = running_loss / len(train_loader)
    #         val_loss = self.evaluate(val_loader)
            
    #         print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {average_train_loss:.4f} - Val Loss: {val_loss:.4f}")

    #         if val_loss < best_val_loss:
    #             best_val_loss = val_loss
    #             best_weights = copy.deepcopy(self.state_dict())
    #             early_stopping_counter = 0
    #         else:
    #             early_stopping_counter += 1
    #             if early_stopping_counter >= patience:
    #                 print("Early stopping triggered.")
    #                 # recover best weights
    #                 self.load_state_dict(best_weights)
    #                 return

    # def evaluate(self, data_loader):
    #     # evaluate loss criterion/ used for estimating validation loss
    #     self.eval()
    #     total_loss = 0.0
    #     with torch.no_grad():
    #         for batch_data in data_loader:
    #             y_batch = batch_data[-1]
    #             x_batch = batch_data[-2]
    #             p_list_batch = batch_data[0:-2]

    #             # forward pass: combine forecasts and solve each newsvendor problem
    #             comb_PDF_hat = self.forward(x_batch, p_list_batch)
    #             comb_CDF_hat = comb_PDF_hat.cumsum(1)

    #             # estimate CRPS (heavyside function)
    #             loss_i = [torch.square( comb_CDF_hat[i] - 1*(self.support >= y_batch[i]) ).sum() for i in range(len(y_batch))]
    #             loss = sum(loss_i)/len(loss_i)
    #             total_loss += loss.item()
                
    #     average_loss = total_loss / len(data_loader)
    #     return average_loss
                