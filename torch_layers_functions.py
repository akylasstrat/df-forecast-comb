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

class MLP(nn.Module):        
    def __init__(self, input_size, hidden_sizes, output_size, activation=nn.ReLU(), constrain_output = True):
        super(MLP, self).__init__()
        """
        Standard MLP for regression
        Args:
            input_size, hidden_sizes, output_size: standard arguments for declaring an MLP
            
            output_size: equal to the number of combination weights, i.e., number of experts we want to combine
            sigmoid_activation: enable sigmoid function as a final layer, to ensure output is in [0,1]
            
        """
        # Initialize learnable weight parameters
        self.num_features = input_size
        self.output_size = output_size
        self.constrain_output = constrain_output
        
        # create sequential model
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(activation)
        
        if self.constrain_output:
            layers.append(activation)
        self.model = nn.Sequential(*layers)
                    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: input tensors/ features
        """
        return self.model(x)

    def predict(self, x):
        # used for inference only
        with torch.no_grad():            
            return self.model(x).detach().numpy()
            
    def estimate_loss(self, y_hat, y_target):
        
        # estimate custom loss function, *elementwise*
        mse_i = torch.square(y_target - y_hat)        
        loss_i = torch.sum(mse_i, 1)
        return loss_i
    
    def epoch_train(self, loader, opt=None):
        """Standard training/evaluation epoch over the dataset"""
        total_loss = 0.
        
        for X,y in loader:
            
            y_hat = self.forward(X)

            #loss = nn.MSELoss()(yp,y)
            loss_i = self.estimate_loss(y_hat, y)                    
            loss = torch.mean(loss_i)
            
            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total_loss += loss.item() * X.shape[0]
            
        return total_loss / len(loader.dataset)

            
    def train_model(self, train_loader, val_loader, optimizer, epochs = 20, patience=5, verbose = 0):
        
        # best_train_loss = float('inf')
        best_val_loss = float('inf')
        early_stopping_counter = 0
        best_weights = copy.deepcopy(self.state_dict())

        for epoch in range(epochs):
            # activate train functionality
            self.train()
            
            average_train_loss = self.epoch_train(train_loader, optimizer)
            val_loss = self.epoch_train(val_loader)
            
            if verbose != -1:
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
                    return
                
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

        for i, batch_data in enumerate(data_loader):

            y_batch = batch_data[-1]
            x_batch = batch_data[-2]
            z_opt_batch = batch_data[0]
                                
            # forward pass: combine forecasts and solve each newsvendor problem
            start_time = time.time()
            z_comb_hat = self.forward(x_batch, z_opt_batch)
            # if i == 0: 
            #     print(f'Forward pass time:{time.time() - start_time}')
            
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

        # newsvendor layer (for i-th observation)
        z = cp.Variable((1), nonneg = True)    
        error = cp.Variable(n_locations)
        prob_weights = cp.Parameter(n_locations, nonneg = True)
        sqrt_prob_weights = cp.Parameter(n_locations, nonneg = True)

        newsv_constraints = [error == self.support - z]
        pinball_loss_expr = cp.maximum(self.crit_fract*(error), (self.crit_fract - 1)*(error))
        newsv_cost = prob_weights@pinball_loss_expr
        
        w_error = cp.multiply(sqrt_prob_weights, error)
        l2_regularization = cp.sum_squares(w_error)

        objective_funct = cp.Minimize( 2*(1-self.risk_aversion)*newsv_cost + 2*(self.risk_aversion)*l2_regularization ) 
        
        newsv_problem = cp.Problem(objective_funct, newsv_constraints)
        self.newsvendor_layer = CvxpyLayer(newsv_problem, parameters=[prob_weights, sqrt_prob_weights],
                                           variables = [z, error] )
    
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
        
        if self.apply_softmax:
            self.model.add_module('softmax', nn.Softmax())

        # Perform the convex combination across input vectors
        combined_pdf = sum(weighted_inputs)
        
        # Pass the combined output to the CVXPY layer
        try:
            cvxpy_output = self.newsvendor_layer(combined_pdf, torch.sqrt(combined_pdf + 1e-4), solver_args={'max_iters':50_000, "solve_method": "ECOS"})
        except:
            cvxpy_output = self.newsvendor_layer(combined_pdf, torch.sqrt(combined_pdf + 1e-4), solver_args={"eps": 1e-8, "max_iters": 10000, "acceleration_lookback": 0})
               
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
                                
            total_loss += loss.item() * y_batch.shape[0]
            
        return total_loss / len(data_loader.dataset)
                    
    def predict_weights(self, x):
        'Forecast combination weights, inference only'
        with torch.no_grad():            
            return self.model(x).detach().numpy()
        
class LinearPoolNewsvendorLayer(nn.Module):        
    def __init__(self, num_inputs, support, gamma, problem = 'reg_trad', feasibility_method = 'projection', 
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
        # self.projection_simplex = projection_simplex
        self.feasibility_method = feasibility_method
        self.regularizer = regularizer
        self.crit_fract = critic_fract
        self.problem = problem
        if (self.problem == 'newsvendor'): 
            self.risk_aversion = 0

        n_locations = len(self.support)
        
        # newsvendor layer (for i-th observation)
        z = cp.Variable((1), nonneg = True)    
        error = cp.Variable(n_locations)
        prob_weights = cp.Parameter(n_locations, nonneg = True)
        sqrt_prob_weights = cp.Parameter(n_locations, nonneg = True)
        
        newsv_constraints = []
        newsv_constraints += [error == self.support - z, z <= self.support.max()]
        
        pinball_loss_expr = cp.maximum(self.crit_fract*(error), (self.crit_fract - 1)*(error))
        newsv_cost = prob_weights@pinball_loss_expr

        w_error = cp.multiply(sqrt_prob_weights, error)
        l2_regularization = cp.sum_squares(w_error)
        
        #l2_regularization = (prob_weights@sq_error)
        objective_funct = cp.Minimize( 2*(1-self.risk_aversion)*newsv_cost + 2*(self.risk_aversion)*l2_regularization ) 
        
        newsv_problem = cp.Problem(objective_funct, newsv_constraints)
        self.newsvendor_layer = CvxpyLayer(newsv_problem, parameters=[prob_weights, sqrt_prob_weights],
                                           variables = [z, error] )

    def get_weights(self):
        if self.feasibility_method == 'softmax':
            return to_np(torch.nn.functional.softmax(self.weights))
        else:
            return to_np(self.weights)
        
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
        if self.feasibility_method == 'softmax':
            softmax_weights = torch.nn.functional.softmax(self.weights, dim = 0)                    
            weighted_inputs = [softmax_weights[i] * input_tensor for i, input_tensor in enumerate(list_inputs)]
        else:
            # Apply the weights element-wise to each input tensor
            weighted_inputs = [self.weights[i] * input_tensor for i, input_tensor in enumerate(list_inputs)]

        # Perform the convex combination across input vectors
        combined_pdf = sum(weighted_inputs)

        # Pass the combined output to the CVXPY layer
        
        try:            
            cvxpy_output = self.newsvendor_layer(combined_pdf, torch.sqrt(combined_pdf + 1e-4), solver_args={'max_iters':50_000, "solve_method": "ECOS"})
        except:
            cvxpy_output = self.newsvendor_layer(combined_pdf, torch.sqrt(combined_pdf + 1e-4), solver_args={"eps": 1e-8, "max_iters": 10000, "acceleration_lookback": 0})
            
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
            
            crps_i = torch.sum(torch.square( cdf_comb_hat - 1*(self.support >= y_batch.reshape(-1,1))), 1).reshape(-1,1)

            sql2_loss_i = torch.square(error_hat)
            
            pinball_loss_i = torch.max(self.crit_fract*error_hat, (self.crit_fract-1)*error_hat)

            loss_i = (1-self.risk_aversion)*pinball_loss_i + self.risk_aversion*sql2_loss_i + self.gamma*crps_i
            loss = torch.mean(loss_i)
            
            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                if self.feasibility_method == 'projection':
                    w_proj = self.simplex_projection(self.weights.clone())
                    with torch.no_grad():
                        self.weights.copy_(torch.FloatTensor(w_proj))
    
            total_loss += loss.item() * y_batch.shape[0]
            
        return total_loss / len(data_loader.dataset)

class LinearPool_VFA_Layer(nn.Module):        
    def __init__(self, approx_model, num_inputs, support, gamma, problem = 'reg_trad', projection_simplex = True, 
                 critic_fract = 0.5, regularizer = 'crps', risk_aversion = 0):
        super(LinearPool_VFA_Layer, self).__init__()

        # Initialize learnable weight parameters
        #self.weights = nn.Parameter(torch.rand(num_inputs), requires_grad=True)
        self.weights = nn.Parameter(torch.FloatTensor((1/num_inputs)*np.ones(num_inputs)).requires_grad_())
        self.num_inputs = num_inputs
        self.support = support
        self.risk_aversion = risk_aversion
        self.gamma = gamma
        self.vfa_model = approx_model
        # self.apply_softmax = apply_softmax
        self.projection_simplex = projection_simplex
        self.regularizer = regularizer
        self.crit_fract = critic_fract
        self.problem = problem

        n_locations = len(self.support)
        
    def forward(self, list_inputs):
        """
        Forward pass of the newvendor layer.

        Args:
            list_inputs: A list of of input tensors/ PDFs.

        Returns:
            torch.Tensor: The convex combination of input tensors/ combination of PDFs.
        """

        # Apply the weights element-wise to each input tensor
        weighted_inputs = [self.weights[i] * input_tensor for i, input_tensor in enumerate(list_inputs)]

        # Perform the convex combination across input vectors
        combined_pdf = sum(weighted_inputs)            
        return combined_pdf
    
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

        w = torch.FloatTensor(self.vfa_model.model[0].weight.detach().numpy())
        bias = torch.FloatTensor(self.vfa_model.model[0].bias.detach().numpy())
        
        for batch_data in data_loader:
            y_batch = batch_data[-1]

            # forward pass: combine forecasts and solve each newsvendor problem
            pdf_comb_hat = self.forward(batch_data[:-1])
            
            cdf_comb_hat = pdf_comb_hat.cumsum(1)
                                        
            # crps_i = sum([torch.square( cdf_comb_hat[i] - 1*(self.support >= y_batch[i]) ).sum() for i in range(len(y_batch))])
            crps_i = torch.sum(torch.square( cdf_comb_hat - 1*(self.support >= y_batch.reshape(-1,1))), 1).reshape(-1,1)
                            
            # loss_i = self.vfa_model.forward(pdf_comb_hat)
            loss_i = pdf_comb_hat@w.T + bias
                
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
    
class LinearPoolSchedLayer(nn.Module):        
    def __init__(self, num_inputs, support, grid, 
                 gamma, regularization = 0, feasibility_method = 'projection', initial_weights = None, clearing_type = 'det', include_network = False, 
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
        self.Pmax_tensor = torch.FloatTensor(self.grid['Pmax'].reshape(-1))
        self.Pmin_tensor = torch.FloatTensor(self.grid['Pmin'].reshape(-1))
        self.gamma = gamma
        self.feasibility_method = feasibility_method
        self.add_network = add_network
        
        n_locations = len(self.support)
        

        #### Stochastic scheduling layer        
        prob_weights = cp.Parameter(n_locations, nonneg = True)

        if self.clearing_type == 'stoch':
            ###### DA variables    
            p_DA = cp.Variable((grid['n_unit']), nonneg = True)
            cost_DA = cp.Variable(1, nonneg = True)
    
            ###### RT variables
            r_up = cp.Variable((grid['n_unit'], n_locations), nonneg = True)
            r_down = cp.Variable((grid['n_unit'], n_locations), nonneg = True)
            slack_up = cp.Variable((n_locations), nonneg = True)
            slack_down = cp.Variable((n_locations), nonneg = True)
                  
            cost_RT = cp.Variable(n_locations)
            
            ## DA constraints
            DA_constraints = [cost_DA == grid['Cost']@p_DA, 
                              p_DA <= grid['Pmax'].reshape(-1)]            

            ## RT constraints            
            RT_constraints = []
            
            for s in range(n_locations):
                # Upper bounds
                RT_constraints += [r_up[:,s] + p_DA <= grid['Pmax'].reshape(-1)]            
                RT_constraints += [r_down[:,s] <= p_DA]                            
                # Balancing per scenario
                RT_constraints += [p_DA.sum() + r_up[:,s].sum() - r_down[:,s].sum() + grid['w_capacity']*self.support[s] 
                                   + slack_up[s] - slack_down[s] == grid['Pd'].sum()]                
            
            ## Ramping limits (only necessary units)
            for g in range(grid['n_unit']):            
                if grid['Pmax'][g] > grid['R_u_max'][g]:                
                    RT_constraints += [r_up[g,:] <= grid['R_u_max'][g].reshape(-1)]                                
                if grid['Pmax'][g] > grid['R_d_max'][g]:                
                    RT_constraints += [r_down[g,:] <= grid['R_d_max'][g].reshape(-1)]            
                
            ## RT cost as expression
            RT_cost_expr = cp.sum([ prob_weights[s]*((grid['C_up'])@r_up[:,s] + (- grid['C_down'])@r_down[:,s]
                                                     + grid['VOLL']*(slack_up[s] + slack_down[s]))  for s in range(n_locations)])
            
            # L2 regularization
            l2_reg_expr = cp.sum([ prob_weights[s]*(cp.sum_squares(r_up[:,s] + r_down[:,s]) + cp.sum_squares(slack_up[s] + slack_down[s]))  for s in range(n_locations)])            
            objective_funct = cp.Minimize( cost_DA +  RT_cost_expr) 
                        
            sched_problem = cp.Problem(objective_funct, DA_constraints + RT_constraints )
             
            self.sched_layer = CvxpyLayer(sched_problem, parameters=[prob_weights], 
                                          variables = [p_DA, r_up, r_down, cost_DA] )
        
        elif self.clearing_type == 'det':
            ###### DA variables    
            p_DA = cp.Variable((grid['n_unit']), nonneg = True)
            slack_DA = cp.Variable(1, nonneg = True)
            cost_DA = cp.Variable(1, nonneg = True)
    
            DA_constraints = [p_DA <= grid['Pmax'].reshape(-1), 
                              p_DA.sum() + grid['w_capacity']*(prob_weights@self.support) + slack_DA.sum() >= grid['Pd'].sum(), 
                              cost_DA == grid['Cost']@p_DA + grid['VOLL']*slack_DA.sum()]
                
            objective_funct = cp.Minimize( cost_DA ) 
            sched_problem = cp.Problem(objective_funct, DA_constraints )
             
            self.sched_layer = CvxpyLayer(sched_problem, parameters=[prob_weights],
                                               variables = [p_DA, slack_DA, cost_DA])
        

        #### RT layer: takes as input parameter the dispatch decisions, optimizes the real-time dispatch        
        p_gen = cp.Parameter(grid['n_unit'], nonneg = True)
        w_actual = cp.Parameter(1, nonneg = True)
                
        ###### RT variables
        r_up = cp.Variable((grid['n_unit']), nonneg = True)
        r_down = cp.Variable((grid['n_unit']), nonneg = True)
        
        g_shed = cp.Variable(1, nonneg = True)
        l_shed = cp.Variable(1, nonneg = True)
        
        cost_RT = cp.Variable(1, nonneg = True)
        
        RT_sched_constraints = []

        RT_sched_constraints += [r_up + p_gen <= grid['Pmax'].reshape(-1) + 1e-5]            
        RT_sched_constraints += [r_down <= p_gen + 1e-5]            
        
        # Ramping constraints (only necessary units)
        for g in range(grid['n_unit']):            
            if grid['Pmax'][g] > grid['R_u_max'][g]:                
                RT_sched_constraints += [r_up[g] <= grid['R_u_max'][g].reshape(-1)]                                
            if grid['Pmax'][g] > grid['R_d_max'][g]:                
                RT_sched_constraints += [r_down[g] <= grid['R_d_max'][g].reshape(-1)]            
        
        # Network flow constraints
        # m.addConstrs(node_inj[:,i] == (grid['node_G']@p_G_i[:,i] + grid['node_L']@slack_i[:,i] -grid['node_L']@curt_i[:,i] 
        #                               - grid['node_L']@node_load_i[:,i]) for i in range(n_samples))    
        # m.addConstrs(PTDF@node_inj[:,i] <= grid['Line_Capacity'].reshape(-1) for i in range(n_samples))
        # m.addConstrs(PTDF@node_inj[:,i] >= -grid['Line_Capacity'].reshape(-1) for i in range(n_samples))
        
        # Balancing constraint
        RT_sched_constraints += [p_gen.sum() + r_up.sum() - r_down.sum() -g_shed.sum() + grid['w_capacity']*w_actual + l_shed.sum() 
                                 == grid['Pd'].sum()]
        
        RT_sched_constraints += [cost_RT == (grid['C_up'])@r_up + (-grid['C_down'])@r_down\
                                  +grid['VOLL']*(g_shed.sum() + l_shed.sum())]
        
        l2_pen = cp.sum_squares(r_up + r_down) + cp.sum_squares(g_shed + l_shed) 
        
        objective_funct = cp.Minimize( cost_RT + 0.001*l2_pen) 
        rt_problem = cp.Problem(objective_funct, RT_sched_constraints)
         
        self.rt_layer = CvxpyLayer(rt_problem, parameters=[p_gen, w_actual],
                                           variables = [r_up, r_down, g_shed, l_shed, cost_RT] )

    def get_weights(self):
        if self.feasibility_method == 'softmax':
            return to_np(torch.nn.functional.softmax(self.weights))
        else:
            return to_np(self.weights)
        
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
        Forward pass of the newvendor layer.

        Args:
            list_inputs: A list of of input tensors/ PDFs.

        Returns:
            torch.Tensor: The convex combination of input tensors/ combination of PDFs.
        """

        if self.feasibility_method == 'softmax':
            softmax_weights = torch.nn.functional.softmax(self.weights, dim = 0)                    
            weighted_inputs = [softmax_weights[i] * input_tensor for i, input_tensor in enumerate(list_inputs)]
        else:
            # Apply the weights element-wise to each input tensor
            weighted_inputs = [self.weights[i] * input_tensor for i, input_tensor in enumerate(list_inputs)]

        # Perform the convex combination across input vectors
        combined_pdf = sum(weighted_inputs)
        
        # Pass the combined output to the CVXPY layer
        try:
            cvxpy_output = self.sched_layer(combined_pdf, solver_args={'max_iters':50_000, "solve_method": "ECOS"})
        except:
            cvxpy_output = self.sched_layer(combined_pdf, solver_args={"eps": 1e-8, "max_iters": 10000, "acceleration_lookback": 0})
        
        # solver_args={'max_iters':50_000, "solve_method": "ECOS"}
        # solver_args={"eps": 1e-8, "max_iters": 10000, "acceleration_lookback": 0}
        return combined_pdf, cvxpy_output

    def epoch_train(self, data_loader, opt=None):
        """Standard training/evaluation epoch over the dataset"""
        # evaluate loss criterion/ used for estimating validation loss
        total_loss = 0.0

        for i, batch_data in enumerate(data_loader):
            

            prob_forecast_batch = batch_data[:self.num_inputs]
            y_batch = batch_data[self.num_inputs].reshape(-1,1)
            perfect_DA_cost_batch = batch_data[self.num_inputs + 1].reshape(-1,1)
            
            # print(len(prob_forecast_batch))
            # print(prob_forecast_batch[0].shape)
            # print(y_batch)
            # print(perfect_DA_cost_batch)
            
            start_time = time.time()
            output_hat = self.forward(prob_forecast_batch)
            
            if i == 0: 
                print(f'Forward pass time:{time.time() - start_time}')
            
            pdf_comb_hat = output_hat[0]
            cdf_comb_hat = pdf_comb_hat.cumsum(1)
            
            decisions_hat = output_hat[1]                
            p_hat = decisions_hat[0]
            
            # Project p_hat to feasible set            
            p_hat_proj = torch.maximum(torch.minimum(p_hat, self.Pmax_tensor), self.Pmin_tensor)

            # solve RT layer, find redispatch cost    
            start_time = time.time()
            try:
                rt_output = self.rt_layer(p_hat_proj, y_batch.reshape(-1,1), 
                                          solver_args={'max_iters':50_000, "solve_method": "ECOS"})
            except:
                rt_output = self.rt_layer(p_hat_proj, y_batch.reshape(-1,1), 
                                          solver_args={"eps": 1e-8, "max_iters": 10000, "acceleration_lookback": 0})
                
            if i == 0: 
                print(f'Forward pass, RT market:{time.time() - start_time}')
            
            # CRPS of combination
            crps_i = torch.sum(torch.square( cdf_comb_hat - 1*(self.support >= y_batch.reshape(-1,1))), 1).reshape(-1,1)

            cost_DA_hat_i = decisions_hat[-1]/ self.Pmax_tensor.sum()
            cost_RT_i = rt_output[-1]/ self.Pmax_tensor.sum()
            loss_i = cost_DA_hat_i + cost_RT_i + self.gamma*crps_i
                        
            # cost_DA_hat_i = decisions_hat[-1]
            # cost_RT_i = rt_output[-1]            
            # relative_regret_i = (cost_DA_hat_i + cost_RT_i - perfect_DA_cost_batch)/perfect_DA_cost_batch
            # loss_i = relative_regret_i + self.gamma*crps_i
            
            loss = torch.mean(loss_i)
            
            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                if self.feasibility_method == 'projection':
                    w_proj = self.simplex_projection(self.weights.clone())
                    with torch.no_grad():
                        self.weights.copy_(torch.FloatTensor(w_proj))

            total_loss += loss.item() * y_batch.shape[0]
        
        return total_loss / len(data_loader.dataset)
    
    
    def train_model(self, train_loader, val_loader, optimizer, epochs = 20, patience=5, validation = False, 
                    relative_tolerance = 0):
        """
        Run gradient-descent algorithm for model training
        """
        
        # Initialize training loss and placeholder variables        
        print('Initialize loss...')
        
        best_train_loss = self.epoch_train(train_loader)
        print(f'Initial loss:{best_train_loss}')
        # best_train_loss = float('inf')
        best_val_loss = float('inf')
        early_stopping_counter = 0
        best_weights = copy.deepcopy(self.state_dict())
                
        for epoch in range(epochs):
            
            print('Current weights')
            if self.feasibility_method == 'softmax':
                print(torch.nn.functional.softmax(self.weights, dim = 0))
            else:
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
                       
   
class LinearPoolCRPSLayer(nn.Module):        
    def __init__(self, num_inputs, support, feasibility_method = 'projection'):
        super(LinearPoolCRPSLayer, self).__init__()

        # Initialize learnable weight parameters
        #self.weights = nn.Parameter(torch.rand(num_inputs), requires_grad=True)
        self.weights = nn.Parameter(torch.FloatTensor((1/num_inputs)*np.ones(num_inputs)).requires_grad_())
        self.num_inputs = num_inputs
        self.support = support
        self.feasibility_method = feasibility_method
    
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
    
    def get_weights(self):
        if self.feasibility_method == 'softmax':
            return to_np(torch.nn.functional.softmax(self.weights))
        else:
            return to_np(self.weights)
        
    def forward(self, list_inputs):
        """
        Forward pass of the linear pool minimizing CRPS.

        Args:
            list_inputs: A list of of input tensors (discrete PDFs).

        Returns:
            torch.Tensor: The convex combination of input tensors.
        """
        
        # Apply the weights element-wise to each input tensor !!!! CDFs
        if self.feasibility_method == 'softmax':
            softmax_weights = torch.nn.functional.softmax(self.weights, dim = 0)
            weighted_inputs = [softmax_weights[i] * input_tensor.cumsum(1) for i, input_tensor in enumerate(list_inputs)]
        else:
            weighted_inputs = [self.weights[i] * input_tensor.cumsum(1) for i, input_tensor in enumerate(list_inputs)]

        #     weights = torch.nn.functional.softmax(self.weights, dim = 0)

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
                        print(self.weights)
                        print(best_weights)
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
                
                if self.feasibility_method == 'projection':
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
