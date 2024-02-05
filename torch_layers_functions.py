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
def create_data_loader(inputs, batch_size, num_workers=0, shuffle=True):
    dataset = MyDataset(*inputs)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle
    )
    return data_loader

class AdaptiveLinearPoolDecisions(nn.Module):        
    def __init__(self, input_size, hidden_sizes, output_size, 
                 support, activation=nn.ReLU(), apply_softmax = True, critic_fract = 0.5, regularizer = 'crps', risk_aversion = 0):
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

    def train_model(self, train_loader, val_loader, optimizer, epochs = 20, patience=5, projection = False):

        best_val_loss = float('inf')
        early_stopping_counter = 0
        best_weights = copy.deepcopy(self.state_dict())

        for epoch in range(epochs):
            # activate train functionality
            self.train()
            running_loss = 0.0
            # sample batch data
            for batch_data in train_loader:
                
                y_batch = batch_data[-1]
                x_batch = batch_data[-2]
                z_opt_batch = batch_data[0]
                
                # clear gradients
                optimizer.zero_grad()
                
                # forward pass: combine forecasts and solve each newsvendor problem
                z_comb_hat = self.forward(x_batch, z_opt_batch)
                 
                error_hat = (y_batch.reshape(-1,1) - z_comb_hat)

                pinball_loss = (self.crit_fract*error_hat[error_hat>0].norm(p=1) + (1-self.crit_fract)*error_hat[error_hat<0].norm(p=1))
                sql2_loss = torch.square(error_hat).sum()
                loss = (1-self.risk_aversion)*pinball_loss + self.risk_aversion*sql2_loss
                
                # backward pass
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            average_train_loss = running_loss / len(train_loader)
            val_loss = self.evaluate(val_loader)
            
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
                
    def evaluate(self, data_loader):
        # evaluate loss criterion/ used for estimating validation loss
        self.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch_data in data_loader:
                y_batch = batch_data[-1]
                x_batch = batch_data[-2]
                z_opt_batch = batch_data[0]

                # forward pass: combine forecasts and solve each newsvendor problem
                z_comb_hat = self.forward(x_batch, z_opt_batch)

                
                # estimate aggregate pinball loss and CRPS (for realization of uncertainty)
                error_hat = (y_batch.reshape(-1,1) - z_comb_hat)

                pinball_loss = (self.crit_fract*error_hat[error_hat>0].norm(p=1) + (1-self.crit_fract)*error_hat[error_hat<0].norm(p=1))
                sql2_loss = torch.square(error_hat).sum()
                loss = (1-self.risk_aversion)*pinball_loss + self.risk_aversion*sql2_loss
                total_loss += loss.item()
                
        average_loss = total_loss / len(data_loader)
        return average_loss
                
    def predict_weights(self, x):
        'Forecast combination weights, inference only'
        with torch.no_grad():            
            return self.model(x).detach().numpy()
        
class AdaptiveLinearPoolNewsvendorLayer(nn.Module):        
    def __init__(self, input_size, hidden_sizes, output_size, 
                 support, gamma, activation=nn.ReLU(), apply_softmax = True, critic_fract = 0.5, regularizer = 'crps', risk_aversion = 0):
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
        z = cp.Variable((1))    
        pinball_loss = cp.Variable(n_locations)
        error = cp.Variable(n_locations)
        #sq_error = cp.Variable(n_locations)

        prob_weights = cp.Parameter(n_locations)
        sqrt_prob_weights = cp.Parameter(n_locations)
            
        newsv_constraints = [z >= 0, z <= 1, error == self.support - z,
                             pinball_loss >= self.crit_fract*(error), 
                             pinball_loss >= (self.crit_fract - 1)*(error)]
        
        newsv_cost = prob_weights@pinball_loss
        
        # define aux variable
        #sq_error = cp.power(error, 2)
        w_error = cp.multiply(sqrt_prob_weights, error)
        l2_regularization = cp.sum_squares(w_error)

        #l2_regularization = self.risk_aversion*(w_error@w_error)

        objective_funct = cp.Minimize( 2*(1-self.risk_aversion)*newsv_cost + 2*self.risk_aversion*l2_regularization ) 
        
        newsv_problem = cp.Problem(objective_funct, newsv_constraints)
        self.newsvendor_layer = CvxpyLayer(newsv_problem, parameters=[prob_weights, sqrt_prob_weights],
                                           variables = [z, pinball_loss, error] )
        
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
        #weighted_inputs = [weights[k,i] * input_tensor for k in range(weights.shape[0]) 
        #                   for i, input_tensor in enumerate(list_inputs)]
        
        weighted_inputs = [torch.tile(weights[:,i:i+1], (1, input_tensor.shape[1])) * input_tensor for i, input_tensor in enumerate(list_inputs)]

        #print(sum(weighted_inputs).shape)
        
        # Perform the convex combination across input vectors
        combined_pdf = sum(weighted_inputs)
        
        # Pass the combined output to the CVXPY layer
        cvxpy_output = self.newsvendor_layer(combined_pdf, torch.sqrt(combined_pdf + 1e-4))
        return combined_pdf, cvxpy_output

    def train_model(self, train_loader, val_loader, optimizer, epochs = 20, patience=5, projection = False, relative_tolerance = 0):
        
        if (projection)and(self.apply_softmax != True):     
            lambda_proj = cp.Variable(self.num_inputs)
            lambda_hat = cp.Parameter(self.num_inputs)
            proj_problem = cp.Problem(cp.Minimize(0.5*cp.sum_squares(lambda_proj-lambda_hat)), [lambda_proj >= 0, lambda_proj.sum()==1])
        
        L_t = []
        best_train_loss = 1e10
        best_val_loss = 1e10
        early_stopping_counter = 0
        best_weights = copy.deepcopy(self.state_dict())

        for epoch in range(epochs):
            # activate train functionality
            self.train()
            running_loss = 0.0
            # sample batch data
            for batch_data in train_loader:
                
                y_batch = batch_data[-1]
                x_batch = batch_data[-2]
                p_list_batch = batch_data[0:-2]
                
                # clear gradients
                optimizer.zero_grad()
                
                # forward pass: combine forecasts and solve each newsvendor problem
                output_hat = self.forward(x_batch, p_list_batch)
                
                pdf_comb_hat = output_hat[0]
                cdf_comb_hat = pdf_comb_hat.cumsum(1)
                
                z_hat = output_hat[1][0]

                error_hat = (y_batch.reshape(-1,1) - z_hat)

                crps_i = sum([torch.square( cdf_comb_hat[i] - 1*(self.support >= y_batch[i]) ).sum() for i in range(len(y_batch))])
                pinball_loss = (self.crit_fract*error_hat[error_hat>0].norm(p=1) + (1-self.crit_fract)*error_hat[error_hat<0].norm(p=1))
                sql2_loss = torch.square(error_hat).sum()
                
                # Total regret (scale CRPS for better trade-off control)
                loss = 2*(1-self.risk_aversion)*pinball_loss + 2*self.risk_aversion*sql2_loss \
                    + self.gamma*crps_i/len(self.support)
                
                # backward pass
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
            

            L_t.append(to_np(self.weights).copy())
                
            average_train_loss = running_loss / len(train_loader)
            val_loss = self.evaluate(val_loader)
            
            print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {average_train_loss:.4f} - Val Loss: {val_loss:.4f}")

            if (val_loss < best_val_loss) and ((best_val_loss-val_loss)/best_val_loss > relative_tolerance):
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
                
    def evaluate(self, data_loader):
        # evaluate loss criterion/ used for estimating validation loss
        self.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch_data in data_loader:
                y_batch = batch_data[-1]
                x_batch = batch_data[-2]
                p_list_batch = batch_data[0:-2]

                # forward pass: combine forecasts and solve each newsvendor problem
                output_hat = self.forward(x_batch, p_list_batch)

                pdf_comb_hat = output_hat[0]
                cdf_comb_hat = pdf_comb_hat.cumsum(1)
                
                z_hat = output_hat[1][0]
                
                # estimate aggregate pinball loss and CRPS (for realization of uncertainty)
                error_hat = (y_batch.reshape(-1,1) - z_hat)

                crps_i = sum([torch.square( cdf_comb_hat[i] - 1*(self.support >= y_batch[i]) ).sum() for i in range(len(y_batch))])
                pinball_loss = (self.crit_fract*error_hat[error_hat>0].norm(p=1) + (1-self.crit_fract)*error_hat[error_hat<0].norm(p=1))
                sql2_loss = torch.square(error_hat).sum()
                
                # Total regret (scale CRPS for better trade-off control)
                loss = 2*(1-self.risk_aversion)*pinball_loss + 2*self.risk_aversion*sql2_loss \
                    + self.gamma*crps_i/len(self.support)

                total_loss += loss.item()
                
        average_loss = total_loss / len(data_loader)
        return average_loss
                
    def predict_weights(self, x):
        'Forecast combination weights, inference only'
        with torch.no_grad():            
            return self.model(x).detach().numpy()
        
class LinearPoolNewsvendorLayer(nn.Module):        
    def __init__(self, num_inputs, support, 
                 gamma, problem = 'reg_trad', apply_softmax = False, critic_fract = 0.5, regularizer = 'crps', risk_aversion = 0):
        super(LinearPoolNewsvendorLayer, self).__init__()

        # Initialize learnable weight parameters
        #self.weights = nn.Parameter(torch.rand(num_inputs), requires_grad=True)
        self.weights = nn.Parameter(torch.FloatTensor((1/num_inputs)*np.ones(num_inputs)).requires_grad_())
        self.num_inputs = num_inputs
        self.support = support
        self.risk_aversion = risk_aversion
        self.gamma = gamma
        self.apply_softmax = apply_softmax
        self.regularizer = regularizer
        self.crit_fract = critic_fract
        self.problem = problem
        
        n_locations = len(self.support)
        
        if self.problem == 'reg_trad':
            # newsvendor layer (for i-th observation)
            z = cp.Variable((1))    
            pinball_loss = cp.Variable(n_locations)
            error = cp.Variable(n_locations)
            prob_weights = cp.Parameter(n_locations)
            sqrt_prob_weights = cp.Parameter(n_locations)

            newsv_constraints = [z >= 0, z <= 1, error == self.support - z,
                                 pinball_loss >= self.crit_fract*(error), 
                                 pinball_loss >= (self.crit_fract - 1)*(error)]
            
            newsv_cost = prob_weights@pinball_loss
            
            # define aux variable
            #w_error = cp.multiply(prob_weights, error)
            #sq_error = cp.power(error, 2)
            w_error = cp.multiply(sqrt_prob_weights, error)
            l2_regularization = cp.sum_squares(w_error)
    
            #l2_regularization = (prob_weights@sq_error)
    
            objective_funct = cp.Minimize( 2*(1-self.risk_aversion)*newsv_cost + 2*(self.risk_aversion)*l2_regularization ) 
            
            newsv_problem = cp.Problem(objective_funct, newsv_constraints)
            self.newsvendor_layer = CvxpyLayer(newsv_problem, parameters=[prob_weights, sqrt_prob_weights],
                                               variables = [z, pinball_loss, error] )
        elif self.problem == 'pwl':
            # newsvendor layer (for i-th observation)
            z = cp.Variable((1))    
            pwl_loss = cp.Variable(n_locations)
            error = cp.Variable(n_locations)
            prob_weights = cp.Parameter(n_locations)
            
            newsv_constraints = [z >= 0, z <= 1, error == self.support - z,
                                 pwl_loss >= 0.2*(error),
                                 pwl_loss >= -0.5*(error),
                                 pwl_loss >= -0.8*(error + 0.3)]
            
            newsv_cost = prob_weights@pwl_loss
    
            objective_funct = cp.Minimize( newsv_cost )             
            newsv_problem = cp.Problem(objective_funct, newsv_constraints)
            
            self.newsvendor_layer = CvxpyLayer(newsv_problem, parameters=[prob_weights],
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
        if self.apply_softmax:
            weights = torch.nn.functional.softmax(self.weights, dim = 0)
        else:
            weights = self.weights

        # Apply the weights element-wise to each input tensor
        weighted_inputs = [weights[i] * input_tensor for i, input_tensor in enumerate(list_inputs)]

        # Perform the convex combination across input vectors
        combined_pdf = sum(weighted_inputs)

        # Pass the combined output to the CVXPY layer
        if self.problem == 'reg_trad':
            cvxpy_output = self.newsvendor_layer(combined_pdf, torch.sqrt(combined_pdf + 1e-4))
        elif self.problem == 'pwl':
            cvxpy_output = self.newsvendor_layer(combined_pdf)
            
        return combined_pdf, cvxpy_output
    
    def train_model(self, train_loader, val_loader, optimizer, epochs = 20, patience=5, projection = True, validation = False, 
                    relative_tolerance = 0):
        # define projection problem for backward pass

        if (projection)and(self.apply_softmax != True):     
            lambda_proj = cp.Variable(self.num_inputs)
            lambda_hat = cp.Parameter(self.num_inputs)
            proj_problem = cp.Problem(cp.Minimize(0.5*cp.sum_squares(lambda_proj-lambda_hat)), [lambda_proj >= 0, lambda_proj.sum()==1])
        
        
        L_t = []
        best_train_loss = 1e7
        best_val_loss = 1e7 
        early_stopping_counter = 0
        best_weights = copy.deepcopy(self.state_dict())

        for epoch in range(epochs):
            # activate train functionality
            self.train()
            running_loss = 0.0
            # sample batch data
            for batch_data in train_loader:
                
                y_batch = batch_data[-1]
                
                # clear gradients
                optimizer.zero_grad()
                
                # forward pass: combine forecasts and solve each newsvendor problem
                output_hat = self.forward(batch_data[:-1])

                pdf_comb_hat = output_hat[0]
                cdf_comb_hat = pdf_comb_hat.cumsum(1)
                
                z_hat = output_hat[1][0]
                
                # estimate aggregate pinball loss and CRPS (for realization of uncertainty)
                error_hat = (y_batch.reshape(-1,1) - z_hat)

                crps_i = sum([torch.square( cdf_comb_hat[i] - 1*(self.support >= y_batch[i]) ).sum() for i in range(len(y_batch))])
                pinball_loss = (self.crit_fract*error_hat[error_hat>0].norm(p=1) + (1-self.crit_fract)*error_hat[error_hat<0].norm(p=1))
                sql2_loss = torch.square(error_hat).sum()
                
                # Total regret (scale CRPS for better trade-off control)
                loss = 2*(1-self.risk_aversion)*pinball_loss + 2*self.risk_aversion*sql2_loss \
                    + self.gamma*crps_i/len(self.support)
                
                # backward pass
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
                

    def evaluate(self, data_loader):
        # evaluate loss criterion/ used for estimating validation loss
        self.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch_data in data_loader:
                y_batch = batch_data[-1]

                # forward pass: combine forecasts and solve each newsvendor problem
                output_hat = self.forward(batch_data[:-1])

                pdf_comb_hat = output_hat[0]
                cdf_comb_hat = pdf_comb_hat.cumsum(1)
                
                z_hat = output_hat[1][0]
                
                # estimate aggregate pinball loss and CRPS (for realization of uncertainty)
                error_hat = (y_batch.reshape(-1,1) - z_hat)

                crps_i = sum([torch.square( cdf_comb_hat[i] - 1*(self.support >= y_batch[i]) ).sum() for i in range(len(y_batch))])
                pinball_loss = (self.crit_fract*error_hat[error_hat>0].norm(p=1) + (1-self.crit_fract)*error_hat[error_hat<0].norm(p=1))
                
                sql2_loss = torch.square(error_hat).sum()
                
                # Total regret (scale CRPS for better trade-off control)
                loss = 2*(1-self.risk_aversion)*pinball_loss + 2*self.risk_aversion*sql2_loss \
                    + self.gamma*crps_i/len(self.support)
                

                total_loss += loss.item()
                
        average_loss = total_loss / len(data_loader)
        return average_loss
    
class LinearPoolCRPSLayer(nn.Module):        
    def __init__(self, num_inputs, support, apply_softmax = False):
        super(LinearPoolCRPSLayer, self).__init__()

        # Initialize learnable weight parameters
        #self.weights = nn.Parameter(torch.rand(num_inputs), requires_grad=True)
        self.weights = nn.Parameter(torch.FloatTensor((1/num_inputs)*np.ones(num_inputs)).requires_grad_())
        self.num_inputs = num_inputs
        self.support = support
        self.apply_softmax = apply_softmax
        
    def forward(self, list_inputs):
        """
        Forward pass of the linear pool minimizing CRPS.

        Args:
            list_inputs: A list of of input tensors (discrete PDFs).

        Returns:
            torch.Tensor: The convex combination of input tensors.
        """
        # Ensure that the weights are in the range [0, 1] using sigmoid activation
        #weights = torch.nn.functional.softmax(self.weights)

        # Ensure that the weights are in the range [0, 1] using sigmoid activation
        if self.apply_softmax:
            weights = torch.nn.functional.softmax(self.weights, dim = 0)
        else:
            weights = self.weights
        
        # Apply the weights element-wise to each input tensor !!!! CDFs
        weighted_inputs = [weights[i] * input_tensor.cumsum(1) for i, input_tensor in enumerate(list_inputs)]

        # Perform the convex combination across input vectors
        combined_CDF = sum(weighted_inputs)

        return combined_CDF
    
    def train_model(self, train_loader, optimizer, epochs = 20, patience=5, projection = True):
        # define projection problem for backward pass
        lambda_proj = cp.Variable(self.num_inputs)
        lambda_hat = cp.Parameter(self.num_inputs)
        proj_problem = cp.Problem(cp.Minimize(0.5*cp.sum_squares(lambda_proj-lambda_hat)), [lambda_proj >= 0, lambda_proj.sum()==1])
        
        
        L_t = []
        best_train_loss = float('inf')

        for epoch in range(epochs):
            # activate train functionality
            self.train()
            running_loss = 0.0
            # sample batch
            for batch_data in train_loader:
                
                y_batch = batch_data[-1]
                
                #cdf_batch = [batch_data[i] for i in range(self.num_inputs)]

                # clear gradients
                optimizer.zero_grad()
                
                # forward pass: combine forecasts and solve each newsvendor problem
                comb_CDF = self.forward(batch_data[:-1])
                
                # estimate CRPS (heavyside function)
                loss_i = [torch.square( comb_CDF[i] - 1*(self.support >= y_batch[i]) ).sum() for i in range(len(y_batch))]
                loss = sum(loss_i)/len(loss_i)

                # Decomposition (see Online learning with the Continuous Ranked Probability Score for ensemble forecasting) 
                #divergence_i = [(weights[j]*torch.norm(self.support - y_batch[i] )) for ]
                
                #loss_i = [weights[j]*torch.abs(self.support >= y_batch[i]) ).sum() for i in range(len(y_batch))]
                #loss = sum(loss_i)/len(loss_i)
                
                # backward pass
                loss.backward()
                optimizer.step()                
                
                if (projection)and(self.apply_softmax != True):     
                    lambda_hat.value = to_np(self.weights)
                    proj_problem.solve(solver = 'GUROBI')
                    # update parameter values
                    with torch.no_grad():
                        self.weights.copy_(torch.FloatTensor(lambda_proj.value))
                
                running_loss += loss.item()
            

            L_t.append(to_np(self.weights).copy())
            average_train_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {average_train_loss:.4f} ")

            if average_train_loss < best_train_loss:
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

        # Apply the weights element-wise to each input tensor !!!! CDFs
        
        #weighted_inputs = [weights[k,i] * input_tensor for k in range(weights.shape[0]) for i, input_tensor in enumerate(list_inputs)]
       # weighted_inputs = [weights[:,i] * input_tensor[:,k] for k in range(len(self.support)) 
       #                    for i, input_tensor in enumerate(list_inputs)]

        weighted_inputs = [torch.tile(weights[:,i:i+1], (1, input_tensor.shape[1])) * input_tensor for i, input_tensor in enumerate(list_inputs)]
        
        # Perform the convex combination across input vectors
        
        combined_PDF = sum(weighted_inputs)
        
        return combined_PDF
    
    def train_model(self, train_loader, val_loader, 
                    optimizer, epochs = 20, patience=5, projection = False):
        
        if (projection)and(self.apply_softmax != True):     
            lambda_proj = cp.Variable(self.num_inputs)
            lambda_hat = cp.Parameter(self.num_inputs)
            proj_problem = cp.Problem(cp.Minimize(0.5*cp.sum_squares(lambda_proj-lambda_hat)), [lambda_proj >= 0, lambda_proj.sum()==1])
        
        L_t = []
        best_train_loss = float('inf')
        best_val_loss = float('inf')
        early_stopping_counter = 0
        best_weights = copy.deepcopy(self.state_dict())

        for epoch in range(epochs):
            # activate train functionality
            self.train()
            running_loss = 0.0
            # sample batch data
            for batch_data in train_loader:
                
                y_batch = batch_data[-1]
                x_batch = batch_data[-2]
                p_list_batch = batch_data[0:-2]
                
                # clear gradients
                optimizer.zero_grad()
                
                # forward pass: predict weights and combine forecasts
                comb_PDF = self.forward(x_batch, p_list_batch)
                comb_CDF = comb_PDF.cumsum(1)
                
                # estimate CRPS (heavyside function)
                loss_i = [torch.square( comb_CDF[i] - 1*(self.support >= y_batch[i]) ).sum() for i in range(len(y_batch))]
                loss = sum(loss_i)/len(loss_i)
                
                # backward pass
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
            

            L_t.append(to_np(self.weights).copy())
                
            average_train_loss = running_loss / len(train_loader)
            val_loss = self.evaluate(val_loader)
            
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

    def evaluate(self, data_loader):
        # evaluate loss criterion/ used for estimating validation loss
        self.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch_data in data_loader:
                y_batch = batch_data[-1]
                x_batch = batch_data[-2]
                p_list_batch = batch_data[0:-2]

                # forward pass: combine forecasts and solve each newsvendor problem
                comb_PDF_hat = self.forward(x_batch, p_list_batch)
                comb_CDF_hat = comb_PDF_hat.cumsum(1)

                # estimate CRPS (heavyside function)
                loss_i = [torch.square( comb_CDF_hat[i] - 1*(self.support >= y_batch[i]) ).sum() for i in range(len(y_batch))]
                loss = sum(loss_i)/len(loss_i)
                total_loss += loss.item()
                
        average_loss = total_loss / len(data_loader)
        return average_loss
                
    def predict_weights(self, x):
        'Forecast combination weights, inference only'
        with torch.no_grad():            
            return self.model(x).detach().numpy()
