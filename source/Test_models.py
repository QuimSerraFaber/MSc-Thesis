import torch.nn as nn
import numpy as np
import torch
from Train import training_single_model, training_parallel_models
from models.Initial_fc_nn import FC_single
from models.FC_nn_single_bounded import FC_single_bounded
from models.FC_nn_parallel import *
from source.models.LSTM import LSTM_single_bounded
from Losses import TAC_loss
from Plots import *

# Set the random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Load the data and define the loss function
#loss = nn.MSELoss()
#loss = nn.L1Loss()
loss = TAC_loss
config = { 
    'data': np.load("data/Generated_Data/simulation_simple_0.01.npz"),
    'model_class': FC_parallel_bounded,
    'loss_function': loss,
    'batch_size': 1024,
    'lr': 0.001,
    'patience': 10,
    'epochs': 150,
    'progress': True,
    'TAC_loss': True,
    'n_models': 1,
    'fast': True
}

# Initialize lists to collect the arrays
results_list = []
n_models = config['n_models']
for i in range(n_models):
    print(f"Training model {i + 1}")
    model, results = training_parallel_models(config)
    # Append the results
    results_list.append(results)

# Plot the mean and variance of the results
plot_mean_variance(results_list, config)
distribution_mean_std(results_list)
scatter_representation(results_list)