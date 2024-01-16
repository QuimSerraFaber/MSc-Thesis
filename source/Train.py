import torch
import torch.optim as optim
from models.Initial_fc_nn import FullyConnectedNN
from Losses import MSE
from Predicted_signal import predict_polynomial_signal

# Initialize the models
models = [FullyConnectedNN(in_features) for _ in range(4)]

# Define the optimizer
optimizers = [optim.Adam(model.parameters(), lr=0.001) for model in models]

# Define training parameters
num_epochs = 10
batch_size = 128

# Define the train function


