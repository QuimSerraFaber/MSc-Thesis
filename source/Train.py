import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# Add parent directory to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.Initial_fc_nn import FullyConnectedNN
from Losses import compute_parameter_loss
from data.Polynomial_test import load_polynomial_data


#inputs = np.random.randn(1000, 10)  # Example input data
#true_params = np.random.randn(1000, 4)  # Example target parameters

# Load data
inputs, true_params = load_polynomial_data(num_samples=10000, x_range=(-50, 50), num_points=200, noise_level=0.1, degree=3)

# Convert data to PyTorch tensors
inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
true_params_tensor = torch.tensor(true_params, dtype=torch.float32)

# Create TensorDataset
dataset = TensorDataset(inputs_tensor, true_params_tensor)

# Create DataLoader
batch_size = 128  # Set batch size
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the models for each parameter
models = [FullyConnectedNN(in_features=inputs.shape[1]) for _ in range(4)]

# Optimizers (one for each model)
optimizers = [optim.Adam(model.parameters(), lr=0.0001) for model in models]


# Training loop
num_epochs = 10  # Set the number of epochs
for epoch in range(num_epochs):
    for batch_inputs, batch_true_params in dataloader:
        for i, model in enumerate(models):
            optimizer = optimizers[i]
            optimizer.zero_grad()

            # Forward pass
            output_param = model(batch_inputs).squeeze()

            # Compute loss for the specific parameter
            loss = compute_parameter_loss(output_param, batch_true_params[:, i], use_absolute=True)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            print(f"Parameter {i+1}, Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")



# Print the shapes of the parameters
print("Input shape:", inputs.shape)
print("True parameters shape:", true_params.shape)

