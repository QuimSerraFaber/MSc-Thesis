import torch
import torch.optim as optim
from models.Initial_fc_nn import FullyConnectedNN
from Losses import MSE, compute_absolute_loss

def load_data():
    # Function to load your data
    # This should return the input data and corresponding true parameters as separate arrays
    return inputs, true_params  # true_params should be a 2D array where each column is a parameter



# Load data
inputs, true_params = load_data()

# Convert data to PyTorch tensors
inputs = torch.tensor(inputs, dtype=torch.float32)
true_params = torch.tensor(true_params, dtype=torch.float32)

# Initialize the models for each parameter
models = [FullyConnectedNN(in_features=inputs.shape[1]) for _ in range(4)]

# Optimizers (one for each model)
optimizers = [optim.Adam(model.parameters(), lr=0.001) for model in models]


# Define training parameters
num_epochs = 10
batch_size = 128

# Training loop
num_epochs = 10  # Set the number of epochs
for epoch in range(num_epochs):
    for i, model in enumerate(models):
        optimizer = optimizers[i]
        optimizer.zero_grad()

        # Forward pass
        output_param = model(inputs).squeeze()

        # Compute loss for the specific parameter
        loss = compute_absolute_loss(output_param, true_params[:, i])

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        print(f"Model {i+1}, Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")



