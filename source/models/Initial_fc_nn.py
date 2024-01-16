import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network model
class FullyConnectedNN(nn.Module):
    def __init__(self, in_features):
        super(FullyConnectedNN, self).__init__()
        self.fc1 = nn.Linear(in_features, 64)  # Adjust in_features to match your data
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(32, 1)  # Output layer

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)  # No activation function in the output layer
        return x
    


