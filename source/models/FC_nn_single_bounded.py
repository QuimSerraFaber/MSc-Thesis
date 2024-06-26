import torch
import torch.nn as nn

class FC_single_bounded(nn.Module):
    def __init__(self, in_features, dropout_rate=0.1):
        super(FC_single_bounded, self).__init__()
        self.fc1 = nn.Linear(in_features, 256)  # Adjust in_features to match the data
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(dropout_rate)  # Dropout layer after the first activation
        self.fc2 = nn.Linear(256, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(dropout_rate)  # Dropout layer after the second activation
        self.fc3 = nn.Linear(64, 4)  # Output layer

    def forward(self, x):
        x = self.dropout1(torch.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(torch.relu(self.bn2(self.fc2(x))))
        x = torch.sigmoid(self.fc3(x))  # Apply sigmoid to all outputs
        
        # Apply different transformations for each output
        a = torch.tensor([7/15, 0.25, 7/6, 7/12])  # Scaling factors
        b = torch.tensor([11/300, -9/250, -5/24, -1/15])  # Shifts
        x = a * x + b # Apply scaling and shifting with a margin b

        return x