import torch
import torch.nn as nn

class LSTM_single_bounded(nn.Module):
    def __init__(self, input_size, hidden_size=256, dropout_rate=0.1):
        super(LSTM_single_bounded, self).__init__()
        self.hidden_size = hidden_size

        # LSTM layer
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # A dropout layer after the LSTM
        self.dropout1 = nn.Dropout(dropout_rate)

        # A fully connected layer
        self.fc2 = nn.Linear(hidden_size, 64)
        # Dropout layer after the second activation
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Output layer
        self.fc3 = nn.Linear(64, 4)
    
    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm1(x)
        x = lstm_out
        
        x = self.dropout1(torch.relu(x))
        x = self.dropout2(torch.relu(self.fc2(x)))
        x = torch.sigmoid(self.fc3(x))  # Apply sigmoid to all outputs
        
        # Apply different transformations for each output
        a = torch.tensor([0.28, 0.15, 0.7, 0.35], device=x.device)  # Scaling factors
        b = torch.tensor([0.13, 0.014, 0.025, 0.05], device=x.device)  # Shifts
        x = a * x + b

        return x