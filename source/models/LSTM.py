import torch
import torch.nn as nn

class LSTM_single(nn.Module):
    def __init__(self, seq_length, hidden_size=50, num_layers=2, output_size=4):
        super(LSTM_single, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(1, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Reshape input to (batch_size, seq_length, input_size)
        x = x.unsqueeze(-1)  # Add an input_size dimension of 1, making it (batch_size, seq_length, 1)
        
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])  # We take the last output only

        return out
    

class LSTM_single_bounded(nn.Module):
    def __init__(self, seq_length, hidden_size=50, num_layers=2, output_size=4):
        super(LSTM_single, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(1, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Reshape input to (batch_size, seq_length, input_size)
        x = x.unsqueeze(-1)  # Add an input_size dimension of 1, making it (batch_size, seq_length, 1)
        
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])  # We take the last output only

        # Apply different transformations for each output
        a = torch.tensor([7/15, 0.25, 7/6, 7/12])  # Scaling factors
        b = torch.tensor([11/300, -9/250, -5/24, -1/15])  # Shifts
        x = a * x + b # Apply scaling and shifting with a margin

        return out
    

class LSTM_parallel(nn.Module):
    def __init__(self, in_features, hidden_size=50, num_layers=2, output_size=1):
        super(LSTM_parallel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(1, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Reshape input to (batch_size, seq_length, input_size)
        x = x.unsqueeze(-1)  # Add an input_size dimension of 1, making it (batch_size, seq_length, 1)
        
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])  # We take the last output only
        return out
    

class LSTM_parallel_bounded(nn.Module):
    def __init__(self, in_features, hidden_size=50, num_layers=2, output_size=1):
        super(LSTM_parallel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(1, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Reshape input to (batch_size, seq_length, input_size)
        x = x.unsqueeze(-1)  # Add an input_size dimension of 1, making it (batch_size, seq_length, 1)
        
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])  # We take the last output only

        out = 7/6 * out   # Scale the output to [0, 7/6]
        
        return out