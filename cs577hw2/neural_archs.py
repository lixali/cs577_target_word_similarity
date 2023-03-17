import torch
import torch.nn as nn


# NOTE: In addition to __init__() and forward(), feel free to add
# other functions or attributes you might need.
class DAN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        # TODO: Declare DAN architecture
        super(DAN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # TODO: Implement DAN forward pass
        # x = torch.mean(x, dim=1)

        # print("enter forard function for DAN")
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x

class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        # TODO: Declare RNN model architecture
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h):
        # TODO: Implement RNN forward pass
        out, h = self.rnn(x, h)
        out = self.fc(out)
        return out, h

class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        # TODO: Declare LSTM model architecture
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # TODO: Implement LSTM forward pass
        
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        c0 = torch.zeros(1, x.size(0), self.hidden_size)
        
        # Forward pass through the LSTM layer
        out, _ = self.lstm(x, (h0, c0))
        
        # Get the last output of the sequence
        out = out[-1, :, :]
        
        # Forward pass through the linear layer
        out = self.fc(out)
        
        return out
