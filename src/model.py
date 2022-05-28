import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1, dropout=0.5):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_directions = 2
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, dtype=torch.double, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size, dtype=torch.double)
        self.fc2 = nn.Linear(hidden_size, output_size, dtype=torch.double)

    def forward(self, x):
        h_0 = torch.randn(self.num_directions * self.num_layers, x.size(0), self.hidden_size, dtype=torch.double).cuda()
        c_0 = torch.randn(self.num_directions * self.num_layers, x.size(0), self.hidden_size, dtype=torch.double).cuda()
        x, _ = self.lstm(x, (h_0, c_0))
        x = F.relu(self.fc1(x[:, -1, :]))
        x = self.fc2(x)
        # x = self.fc2(x[:, -1, :])
        return x
