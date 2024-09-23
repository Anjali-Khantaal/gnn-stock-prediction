# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class TemporalGNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super(TemporalGNNModel, self).__init__()
        self.gcn = GCNConv(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x_list, edge_index):
        gcn_outputs = []
        for x in x_list:
            x = self.gcn(x, edge_index)
            x = F.relu(x)
            # Convert to dense tensor if it's sparse
            if x.is_sparse:
                x = x.to_dense()
            x = x.unsqueeze(0)
            gcn_outputs.append(x)
        gcn_sequence = torch.cat(gcn_outputs, dim=0)
        gcn_sequence = gcn_sequence.permute(1, 0, 2)
        lstm_out, _ = self.lstm(gcn_sequence)
        out = self.fc(lstm_out[:, -1, :])
        return out.squeeze()
