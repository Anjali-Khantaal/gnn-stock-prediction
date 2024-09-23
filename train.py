# train.py

import torch
import torch.nn as nn
from model import TemporalGNNModel
from torch_geometric.utils import from_networkx

def prepare_dataset(data, G, features, sequence_length=5):
    edge_index = from_networkx(G).edge_index
    graph_data_list = []

    dates = sorted(list(set().union(*(df.index for df in data.values()))))

    for date in dates:
        node_features = []
        node_labels = []
        valid = True
        for ticker in data:
            df = data[ticker]
            if date not in df.index:
                valid = False
                break
            feature = df.loc[date, features].values
            label = df.loc[date, 'Close']
            node_features.append(feature)
            node_labels.append(label)
        if valid:
            x = torch.tensor(node_features, dtype=torch.float)
            y = torch.tensor(node_labels, dtype=torch.float)
            graph_data_list.append((x, y))

    # Create sequences
    inputs = []
    targets = []
    for i in range(len(graph_data_list) - sequence_length):
        x_seq = [graph_data_list[j][0] for j in range(i, i + sequence_length)]
        y = graph_data_list[i + sequence_length][1]
        inputs.append(x_seq)
        targets.append(y)

    return inputs, targets, edge_index

def train_model(inputs, targets, edge_index, input_dim):
    model = TemporalGNNModel(input_dim=input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    dataset = list(zip(inputs, targets))

    model.train()
    for epoch in range(50):
        total_loss = 0
        for x_seq, y in dataset:
            optimizer.zero_grad()
            out = model(x_seq, edge_index)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(dataset)}')

    return model
