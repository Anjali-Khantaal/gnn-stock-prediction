# evaluate.py

import torch
import torch.nn as nn

def evaluate_model(model, inputs, targets, edge_index):
    model.eval()
    total_loss = 0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for x_seq, y in zip(inputs, targets):
            out = model(x_seq, edge_index)
            loss = criterion(out, y)
            total_loss += loss.item()
    print(f'Evaluation Loss: {total_loss/len(inputs)}')
