
import torch
import torch.nn as nn
import math
class MLPColorPredictor(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=512, output_dim=3, num_layers=4):
        super(MLPColorPredictor, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, output_dim))  # RGB color output (3 channels)
        layers.append(nn.Sigmoid())  # Color values in range [0, 1]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class PositionEncodingMLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=512, output_dim=3, num_layers=4, encoding_dim=10):
        super(PositionEncodingMLP, self).__init__()
        self.encoding_dim = encoding_dim

        # MLP layers
        layers = []
        layers.append(nn.Linear(input_dim * 2 * encoding_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, output_dim))  # RGB output
        layers.append(nn.Sigmoid())  # Color values in range [0, 1]

        self.model = nn.Sequential(*layers)

    def position_encoding(self, x):
        encoded = []
        for i in range(self.encoding_dim):
            encoded.append(torch.sin((2 ** i) * math.pi * x))
            encoded.append(torch.cos((2 ** i) * math.pi * x))
        return torch.cat(encoded, dim=-1)

    def forward(self, x):
        # Apply position encoding to each (x, y) pair
        encoded_x = self.position_encoding(x)
        return self.model(encoded_x)
