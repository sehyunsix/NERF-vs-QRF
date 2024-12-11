import torch
import torch.nn as nn
import math
import pennylane as qml
import torch.nn.functional as F


dev = qml.device("default.qubit", wires=3)


class MLPColorPredictor(nn.Module):
    def __init__(self, num_layers=4):
        input_dim = 2
        hidden_dim = 8
        output_dim = 3
        super(MLPColorPredictor, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(
            nn.Linear(hidden_dim, output_dim)
        )  # RGB color output (3 channels)
        layers.append(nn.Sigmoid())  # Color values in range [0, 1]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class PositionEncodingMLP(nn.Module):
    def __init__(
        self, input_dim=2, hidden_dim=512, output_dim=3, num_layers=4, encoding_dim=10
    ):
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
        jisu = 1
        for i in range(self.encoding_dim):
            encoded.append(torch.sin(jisu * math.pi * x))
            encoded.append(torch.cos(jisu * math.pi * x))
            jisu *= 2
        return torch.cat(encoded, dim=-1)

    def forward(self, x):
        # Apply position encoding to each (x, y) pair
        encoded_x = self.position_encoding(x)
        return self.model(encoded_x)


# dev = qml.device("default.qubit.torch", wires=3, torch_device="cpu")


# Define the PyTorch model
class QModel(nn.Module):

    def __init__(self, n_layer):
        super(QModel, self).__init__()
        # Initialize trainable parameters (theta) for the quantum circuit
        self.n_wires = 3
        self.n_layers = n_layer
        self.n_qubit = self.n_wires
        self.theta = nn.Parameter(
            torch.rand(self.n_layers * self.n_wires * 2, dtype=torch.float32),
            requires_grad=True,
        )  # 9 trainable parameters

    @qml.qnode(dev, interface="torch")  # , diff_method="parameter-shift")
    def quantum_circuit_n_qubit(self, x, theta):
        # Encoding circuit: rotate qubits based on input data (x, y)
        qml.AngleEmbedding(x, wires=[0, 1], rotation="X")
        # Parameterized quantum circuit with entanglement
        for layer_count in range(self.n_layers):
            for i in range(self.n_wires - 1):
                qml.CNOT(wires=[i, i + 1])

            # for i in range(self.n_wires):
            #     qml.RX(theta[layer_count * self.n_wires + i * 2 + 1], wires=i)

            # for i in range(self.n_wires - 1):
            #     qml.CNOT(wires=[i, i + 1])

            for i in range(self.n_wires):
                qml.RY(theta[layer_count * self.n_wires + i * 2], wires=i)

        return [qml.expval(qml.PauliZ(i)) for i in range(3)]

    def forward(self, x):
        return F.relu(
            torch.stack((self.quantum_circuit_n_qubit(self, x=x, theta=self.theta))).T
        )
