import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

# WIRES = 3
# LAYER = 16
# Define the quantum device with 3 qubits
dev = qml.device("default.qubit", wires=3)
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

            for i in range(self.n_wires):
                qml.RX(theta[layer_count * self.n_wires + i * 2 + 1], wires=i)

            for i in range(self.n_wires - 1):
                qml.CNOT(wires=[i, i + 1])

            for i in range(self.n_wires):
                qml.RY(theta[layer_count * self.n_wires + i * 2], wires=i)

        return [qml.expval(qml.PauliZ(i)) for i in range(3)]

    def forward(self, x):
        return F.relu(
            torch.stack((self.quantum_circuit_n_qubit(self, x=x, theta=self.theta))).T
        )
