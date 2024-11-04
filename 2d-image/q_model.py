import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WIRES = 3
<<<<<<< HEAD
LAYER = 16
=======
LAYER = 1
>>>>>>> 63edeb7d2fc2da68720419da7ff2d6a6de90fe94
# Define the quantum device with 3 qubits
dev = qml.device("default.qubit", wires=WIRES)
# dev = qml.device("default.qubit.torch", wires=3, torch_device="cpu")

<<<<<<< HEAD
=======

# Define the quantum circuit (qnode) with autograd support for PyTorch
@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def quantum_circuit(x, y, theta):
    # Encoding circuit: rotate qubits based on input data (x, y)

    x = 2 * np.pi * x - np.pi  # x in [0, 1] ~> [-pi, pi]
    y = 2 * np.pi * y - np.pi  # y in [0, 1] ~> [-pi, pi]

    qml.RX(x, wires=0)
    qml.RX(y, wires=1)

    # Parameterized quantum circuit with entanglement
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])

    qml.RZ(theta[0], wires=0)
    qml.RZ(theta[1], wires=1)
    qml.RZ(theta[2], wires=2)

    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])

    qml.RZ(theta[3], wires=0)
    qml.RZ(theta[4], wires=1)
    qml.RZ(theta[5], wires=2)

    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])

    qml.RZ(theta[6], wires=0)
    qml.RZ(theta[7], wires=1)
    qml.RZ(theta[8], wires=2)

    result = [qml.expval(qml.PauliZ(i)) for i in range(3)]  ## 여기에서 grad 날라감 ㅅㅂ

    # result_tensor = torch.tensor(result, dtype=torch.float32)

    # Measurement: expectation values of the PauliZ operator for each wire
    return result


@qml.qnode(dev, interface="torch")  # , diff_method="parameter-shift")
def quantum_circuit_n_qubit(x, y, theta):
    # Encoding circuit: rotate qubits based on input data (x, y)

    x = 2 * np.pi * x - np.pi  # x in [0, 1] ~> [-pi, pi]
    y = 2 * np.pi * y - np.pi  # y in [0, 1] ~> [-pi, pi]

    qml.RX(x, wires=0)
    qml.RX(y, wires=1)

    # Parameterized quantum circuit with entanglement
    for layer_count in range(LAYER):
        for i in range(WIRES - 1):
            qml.CNOT(wires=[i, i + 1])

        # for i in range(WIRES):
        #     qml.RZ(theta[layer_count * WIRES + i], wires=i)

        # for i in range(WIRES - 1):
        #     qml.CNOT(wires=[i, i + 1])

        for i in range(WIRES):
            qml.RX(theta[layer_count * WIRES + i * 1], wires=i)

        for i in range(WIRES - 1):
            qml.CNOT(wires=[i, i + 1])

        for i in range(WIRES):
            qml.RY(theta[layer_count * WIRES + i * 2], wires=i)
        # for i in range(WIRES):
        #     qml.AmplitudeDamping(0.1, wires=i)

    result = [qml.expval(qml.PauliZ(i)) for i in range(3)]  ## 여기에서 grad 날라감 ㅅㅂ

    # result_tensor = torch.tensor(result, dtype=torch.float32)

    # Measurement: expectation values of the PauliZ operator for each wire
    return result
>>>>>>> 63edeb7d2fc2da68720419da7ff2d6a6de90fe94


# Define the PyTorch model
class QModel(nn.Module):

    def __init__(self, n_wire, n_layer):
        super(QModel, self).__init__()
        # Initialize trainable parameters (theta) for the quantum circuit
<<<<<<< HEAD
        self.n_wires = n_wire
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

        ## 여기에서 grad 날라감 ㅅㅂ
        return [qml.expval(qml.PauliZ(i)) for i in range(3)]

    def forward(self, x):
        return F.relu(
            torch.stack((self.quantum_circuit_n_qubit(self, x=x, theta=self.theta))).T
        )
=======
        self.n_qubit = WIRES
        self.theta = nn.Parameter(
            torch.rand(LAYER * WIRES * 2, dtype=torch.float32),
            requires_grad=True,
        )  # 9 trainable parameters

    def forward(self, x):
        # Split input x into two parts for X and Y encoding in the quantum circuit
        x1, x2 = x[:, 0], x[:, 1]

        # Apply the quantum circuit to each pair of inputs (x1, x2)åå
        outputs = [
            torch.stack(quantum_circuit_n_qubit(x1[i], x2[i], self.theta))
            for i in range(x.shape[0])
        ]

        # outputs = torch.stack(quantum_circuit(x1, x2, self.theta)).reshape((-1, 3))

        # outputs = torch.stack(
        #     [(e + 1.0) / 2.0 for e in quantum_circuit_n_qubit(x1, x2, self.theta)]
        # ).T

        # outputs = torch.stack([torch.tensor(x1 ,requires_grad=True)*2,torch.tensor(x1,requires_grad=True)*3,torch.tensor(x1,requires_grad=True)*4])
        # print(outputs)

        # Convert the output to a tensor for compatibility with PyTorchå
        # return outputs
        return torch.stack(
            [(o + 1.0) / 2.0 for o in outputs]
        )  # torch.tensor(outputs, dtype=torch.float32)
>>>>>>> 63edeb7d2fc2da68720419da7ff2d6a6de90fe94
