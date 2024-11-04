import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn

# from tqdm import tqdm


class sin_ml(nn.Module):
    def __init__(self, hidden_dim, num_layer):
        """
        hidden_dim(int) : hidden layer의 노드 개수
        num_layer(int) : 총 layer의 개수 (입력 layer 1개 + hidden_layer 개수 + output layer 1개)
        """
        super(sin_ml, self).__init__()

        linear_list = []
        linear_list.append(nn.Linear(1, hidden_dim))
        linear_list.append(nn.ReLU(hidden_dim))
        for _ in range(num_layer - 2):
            linear_list.append(nn.Linear(hidden_dim, hidden_dim))
            linear_list.append(nn.ReLU(hidden_dim))
        linear_list.append(nn.Linear(hidden_dim, 1))
        linear_list.append(nn.Tanh())

        self.linear = nn.Sequential(*linear_list)

    def forward(self, x):
        return self.linear(x)


class sin_qml(nn.Module):
    def __init__(self, num_qubit, num_layer):
        """
        nun_qubit(int) : 사용할 qubit 개수
        num_layer(int) : 총 layer의 개수 (quantum layer 개수, input layer 제외)
        """
        super(sin_qml, self).__init__()

        self.num_qubit = num_qubit
        self.num_layer = num_layer

        self.required_parameters = 2 * self.num_qubit * (num_layer - 1)
        # self.theta = torch.rand(self.required_parameters)
        self.theta = nn.Parameter(
            torch.rand(self.required_parameters), requires_grad=True
        )

        # self.input_layer = nn.Linear(1, 2 * num_qubit)

        self.device = qml.device("default.qubit", wires=num_qubit)

        obs = qml.PauliZ(0)
        for i in range(1, self.num_qubit):
            obs = obs @ qml.PauliZ(i)

        self.obs = obs

    def quantum_circuit(self, theta):
        for i in range(self.num_qubit):
            qml.RX(theta[i], wires=i)
            qml.RY(theta[i + 1], wires=i)

        for i in range(self.num_qubit - 1):
            qml.CNOT(wires=[i, i + 1])

    def pqc(self, x, encoding_theta=[], chk=False):
        @qml.qnode(self.device, interface="torch")
        def inner_pqc():
            if encoding_theta != []:
                self.quantum_circuit(theta=encoding_theta)
            else:
                qml.AngleEmbedding(x, wires=range(self.num_qubit))
                for i in range(self.num_qubit - 1):
                    qml.CNOT(wires=[i, i + 1])
            for i in range(0, self.required_parameters, 2 * self.num_qubit):
                self.quantum_circuit(theta=self.theta[i : i + 2 * self.num_qubit])
            # obs = qml.PauliZ(0)
            # for i in range(1, self.num_qubit):
            #     obs = obs @ qml.PauliZ(i)

            return qml.expval(qml.PauliZ(0))

        if chk:
            qml.draw_mpl(inner_pqc)()
        return inner_pqc()

    def forward(self, x):
        encoding_theta = []
        # encoding_theta = self.input_layer(x)
        # print('before enc_theta shape :', encoding_theta.shape)
        # encoding_theta = encoding_theta.reshape(2 * self.num_qubit, -1)
        # print('after enc_theta shape :', encoding_theta.shape)
        output = self.pqc(x, encoding_theta=encoding_theta)
        # print('before output reshape :', output.shape)
        output = output.reshape(-1, 1)
        # print('after output reshape :', output.shape)
        return output.float()
