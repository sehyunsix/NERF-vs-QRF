import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from qrelu import QuantumReLU
from config import *


class sin_qml(nn.Module):
    def __init__(self, num_qubit, num_layer, use_qrelu):
        """
        nun_qubit(int) : 사용할 qubit 개수
        num_layer(int) : 총 layer의 개수 (quantum layer 개수, input layer 제외)
        """
        super(sin_qml, self).__init__()
        self.use_qrelu = use_qrelu
        self.num_qubit = num_qubit
        self.num_layer = num_layer
        self.qrelu = QuantumReLU(modified=True)
        self.required_parameters = 2 * self.num_qubit * (num_layer - 1)
        # self.theta = torch.rand(self.required_parameters)
        self.theta = nn.Parameter(
            torch.rand(self.required_parameters), requires_grad=True
        )
        self.init_theta = nn.Parameter(torch.rand(1), requires_grad=True)

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
            for i in range(self.num_qubit - 1):
                qml.CNOT(wires=[i, i + 1])

            # obs = [qml.PauliZ(i) for i in range(self.num_qubit)]

            # for i in range(1, self.num_qubit):
            #     obs = obs @ qml.PauliZ(i)

            return [qml.expval(qml.PauliZ(i)) for i in range(3)]

        if chk:
            qml.draw_mpl(inner_pqc)()
        return inner_pqc()

    def forward(self, x):
        encoding_theta = []
        # encoding_theta = self.input_layer(x)
        # print('before enc_theta shape :', encoding_theta.shape)
        # encoding_theta = encoding_theta.reshape(2 * self.num_qubit, -1)

        # encoding_theta = torch.tanh(encoding_theta)
        # print('after enc_theta shape :', encoding_theta.shape)
        output = self.pqc((1 / self.init_theta) * x, encoding_theta=encoding_theta)
        # print("shaep output ", len(output))
        output = torch.stack((output))
        if self.use_qrelu:
            output = self.qrelu(output)
        # print('before output reshape :', output.shape)
        # output = output.T
        # torch.reshape(output, (128, 3))
        # print("after output reshape :", output.shape)
        return output.T
