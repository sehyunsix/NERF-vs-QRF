import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn


class sin_ml(nn.Module):
    def __init__(self, num_layer):
        """
        hidden_dim(int) : hidden layer의 노드 개수
        num_layer(int) : 총 layer의 개수 (입력 layer 1개 + hidden_layer 개수 + output layer 1개)
        """
        super(sin_ml, self).__init__()
        hidden_dim = 8
        linear_list = []
        linear_list.append(nn.Linear(1, hidden_dim))
        linear_list.append(nn.ReLU(hidden_dim))
        for _ in range(num_layer - 2):
            linear_list.append(nn.Linear(hidden_dim, hidden_dim))
            linear_list.append(nn.ReLU(hidden_dim))
        linear_list.append(nn.Linear(hidden_dim, 1))
        linear_list.append(nn.Tanh())  # For sin

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

        ## Instance Initialize ##
        self.num_qubit = num_qubit
        self.num_layer = num_layer

        ## Parameter Generation ##
        self.required_parameters = 2 * self.num_qubit * (num_layer - 1)
        self.theta = nn.Parameter(
            torch.rand(self.required_parameters), requires_grad=True
        )  # parameter 초기화 시 범위(-pi ~ pi?) 고려

        ## Quantum Device Initialize ##
        self.device = qml.device("default.qubit", wires=num_qubit)

    def quantum_circuit(self, theta):
        """
        theta(list or tensor) : 1개의 layer에 대한 (2 * num_qubit)개의 parameter set
        """
        for i in range(self.num_qubit):
            qml.RX(theta[2 * i], wires=i)
            qml.RY(theta[2 * i + 1], wires=i)

        for i in range(self.num_qubit - 1):
            qml.CNOT(wires=[i, i + 1])

    def pqc(self, x, chk=False):
        """
        TODO
            inner_pqc를 통해 self.device에서 quantum circuit을 동작시킴
            1. AngleEmbedding 메소드를 통해 x에 대한 Embedding
            2. num_layer만큼의 layer 반복, 이때 self.theta 사용
            3. (Option) Expectation Value의 대상이 되는 Observable 생성
            4. Measure (qml.expval)
        Args
            x(float32 or tensor(batch, 1)) : sin의 input으로 주어진 값
            chk(bool) : Quantum Circuit Draw 여부
        """

        @qml.qnode(self.device, interface="torch")
        def inner_pqc():
            ## Embedding Section ##
            qml.AngleEmbedding(x, wires=range(self.num_qubit))
            for i in range(self.num_qubit - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.Barrier()

            ## Layer Iteration ##
            for i in range(0, self.required_parameters, 2 * self.num_qubit):
                self.quantum_circuit(theta=self.theta[i : i + 2 * self.num_qubit])
                qml.Barrier()

            ## Observable Generation ##
            obs = qml.PauliZ(0)
            for i in range(1, self.num_qubit):
                obs = obs @ qml.PauliZ(i)

            ## Measure ##
            return qml.expval(obs)  # qml.expval(qml.PauliZ(0))

        # qml.draw_mpl(inner_pqc)()

        return inner_pqc()

    def forward(self, x):
        output = self.pqc(x)
        output = output.reshape(-1, 1)
        return output.float()  # -1 ~ 1까지만의 value 가질 수 있음
