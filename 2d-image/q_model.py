import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn

# Define the quantum device with 3 qubits
dev = qml.device("default.qubit", wires = 10)
# dev = qml.device("default.qubit.torch", wires=3, torch_device="cpu")

# Define the quantum circuit (qnode) with autograd support for PyTorch
@qml.qnode(dev, interface="torch") #, diff_method="parameter-shift")
def quantum_circuit(x, y, theta):
    # Encoding circuit: rotate qubits based on input data (x, y)

    x = 2 * np.pi * x - np.pi # x in [0, 1] ~> [-pi, pi]
    y = 2 * np.pi * y - np.pi # y in [0, 1] ~> [-pi, pi]    

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

    result = [qml.expval(qml.PauliZ(i)) for i in range(3)] ## 여기에서 grad 날라감 ㅅㅂ

    # result_tensor = torch.tensor(result, dtype=torch.float32)

    # Measurement: expectation values of the PauliZ operator for each wire
    return result

@qml.qnode(dev, interface="torch") #, diff_method="parameter-shift")
def quantum_circuit_n_qubit(x, y, theta, n = 10):
    # Encoding circuit: rotate qubits based on input data (x, y)

    x = 2 * np.pi * x - np.pi # x in [0, 1] ~> [-pi, pi]
    y = 2 * np.pi * y - np.pi # y in [0, 1] ~> [-pi, pi]    

    qml.RX(x, wires=0)
    qml.RX(y, wires=1)
    
    # Parameterized quantum circuit with entanglement
    for layer_count in range(3):
        for i in range(n - 1):
            qml.CNOT(wires = [i, i + 1])

        for i in range(n):
            qml.RZ(theta[layer_count * 3 + i], wires = i)
    


    result = [qml.expval(qml.PauliZ(i)) for i in range(3)] ## 여기에서 grad 날라감 ㅅㅂ

    # result_tensor = torch.tensor(result, dtype=torch.float32)

    # Measurement: expectation values of the PauliZ operator for each wire
    return result

# Define the PyTorch model
class QModel(nn.Module):
    def __init__(self):
        super(QModel, self).__init__()
        # Initialize trainable parameters (theta) for the quantum circuit
        self.n_qubit = 10
        self.theta = nn.Parameter(torch.rand(3 * self.n_qubit, dtype=torch.float64) * 2 * np.pi, requires_grad=True)  # 9 trainable parameters

    def forward(self, x):
        # Split input x into two parts for X and Y encoding in the quantum circuit
        x1, x2 = x[:, 0], x[:, 1]
        
        # Apply the quantum circuit to each pair of inputs (x1, x2)åå
        # outputs = [quantum_circuit(x1[i], x2[i], self.theta) for i in range(x.shape[0])]

        # outputs = torch.stack(quantum_circuit(x1, x2, self.theta)).reshape((-1, 3))


        outputs = torch.stack([(e + 1.0) / 2.0 for e in quantum_circuit_n_qubit(x1, x2, self.theta, n = self.n_qubit)]).T
    

        # outputs = torch.stack([torch.tensor(x1 ,requires_grad=True)*2,torch.tensor(x1,requires_grad=True)*3,torch.tensor(x1,requires_grad=True)*4])
        # print(outputs)

        # Convert the output to a tensor for compatibility with PyTorchå
        return outputs
        # return torch.stack([torch.tensor(o,requires_grad=True ,dtype=torch.float64) for o in outputs]) # torch.tensor(outputs, dtype=torch.float32)