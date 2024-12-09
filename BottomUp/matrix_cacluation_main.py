import numpy as np
import sympy as sp

sp.init_printing()


def RX(params):
    params = sp.symbols(
        params,
        real=True,
    )

    _RX = np.array(
        [
            [sp.cos(params / 2), -sp.I * sp.sin(params / 2)],
            [-sp.I * sp.sin(params / 2), sp.cos(params / 2)],
        ]
    )
    return _RX


def RY(params):
    params = sp.symbols(
        params,
        real=True,
    )
    _RY = np.array(
        [
            [sp.cos(params / 2), -1 * sp.sin(params / 2)],
            [sp.sin(params / 2), sp.cos(params / 2)],
        ]
    )
    return _RY


def Z_3(s="IZI"):
    m_dict = {"Z": np.array([[1, 0], [0, -1]]), "I": np.eye(2)}
    result = np.kron(np.kron(m_dict[s[0]], m_dict[s[1]]), m_dict[s[2]])
    return result


def pqc(layer):
    C_NOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    I_CNOT = np.kron(np.eye(2), C_NOT)
    CNOT_I = np.kron(C_NOT, np.eye(2))
    angle_embeding = np.kron(np.kron(RX("x1"), RX("x2")), np.eye(2))
    cnot_layer = I_CNOT @ CNOT_I
    tmp = np.eye(8)
    for i in range(layer):
        pqc_layer = np.kron(np.kron(RY(f"t{i*3+1}"), RY(f"t{i*3+2}")), RY(f"t{i*3+3}"))
        tmp = tmp @ pqc_layer @ cnot_layer
    tmp = tmp @ angle_embeding
    return tmp


if __name__ == "__main__":
    ans = pqc(3)
    x = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    res = ans @ x

    x1 = np.vdot(res, Z_3("ZII") @ res)
    x2 = np.vdot(res, Z_3("IZI") @ res)
    x3 = np.vdot(res, Z_3("IIZ") @ res)

    print(sp.latex(x3))
