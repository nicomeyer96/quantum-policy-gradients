from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

"""
Instantiates variational quantum circuit
"""


def _equal_superposition(vqc, n_qubits):
    """
    Create initially equal superposition of all computational basis states

    :param qc: VQC handle
    :param n_qubits: number of qubits
    """

    for i in range(n_qubits):
        vqc.h(i)


def _variational_layer(vqc, params_variational, n_qubits, depth):
    """
    Variational layer

    :param vqc: VQC handle
    :param params_variational: variational parameters
    :param n_qubits: number of qubits
    :param depth: current layer
    """

    for i in range(n_qubits):
        idx = (2 * n_qubits) * depth + 2 * i
        vqc.rz(params_variational[idx], i)
        vqc.ry(params_variational[idx+1], i)


def _entanglement_layer(vqc, n_qubits):
    """
    Entangling layer (all-to-all)

    :param vqc: VQC handle
    :param n_qubits: number of qubits
    """

    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            vqc.cz(i, j)


def _encoding_layer(vqc, params_scaling, params_encoding, n_qubits, depth):
    """
    Feature map with first-order angle encoding

    :param vqc: VQC handle
    :param params_scaling: scaling parameters
    :param params_encoding: state encoding parameters
    :param n_qubits: number of qubits
    :param depth: current layer
    """

    for i in range(n_qubits):
        idx = (2 * n_qubits) * depth + 2 * i
        vqc.ry(params_scaling[idx] * params_encoding[i], i)
        vqc.rz(params_scaling[idx+1] * params_encoding[i], i)


def construct_circuit(n_qubits, depth):
    """
    Construct VQC

    :param n_qubits: number of qubits
    :param depth: depth of VQC
    :return: VQC, learnable parameters, state encoding parameters, # learnable parameters, # state encoding parameters
    """

    # number of elements for the different parameter sets
    n_params_variational = 2 * n_qubits * (depth+1)
    n_params_scaling = 2 * n_qubits * depth
    n_params_encoding = n_qubits

    # container for the different parameter sets
    params_variational = ParameterVector('\u03B8', length=n_params_variational)
    params_scaling = ParameterVector('\u03BB', length=n_params_scaling)
    params_encoding = ParameterVector('s', length=n_params_encoding)

    vqc = QuantumCircuit(n_qubits)
    _equal_superposition(vqc, n_qubits=n_qubits)
    vqc.barrier()
    for d in range(depth):
        _variational_layer(vqc, params_variational=params_variational, n_qubits=n_qubits, depth=d)
        _entanglement_layer(vqc, n_qubits=n_qubits)
        vqc.barrier()
        _encoding_layer(vqc, params_scaling=params_scaling, params_encoding=params_encoding, n_qubits=n_qubits, depth=d)
        vqc.barrier()
    _variational_layer(vqc, params_variational=params_variational, n_qubits=n_qubits, depth=depth)
    _entanglement_layer(vqc, n_qubits=n_qubits)

    return vqc, params_variational, params_scaling, params_encoding, n_params_variational, n_params_scaling
