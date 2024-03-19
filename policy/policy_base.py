import numpy as np
from abc import ABC, abstractmethod

from qiskit import Aer
from qiskit.opflow.gradients import Gradient
from qiskit.utils import QuantumInstance

from policy.circuit import construct_circuit


class Policy(ABC):
    """
    Interface for policy classes
    """

    def __init__(self, n_qubits, n_actions, depth):
        """
        Initialize policy base class

        :param n_qubits: number of qubits
        :param n_actions: number of actions
        :param depth: depth of VQC
        """

        # This simulator handles noise-free simulation quite efficiently
        self.simulator = Aer.get_backend('statevector_simulator')

        # It seems to be a bit faster to parallelize over shots instead of circuits
        # (circuit parallelization only relevant for gradient computation with parameter-shift rule)
        # -> overall, the parallelization capabilities of qiskit are not very sophisticated, it is much more efficient
        #    to parallelize within the QPG algorithm itself, i.e. compute the log-policy gradients parallel over the
        #    trajectory. However, this permits usage of the TorchConnector, so we choose not to use this approach for
        #    this implementation which should demonstrate only the principles
        self.simulator.set_options(max_parallel_experiments=1, max_parallel_shots=0)

        self.quantum_instance = QuantumInstance(self.simulator)

        # Works a bit faster than 'param_shift', the results re completely equivalent for noise-free simulation
        self.gradient = Gradient(grad_method='lin_comb')

        # Constructs array [0, 1, .... |A|-1], which simplifies sampling an action from the extracted policy later on
        self.action_space = np.arange(n_actions)

        self.vqc, self.params_variational, self.params_scaling, self.params_encoding,\
            self.n_params_variational, self.n_params_scaling = construct_circuit(n_qubits, depth)

    @abstractmethod
    def select_action(self, state):
        """
        Abstract method to select action

        :param state: observed environment state
        :return: action, policy
        """

        pass

    @abstractmethod
    def get_model_weights(self):
        """
        Abstract method to get model weights

        :return: variational weights, scaling weights
        """

        pass
