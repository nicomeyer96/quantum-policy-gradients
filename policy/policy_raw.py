import numpy as np

import torch
from torch import Tensor

from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit_machine_learning.connectors import TorchConnector

from policy.policy_base import Policy
from policy.pp_functions import PPFunction


class RawPolicy(Policy):
    """
    Raw Policy with Different Action Decoding Techniques
    Based on the paper "Quantum Policy Gradient Algorithm with Optimized Action Decoding", Meyer et al.
    (Original QPG algorithm from "Variational quantum policy for reinforcement learning", Jerbi et al.)
    """

    def __init__(self, n_qubits, n_actions, depth, args):
        """
        Initialize RAW-VQC policy class

        :param n_qubits: number of qubits
        :param n_actions: number of actions
        :param depth: depth of VQC
        :param args:
        """

        super().__init__(n_qubits, n_actions, depth)

        # Initialize PPFunction class, which helps to generate a specific post-processing function for action decoding
        pp = PPFunction(n_qubits=n_qubits, n_actions=n_actions)

        # Generate the post-processing function selected by the user
        pp_function = pp.select_pp_function(args.pp_function, args.q_local)

        # Construct a quantum circuit, that returns the observed computational bitstring (in decimal representation)
        # (the 'weight_params' have to be converted to lists, as qiskit_machine_learning does not currently allow
        #  to combine multiple ParameterVectors)
        # The output shape is implicitly already defined by the post-processing function, but additionally has to be
        # given via 'output_shape', for the CircuitQNN to work correctly.
        qnn = CircuitQNN(self.vqc, input_params=self.params_encoding,
                         weight_params=[p for p in self.params_variational]+[p for p in self.params_scaling],
                         sparse=False, sampling=False, interpret=pp_function, output_shape=n_actions,
                         gradient=self.gradient, quantum_instance=self.quantum_instance, input_gradients=False)

        # Initialize the variational parameters to N(0.0, 0.1)
        # -> this works good for CartPole environments, but may not be the best choice for other problems
        initial_param_values_variational = torch.normal(mean=0.0, std=0.1, size=(self.n_params_variational,),
                                                        dtype=torch.float64)
        # Initialize the scaling parameters to all one's
        initial_param_values_scaling = torch.ones(self.n_params_scaling, dtype=torch.float64)
        # Combine both initializations, as qiskit_machine_learning is currently only able to handle one set of learnable
        # parameters
        initial_param_values = torch.cat((initial_param_values_variational, initial_param_values_scaling), dim=0)

        # Use the TorchConnector to create a hybrid model, that can be used as any PyToch model
        self.model = TorchConnector(qnn, initial_weights=initial_param_values)

    def _forward_pass_torch(self, state):
        """
        Compute expectation values of VQC with TorchConnector, parameters are handled by PyTorch

        :param state: environment state to encode
        :return: expectation values (= policy)
        """

        return self.model.forward((Tensor(state)))

    def select_action(self, state):
        """
        Extract expectation values by sampling the VQC;
        Postprocess this values to generate the policy (trivial, as RAW-VQC policy inherently defines a PDF);
        Draw the action to execute from the distribution;

        :param state: observed environment state
        :return: action, policy
        """

        # As we sample from the complete computational basis, no renormalization is necessary and the result can be
        # directly used as policy
        policy = self._forward_pass_torch(state)
        # Sample an action, based on the current policy
        action = np.random.choice(self.action_space, p=policy.detach().clone().numpy())
        return action, policy

    def get_model_weights(self):
        """
        Return trainable weights of the quantum model;

        :return: trainable weights
        """

        # (it would be desirable to return variational and scaling weights separately, but this is currently not
        #  supported by qiskit_machine_learning. Separating the parameter sets by hand implies that we no longer have
        #  a leaf tensor, which makes training via PyTorch very difficult.)
        return self.model.weight
