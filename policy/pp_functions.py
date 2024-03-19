import numpy as np


class PPFunction:
    """
    Defines different post-processing functions, that were introduced in
    "Quantum Policy Gradient Algorithm with Optimized Action Decoding", Meyer et al.
    """

    def __init__(self, n_qubits, n_actions):
        """
        Initialize Post-Processing Function class

        :param n_qubits: number of qubits
        :param n_actions: number of actions
        """

        # Number of bits necessary to encode n_actions
        log_n_actions = np.log2(n_actions)

        if log_n_actions < 1.0 or not (np.floor(log_n_actions) == np.ceil(log_n_actions)):
            raise NotImplementedError('Number of actions needs to be a power of two!')

        if log_n_actions > n_qubits:
            raise ValueError('Number of actions exceeds number of basis states!')

        self.n_qubits = n_qubits
        self.n_actions = n_actions

        # As the previous tests went successfully, log_n_actions is guaranteed to be a natural number
        self.log_n_actions = int(log_n_actions)

        # Necessary for definition of 'optimal' (global) post-processing function
        # (see "Quantum Policy Gradient Algorithm with Optimized Action Decoding", Meyer et al.)
        self.m = self.log_n_actions - 1

        # This indicator is set, whenever a post-processing function (for two actions) should be defined on the tensored
        # measurement of a subset of the qubits
        self.q = None

    def _pp_glob(self, b_dec):
        """
        Global (i.e. optimal w.r.t. globality measure) post-processing function for self.n_action actions

        :param b_dec: decimal representation of observed bitstring
        :return: action associated with observed bitstring
        """

        b_bin = '{:b}'.format(b_dec).zfill(self.n_qubits)
        initial_bits = b_bin[:self.m]
        parity_bits = b_bin[self.m:]
        action_label = initial_bits + str(parity_bits.count('1') % 2)
        return int(action_label, 2)

    def _pp_loc(self, b_dec):
        """
        Local (i.e. worst w.r.t. globality measure) post-processing function for self.n_action actions

        :param b_dec: decimal representation of observed bitstring
        :return: action associated with observed bitstring
        """

        b_bin = '{:b}'.format(b_dec).zfill(self.n_qubits)
        action_label = b_bin[:self.log_n_actions]
        return int(action_label, 2)

    def _pp_q_loc(self, b_dec):
        """
        Q-Local post-processing function for 2 actions

        :param b_dec: decimal representation of observed bitstring
        :return: action associated with observed bitstring
        """

        b_bin = '{:b}'.format(b_dec).zfill(self.n_qubits)
        action_label = b_bin[:self.q]
        return action_label.count('1') % 2

    def select_pp_function(self, pp_function, q_local):
        """
        Select, construct, and return post-processing function selected by user

        :param pp_function: Type of post-processing function to use
        :param q_local: Whether q-local post-processing function should be used, and on how many qubits
        :return: post-processing function
        """

        if 0 == q_local:
            if 'glob' == pp_function:
                return self._pp_glob
            elif 'loc' == pp_function:
                return self._pp_loc
            else:
                raise NotImplementedError('Post-processing function `{}` not implemented!'.format(pp_function))
        else:
            if not 2 == self.n_actions:
                raise ValueError('The q_local post-processing functions are only available for two action environments!')
            if q_local < 1 or q_local > self.n_qubits:
                raise ValueError('`{}` is not a valid choice for a q-local observable!'.format(q_local))
            if 1 == q_local:
                print('Note: --q_local=1 is equivalent to using --pp_function=loc')
            if self.n_qubits == q_local:
                print('Note: --q_local={} is equivalent to using --pp_function=glob'.format(q_local))
            self.q = q_local
            return self._pp_q_loc
