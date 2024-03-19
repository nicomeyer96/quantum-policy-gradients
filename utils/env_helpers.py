import numpy as np


def stateprep(raw_state, env_name):
    """
    Prepare state observed from environment in a way, that it can be encoded into the VQC

    :param raw_state: State observed from environment
    :param env_name: Used environment
    :return: Prepared state that can be encoded into the VQC
    """

    if env_name in ['CartPole-v0', 'CartPole-v1']:
        # Bring all values (close) to range [-1, 1]
        state = np.copy(raw_state)
        state[0] /= 2.4
        state[1] /= 2.5
        state[2] /= 0.21
        state[3] /= 2.5
        return state
    else:
        raise NotImplementedError
