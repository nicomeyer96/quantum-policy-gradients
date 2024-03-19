import numpy as np

import torch


class Data:

    def __init__(self):
        """
        Data container for one trajectory
        """

        self.policy_values = []
        self.reward_values = []

    def append(self, policy_value, reward_value):
        """
        Append data from current timestep

        :param policy_value: Policy value of executed action
        :param reward_value: Received reward value
        """

        self.policy_values.append(policy_value)
        self.reward_values.append(reward_value)

    def get_policy_values(self):
        """
        :return: Policy values of trajectory
        """

        return self.policy_values

    def get_reward_values(self):
        """
        :return: Reward values of trajectory
        """

        return self.reward_values

    def reset(self):
        """
        Reset dataholder
        """

        self.policy_values = []
        self.reward_values = []


class BatchData:

    def __init__(self, batch_size, gamma):
        """
        Data container for one batch

        :param batch_size: Number of episodes in one batch
        :param gamma: Discount parameter
        """

        self.batch_policy_values = []
        self.batch_reward_values = []
        self.training_history = []

        self.gamma = gamma
        self.batch_counter = 0

    def extend(self, trajectory):
        """
        Add data from one episode to container

        :param trajectory: Data from current episode
        """

        self.batch_policy_values.extend(trajectory.get_policy_values())
        discounted_rewards = discount_and_cumulate_rewards(trajectory.get_reward_values(), self.gamma)
        self.batch_reward_values.extend(discounted_rewards)
        cumulated_reward = sum(trajectory.get_reward_values())
        self.training_history.append(cumulated_reward)
        self.batch_counter += 1
        return cumulated_reward

    def get_batch_counter(self):
        """
        :return: Current number of acquired trajectories
        """

        return self.batch_counter

    def get_batch_policy_values(self):
        """
        :return: Acquired policy values
        """

        return self.batch_policy_values

    def get_batch_reward_values(self):
        """
        :return: Acquired reward values
        """

        return self.batch_reward_values

    def get_mean_training_history(self):
        """
        :return: Averaged cumulated reward of current batch
        """

        return np.mean(self.training_history)

    def reset(self):
        """
        Reset dataholder
        """

        self.batch_policy_values = []
        self.batch_reward_values = []
        self.training_history = []
        self.batch_counter = 0


def discount_and_cumulate_rewards(rewards, gamma):
    """
    Computes the cumulated, discounted reward

    :param rewards: List of rewards of an episode
    :param gamma: Discount factor
    :return: List of cumulated, discounted rewards
    """

    discounted_cumulated_rewards = []
    running_cumulated_reward = 0
    for t in range(len(rewards)-1, -1, -1):
        running_cumulated_reward = rewards[t] + gamma * running_cumulated_reward
        discounted_cumulated_rewards.append(running_cumulated_reward)

    discounted_rewards = discounted_cumulated_rewards[::-1]

    return discounted_rewards


def cost(rewards, policies):
    """
    Cost function used to update parameters, i.e. avg[ \nabla ln pi (A_i | S_i) G_i ]

    :param rewards: List of cumulated (discounted) rewards -> G_i
    :param policies: List of policy values -> pi (A_i | S_i)
    :return: Loss, averaged over several batches / trajectories
    """

    akt_loss = 0
    log_policies = [torch.log(p) for p in policies]
    for i in range(len(policies)):
        akt_loss += log_policies[i] * rewards[i]
    akt_loss /= len(policies)

    # Return negative value, as we need to do gradient ASCENT
    return -akt_loss
