import numpy as np
import gym
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from policy import RawPolicy
from utils import parse_args, construct_exp_name, stateprep, Data, BatchData, cost


def policy_gradient(env, model, opt, writer, args):
    """
    Quantum Policy Gradient Algorithm

    :param env: Environment handle
    :param model: Hybrid model handle
    :param opt: Optimizer handle
    :param writer: Tensorboard handle
    :param args: Algorithmic hyperparameters
    """

    # instantiate batch data holder
    batch_data = BatchData(batch_size=args.batch_size, gamma=args.gamma)
    # data holder for one episode
    data = Data()

    start_time = time.time()
    for episode in range(args.episodes):

        done = False
        raw_state = env.reset()[0]

        data.reset()

        step_counter = 0
        while not done:

            state = stateprep(raw_state, env_name=args.environment)

            # Sample action from current policy
            action, policy = model.select_action(state)

            # Execute action
            raw_state, reward, done, _, _ = env.step(action)

            # Store relevant information
            data.append(policy[action], reward)

            # This is necessary, as the current Gym version of CartPole unfortunately did remove the 'max_episode_steps'
            # flag, which defines the finite horizon of the environment
            step_counter += 1
            if args.environment == 'CartPole-v0' and 200 == step_counter:
                done = True
            if args.environment == 'CartPole-v1' and 500 == step_counter:
                done = True

        # Cumulated reward of the current episode
        cumulated_reward = batch_data.extend(data)
        writer.add_scalar('Reward/Episode', cumulated_reward, episode + 1)

        # Update, one the desired number of trajectories has been acquired
        if args.batch_size == batch_data.get_batch_counter():

            def closure():
                opt.zero_grad()
                loss = cost(batch_data.get_batch_reward_values(), batch_data.get_batch_policy_values())
                loss.backward()
                return loss

            # Perform optimization
            opt.step(closure)

            # Averaged cumulated reward of current batch
            mean_reward_batch = batch_data.get_mean_training_history()

            # Print results to console
            print("TRAIN Ep: {}   |   Average of last {}: {:.2f}   |   --- {:.2f} seconds ---".format(
                episode + 1, args.batch_size, mean_reward_batch, time.time() - start_time))

            # Write results to tensorboard
            writer.add_scalar('Reward/Batch', mean_reward_batch, (episode/args.batch_size)+1)
            writer.add_scalar('Time/Batch', time.time() - start_time, (episode / args.batch_size)+1)

            # Empty dataholder
            batch_data.reset()

            start_time = time.time()


def run():

    overall_time = time.time()

    args = parse_args()

    # get name of folder for results
    exp_name = construct_exp_name(args)
    print("Saving training history to ./results/{}".format(exp_name))

    writer = SummaryWriter("results/{}".format(exp_name))

    env = gym.make(args.environment)
    # preset for CartPole environment
    n_qubits = 4
    n_actions = 2

    # Construct RAW-VQC policy model
    model = RawPolicy(n_qubits=n_qubits, n_actions=n_actions, depth=args.depth, args=args)\

    # Get trainable weights and prepare for optimization. Unfortunately, qiskit_machine_learning currently does not
    # allow to define multiple parameter sets. Consequently, one has to go with the same learning rate for all parameter
    # sets, if the TorchConnector should be used.
    model_weights = model.get_model_weights()
    opt = torch.optim.Adam([model_weights], lr=args.learning_rate, amsgrad=True)

    # Run the QPG algorithm
    policy_gradient(env, model, opt, writer, args)

    writer.add_scalar('Time/Overall', time.time() - overall_time, 0)
    writer.close()

    # store trained parameters
    with open("results/{}/trained_weights.npy".format(exp_name), 'wb') as ff:
        np.save(ff, model.get_model_weights().detach().clone().numpy())
    print("Saving trained weights to ./results/{}/trained_weights.npy".format(exp_name))


if __name__ == '__main__':
    run()
