import argparse


def parse_args():
    """
    Parse input arguments

    :return: Parsed arguments
    """

    _environment_choices = ['CartPole-v0', 'CartPole-v1']
    _pp_function_choices = ['glob', 'loc']

    parser = argparse.ArgumentParser(description='Quantum Policy Gradient Algorithm')

    parser.add_argument('--environment', '-env', default='CartPole-v0', type=str, choices=_environment_choices,
                        help='Environment to train on'
                             + ' | '.join(_environment_choices)
                             + ' (default: CartPole-v0)')

    parser.add_argument('--episodes', '-ep', default=1000, type=int, metavar='N',
                        help='Number of episodes to train for (default: 1000)')

    parser.add_argument('--batch_size', '-bs', default=10, type=int, metavar='N',
                        help='Number of trajectories to compute for each update (default: 10)')

    parser.add_argument('--gamma', '-gamma', default=0.99, type=float, metavar='F',
                        help='Gamma value to use for discounted rewards (default: 0.99)')

    parser.add_argument('--learning_rate', '-lr', default=0.02, type=float, metavar='F',
                        help='Learning rate (default: 0.02)')

    parser.add_argument('--depth', '-d', default=1, type=int, metavar='N',
                        help='VQC circuit depth (default: 1)')

    parser.add_argument('--pp_function', '-ppf', default='glob', type=str,
                        help='Post-processing function to use:'
                             + ' | '.join(_pp_function_choices)
                             + ' (default: glob)')

    parser.add_argument('--q_local', '-ql', default=0, type=int, metavar='N',
                        help='Use q-local post-processing function with the selected number of qubits. '
                             'This only works for two actions. (The value `0` indicates, that `--pp_function` '
                             'is used`.) (default: 0)')

    parser.add_argument('--exp_name', '-name', default='0', type=str,
                        help='Folder postfix for saving experimental results (default: 0)')

    args = parser.parse_args()

    return args


def construct_exp_name(args):

    exp_name = str(args.environment) + '_eps=' + str(args.episodes) + '_bs=' + str(args.batch_size) \
               + '_gamma=' + str(args.gamma) + '_lr=' + str(args.learning_rate) + '_d=' + str(args.depth)
    if 0 == args.q_local:
        exp_name += '_pp=' + str(args.pp_function)
    else:
        exp_name += '_pp=loc' + str(args.q_local)
    exp_name += '_' + str(args.exp_name)

    return exp_name
