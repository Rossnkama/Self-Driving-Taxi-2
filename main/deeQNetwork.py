# Optimiser to perform stochastic gradient decent
import torch
import torch.optim as optim
from architecture import NeuralNetworkArchitecture
from prioritised_replay import DQNPrioritisedReplay
import torch.nn.functional as f


class DeepQNetwork(object):
    """ DeepQNetwork
        ------------
        - This is the deep Q-network that allows Q-learning for Q-learning
          to take place.
    """
    def __init__(self, input_neurons, num_actions, gamma):
        """
            :param input_neurons: Number of inputs our neural network takes
            :param num_actions: Number of actions the network returns
        """

        # Discount factor
        self.gamma = gamma

        # Let's us plot a graph of mean rewards
        self.rewards_window = []

        # Deep Q Network model

