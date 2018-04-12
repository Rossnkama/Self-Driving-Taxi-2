# Optimiser to perform stochastic gradient decent
import torch
import torch.optim as optim
from architecture import NeuralNetworkArchitecture
from prioritised_replay import DQNPrioritisedReplay
import torch.nn.functional as f


def get_loss(real, expected):
    """ Get's us the TD-Error

        :param real: Our real value experienced
        :param expected: Our network's hypothesised value
        :return: The TD-Error (In this case)
    """
    return f.smooth_l1_loss(real, expected)


class DeepQNetwork(object):

    def __init__(self, input_size, output_size):
        """ The Q-Learning of our ANN

            :param input_size: How many input's will the ANN take
            :param output_size: How many outputs will the ANN give
        """
        self.gamma = gamma
        self.reward_window = []
        self.model = NeuralNetworkArchitecture(input_size, output_size)
        self.memory = DQNPrioritisedReplay(10000)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        """ Where the back propagation happens

            :param batch_state: Batches of states ordered by temporal magnitude to one another
            :type batch_state: LongTensor

            :param batch_next_state: Batches of next states ordered by temporal magnitude to one another
            :type batch_next_state: LongTensor

            :param batch_reward: Batches of rewards ordered by temporal magnitude to one another
            :type batch_reward: LongTensor

            :param batch_action: Batches of actions ordered by temporal magnitude to one another
            :type batch_action: LongTensor
        """
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma * next_outputs + batch_reward
        td_loss = get_loss(outputs, target)
        td_loss.backward(retain_variables=True)
        self.optimiser.step()

