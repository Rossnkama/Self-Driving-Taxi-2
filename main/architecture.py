import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable


class NeuralNetworkArchitecture(nn.Module):

    def __init__(self, input_nodes, num_actions):
        """ Defines the structure of our neural network on initialisation.

            :param input_nodes: The number of sensors on the car
            :param num_actions: The number of actions which can be taken

            :type input_nodes: int
            :type num_actions: int
        """
        super(NeuralNetworkArchitecture, self).__init__()

        # Initialising the new variables attached to the NNA object
        self.input_nodes = input_nodes
        self.num_actions = num_actions

        self.full_connection_1 = nn.Linear(input_nodes, 30)
        self.full_connection_2 = nn.Linear(30, num_actions)

    def forward(self, sensor_data):
        """ Gives us a list of Q-values from the current state of the car

            :param sensor_data: The current state of the vehicle as data from sensors
            :type sensor_data: list

            :return: The expected values of each possible action
            :rtype: list
        """
        hidden_node_values = f.relu(self.full_connection_1(sensor_data))
        q_values = self.full_connection_2(hidden_node_values)
        return q_values
