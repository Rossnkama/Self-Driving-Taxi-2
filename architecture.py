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
    def __init__(self, input_nodes, output_nodes):
        """ Defines the structure of our neural network on initialisation.

            :param input_nodes: The number of sensors on the car
            :param output_nodes: The number of actions which can be taken

            :type input_nodes: int
            :type output_nodes: int
        """
        super(NeuralNetworkArchitecture, self).__init__()

        # Initialising the new variables attached to the NNA object
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes

        self.full_connection_1 = nn.Linear(input_nodes, 30)
        self.full_connection_2 = nn.Linear(30, output_nodes)
