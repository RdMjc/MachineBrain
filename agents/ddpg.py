import torch
import numpy as np
import copy

from .buffers import ContinuousActionSpaceBuffer


class QNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim, learning_rate):
        super().__init__()
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._learning_rate = learning_rate

        # create layers
        self._fc_layer1 = torch.nn.Linear(self._state_dim + self._action_dim, 128)
        self._activation1 = torch.nn.ReLU()
        self._fc_layer2 = torch.nn.Linear(128, 128)
        self._activation2 = torch.nn.ReLU()
        self._fc_layer3 = torch.nn.Linear(128, 1)

        # create optimizer
        self.optimizer = torch.optim.SGD(params=self.parameters(), lr=self._learning_rate)

    def forward(self, state, action):
        x = torch.cat((state, action), 1)
        x = self._fc_layer1(x)
        x = self._activation1(x)
        x = self._fc_layer2(x)
        x = self._activation2(x)
        x = self._fc_layer3(x)
        return x


class PolicyNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim, learning_rate):
        super().__init__()

        self._state_dim = state_dim
        self._action_dim = action_dim
        self._learning_rate = learning_rate

        # create layers
        self._fc_layer1 = torch.nn.Linear(self._state_dim, 128)
        self._activation1 = torch.nn.ReLU()
        self._fc_layer2 = torch.nn.Linear(128, 128)
        self._activation2 = torch.nn.ReLU()
        self._fc_layer3 = torch.nn.Linear(128, self._action_dim)

        # create optimizer
        self.optimizer = torch.optim.SGD(params=self.parameters(), lr=self._learning_rate)

    def forward(self, state):
        x = self._fc_layer1(state)
        x = self._activation1(x)
        x = self._fc_layer2(x)
        x = self._activation2(x)
        x = self._fc_layer3(x)
        return x


class DDPGAgent():
    def __init__(self,
                 env,
                 q_learning_rate,
                 policy_learning_rate,
                 batch_size,
                 buffer_size,
                 noise,
                 polyak):
        """
        Implements Deep Deterministic Policy Gradient Algorithm.
        "https://spinningup.openai.com/en/latest/algorithms/ddpg.html"

        Lillicrap, Timothy P., et al.
        "Continuous control with deep reinforcement learning."
        arXiv preprint arXiv:1509.02971 (2015).

        :param env: The GYM environment of the problem
        :param q_learning_rate: Learning rate of Q function
        :param policy_learning_rate: Learning rate of policy
        :param batch_size: Batch size to use in training phase
        :param buffer_size: The size of the experience replay buffer
        :param noise: The standard deviation of the noise which will be added to actions in training time for exploration
        :param polyak: The parameters which determines the copying rate of online network weights to target network weights
        """

        self._env = env

        self._q_learning_rate = q_learning_rate
        self._policy_learning_rate = policy_learning_rate

        self._batch_size = batch_size
        self._buffer_size = buffer_size

        self._noise = noise
        self._polyak = polyak

        self._q_network = QNetwork(self._env.observation_space.shape[0], self._env.action_space.shape[0])
        self._q_network_target = copy.deepcopy(self._q_network)