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
        self._fc_layer1 = torch.nn.Linear(self._state_dim + self._action_dim, 256)
        self._activation1 = torch.nn.ReLU()
        self._fc_layer2 = torch.nn.Linear(256, 128)
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
        self._fc_layer1 = torch.nn.Linear(self._state_dim, 256)
        self._activation1 = torch.nn.ReLU()
        self._fc_layer2 = torch.nn.Linear(256, 128)
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
                 gamma,
                 q_learning_rate,
                 policy_learning_rate,
                 batch_size,
                 buffer_size,
                 noise,
                 noise_decay_rate,
                 polyak):
        """
        Implements Deep Deterministic Policy Gradient Algorithm.
        "https://spinningup.openai.com/en/latest/algorithms/ddpg.html"

        Lillicrap, Timothy P., et al.
        "Continuous control with deep reinforcement learning."
        arXiv preprint arXiv:1509.02971 (2015).

        :param env: The GYM environment of the problem
        :param gamma: Discount rate
        :param q_learning_rate: Learning rate of Q function
        :param policy_learning_rate: Learning rate of policy
        :param batch_size: Batch size to use in training phase
        :param buffer_size: The size of the experience replay buffer
        :param noise: The standard deviation of the noise which will be added to actions in training time for exploration
        :param noise_decay_rate: The rate which the noise will decay during training (It is applied after each run of train method)
        :param polyak: The parameters which determines the copying rate of online network weights to target network weights
        """
        # set device GPU/CPU
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self._env = env
        self._gamma = gamma

        self._q_learning_rate = q_learning_rate
        self._policy_learning_rate = policy_learning_rate

        self._batch_size = batch_size
        self._buffer_size = buffer_size

        self._noise = noise
        self._noise_decay_rate = noise_decay_rate
        self._polyak = polyak

        # create buffer
        self._buffer = ContinuousActionSpaceBuffer(self._env.observation_space.shape,
                                                   self._env.action_space.shape,
                                                   self._buffer_size)

        # create Q network and its target
        self._q_network = QNetwork(self._env.observation_space.shape[0],
                                   self._env.action_space.shape[0],
                                   learning_rate=self._q_learning_rate).to(self._device)
        self._q_network_target = copy.deepcopy(self._q_network).to(self._device)

        # create policy network and its target
        self._policy_network = PolicyNetwork(self._env.observation_space.shape[0],
                                             self._env.action_space.shape[0],
                                             learning_rate=self._policy_learning_rate).to(self._device)
        self._policy_network_target = copy.deepcopy(self._policy_network).to(self._device)

        # initialize training counter
        self._train_counter = 0

    def choose_action(self, state, training=False):
        state = state[np.newaxis, :]
        state = torch.from_numpy(state.astype(np.float32))
        state = state.to(self._device)
        # if training
        if training:
            # push action into policy network
            action = self._policy_network(state)[0]

            # add gaussian noise
            print(action)
            print(self._noise)
            print(torch.normal(0, self._noise, size=action.shape))
            action = action + torch.normal(0, self._noise, size=action.shape).to(self._device)

            # clip final action into environment action low-high boundary
            low = torch.from_numpy(self._env.action_space.low).to(self._device)
            high = torch.from_numpy(self._env.action_space.high).to(self._device)
            action = torch.clip(action, min=low, max=high)

            return action.detach().numpy()
        else:
            # no need for gradient track
            with torch.no_grad():
                action = self._policy_network(state)[0]
            return action.numpy()

    def train(self):
        # check if there as enough samples in the buffer
        if self._buffer.get_buffer_counter() < self._batch_size:
            return

        # sample a batch from buffer
        states, actions, next_states, rewards, dones = self._buffer.sample(self._batch_size)

        # convert to tensors
        states = torch.from_numpy(states.astype(np.float32)).to(self._device)
        actions = torch.from_numpy(actions.astype(np.float32)).to(self._device)
        next_states = torch.from_numpy(next_states.astype(np.float32)).to(self._device)
        rewards = torch.from_numpy(rewards.astype(np.float32)).to(self._device)
        dones = torch.from_numpy(dones.astype(np.int8)).to(self._device)

        # compute targets
        mu_next_states = self._policy_network_target(next_states)
        y = rewards + self._gamma * (1 - dones) * (self._q_network_target(next_states, mu_next_states))

        # update Q function
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(y, self._q_network(states, actions))
        self._q_network.optimizer.zero_grad()
        loss.backward()
        # clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self._q_network.parameters(), 1)
        self._q_network.optimizer.step()



        # update policy
        meanval = -torch.mean(self._q_network(states, self._policy_network(states)))
        self._policy_network.optimizer.zero_grad()
        meanval.backward()
        # clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self._policy_network.parameters(), 1)
        self._policy_network.optimizer.step()

        # update target network weights
        self._update_target_network_weights()

        # update noise
        self._noise *= self._noise_decay_rate

    def store_transition(self, state, action, next_state, reward, done):
        self._buffer.store_transition(state, action, next_state, reward, done)

    def _update_target_network_weights(self):
        # copy Q weights
        for target_network_param, network_param in zip(self._q_network_target.parameters(), self._q_network.parameters()):
            target_network_param.data.copy_((1 - self._polyak) * network_param.data + self._polyak*target_network_param.data)

        # copy policy weights
        for target_network_param, network_param in zip(self._policy_network_target.parameters(), self._policy_network.parameters()):
            target_network_param.data.copy_((1 - self._polyak) * network_param.data + self._polyak*target_network_param.data)
