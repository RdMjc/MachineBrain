import torch
import numpy as np
import copy

from agents.buffers import ContinuousActionSpaceBuffer


class QNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim, learning_rate):
        super().__init__()
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._learning_rate = learning_rate

        # create layers
        self._fc_layer1 = torch.nn.Linear(self._state_dim + self._action_dim, 512)
        self._activation1 = torch.nn.ReLU()
        self._fc_layer2 = torch.nn.Linear(512, 256)
        self._activation2 = torch.nn.ReLU()
        self._fc_layer3 = torch.nn.Linear(256, 1)

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
        self._fc_layer1 = torch.nn.Linear(self._state_dim, 512)
        self._activation1 = torch.nn.ReLU()
        self._fc_layer2 = torch.nn.Linear(512, 256)
        self._activation2 = torch.nn.ReLU()
        self._fc_layer3 = torch.nn.Linear(256, self._action_dim)
        self._tanh_layer = torch.nn.Tanh()

        # create optimizer
        self.optimizer = torch.optim.SGD(params=self.parameters(), lr=self._learning_rate)

    def forward(self, state):
        x = self._fc_layer1(state)
        x = self._activation1(x)
        x = self._fc_layer2(x)
        x = self._activation2(x)
        x = self._fc_layer3(x)
        x = self._tanh_layer(x)
        return x


class TD3Agent():
    def __init__(self,
                 env,
                 gamma,
                 q_learning_rate,
                 policy_learning_rate,
                 batch_size,
                 buffer_size,
                 noise,
                 noise_decay_rate,
                 policy_delay,
                 policy_noise,
                 target_noise_clip,
                 polyak):
        """
        Implement Twin delayed Deep Deterministic Policy Gradient algorithm.
        https://spinningup.openai.com/en/latest/algorithms/td3.html

        :param env: The GYM environment of the problem
        :param gamma: Discount rate
        :param q_learning_rate: Learning rate of Q function
        :param policy_learning_rate: Learning rate of policy
        :param batch_size: Batch size to use in training phase
        :param buffer_size: The size of the experience replay buffer
        :param noise: The standard deviation of the noise which will be added to actions in training time for exploration
        :param noise_decay_rate: The rate which the noise will decay during training (It is applied after each run of train method)
        :param policy_delay: The amount of step required to update policy (per how many steps to update policy)
        :param policy_noise: The noise STD which is used to smooth target policy
        :param target_noise_clip: "c" in equations (in clipping)
        :param polyak: The parameters which determines the copying rate of online network weights to target network weights
        """
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self._env = env
        self._gamma = gamma

        self._q_learning_rate = q_learning_rate
        self._policy_learning_rate = policy_learning_rate

        self._batch_size = batch_size
        self._buffer_size = buffer_size

        self._noise = noise
        self._noise_decay_rate = noise_decay_rate
        self._policy_delay = policy_delay
        self._target_noise_clip = target_noise_clip
        self._policy_noise = policy_noise
        self._polyak = polyak

        # create buffer
        self._buffer = ContinuousActionSpaceBuffer(self._env.observation_space.shape,
                                                   self._env.action_space.shape,
                                                   self._buffer_size)

        # create Q network 1 and its target
        self._q_network1 = QNetwork(self._env.observation_space.shape[0],
                                   self._env.action_space.shape[0],
                                   learning_rate=self._q_learning_rate).to(self._device)
        self._q_network1_target = copy.deepcopy(self._q_network1).to(self._device)

        # create Q network 2 and its target
        self._q_network2 = QNetwork(self._env.observation_space.shape[0],
                                    self._env.action_space.shape[0],
                                    learning_rate=self._q_learning_rate).to(self._device)
        self._q_network2_target = copy.deepcopy(self._q_network2).to(self._device)

        # create policy network and its target
        self._policy_network = PolicyNetwork(self._env.observation_space.shape[0],
                                             self._env.action_space.shape[0],
                                             learning_rate=self._policy_learning_rate).to(self._device)
        self._policy_network_target = copy.deepcopy(self._policy_network).to(self._device)

        # initialize training counter
        self._train_counter = 0

    def choose_action(self, state, uniform_random=False, training=False):
        state = state[np.newaxis, :]
        state = torch.from_numpy(state.astype(np.float32))
        state = state.to(self._device)
        if uniform_random:
            #action = torch.normal(0.0, high/2, size=(self._env.action_space.shape[0], 1))
            action = np.random.normal(0, self._env.action_space.high/2, size=(self._env.action_space.shape[0]))
            return action
        # if training
        if training:
            # push action into policy network
            action = self._policy_network(state)[0]

            # add gaussian noise
            action = action + torch.normal(0, self._noise, size=action.shape).to(self._device)

            # clip final action into environment action low-high boundary
            low = torch.from_numpy(self._env.action_space.low).to(self._device)
            high = torch.from_numpy(self._env.action_space.high).to(self._device)
            action = torch.clip(action, min=low, max=high)

            return action.detach().cpu().numpy()
        else:
            # no need for gradient track
            with torch.no_grad():
                action = self._policy_network(state)[0]
            return action.cpu().numpy()

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

        # compute target actions
        with torch.no_grad():
            mu_ = self._policy_network_target(next_states)
            noise = torch.rand_like(actions).to(self._device) * self._policy_noise
            noise.clamp_(-self._target_noise_clip, self._target_noise_clip)
            actions_target = mu_ + noise
            low = torch.from_numpy(self._env.action_space.low).to(self._device)
            high = torch.from_numpy(self._env.action_space.high).to(self._device)
            actions_target.clamp_(low, high)

            ### compute targets
            # get Q1 values
            Q1_values = self._q_network1_target(next_states, actions_target)
            # get Q2 values
            Q2_values = self._q_network2_target(next_states, actions_target)

            # horizontally stack both values to later take their min in each row
            Q_values = torch.column_stack([Q1_values, Q2_values])

            # get minimum of two Q values
            Q_min, indices = torch.min(Q_values, dim=1, keepdim=True)

            # calculate y
            #y = rewards + self._gamma * (1 - dones) * Q_min
            y = rewards + self._gamma * torch.mul((1-dones), Q_min)

        # update Q1
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(y, self._q_network1(states, actions))
        self._q_network1.optimizer.zero_grad()
        loss.backward()
        # clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self._q_network1.parameters(), 1)
        self._q_network1.optimizer.step()

        # update Q2
        loss_fn2 = torch.nn.MSELoss()
        loss2 = loss_fn2(y, self._q_network2(states, actions))
        self._q_network2.optimizer.zero_grad()
        loss2.backward()
        # clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self._q_network2.parameters(), 1)
        self._q_network2.optimizer.step()

        # if time to update policy
        if self._train_counter % self._policy_delay == 0:
            meanval = -torch.mean(self._q_network1(states, self._policy_network(states)))
            self._policy_network.optimizer.zero_grad()
            meanval.backward()
            # clip gradients to prevent exploding gradients
            #torch.nn.utils.clip_grad_norm_(self._policy_network.parameters(), 1)
            self._policy_network.optimizer.step()

            # update target networks
            self._update_target_network_weights()

        self._train_counter += 1

    def store_transition(self, state, action, next_state, reward, done):
        self._buffer.store_transition(state, action, next_state, reward, done)

    def _update_target_network_weights(self):
        # copy Q1 weights
        for target_network_param, network_param in zip(self._q_network1_target.parameters(), self._q_network1.parameters()):
            target_network_param.data.copy_((1 - self._polyak) * network_param.data + self._polyak*target_network_param.data)

        # copy Q2 weights
        for target_network_param, network_param in zip(self._q_network2_target.parameters(), self._q_network2.parameters()):
            target_network_param.data.copy_((1 - self._polyak) * network_param.data + self._polyak*target_network_param.data)

        # copy policy weights
        for target_network_param, network_param in zip(self._policy_network_target.parameters(), self._policy_network.parameters()):
            target_network_param.data.copy_((1 - self._polyak) * network_param.data + self._polyak*target_network_param.data)
