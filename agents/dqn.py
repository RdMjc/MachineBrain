import torch
import numpy as np
import copy
import gym

from agents.buffers import DiscreteActionSpaceBuffer


class DQNAgent():
    def __init__(self, env, gamma, learning_rate, batch_size,
                 epsilon, epsilon_decay, epsilon_min, buffer_size,
                 target_network_update_per_step):
        """
        Implements Deep Q Network agent.
        :param env: Environment variable in OpenAI Gym format
        :param gamma: discount factor
        :param learning_rate: learning rate of optimizer
        :param batch_size:
        :param epsilon: random probability start of epsilon-greedy policy
        :param epsilon_decay: decaying factor of epsilon
        :param epsilon_min: minimum epsilon value in training phase
        :param buffer_size: size of the experience replay buffer
        :param target_network_update_per_step: the number of steps at which the target network is updated to
        have equal weights as actual neural network
        """
        self._env = env
        self._gamma = gamma
        self._learning_rate = learning_rate
        self._batch_size = batch_size

        self._epsilon = epsilon
        self._epsilon_decay = epsilon_decay
        self._epsilon_min = epsilon_min

        self._train_counter = 0

        self._buffer_size = buffer_size

        self._target_network_update_per_step = target_network_update_per_step

        # create buffer
        self._buffer = DiscreteActionSpaceBuffer(env.observation_space.shape,
                                                         env.action_space.n,
                                                         self._buffer_size)

        # create Q network
        self._q_network = torch.nn.Sequential(
            torch.nn.Linear(self._env.observation_space.shape[0], 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, self._env.action_space.n)
        )

        # create target Q network
        self._q_network_target = copy.deepcopy(self._q_network)

        # create optimizer
        self._optimizer = torch.optim.SGD(self._q_network.parameters(), lr=self._learning_rate)
        self._loss = torch.nn.MSELoss()

    def _update_target_network(self):
        self._q_network_target.load_state_dict(self._q_network.state_dict())

    def store_experience(self, state, action, new_state, reward, done):
        self._buffer.store_transition(state, action, new_state, reward, done)

    @torch.no_grad()
    def choose_action(self, state, training=False):
        state = state[np.newaxis, :]
        state = torch.from_numpy(state.astype(np.float32))
        if training:
            rand = np.random.random_sample()
            if rand < self._epsilon:
                action = np.random.choice(self._env.action_space.n)
                return action
            else:
                action = np.argmax(self._q_network(state))
                return action.detach().numpy()
        else:
            return np.argmax(self._q_network(state)).detach().numpy()

    def train(self):
        # if there is no enough samples, then don't learn
        if self._buffer.get_buffer_counter() < self._batch_size:
            return

        # sample from buffer
        states, actions, next_states, rewards, dones = self._buffer.sample(self._batch_size)

        # convert numpy arrays to tensors
        states = torch.from_numpy(states.astype(np.float32))
        actions = torch.from_numpy(actions.astype(np.int64))
        next_states = torch.from_numpy(next_states.astype(np.float32))
        rewards = torch.from_numpy(rewards.astype(np.float32))
        dones = torch.from_numpy(dones.astype(np.int8))

        # get Q(s) and Q(s')
        q_states = self._q_network(states)
        q_next_states = self._q_network_target(next_states)

        # calculate y = r + gamma * max_over_a (Q(s'))
        max_over_a_q_next_states, _ = torch.max(q_next_states, dim=1, keepdim=True)
        y = rewards + self._gamma * torch.mul(max_over_a_q_next_states, (1 - dones))

        q_a = q_states.gather(1, actions)

        # apply gradient descent
        loss = self._loss(y, q_a)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        # update epsilon
        self._epsilon *= self._epsilon_decay
        self._epsilon = max(self._epsilon, self._epsilon_min)

        # update target network
        if self._train_counter > 0 and self._train_counter % self._target_network_update_per_step == 0:
            self._update_target_network()

        self._train_counter += 1

    def get_epsilon(self):
        return self._epsilon

if __name__ == "__main__":
    env = gym.make("LunarLander-v2")

    dqn_agent = DQNAgent(env, gamma=0.99, learning_rate=0.001, batch_size=64,
                         epsilon=0.90, epsilon_decay=0.99, epsilon_min=0.01, buffer_size=100000,
                         target_network_update_per_step=100)

    episodes = 10000
    for episode in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0
        total_steps = 0
        while not done:
            action = dqn_agent.choose_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            dqn_agent.store_experience(state, action, next_state, reward, done)
            state = next_state

            total_steps += 1
            dqn_agent.train()

        print(f"Episode {episode}, total reward {total_reward}, total steps {total_steps}")


