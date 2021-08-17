import torch
import numpy as np

from agents.dqn import DQNAgent


class DoubleDQNAgent(DQNAgent):
    """
    Implements the algorithm provided in the paper:
        Deep Reinforcement Learning with Double Q-learning,
        Hado van Hasselt and Arthur Guez and David Silver,
        2015.

    Inherits from DQNAgent since only the calculation of the target (y) is different from DQN algorithm.
    """
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

        ### get Q(s) and Q(s')
        q_states = self._q_network(states)
        q_next_states = self._q_network_target(next_states)

        # get argmax(over a) Q(s', a)
        # use online network for action selection contrary to DQN
        next_state_actions = torch.argmax(self._q_network(next_states), dim=1, keepdim=True)

        # calculate y = r + gamma * Q(s', next_state_actions)
        y = rewards + self._gamma * torch.mul(q_next_states.gather(1, next_state_actions), (1 - dones))

        # get estimates
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
