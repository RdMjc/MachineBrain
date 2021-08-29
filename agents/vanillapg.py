import torch
import numpy as np
from torch.distributions import Categorical


class VanillaPGAgent():
    def __init__(self, env, gamma, policy_learning_rate, critic_learning_rate):
        """
        Implements the vanilla policy gradient algorithm with an advantage function.
        :param env: The environment in OpenAI Gym form
        :param gamma: The discount rate
        :param policy_learning_rate: Policy learning rate of the optimizer
        :param critic_learning_rate: Critic (Value function) learning rate of the optimizer
        """
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self._env = env
        self._gamma = gamma
        self._policy_learning_rate = policy_learning_rate
        self._critic_learning_rate = critic_learning_rate

        # create policy network
        self._network_policy = torch.nn.Sequential(torch.nn.Linear(self._env.observation_space.shape[0], 256),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Linear(256, 256),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Linear(256, self._env.action_space.n),
                                                    torch.nn.Softmax()
                                                    ).to(self._device)

        #self._optimizer_policy = torch.optim.SGD(self._network_policy.parameters(), lr=self._policy_learning_rate)
        self._optimizer_policy = torch.optim.Adam(self._network_policy.parameters(), lr=self._policy_learning_rate)

        self._network_critic = torch.nn.Sequential(torch.nn.Linear(self._env.observation_space.shape[0], 256),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Linear(256, 256),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Linear(256, 1)
                                                    ).to(self._device)

        #self._optimizer_critic = torch.optim.SGD(self._network_critic.parameters(), lr=self._critic_learning_rate)
        self._optimizer_critic = torch.optim.Adam(self._network_critic.parameters(), lr=self._critic_learning_rate)

        # create buffer
        self._buffer_states = []
        self._buffer_actions = []
        self._buffer_rewards = []

    def store_transition(self, state, action, next_state, reward, done):
        self._buffer_states.append(state)
        self._buffer_actions.append(action)
        self._buffer_rewards.append(reward)

    def choose_action(self, state):
        state = state[np.newaxis, :]
        state = torch.from_numpy(state.astype(np.float32))
        state = state.to(self._device)

        # get logits
        logits = self._network_policy(state)[0]
        logits.clamp_(min=0.000001, max=1 - 0.0000001)

        # create probability distribution from logits
        probabilities = Categorical(logits)

        # return a sample action from the distribution
        return probabilities.sample().cpu().detach().item()

    def train(self):
        states = self._buffer_states
        actions = self._buffer_actions
        rewards = self._buffer_rewards

        states = torch.tensor(states, dtype=torch.float32).to(self._device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self._device)

        # get G(t) functions which is discount sum of rewards from time t to end
        G = []
        reward_sum = 0
        self._buffer_rewards.reverse()
        for reward in self._buffer_rewards:
            reward_sum = reward + self._gamma * reward_sum
            G.append(reward_sum)
        G.reverse()

        # convert rewards to tensors
        rewards = torch.tensor(G, dtype=torch.float32).to(self._device)

        # calculate advantages
        advantages = reward - self._network_critic(states)

        # calculate "policy loss"
        logits = self._network_policy(states)
        probabilities = Categorical(logits)
        log_probs = probabilities.log_prob(actions)

        policy_loss = -torch.sum(log_probs * advantages)

        # apply gradient ascent
        self._optimizer_policy.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._network_policy.parameters(), 1)
        self._optimizer_policy.step()

        # calculate critic loss
        critic_loss = torch.nn.MSELoss()(self._network_critic(states), rewards)

        # apply gradient descent
        self._optimizer_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._network_critic.parameters(), 1)
        self._optimizer_critic.step()

        # clear the buffer
        self._buffer_states.clear()
        self._buffer_actions.clear()
        self._buffer_rewards.clear()