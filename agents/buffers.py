import numpy as np


class BaseBuffer():
    def __init__(self, state_space, action_space, size):
        """
        An experience replay buffer which stores experiences.
        Other buffer types (which are DiscreteAction and ContinuousAction buffers)
        inherit from this BaseBuffer and just change the definition of _buffer_action
        :param state_space: Tuple that specifies the shape of the observations (states)
        :param action_space: Tuple or scalar that specifies action space
        :param size: Size of the buffer
        """
        self._state_space = state_space
        self._action_space = action_space
        self._size = size

        #initialize buffer counter
        self._buffer_counter = 0

        # initialize buffer
        self._init_buffer()

    def store_transition(self, state, action, next_state, reward, done):
        """
        Stores an MDP transition in the buffer
        :param state:
        :param action:
        :param next_state:
        :param reward:
        :param done:
        :return:
        """
        index = self._buffer_counter % self._size

        self._buffer_state[index] = state
        self._buffer_action[index] = action
        self._buffer_next_state[index] = next_state
        self._buffer_reward[index] = reward
        self._buffer_done[index] = int(done)

        # increase counter
        self._buffer_counter += 1

    def sample(self, batch_size):
        """
        Samples batch_size amount of sample from the buffer and returns as a tuple of
        states, actions, next_states, rewards, dones
        :param batch_size:
        :return:
        """
        if self._buffer_counter < batch_size:
            raise(ValueError("Not enough samples in the buffer"))

        # get min of buffer counter and capacity since buffer counter goes unlimited
        limit = min(self._buffer_counter, self._size)
        batch_indices = np.random.choice(limit, batch_size)
        return self._buffer_state[batch_indices], \
               self._buffer_action[batch_indices], \
               self._buffer_next_state[batch_indices], \
               self._buffer_reward[batch_indices], \
               self._buffer_done[batch_indices]

    # def get_buffer(self):
    #     """
    #     Returns entire buffer
    #     :return:
    #     """
    #     #return every sample in buffer
    #     return self._buffer_state[:], \
    #            self._buffer_action[:], \
    #            self._buffer_next_state[:], \
    #            self._buffer_reward[:], \
    #            self._buffer_done[:]

    def clear_buffer(self):
        # setting counter to 0 is equivalent to clearing
        self._buffer_counter = 0

    def get_buffer_counter(self):
        return self._buffer_counter


class DiscreteActionSpaceBuffer(BaseBuffer):
    """
    Discrete action space buffer
    Inherits from BaseBuffer
    Just changes how _buffer_state is initialized
    """
    def _init_buffer(self):
        # create buffer variables
        self._buffer_state = np.zeros(shape=(self._size, *(self._state_space)))
        self._buffer_action = np.zeros(shape=(self._size, 1))
        self._buffer_next_state = np.zeros(shape=(self._size, *(self._state_space)))
        self._buffer_reward = np.zeros(shape=(self._size, 1))
        self._buffer_done = np.zeros(shape=(self._size, 1))


class ContinuousActionSpaceBuffer(BaseBuffer):
    """
    Continuous action space buffer
    Inherits from BaseBuffer
    Just changes how _buffer_state is initialized
    """
    def _init_buffer(self):
        # create buffer variables
        self._buffer_state = np.zeros(shape=(self._size, *(self._state_space)))
        self._buffer_action = np.zeros(shape=(self._size, *(self._action_space)))
        self._buffer_next_state = np.zeros(shape=(self._size, *(self._state_space)))
        self._buffer_reward = np.zeros(shape=(self._size, 1))
        self._buffer_done = np.zeros(shape=(self._size, 1))
