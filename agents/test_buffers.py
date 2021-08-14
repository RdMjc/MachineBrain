import buffers
import unittest
import gym


class TestBuffers(unittest.TestCase):
    def setUp(self) -> None:
        # create two environments
        # 1 - discrete action space
        # 2 - continuous action space
        self.env_discrete_action_space = gym.make("AirRaid-v0")
        self.env_discrete_action_space.reset()

        self.env_continuous_action_space = gym.make("LunarLanderContinuous-v2")
        self.env_continuous_action_space.reset()

    def test_discrete_buffer(self):
        buffer = buffers.DiscreteActionSpaceBuffer(self.env_discrete_action_space.observation_space.shape, self.env_discrete_action_space.action_space.shape, 100)

        for i in range(150):
            state = self.env_discrete_action_space.observation_space.sample()
            action = self.env_discrete_action_space.action_space.sample()
            next_state = self.env_discrete_action_space.observation_space.sample()
            reward = i
            done = False
            buffer.store_transition(state, action, next_state, reward, done)

        states, actions, next_states, rewards, dones = buffer.sample(10)
        assert (states[0].shape == self.env_discrete_action_space.observation_space.shape)
        assert (states[1].shape == self.env_discrete_action_space.observation_space.shape)
        assert (states[9].shape == self.env_discrete_action_space.observation_space.shape)

if __name__ == "__main__":
    unittest.main()