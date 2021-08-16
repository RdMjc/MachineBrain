"""
Contains example uses of provided RL agents.
"""
import gym
import agents.dqn
import agents.doubledqn
import agents.ddpg

def DQNexample():
    env = gym.make("LunarLander-v2")

    dqn_agent = agents.dqn.DQNAgent(env, gamma=0.99, learning_rate=0.001, batch_size=64,
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



def DoubleDQNexample():
    env = gym.make("LunarLander-v2")

    dqn_agent = agents.doubledqn.DoubleDQNAgent(env, gamma=0.99, learning_rate=0.001, batch_size=64,
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


def DDPGexample():
    env = gym.make("LunarLanderContinuous-v2")

    agent = agents.ddpg.DDPGAgent(env=env, gamma=0.99, q_learning_rate=0.001, policy_learning_rate=0.0001,
                                  batch_size=64, buffer_size=100000, noise=0.1, polyak=0.95)

    episodes = 10000
    for episode in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0
        total_steps = 0
        while not done:
            action = agent.choose_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            agent.store_transition(state, action, next_state, reward, done)
            state = next_state

            total_steps += 1
            agent.train()

        print(f"Episode {episode}, total reward {total_reward}, total steps {total_steps}")




if __name__ == "__main__":
    DDPGexample()