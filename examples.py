"""
Contains example uses of provided RL agents.
"""
import gym
import agents.dqn
import agents.doubledqn
import agents.ddpg
import agents.td3


def evaluate(env, agent, simulate=False):
    done = False
    state = env.reset()
    total_reward = 0
    while not done:
        action = agent.choose_action(state, training=False)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        state = next_state
        if simulate:
            env.render()

    print(f"### Evaluation Reward {total_reward}")

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
    #env = gym.make("LunarLanderContinuous-v2")
    env = gym.make("BipedalWalker-v3")
    #env = gym.make("Pendulum-v0")

    agent = agents.ddpg.DDPGAgent(env=env, gamma=0.99, q_learning_rate=0.001, policy_learning_rate=0.0001,
                                  batch_size=256, buffer_size=200000, noise=0.1, noise_decay_rate=0.99,
                                  polyak=0.999)

    episodes = 10000
    for episode in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0
        total_steps = 0
        while not done:
            action = agent.choose_action(state, training=True)
            #print(action)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            agent.store_transition(state, action, next_state, reward, done)
            state = next_state

            total_steps += 1
            agent.train()

        print(f"Episode {episode}, total reward {total_reward}, total steps {total_steps}")


def TD3example():
    env = gym.make("LunarLanderContinuous-v2")
    #env = gym.make("BipedalWalker-v3")
    #env = gym.make("Pendulum-v0")

    agent = agents.td3.TD3Agent(env=env, gamma=0.99, q_learning_rate=0.001, policy_learning_rate=0.001,
                                  batch_size=128, buffer_size=400000, noise=0.1, noise_decay_rate=0.999,
                                 policy_delay=2, target_noise_clip=0.5, policy_noise=0.2, polyak=0.995)

    episodes = 10000
    for episode in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0
        total_steps = 0
        while not done:
            if episode < 50:
                action = agent.choose_action(state, uniform_random=True)
            else:
                action = agent.choose_action(state, training=True)
            #print(action)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            agent.store_transition(state, action, next_state, reward, done)
            state = next_state

            total_steps += 1
            if episode > 50:
                agent.train()

        if episode > 50 and episode % 50 == 0:
            evaluate(env, agent)
        #if episode > 50 and episode % 50 == 0:
        if episode > 0 and episode % 50 == 0:
            evaluate(env, agent, simulate=True)

        print(f"Episode {episode}, total reward {total_reward}, total steps {total_steps}")


if __name__ == "__main__":
    TD3example()