import gym
import time
import random
import numpy


class DummyAgent:
    def __init__(self, env):
        self.env = env

    def step(self, state, action, reward, next_state, done):
        pass

    def select_action(self, _):
        return self.env.action_space.sample()


class Agent1:
    def __init__(self, env):
        self.state_space_dim = env.state
        self.env = env

    def step(self, state, action, reward, next_state, done):
        pass

    def select_action(self, state):
        return self.env.action_space.sample()


def visualize(env, agent):
    state = env.reset()
    cum_episode_reward = 0
    env.render()
    for i in range(1000):
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        cum_episode_reward += reward
        env.render()
        state = next_state
        if done:
            break
    return cum_episode_reward


def main():
    env = gym.make('LunarLanderContinuous-v2')
    agent = DummyAgent(env)

    cum_reward = visualize(env, agent)
    print(cum_reward)
    time.sleep(3)


if __name__ == '__main__':
    main()
