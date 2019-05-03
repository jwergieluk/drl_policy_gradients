import sys
import math
import numpy
import time
import subprocess
import gym
import random
import networkx
import itertools


def encode_state(taxi_row, taxi_col, pass_loc, dest_idx):
    # (5) 5, 5, 4
    i = taxi_row
    i *= 5
    i += taxi_col
    i *= 5
    i += pass_loc
    i *= 4
    i += dest_idx
    return i


def decode_state(i):
    out = [i % 4]
    i = i // 4
    out.append(i % 5)
    i = i // 5
    out.append(i % 5)
    i = i // 5
    out.append(i)
    assert 0 <= i < 5
    return reversed(out)


def test_taxi_policy_eval1():
    policy_eval = TaxiPolicyEval()
    for i, (r, c) in enumerate(policy_eval.locs):
        for j in range(4):
            assert policy_eval.action_is_optimal(encode_state(r, c, i, j), 4)
        assert policy_eval.action_is_optimal(encode_state(r, c, 4, i), 5)
        for j in range(4):
            if i == j:
                assert policy_eval.action_is_optimal(encode_state(r, c, 4, j), 5)
            else:
                assert not policy_eval.action_is_optimal(encode_state(r, c, 4, j), 5)


def test_taxi_policy_eval2():
    policy_eval = TaxiPolicyEval()
    for r, c, p, d in itertools.product(range(5), range(5), range(4), range(3)):
        if (r, c) in policy_eval.locs:
            continue
        assert not policy_eval.action_is_optimal(encode_state(r, c, p, d), 4)
        assert not policy_eval.action_is_optimal(encode_state(r, c, p, d), 5)


class TaxiPolicyEval:
    """
    +---------+
    |R: | : :G|
    | : : : : |
    | : : : : |
    | | : | : |
    |Y| : |B: |
    +---------+ """

    edges = [((1, 2), (0, 2)), ((1, 2), (1, 1)), ((1, 2), (2, 2)), ((1, 2), (1, 3)), ((0, 2), (0, 3)), ((0, 3), (1, 3)), ((0, 3), (0, 4)), ((1, 3), (2, 3)), ((1, 3), (1, 4)), ((2, 3), (2, 2)), ((2, 3), (3, 3)), ((2, 3), (2, 4)), ((2, 2), (3, 2)), ((2, 2), (2, 1)), ((3, 2), (3, 1)), ((3, 2), (4, 2)), ((3, 1), (2, 1)), ((3, 1), (4, 1)), ((2, 1), (2, 0)), ((2, 1), (1, 1)), ((2, 0), (1, 0)), ((2, 0), (3, 0)), ((1, 0), (0, 0)), ((1, 0), (1, 1)), ((0, 0), (0, 1)), ((1, 1), (0, 1)), ((3, 0), (4, 0)), ((4, 2), (4, 1)), ((3, 3), (3, 4)), ((3, 3), (4, 3)), ((2, 4), (3, 4)), ((2, 4), (1, 4)), ((3, 4), (4, 4)), ((1, 4), (0, 4)), ((4, 4), (4, 3))]
    locs = [(0, 0), (0, 4), (4, 0), (4, 3)]
    drive = {0: (1, 0), 1: (-1, 0), 2: (0, 1), 3: (0, -1)}
    pass_loc_decode = {0: 'R', 1: 'G', 2: 'Y', 3: 'B', 4: 'T'}
    dest_decode = ['R', 'G', 'Y', 'B']
    action_decode = {0: 'south', 1: 'north', 2: 'east', 3: 'west', 4: 'pickup', 5: 'dropoff'}

    def __init__(self):
        self.graph = networkx.Graph()
        self.graph.add_edges_from(self.edges)

    def expected_end_reward(self):
        expectation = -1.0 + 20.0
        path_lengths = 0.0
        path_no = 0.0
        for r0, c0, pickup_index, dropoff_index in itertools.product(
                range(5), range(5), range(4), range(4)):
            if pickup_index == dropoff_index:
                continue
            path_lengths += networkx.shortest_path_length(self.graph, (r0, c0), self.locs[pickup_index])
            path_lengths += networkx.shortest_path_length(self.graph, self.locs[pickup_index], self.locs[dropoff_index])
            path_no += 1.0
        expectation -= path_lengths/path_no
        return expectation

    def action_is_optimal(self, state, action, verbose=False):
        r, c, pass_loc, dest_idx = decode_state(state)
        if verbose:
            sys.stdout.write(f'state {state}: {r} {c} {self.pass_loc_decode[pass_loc]} {self.dest_decode[dest_idx]}  ')
            sys.stdout.write(f'action {action}: {self.action_decode[action]}  ')
        if pass_loc == dest_idx:
            return True

        pass_in_taxi = pass_loc == 4
        goal_tile = self.locs[dest_idx] if pass_in_taxi else self.locs[pass_loc]
        if (r, c) == goal_tile:
            if pass_in_taxi:
                return action == 5
            return action == 4
        if action in (4, 5):
            return False

        r1 = r + self.drive[action][0]
        c1 = c + self.drive[action][1]
        for path in networkx.all_shortest_paths(self.graph, source=(r, c), target=goal_tile):
            if (r1, c1) == path[1]:
                return True
        else:
            return False

    def distance_to_optimal(self, policy):
        distance = 0
        for s, a in enumerate(policy):
            if not self.action_is_optimal(s, a, True):
                distance += 1
                print('not opt')
            else:
                print('')
        return distance

    def get_optimal_policy(self):
        policy = numpy.zeros(500)
        for state in range(len(policy)):
            policy[state] = -999
            for action in range(6):
                if self.action_is_optimal(state, action):
                    policy[state] = action
            assert policy[state] in list(range(6))
        return policy.astype('int').tolist()


class PolicyExecutor:
    def __init__(self, policy):
        self.policy = policy

    def step(self, state, action, reward, next_state, done):
        pass

    def select_action(self, state):
        return self.policy[state]


class Agent1:
    def __init__(self, env, alpha: float, gamma: float, epsilon: float, update_method: str):
        self.dim_s = env.observation_space.n
        self.dim_a = env.action_space.n
        self.Q = numpy.random.randn(self.dim_s, self.dim_a) + 15.0
        self.alpha = alpha
        self.gamma = gamma
        self.episodes_no = 1.0
        self.epsilon0 = epsilon
        if not hasattr(self, update_method):
            raise ValueError(f'Update method {update_method} not implemented')
        self.update_method = getattr(self, update_method)
        self.test_phase = False
        self.graph = networkx.Graph()

    def step(self, state, action, reward, next_state, done):
        if done:
            self.episodes_no += 1.0
        if not self.test_phase:
            self.update_method(state, action, reward, next_state)

    def sarsa(self, state, action, reward, next_state):
        """ Sarsa on-policy Q-table update rule """
        next_action = self.select_action(next_state)
        self.Q[state, action] = self.Q[state, action]*(1.0 - self.alpha) + self.alpha*(
                reward + self.gamma*self.Q[next_state, next_action])

    def sarsamax(self, state, action, reward, next_state):
        """ Sarsamax aka Q-learning off-policy Q-table update rule """
        next_action = numpy.argmax(self.Q[next_state, :])
        self.Q[state, action] = self.Q[state, action]*(1.0 - self.alpha) + self.alpha*(
                reward + self.gamma*self.Q[next_state, next_action])

    def expected_sarsa(self, state, action, reward, next_state):
        """ Expected Sarsa on-policy Q-table update rule """
        next_action = numpy.argmax(self.Q[next_state, :])
        policy_vector_for_next_state = numpy.repeat(self.epsilon()/self.dim_a, self.dim_a)
        policy_vector_for_next_state[next_action] += 1.0 - self.epsilon()
        self.Q[state, action] = self.Q[state, action] * (1.0 - self.alpha) + self.alpha * (
                reward + self.gamma * numpy.dot(policy_vector_for_next_state, self.Q[next_state, :]))

    def epsilon(self):
        if self.test_phase:
            return 0.0
        n = self.episodes_no
        if n < 11000.0:
            return 0.1
        return self.epsilon0

    def select_action(self, state):
        if random.random() <= self.epsilon():
            return random.randint(0, self.dim_a-1)
        return numpy.argmax(self.Q[state, :])

    def get_policy(self):
        return numpy.argmax(self.Q, axis=1)


def run(env, agent, max_episodes=20000, verbose=True):
    end_rewards = []
    best_avg_reward = -math.inf
    for i in range(max_episodes):
        state = env.reset()
        cum_reward = 0
        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            cum_reward += reward
            agent.step(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        end_rewards.append(cum_reward)
        avg = numpy.mean(end_rewards[-100:])
        best_avg_reward = avg if avg > best_avg_reward else best_avg_reward
        if i % 500 == 0 and verbose:
            print(i, best_avg_reward)
    return numpy.mean(end_rewards)


def render(env, i, reward, cum_reward):
    subprocess.call('clear', shell=False)
    env.render()
    print(f'ep {i}  last_reward {reward}  cum_reward {cum_reward}')
    time.sleep(1.0)


def visualize(env, agent):
    state = env.reset()
    cum_episode_reward = 0
    render(env, -1, 0, cum_episode_reward)
    for i in range(1000):
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        cum_episode_reward += reward
        render(env, i, reward, cum_episode_reward)
        state = next_state
        if done:
            break
    return cum_episode_reward


def main():
    environment = gym.make('Taxi-v2')

    policy_eval = TaxiPolicyEval()
    # optimal_agent = PolicyExecutor(policy_eval.get_optimal_policy())
    print('expected end reward: ', policy_eval.expected_end_reward())

    agent = Agent1(environment, alpha=0.05, gamma=0.9, epsilon=0.0, update_method='expected_sarsa')
    visualize(environment, agent)
    run(environment, agent, max_episodes=20000)

    agent.test_phase = True
    mean_end_reward = run(environment, agent, max_episodes=10000, verbose=False)
    print('mean_end_reward: ', mean_end_reward)

    test_agent = PolicyExecutor(agent.get_policy())
    mean_end_reward = run(environment, test_agent, max_episodes=10000, verbose=False)
    print('test_agent mean_end_reward: ', mean_end_reward)

    print('distance to an optimal policy: ', policy_eval.distance_to_optimal(agent.get_policy()))


if __name__ == '__main__':
    main()

