import math
import datetime
import os
import pandas
import matplotlib.pyplot as plt
import numpy
import random
from unityagents import UnityEnvironment
import torch
import torch.nn
import torch.optim
from collections import deque, namedtuple
import click


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size):
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size: int, device):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=batch_size)

        states = torch.from_numpy(numpy.vstack([e.state for e in experiences])).float().to(device)
        actions = torch.from_numpy(numpy.vstack([e.action for e in experiences])).long().to(device)
        rewards = torch.from_numpy(numpy.vstack([e.reward for e in experiences])).float().to(device)
        next_states = torch.from_numpy(numpy.vstack([e.next_state for e in experiences])).float().to(device)
        dones = torch.from_numpy(numpy.vstack([e.done for e in experiences]).astype(numpy.uint8)).float().to(device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class QNet(torch.nn.Module):
    """ Deep Q Network approximating the state-action value function """
    def __init__(self, input_dim: int, action_no):
        super().__init__()
        self._net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 96),
            torch.nn.ReLU(),
            torch.nn.Linear(96, 96),
            torch.nn.ReLU(),
            torch.nn.Linear(96, action_no)
        )

    def forward(self, x):
        return self._net(x)


class Agent0:
    LEARNING_RATE = 0.0005
    UPDATE_EVERY = 4
    REPLAY_BUFFER_SIZE = 100_000
    BATCH_SIZE = 128
    GAMMA = 0.99

    def __init__(self, state_space_dim: int, no_actions: int, device):
        self.state_space_dim = state_space_dim
        self.no_actions = no_actions
        self.device = device

        self.q_net = QNet(self.state_space_dim, self.no_actions)
        self.q_net.to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.LEARNING_RATE)
        self.loss = torch.nn.MSELoss()

        self._replay_buffer = ReplayBuffer(self.REPLAY_BUFFER_SIZE)
        self.t = 1   # counts the calls to the learn() method

    def load_weights(self, file_name: str):
        """ Loads the DQN weights from a file and sets the Agent to test mode """
        self.q_net.load_state_dict(torch.load(file_name))
        self.q_net.eval()
        self.t = 1800.0*300.0

    def save_weights(self, file_name: str):
        """ Save DQN weights to file """
        torch.save(self.q_net.state_dict(), file_name)

    def epsilon(self):
        """ Returns the probability of taking a random action during the training time """
        return math.exp(-self.t*0.00002)

    def get_action(self, state):
        """ Produce an optimal action for a given state """
        if random.random() <= self.epsilon():
            return random.randint(0, self.no_actions-1)

        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.q_net.eval()
        with torch.no_grad():
            action_values = self.q_net(state)
        self.q_net.train()
        return numpy.argmax(action_values.cpu().detach().numpy())

    def learn(self, state, action, reward, next_state, done):
        self.t += 1
        self._replay_buffer.add(state, action, reward, next_state, done)

        if self.t % self.UPDATE_EVERY != 0:
            return
        if len(self._replay_buffer) < self.BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self._replay_buffer.sample(self.BATCH_SIZE, self.device)

        # Get max predicted Q values (for next states) from target model
        q_targets_next = self.q_net(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        q_targets = rewards + (self.GAMMA * q_targets_next * (1 - dones))
        # Get expected Q values from local model
        q_expected = self.q_net(states).gather(1, actions)

        loss_value = self.loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss_value.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        # self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class UnityEnvWrapper:
    """ This class provides gym-like wrapper around the unity environment """

    def __init__(self, env_file: str = 'Banana_Linux_NoVis/Banana.x86_64'):
        self._env = UnityEnvironment(file_name=env_file)
        self._brain_name = self._env.brain_names[0]
        self._brain = self._env.brains[self._brain_name]

        env_info = self._env.reset(train_mode=True)[self._brain_name]
        state = env_info.vector_observations[0]

        self.state_space_dim = len(state)
        self.action_space_size = self._brain.vector_action_space_size

    def reset(self, train_mode: bool = False):
        env_info = self._env.reset(train_mode)[self._brain_name]
        state = env_info.vector_observations[0]
        return state

    def step(self, action):
        env_info = self._env.step(action)[self._brain_name]  # send the action to the environment
        next_state = env_info.vector_observations[0]  # get the next state
        reward = env_info.rewards[0]  # get the reward
        done = env_info.local_done[0]  # see if episode has finished
        return next_state, reward, done, None

    def close(self):
        self._env.close()


def train(max_episodes: int):
    """ Train the agent using a head-less environment and save the DQN weights when done """
    env = UnityEnvWrapper('Banana_Linux_NoVis/Banana.x86_64')
    agent = Agent0(env.state_space_dim, env.action_space_size, DEVICE)

    data = []
    scores = []
    for episode in range(1, max_episodes):
        state = env.reset(train_mode=True)
        score = 0
        for step in range(1, 300):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state, done)

            score += reward
            state = next_state
            if done:
                break
        scores.append(score)
        rolling_average_score = sum(scores[-100:])/min(episode, 100)
        data.append([score, rolling_average_score])
        print(f'Episode {episode}. Final score {score}. Average score (last 100 episodes) {rolling_average_score}.')
    # Save weights and score series
    now_str = datetime.datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')
    os.makedirs('runs', exist_ok=True)
    agent.save_weights(f'runs/weights-{now_str}.bin')

    # Plot average scores
    df = pandas.DataFrame(data=data, index=range(1, max_episodes), columns=['score', 'rolling_avg_score'])
    df.to_csv(f'runs/scores-{now_str}.csv')
    plt.figure(figsize=(8, 6), dpi=120)
    plt.tight_layout()
    df['rolling_avg_score'].plot(grid=True, colormap='cubehelix')
    plt.savefig(f'runs/scores-{now_str}.png')


def test(weights_file_name: str):
    """ Load DQN weights and run the agent """
    env = UnityEnvWrapper('Banana_Linux/Banana.x86_64')
    agent = Agent0(env.state_space_dim, env.action_space_size, DEVICE)
    agent.load_weights(weights_file_name)

    state = env.reset(train_mode=False)
    score = 0
    for step in range(1, 300):
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)

        print(f'Step {step}. Action {action}. Reward {reward}.')

        score += reward
        state = next_state
        if done:
            break
    print(f'Final score {score}.')
    env.close()


@click.group()
@click.version_option()
def cli():
    """ deep_banana_eater -- command line interface """


@cli.command('train')
@click.option('--max-episodes', type=click.INT, default=2000)
def train_command(max_episodes: int):
    """ Train the agent using a head-less environment and save the DQN weights when done """
    train(max_episodes)


@cli.command('test')
@click.option('--load-weights-from', type=click.Path(dir_okay=False, file_okay=True, readable=True, exists=True))
def test_command(load_weights_from: str):
    """ Load DQN weights and run the agent """
    test(load_weights_from)


if __name__ == '__main__':
    cli()
