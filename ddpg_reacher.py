# coding=utf-8
import copy
import datetime
import pandas
import numpy
import random
from unityagents import UnityEnvironment

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import deque, namedtuple
import click


BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
EPSILON = 1.0
EPSILON_DECAY = 1e-6

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=256):
        super().__init__()
        self.seed = torch.manual_seed(seed)

        self._pi_net = torch.nn.Sequential(
            nn.BatchNorm1d(state_size),
            nn.Linear(state_size, fc1_units),
            nn.BatchNorm1d(fc1_units),
            nn.ReLU(),
            nn.Linear(fc1_units, fc2_units),
            nn.BatchNorm1d(fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, action_size),
            nn.Tanh()
        )

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""

        return self._pi_net(state)


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, state_layer_units=256, state_action_layer_units=256):
        super().__init__()
        self.seed = torch.manual_seed(seed)

        self._q_state_net = nn.Sequential(
            nn.BatchNorm1d(state_size),
            nn.Linear(state_size, state_layer_units),
            nn.BatchNorm1d(state_layer_units),
            nn.ReLU()
        )

        self._q_net = nn.Sequential(
            nn.Linear(state_layer_units + action_size, state_action_layer_units),
            nn.BatchNorm1d(state_action_layer_units),
            nn.ReLU(),
            nn.Linear(state_action_layer_units, 1),
        )

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""

        state_rep = self._q_state_net(state)
        return self._q_net(torch.cat((state_rep, action), dim=1))


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, random_seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.epsilon = EPSILON

        # Actor Network
        self.actor_local = Actor(state_size, action_size, random_seed).to(DEVICE)
        self.actor_target = Actor(state_size, action_size, random_seed).to(DEVICE)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network
        self.critic_local = Critic(state_size, action_size, random_seed).to(DEVICE)
        self.critic_target = Critic(state_size, action_size, random_seed).to(DEVICE)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

        self.copy_weights(self.actor_target, self.actor_local)
        self.copy_weights(self.critic_target, self.critic_local)

    def step(self, states, actions, rewards, next_states, dones, timestep):
        """Save experience in replay memory, and use random sample from buffer to learn."""

        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and timestep % 20 == 0:
            for _ in range(10):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""

        state = torch.from_numpy(state).float().to(DEVICE)

        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        if add_noise:
            action += self.epsilon * self.noise.sample()

        return action

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)

        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

        # ---------------------------- update noise ---------------------------- #
        self.epsilon -= EPSILON_DECAY
        self.noise.reset()

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def copy_weights(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.state = None
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(DEVICE)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(DEVICE)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(DEVICE)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(DEVICE)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class UnityMultiAgentEnvWrapper:
    """ This class provides gym-like wrapper around the unity environment with multiple agents """

    def __init__(self, env_file: str, train_mode: bool = False):
        self._env = UnityEnvironment(file_name=env_file)
        self._train_mode = train_mode
        self._brain_name = self._env.brain_names[0]
        self._brain = self._env.brains[self._brain_name]

        env_info = self._env.reset(train_mode=True)[self._brain_name]

        self.num_agents = len(env_info.agents)
        self.state_space_dim = env_info.vector_observations.shape[1]
        self.action_space_dim = self._brain.vector_action_space_size

    def reset(self):
        env_info = self._env.reset(self._train_mode)[self._brain_name]
        state = env_info.vector_observations
        return state

    def step(self, action):
        env_info = self._env.step(action)[self._brain_name]  # send the action to the environment
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        return next_states, rewards, dones, None

    def close(self):
        self._env.close()


def train(max_episodes: int, episode_len: int = 3000):
    """ Train the agent using a head-less environment and save the weights when done """

    env = UnityMultiAgentEnvWrapper('Reacher_Linux_NoVis/Reacher.x86_64', train_mode=True)
    agent = Agent(state_size=env.state_space_dim, action_size=env.action_space_dim, random_seed=8276364)

    data = []
    scores = []
    episode = 0
    for episode in range(1, max_episodes):
        states = env.reset()
        score = numpy.zeros(env.num_agents)
        agent.reset()

        for step in range(episode_len):
            actions = agent.act(states)
            next_states, rewards, dones, _ = env.step(actions)
            agent.step(states, actions, rewards, next_states, dones, step)

            score += rewards
            states = next_states
            if numpy.any(dones):
                break

        scores.append(score.mean())
        rolling_average_score = sum(scores[-100:])/min(episode, 100)
        data.append([score.mean(), rolling_average_score])
        print(f'Episode {episode} ({step} steps). Final score {score.mean():.2f}. '
              f'Average score (last 100 episodes) {rolling_average_score:.2f}. '
              f'Replay buffer size {len(agent.memory)}.')

        if episode % 10 == 0:
            torch.save(agent.actor_local.state_dict(), f'checkpoint_actor_{episode}.pth')
            torch.save(agent.critic_local.state_dict(), f'checkpoint_critic_{episode}.pth')

    df = pandas.DataFrame(data=data, index=range(1, episode+1), columns=['score', 'rolling_avg_score'])
    now_str = datetime.datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')
    df.to_csv(f'scores-{now_str}.csv')

    env.close()


def test(actor_weights_file: str, critic_weights_file: str):
    """ Load model parameters and run the agent """
    env = UnityMultiAgentEnvWrapper('Reacher_Linux/Reacher.x86_64', train_mode=False)
    agent = Agent(state_size=env.state_space_dim, action_size=env.action_space_dim, random_seed=8276364)

    agent.actor_local.load_state_dict(torch.load(actor_weights_file))
    agent.critic_local.load_state_dict(torch.load(critic_weights_file))
    agent.actor_local.eval()
    agent.critic_local.eval()

    states = env.reset()
    score = numpy.zeros(env.num_agents)

    for step in range(3000):
        actions = agent.act(states, add_noise=False)
        next_states, rewards, dones, _ = env.step(actions)
        agent.step(states, actions, rewards, next_states, dones, step)

        score += rewards
        states = next_states
        if numpy.any(dones):
            break

    print(f'Final score {score.mean()}.')
    env.close()


@click.group()
@click.version_option()
def cli():
    """ drl_policy_gradients -- command line interface """


@cli.command('train')
@click.option('--max-episodes', type=click.INT, default=251)
def train_command(max_episodes: int):
    """ Train the agent using a head-less environment and save the DQN weights when done """
    train(max_episodes)


@cli.command('test')
@click.option('--actor-weights-file', default='checkpoint_actor.pth',
              type=click.Path(dir_okay=False, file_okay=True, readable=True, exists=True))
@click.option('--critic-weights-file', default='checkpoint_critic.pth',
              type=click.Path(dir_okay=False, file_okay=True, readable=True, exists=True))
def test_command(actor_weights_file: str, critic_weights_file: str):
    """ Load DQN weights and run the agent """
    test(actor_weights_file, critic_weights_file)


if __name__ == '__main__':
    cli()
