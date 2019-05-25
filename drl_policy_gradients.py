# coding=utf-8
import math
import datetime
import os
import copy
import pandas
import matplotlib.pyplot as plt
import numpy
import random
from unityagents import UnityEnvironment
import torch
import torch.nn
import torch.nn.functional
import torch.optim
from collections import deque, namedtuple
import click


BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 3e-4        # learning rate of the critic
WEIGHT_DECAY = 0.0001   # L2 weight decay


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

        states = torch.from_numpy(numpy.vstack([e.state for e in experiences if e is not None])).float().to(DEVICE)
        actions = torch.from_numpy(numpy.vstack([e.action for e in experiences if e is not None])).float().to(DEVICE)
        rewards = torch.from_numpy(numpy.vstack([e.reward for e in experiences if e is not None])).float().to(DEVICE)
        next_states = torch.from_numpy(numpy.vstack([e.next_state for e in experiences if e is not None])).float().to(
            DEVICE)
        dones = torch.from_numpy(
            numpy.vstack([e.done for e in experiences if e is not None]).astype(numpy.uint8)).float().to(DEVICE)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / numpy.sqrt(fan_in)
    return -lim, lim


class Actor(torch.nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc_units=256):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = torch.nn.Linear(state_size, fc_units)
        self.fc2 = torch.nn.Linear(fc_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = torch.nn.functional.relu(self.fc1(state))
        return torch.nn.functional.tanh(self.fc2(x))


class Critic(torch.nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=256, fc2_units=256, fc3_units=128):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = torch.nn.Linear(state_size, fcs1_units)
        self.fc2 = torch.nn.Linear(fcs1_units + action_size, fc2_units)
        self.fc3 = torch.nn.Linear(fc2_units, fc3_units)
        self.fc4 = torch.nn.Linear(fc3_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = torch.nn.functional.leaky_relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = torch.nn.functional.leaky_relu(self.fc2(x))
        x = torch.nn.functional.leaky_relu(self.fc3(x))
        return self.fc4(x)


class Agent:
    def __init__(self, state_size, action_size, random_seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(DEVICE)
        self.actor_target = Actor(state_size, action_size, random_seed).to(DEVICE)
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(DEVICE)
        self.critic_target = Critic(state_size, action_size, random_seed).to(DEVICE)
        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC,
                                                 weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
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
            action += self.noise.sample()
        return numpy.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))
        # Compute critic loss
        q_expected = self.critic_local(states, actions)
        critic_loss = torch.nn.functional.mse_loss(q_expected, q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
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

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * numpy.ones(size)
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
        dx = self.theta * (self.mu - x) + self.sigma * numpy.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


class UnityEnvWrapper:
    """ This class provides gym-like wrapper around the unity environment """

    def __init__(self, env_file: str):
        self._env = UnityEnvironment(file_name=env_file)
        self._brain_name = self._env.brain_names[0]
        self._brain = self._env.brains[self._brain_name]

        env_info = self._env.reset(train_mode=True)[self._brain_name]
        state = env_info.vector_observations[0]

        self.state_space_dim = len(state)
        self.action_space_dim = self._brain.vector_action_space_size

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


def train(max_episodes: int, episode_len: int = 300):
    """ Train the agent using a head-less environment and save the DQN weights when done """
    env = UnityEnvWrapper('Reacher_Linux_NoVis/Reacher.x86_64')
    agent = Agent(state_size=env.state_space_dim, action_size=env.action_space_dim, random_seed=10)

    data = []
    scores = []
    for episode in range(1, max_episodes):
        state = env.reset()
        agent.reset()
        # state = env.reset(train_mode=True)
        score = 0
        for step in range(1, episode_len):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)

            score += reward
            state = next_state
            if done:
                break
        scores.append(score)
        rolling_average_score = sum(scores[-100:])/min(episode, 100)
        data.append([score, rolling_average_score])
        print(f'Episode {episode}. Final score {score}. Average score (last 100 episodes) {rolling_average_score}.')
    # Save weights and score series
    #now_str = datetime.datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')
    #os.makedirs('runs', exist_ok=True)
    #agent.save_weights(f'runs/weights-{now_str}.bin')

    # Plot average scores
    #df = pandas.DataFrame(data=data, index=range(1, max_episodes), columns=['score', 'rolling_avg_score'])
    #df.to_csv(f'runs/scores-{now_str}.csv')
    #plt.figure(figsize=(8, 6), dpi=120)
    #plt.tight_layout()
    #df['rolling_avg_score'].plot(grid=True, colormap='cubehelix')
    #plt.savefig(f'runs/scores-{now_str}.png')

def test(weights_file_name: str):
    """ Load DQN weights and run the agent """
    env = UnityEnvWrapper('Reacher_Linux/Reacher.x86_64')
    agent = Agent0(env.state_space_dim, env.action_space_dim, DEVICE)
    if weights_file_name is not None:
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
    """ drl_policy_gradients -- command line interface """


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
