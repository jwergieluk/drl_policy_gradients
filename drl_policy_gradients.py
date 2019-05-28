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
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
EPSILON = 1.0
EPSILON_DECAY = 1e-6

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Actor(torch.nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.bn0 = torch.nn.BatchNorm1d(state_size)
        self.fc1 = torch.nn.Linear(state_size, fc1_units)
        self.bn1 = torch.nn.BatchNorm1d(fc1_units)
        self.fc2 = torch.nn.Linear(fc1_units, fc2_units)
        self.bn2 = torch.nn.BatchNorm1d(fc2_units)
        self.fc3 = torch.nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = self.bn0(state)
        x = torch.nn.functional.relu(self.bn1(self.fc1(x)))
        x = torch.nn.functional.relu(self.bn2(self.fc2(x)))
        return torch.tanh(self.fc3(x))


class Critic(torch.nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=128, fc2_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
            fc3_units (int): Number of nodes in the third hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.bn0 = torch.nn.BatchNorm1d(state_size)
        self.fcs1 = torch.nn.Linear(state_size, fcs1_units)
        self.fc2 = torch.nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = torch.nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        state = self.bn0(state)
        xs = torch.nn.functional.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = torch.nn.functional.relu(self.fc2(x))
        return self.fc3(x)


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size):
        """Initialize an Agent object.
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(10)
        self.epsilon = EPSILON

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, 10).to(device)
        self.actor_target = Actor(state_size, action_size, 10).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, 10).to(device)
        self.critic_target = Critic(state_size, action_size, 10).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, 10)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)

        # Make sure target is with the same weight as the source
        self.hard_update(self.actor_target, self.actor_local)
        self.hard_update(self.critic_target, self.critic_local)

    def step(self, states, actions, rewards, next_states, dones, timestep):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and timestep % 20 == 0:
            for _ in range(10):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""

        state = torch.from_numpy(state).float().to(device)

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
        Q_targets = r + ? * critic_target(next_state, actor_target(next_state))
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
        Q_targets_next = self.critic_target(next_states, actions_next)

        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = torch.nn.functional.mse_loss(Q_expected, Q_targets)

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
        """Soft update model parameters.
        ?_target = t*?_local + (1 - t)*?_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


class Actor0(torch.nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_units=256):
        super(Actor0, self).__init__()

        self._pi_net = torch.nn.Sequential(
            # torch.nn.BatchNorm1d(state_size),
            torch.nn.Linear(state_size, hidden_units),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_units, action_size),
            torch.nn.Tanh()
        )

        # self.fc1 = torch.nn.Linear(state_size, fc_units)
        # self.fc2 = torch.nn.Linear(fc_units, action_size)
        # self.reset_parameters()

    #def reset_parameters(self):
    #    self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
    #    self.fc2.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """ Build an actor (policy) network that maps states -> actions. """
        return self._pi_net(state)
        # x = torch.nn.functional.relu(self.fc1(state))
        # return torch.nn.functional.tanh(self.fc2(x))


class Critic0(torch.nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, hidden_units: int = 64):
        super(Critic0, self).__init__()

        self._q_net = torch.nn.Sequential(
            # torch.nn.BatchNorm1d(state_size),
            torch.nn.Linear(state_size + action_size, hidden_units),
            torch.nn.ReLU(),
            # torch.nn.BatchNorm1d(hidden_units),
            torch.nn.Linear(hidden_units, hidden_units),
            torch.nn.ReLU(),
            # torch.nn.BatchNorm1d(hidden_units),
            torch.nn.Linear(hidden_units, 1),
        )

        #self.fcs1 = torch.nn.Linear(state_size, fcs1_units)
        #self.fc2 = torch.nn.Linear(fcs1_units + action_size, fc2_units)
        #self.fc3 = torch.nn.Linear(fc2_units, fc3_units)
        #self.fc4 = torch.nn.Linear(fc3_units, 1)
        #self.reset_parameters()

    #def reset_parameters(self):
    #    self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
    #    self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
    #    self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
    #    self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        return self._q_net(torch.cat((state, action), dim=1))

        # xs = torch.nn.functional.leaky_relu(self.fcs1(state))
        # x = torch.cat((xs, action), dim=1)
        # x = torch.nn.functional.leaky_relu(self.fc2(x))
        # x = torch.nn.functional.leaky_relu(self.fc3(x))
        # return self.fc4(x)


class Agent0:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # Actor Network (w/ Target Network)
        self.actor_local = Actor0(state_size, action_size, 512).to(DEVICE)
        self.actor_target = Actor0(state_size, action_size, 512).to(DEVICE)
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic0(state_size, action_size, 512).to(DEVICE)
        self.critic_target = Critic0(state_size, action_size, 512).to(DEVICE)
        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC,
                                                 weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size)
        self.epsilon = 1.0

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)

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
            action += self.epsilon * self.noise.sample()

        return action  # numpy.clip(action, -1, 1)

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

    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * numpy.ones(size)
        self.theta = theta
        self.sigma = sigma
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


def ddpg(n_episodes=3000):
    scores_deque = deque(maxlen=100)
    scores_global = []

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        states = env_info.vector_observations  # get the current state (for each agent)
        scores = np.zeros(num_agents)  # initialize the score (for each agent)
        agent.reset()

        score_average = 0
        timestep = time.time()
        for t in range(3000):
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]  # send all actions to tne environment
            next_states = env_info.vector_observations  # get next state (for each agent)
            rewards = env_info.rewards  # get reward (for each agent)
            dones = env_info.local_done  # see if episode finished
            agent.step(states, actions, rewards, next_states, dones, t)
            states = next_states  # roll over states to next time step
            scores += rewards  # update the score (for each agent)
            if np.any(dones):  # exit loop if episode finished
                break

        score = np.mean(scores)
        scores_deque.append(score)
        score_average = np.mean(scores_deque)
        scores_global.append(score)

        print('\rEpisode {}, Average Score: {:.2f}, Max: {:.2f}, Min: {:.2f}, Time: {:.2f}' \
              .format(i_episode, score_average, np.max(scores), np.min(scores), time.time() - timestep), end="\n")

        if i_episode % 10 == 0:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        if score_average > 32:
            break

    return scores_global


def train(max_episodes: int, episode_len: int = 3000):
    """ Train the agent using a head-less environment and save the DQN weights when done """

    env = UnityMultiAgentEnvWrapper('Reacher_Linux_NoVis/Reacher.x86_64')
    agent = Agent(state_size=env.state_space_dim, action_size=env.action_space_dim)

    data = []
    scores = []
    for episode in range(1, max_episodes):
        states = env.reset()
        score = numpy.zeros(env.num_agents)
        agent.reset()

        for step in range(1, episode_len):
            actions = agent.act(states)
            next_states, rewards, dones, _ = env.step(actions)
            agent.step(states, actions, rewards, next_states, dones, step)

            score += rewards
            states = next_states
            if numpy.any(dones):
                break

        scores.append(score.mean())
        rolling_average_score = sum(scores[-100:])/min(episode, 100)
        data.append([score, rolling_average_score])
        print(f'Episode {episode}. Final score {score.mean()}. Average score (last 100 episodes) {rolling_average_score}.')

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
    env = UnityMultiAgentEnvWrapper('Reacher_Linux/Reacher.x86_64')
    agent = Agent(env.state_space_dim, env.action_space_dim, DEVICE)
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
