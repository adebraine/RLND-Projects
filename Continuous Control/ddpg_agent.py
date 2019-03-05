import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

NEGATIVE_SLOPE = 0.01
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 32       # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 3e-4        # learning rate of the critic
WEIGHT_DECAY = 0.0001   # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

#%%
class My_Network(nn.Module):

    def __init__(self, state_size, action_size, hidden_layers, random_seed, critic=False):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        super(My_Network, self).__init__()
        self.critic = critic
        self.seed = torch.manual_seed(random_seed)

        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_layers[0])])
        if self.critic:
            hidden_layers[0] = hidden_layers[0]+action_size
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        if self.critic:
            output_size = 1
        else:
            output_size = action_size
        self.output = nn.Linear(hidden_layers[-1], output_size)
        self.batchnorm = nn.BatchNorm1d(state_size)
        self.reset_parameters()
        if self.critic:
            hidden_layers[0] = hidden_layers[0]-action_size

    def initialize(self, func, K=False):
        if K:
            torch.nn.init.kaiming_normal_(func.weight.data, a=NEGATIVE_SLOPE, mode='fan_in')
        else:
            func.weight.data.uniform_(*hidden_init(func))

    def acti_func(self, func, leak=False):
        if not leak:
            return F.relu(func)
        else:
            return F.leaky_relu(func)

    def reset_parameters(self):
        for layer in self.hidden_layers[:-1]:
            self.initialize(layer, K=True)
        
        if self.critic:
            self.hidden_layers[-1].weight.data.uniform_(-3e-3, 3e-3)
        else:
            self.initialize(self.hidden_layers[-1], K=True)
            self.output.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action=None, bnorm=True):
        """Build a network that maps state -> action values."""
        if bnorm:
            state = self.batchnorm(state)
        if self.critic:
            state = self.acti_func(self.hidden_layers[0](state), leak=True)
            state = torch.cat((state, action), dim=1)

            for layer in self.hidden_layers[1:]:
                state = self.acti_func(layer(state), leak=True)
            state = self.output(state)
        else:
            for layer in self.hidden_layers:
                state = self.acti_func(layer(state))
            state = F.tanh(self.output(state))

        return state

#%%
class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, hidden_layers, n_agents=1, random_seed=0, sampling=10):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.sampling = sampling

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.n_agents = n_agents

        # Actor Network (w/ Target Network)
        self.actor_local = My_Network(state_size, action_size, hidden_layers['actor'], random_seed=random_seed, critic=False).to(device)
        self.actor_target = My_Network(state_size, action_size, hidden_layers['actor'], random_seed=random_seed, critic=False).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = My_Network(state_size, action_size, hidden_layers['critic'], random_seed=random_seed, critic=True).to(device)
        self.critic_target = My_Network(state_size, action_size, hidden_layers['critic'], random_seed=random_seed, critic=True).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
    
    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        for i in zip(states, actions, rewards, next_states, dones):
            self.memory.add(*i)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            for i in range(10):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""
        states = torch.from_numpy(states).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            actions += [self.noise.sample()]*self.n_agents
        return np.clip(actions, -1, 1)

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
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
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
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

#%%
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
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

#%%
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
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

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)