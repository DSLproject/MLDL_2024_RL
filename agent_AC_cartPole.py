import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical


def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        """
            Actor network
        """
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)

        """
            Critic network
        """
        # TASK 3: critic network for actor-critic algorithm
        self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_critic = torch.nn.Linear(self.hidden, 1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        """
            Actor
        """
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_logits = self.fc3_actor_mean(x_actor)
        action_probabilities = F.softmax(action_logits, dim=-1)
        action_distribution = Categorical(action_probabilities)

        """
            Critic
        """
        # TASK 3: forward in the critic network
        x_critic = self.tanh(self.fc1_critic(x))
        x_critic = self.tanh(self.fc2_critic(x_critic))
        value = self.fc3_critic(x_critic)

        return action_distribution, value


class Agent(object):
    def __init__(self, policy, device='cpu'):
        self.eps = np.finfo(np.float32).eps.item()

        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        self.gamma = 0.99
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.critic_values = []
        self.rewards = []
        self.done = []

    def update_policy(self):
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(
            self.train_device).squeeze(-1)
        critic_values = torch.stack(self.critic_values, dim=0).to(
            self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(
            self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(
            self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(
            self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)

        self.states, self.next_states, self.action_log_probs, self.rewards, self.done = [], [], [], [], []

        #
        # TASK 3:
        # Compute boostrapped discounted return estimates
        discounted_returns = discount_rewards(rewards, self.gamma)

        policy_losses = []
        value_losses = []
        for log_prob, critic_value, reward in zip(action_log_probs, critic_values, discounted_returns):
            advantage = reward - critic_value.item()
            # calculate actor (policy) loss
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(critic_value, reward))

        # reset gradients
        self.optimizer.zero_grad()

        # sum up all the values of policy_losses and value_losses
        loss = torch.stack(policy_losses).sum() + \
            torch.stack(value_losses).sum()
        # perform backprop
        loss.backward()
        self.optimizer.step()

        return

    def get_action(self, state, evaluation=False):
        """ state -> action (3-d), action_log_densities """
        x = torch.from_numpy(state).float().to(self.train_device)

        action_distribution, value_critic = self.policy(x)

        if evaluation:  # Return mean
            action = action_distribution.probs.argmax()
        else:   # Sample from the distribution
            action = action_distribution.sample()

        action_log_prob = action_distribution.log_prob(action)
        return action.item(), action_log_prob, value_critic

    def store_outcome(self, state, next_state, action_log_prob, critic_value, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.critic_values.append(critic_value)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)

    def reset(self):
        self.states.clear()
        self.next_states.clear()
        self.action_log_probs.clear()
        self.critic_values.clear()
        self.rewards.clear()
        self.done.clear()
