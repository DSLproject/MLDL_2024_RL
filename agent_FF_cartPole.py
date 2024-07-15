import math
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F


# Compute the discounted cumulative reward
def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def loss_fn(y, theta, sign):
    # If there are no samples set the loss to 0
    if y.shape[0] == 0:
        return torch.tensor(0)

    logits = torch.square(y).mean(dim=-1) - theta
    loss = (-logits * sign).mean()
    return loss


class FFLayer(torch.nn.Module):
    """
    Forward-Forward linear layer (ReLU activated)
    """

    def __init__(self, in_features, out_features, normalize=True):
        super().__init__()
        self.normalize = normalize

        self.linear = torch.nn.Linear(in_features, out_features, bias=True)
        self.activation = torch.nn.ReLU()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)

        self.init_weights()

    # He uniform initialization
    def init_weights(self):
        torch.nn.init.kaiming_uniform_(
            self.linear.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)

        return self.activation(self.linear(x))

    # Train the layer with the given target
    def ff_train(self, input_tensor, signs, theta):
        y = self(input_tensor.detach())

        y_pos = y[torch.where(signs == 1)]
        y_neg = y[torch.where(signs == -1)]

        loss_pos = loss_fn(y_pos, theta, sign=1)
        loss_neg = loss_fn(y_neg, theta, sign=-1)

        self.optimizer.zero_grad()
        loss = loss_pos + loss_neg
        loss.backward()
        self.optimizer.step()

    # Compute the goodness of the layer with the given input and theta
    @torch.no_grad()
    def positive_eval(self, input_tensor, theta):
        y = self(input_tensor)
        return y, torch.square(y).mean(dim=-1) - theta


class FFActor(torch.nn.Module):
    """
    Forward-Forward Policy network
    """

    def __init__(self, state_space, action_space, max_iter, theta=0.0):
        super().__init__()

        self.hidden = 64
        self.layers = torch.nn.ModuleList()
        # The input layer is composed of the state concatenated with the possible actions
        # In this first layer the input is NOT normalized
        self.layers.append(
            FFLayer(state_space + action_space, self.hidden, normalize=False))
        self.layers.append(FFLayer(self.hidden, self.hidden, normalize=True))

        self.theta = theta             # FF parameter
        self.first_layer_factor = 0.5  # Coefficient of the goodness of the first layer

        # Keep track for the progressive training
        self.curr_phase = 0
        self.max_phases = len(self.layers) - 1
        self.curr_iter = 0
        self.step_iter = math.floor(max_iter / len(self.layers))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    # Train the network in a progressive manner (one layer at a time)
    def train(self, states, actions, crit_values):
        # Convert the inputs into a "dataset"
        data = torch.cat((states, actions), dim=1)

        """
        Compute the sign as:
            Positive (+1) if there is an improvement in the value function
            Negative (-1) if the value function is decreasing (the last action is always bad)
        Improvement: if the episode is truncated (the end is reached) the last sign should be +1
        """
        current_elements = crit_values[:-1]
        next_elements = crit_values[1:]
        comparison_result = torch.where(
            next_elements > current_elements, 1, -1)
        signs = torch.cat((comparison_result, torch.tensor([-1])))

        # Lift data to the layer to train
        for i in range(self.curr_phase):
            data = self.layers[i](data)

        # Train the layer
        self.layers[self.curr_phase].ff_train(data, signs, self.theta)

        # Keep track of which layer to train
        if self.curr_phase != self.max_phases:
            self.curr_iter += 1
            if self.curr_iter > self.step_iter:
                # Freeze the previous layer
                for param in self.layers[self.curr_phase].parameters():
                    param.requires_grad = False
                self.curr_phase += 1
                self.curr_iter = 0

    # Compute the goodness of the entire network with the given input and theta
    # The goodness function used is the sum of all the layer goodnesses,
    # with the first layer scaled down by 'first_layer_factor'.
    @torch.no_grad()
    def positive_eval(self, input_tensor):
        accumulated_goodness = 0
        for i in range(self.curr_phase + 1):
            input_tensor, goodness = self.layers[i].positive_eval(
                input_tensor, self.theta)
            accumulated_goodness += goodness * \
                (self.first_layer_factor if i == 0 else 1)
        return accumulated_goodness

    # Iterate over all the possible actions and choose the one with max goodness
    @torch.no_grad()
    def getBestAction(self, x):
        action0 = torch.tensor([1, 0])
        action1 = torch.tensor([0, 1])
        goodness_0 = self.positive_eval(torch.cat((x, action0)))
        goodness_1 = self.positive_eval(torch.cat((x, action1)))

        # Resolve ties randomly
        if goodness_0 == goodness_1:
            goodness_0 += np.random.normal()

        if goodness_0 >= goodness_1:
            return action0
        else:
            return action1


class Critic(torch.nn.Module):
    """
    Critic network (backpropagation)
    """

    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        self.critic_fc1 = torch.nn.Linear(state_space, self.hidden)
        self.critic_fc2 = torch.nn.Linear(self.hidden, self.hidden)
        self.critic_fc3 = torch.nn.Linear(self.hidden, 1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.tanh(self.critic_fc1(x))
        x = self.tanh(self.critic_fc2(x))
        x = self.critic_fc3(x)
        return x


class Agent(object):
    """
    RL actor-critic agent.
    Composed of: policy (FFActor) + value function (Critic)
    """

    def __init__(self, actor, critic, device='cpu'):
        self.train_device = device

        self.actor = actor.to(self.train_device)
        self.critic = critic.to(self.train_device)
        self.critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)

        self.gamma = 0.99  # Discount factor for the rewards

        # Save values for batch training
        self.states = []
        self.actions = []
        self.critic_values = []
        self.rewards = []
        self.done = []

    # Update the policy and value networks in batch (at the end of the episode)
    def update_policy(self):
        states = torch.stack(self.states, dim=0).to(
            self.train_device).squeeze(-1)
        actions = torch.stack(self.actions, dim=0).to(
            self.train_device).squeeze(-1)
        critic_values = torch.stack(self.critic_values, dim=0).to(
            self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(
            self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)

        # Clear internal memory
        self.reset()

        # Compute boostrapped discounted return estimates
        discounted_returns = discount_rewards(rewards, self.gamma)

        # Calculate critic (value) loss using L1 smooth loss
        value_losses = []
        for critic_value, reward in zip(critic_values, discounted_returns):
            value_losses.append(F.smooth_l1_loss(critic_value, reward))

        # Train actor with progressive forward forward
        self.actor.train(states, actions, critic_values)

        # perform backprop for critic
        self.critic_opt.zero_grad()
        loss = torch.stack(value_losses).sum()
        loss.backward()
        self.critic_opt.step()

    # Get pair (action, value)
    def get_action(self, state):
        x = torch.from_numpy(state).float().to(self.train_device)
        return self.actor.getBestAction(x), self.critic(x)

    # Convert the action from one-hot to 0/1
    def convertAction(self, action):
        return int(action[0] < action[1])

    def store_outcome(self, state, action, critic_value, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.actions.append(action)
        self.critic_values.append(critic_value)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)

    def reset(self):
        self.states.clear()
        self.actions.clear()
        self.critic_values.clear()
        self.rewards.clear()
        self.done.clear()
