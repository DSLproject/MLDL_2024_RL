"""
	Train an RL agent on the OpenAI Gym Hopper environment using
    the Actor-critic algorithm
"""

from env.custom_hopper import *
import gym
import argparse
import matplotlib.pyplot as plt
import torch
import time
import cv2 as cv

from agent_AC import Agent, Policy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=100000,
                        type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=10000,
                        type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str,
                        help='network device [cpu, cuda]')
    return parser.parse_args()


args = parse_args()


def train_agent(env_name, n_episodes, print_every):
    env = gym.make(env_name)

    print('Action space:', env.action_space)
    print('State space:', env.observation_space)
    print('Dynamics parameters:', env.get_parameters())

    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    policy = Policy(observation_space_dim, action_space_dim)
    agent = Agent(policy, device=args.device)

    rewards = []
    eps_len = []

    start_time = time.time()

    for episode in range(n_episodes):
        done = False
        ep_return = 0
        ep_len = 0
        ep_actions = []

        state = env.reset()  # Reset the environment and observe the initial state

        while not done:  # Loop until the episode is over
            action, action_probabilities, critic_value = agent.get_action(
                state)
            previous_state = state

            state, reward, done, info = env.step(action.detach().cpu().numpy())

            agent.store_outcome(previous_state, state,
                                action_probabilities, critic_value, reward, done)

            ep_return += reward
            ep_len += 1
            ep_actions.append(action.detach().cpu().numpy())

        rewards.append(ep_return)
        eps_len.append(ep_len)

        # Train after each episode
        agent.update_policy()

        agent.reset()  # Clear the internal memory

        if (episode+1) % print_every == 0:
            print('Training episode:', episode)
            print('Episode return:', ep_return)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time //
          60} minutes and {elapsed_time % 60} seconds")

    np.save(f'results/Actor-Critic/rewards_{env_name}.npy', rewards)
    np.save(f'results/Actor-Critic/episodes_{env_name}.npy', ep_len)

    torch.save(agent.policy.state_dict(),
               f"results/Actor-Critic/model_ac_{env_name}.mdl")

    return agent


def test_agent(agent, env_name, n_episodes=50):
    env = gym.make(env_name)
    total_returns = []

    for episode in range(n_episodes):
        done = False
        tot_reward = 0

        state = env.reset()

        while not done:
            action, _ = agent.get_action(state, evaluation=True)
            state, reward, done, info = env.step(action.detach().cpu().numpy())
            tot_reward += reward

        total_returns.append(tot_reward)

    average_return = np.mean(total_returns)
    print(f'Average return over {n_episodes} episodes in {
          env_name}: {average_return}')
    return average_return


def visualize_model(path, env_name):
    env = gym.make(env_name)

    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    policy = Policy(observation_space_dim, action_space_dim)
    agent = Agent(policy, device=args.device)
    agent.policy.load_state_dict(torch.load(path))

    render = True

    for episode in range(10):
        done = False
        tot_reward = 0

        state = env.reset()  # Reset the environment and observe the initial state

        while True:  # Loop until the episode is over
            if render:  # Render the environment
                # env.render()

                img = env.render(mode='rgb_array')
                cv.imshow('frame', img)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    return

            action, _ = agent.get_action(state, evaluation=True)

            state, reward, done, info = env.step(action.detach().cpu().numpy())
            tot_reward += reward

            if done:
                print('Return: ', tot_reward)
                break


def main():
    # Train on source environment
    source_agent = train_agent(
        'CustomHopper-source-v0', args.n_episodes, args.print_every)

    # Test configurations
    print("Testing source-trained agent on source environment:")
    source_on_source = test_agent(source_agent, 'CustomHopper-source-v0')

    print("Testing source-trained agent on target environment:")
    source_on_target = test_agent(source_agent, 'CustomHopper-target-v0')


if __name__ == '__main__':
    # main()
    visualize_model(
        "results/Actor-Critic/model_ac_CustomHopper-source-v0.mdl", 'CustomHopper-source-v0')
