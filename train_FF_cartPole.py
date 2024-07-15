"""
	Train an RL agent on the OpenAI Gym Cart Pole environment using
	the Forward-Forward algorithm
"""

from env.custom_hopper import *
import gym
import argparse
import matplotlib.pyplot as plt
import torch
import time
import cv2 as cv

from agent_FF_cartPole import Agent, FFActor, Critic


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=500, type=int,
                        help='Number of training episodes')
    parser.add_argument('--print-every', default=100,
                        type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str,
                        help='network device [cpu, cuda]')
    return parser.parse_args()


args = parse_args()


def train_agent(env_name, n_episodes, print_every, count):
    env = gym.make(env_name)

    print('Action space:', env.action_space)
    print('State space:', env.observation_space)
    observation_space_dim = env.observation_space.shape[0]
    action_space_dim = env.action_space.n
    actor = FFActor(observation_space_dim, action_space_dim, args.n_episodes)
    critic = Critic(observation_space_dim, action_space_dim)
    agent = Agent(actor, critic, device=args.device)

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
            action, value = agent.get_action(state)
            action_bin = agent.convertAction(action)

            next_state, reward, done, info = env.step(action_bin)

            agent.store_outcome(state, action, value, reward, done)

            state = next_state
            ep_return += reward
            ep_len += 1
            ep_actions.append(action)

        rewards.append(ep_return)
        eps_len.append(ep_len)

        # Train after each episode
        agent.update_policy()

        if (episode+1) % print_every == 0:
            print('Training episode:', episode)
            print('Episode return:', ep_return)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time //
          60} minutes and {elapsed_time % 60} seconds")

    plt.plot(rewards)
    plt.title('Return per episode')
    plt.grid(True)
    plt.show()

    # np.save(f'results/ff/rewards_{count}_TEST.npy', rewards)

    torch.save(agent.actor.state_dict(),
               f"models/ff/model_ff_cartPole_{env_name}_a.mdl")
    torch.save(agent.critic.state_dict(),
               f"models/ff/model_ff_cartPole_{env_name}_c.mdl")

    return agent


def test_agent(agent, env_name, n_episodes=50):
    env = gym.make(env_name)

    total_returns = []

    for episode in range(n_episodes):
        done = False
        tot_reward = 0

        state = env.reset()

        while not done:
            # Get the action value pair
            action, value = agent.get_action(state)
            action_bin = agent.convertAction(action)

            # Step the environment
            state, reward, done, _ = env.step(action_bin)

            tot_reward += reward

        total_returns.append(tot_reward)

    total_returns = np.array(total_returns)
    average_return = np.mean(total_returns)
    median_return = np.median(total_returns)
    stddev_return = np.std(total_returns)
    print(f'Statistics of return over {n_episodes} episodes in {env_name}:')
    print(f'Average: {average_return}')
    print(f'Std dev: {stddev_return}')
    print(f'Median : {median_return}')
    return average_return


def visualize_model(path, env_name):
    env = gym.make(env_name)

    observation_space_dim = env.observation_space.shape[0]
    action_space_dim = env.action_space.n

    actor = FFActor(observation_space_dim, action_space_dim, args.n_episodes)
    critic = Critic(observation_space_dim, action_space_dim)
    agent = Agent(actor, critic, device=args.device)
    agent.actor.load_state_dict(torch.load(path + "_a.mdl"))
    agent.critic.load_state_dict(torch.load(path + "_c.mdl"))

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
                if cv.waitKey(16) & 0xFF == ord('q'):
                    return

            # Get the action value pair
            action, value = agent.get_action(state)
            action_bin = agent.convertAction(action)

            # Step the environment
            state, reward, done, _ = env.step(action_bin)

            tot_reward += reward

            if done:
                print('Return: ', tot_reward)
                break


def main():
    # Train on source environment
    for i in range(1):
        source_agent = train_agent(
            'CartPole-v0', args.n_episodes, args.print_every, i)
        source_on_source = test_agent(source_agent, 'CartPole-v0')


if __name__ == '__main__':
    main()
    visualize_model("models/ff/model_ff_cartPole_CartPole-v0", 'CartPole-v0')
