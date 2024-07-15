"""
    Train an RL agent on the OpenAI Gym Hopper environment using
    the Soft Actor-Critic algorithm
"""

import gym
import os
from env.custom_hopper import *
import env.mujoco_env as menv
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
import pandas as pd
import itertools
import cv2 as cv
from env.mujoco_env import MujocoEnv


def findHyperparameters():
    total_timesteps = int(50000)

    # Hyperparameters grid
    hyperparameters = {
        'learning_rate': [1e-4, 3e-4],
        'buffer_size': [int(1e6), int(5e6)],
        'batch_size': [256, 512]
    }

    # Create the training environment
    train_env_source = gym.make('CustomHopper-source-v0')

    # Create the test environment
    test_env_source = make_vec_env("CustomHopper-source-v0")

    best_reward_s_s = float("-inf")
    best_params = None
    best_model = None

    # Generate all combinations of hyperparameters
    hyperparameter_combinations = list(
        itertools.product(*hyperparameters.values()))

    for params in hyperparameter_combinations:

        lr, buffer_size, batch_size = params

        # Define and train the SAC agent
        model_source = SAC('MlpPolicy', train_env_source, learning_rate=lr,
                           buffer_size=buffer_size, batch_size=batch_size, verbose=1)
        model_source.learn(total_timesteps=total_timesteps)

        # Evaluate the trained model on the test environment
        mean_reward_s_s, _ = evaluate_policy(
            model_source, test_env_source, n_eval_episodes=50)

        print(params)
        print(f"Source → Source: {mean_reward_s_s}")

        # Update the best parameters if necessary
        if mean_reward_s_s > best_reward_s_s:
            best_reward_s_s = mean_reward_s_s
            best_params = params
            best_model = model_source

    print(f"Source → Source: {best_reward_s_s}")

    print(f"\nHyperparameters:")
    print(f"\tLearning rate: {best_params[0]}")
    print(f"\tBuffer size: {best_params[1]}")
    print(f"\tBatch size: {best_params[2]}")

    if best_model is not None:
        best_model.save(
            f"results/SAC/model_SAC_{best_params[0]}_{best_params[1]}_{best_params[2]}.zip")
        print("Best model saved successfully.")


def trainAndTestSourceTarget(params):

    log_dir = "results/SAC/logs/"
    os.makedirs(log_dir, exist_ok=True)

    total_timesteps = int(400000)

    # Create the training environment
    train_env_source = gym.make('CustomHopper-source-v0')
    train_env_target = gym.make('CustomHopper-target-v0')

    train_env_source = Monitor(train_env_source, log_dir)

    # Create the test environment
    test_env_source = make_vec_env("CustomHopper-source-v0")
    test_env_target = make_vec_env("CustomHopper-target-v0")

    lr, buffer_size, batch_size = params

    # Define and train the SAC agent
    model_source = SAC('MlpPolicy', train_env_source, learning_rate=lr,
                       buffer_size=buffer_size, batch_size=batch_size, verbose=1)
    model_source.learn(total_timesteps=total_timesteps)

    model_target = SAC('MlpPolicy', train_env_target, learning_rate=lr,
                       buffer_size=buffer_size, batch_size=batch_size, verbose=1)
    model_target.learn(total_timesteps=total_timesteps)

    # Get rewards and episode lengths
    rewards = train_env_source.get_episode_rewards()
    lengths = train_env_source.get_episode_lengths()

    np.save(f'results/SAC/sac400k_rewards_source_1.npy', rewards)
    np.save(f'results/SAC/sac400k_episodes_source_1.npy', lengths)

    # Evaluate the trained model on the test environment
    mean_reward_s_s, _ = evaluate_policy(
        model_source, test_env_source, n_eval_episodes=50)
    mean_reward_s_t, _ = evaluate_policy(
        model_source, test_env_target, n_eval_episodes=50)
    mean_reward_t_t, _ = evaluate_policy(
        model_target, test_env_target, n_eval_episodes=50)

    print(params)
    print(f"Source → Source: {mean_reward_s_s}")
    print(f"Source → Target: {mean_reward_s_t}")
    print(f"Target → Target: {mean_reward_t_t}")

    # Save models
    model_source.save(
        f"results/SAC/sac400k_model_source_{lr}_{buffer_size}_{batch_size}_1.zip")
    model_target.save(
        f"results/SAC/sak400k_model_target_{lr}_{buffer_size}_{batch_size}.zip")


def visualize_hopper(path, env_name, num_episodes=25):
    env = gym.make(env_name)

    # Load the model
    model = SAC.load(path)

    print(model.policy)

    for _ in range(num_episodes):
        obs = env.reset()
        done = False

        tot_reward = 0

        while not done:
            # env.render()
            img = env.render(mode='rgb_array')
            cv.imshow('frame', img)

            if cv.waitKey(1) & 0xFF == ord('q'):
                return

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)

            tot_reward += reward

            if done:
                print('Return: ', tot_reward)
                break

    env.close()


if __name__ == '__main__':

    params = findHyperparameters()
    # params = [3e-4, int(5e6), 256]

    trainAndTestSourceTarget(params)

    # visualize_hopper("results/SAC/sac_model_source_0.0003_5000000_256_1.zip", 'CustomHopper-source-v0')
