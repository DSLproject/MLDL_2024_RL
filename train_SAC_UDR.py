"""
    Train an RL agent on the OpenAI Gym Hopper environment using
    the Soft Actor-Critic algorithm with Uniform Domain Randomization 
"""

import gym
from env.custom_hopper import *
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from env.customCallback import *
import torch


# Functions to modify the body's mass
'''
def normal(train_env_source):

    val=[3.92699082, 2.71433605, 5.0893801]
    mass = torch.tensor(val, dtype=torch.float32)
    train_env_source.sim.model.body_mass[2:] = torch.normal(mean=mass, std=0.1).numpy()
'''


def uniform(train_env_source, param):
    val = [3.92699082, 2.71433605, 5.0893801]
    mass = torch.tensor(val, dtype=torch.float32)
    low = (1-param) * mass
    high = (1+param) * mass
    new_masses = torch.rand_like(mass) * (high - low) + low
    train_env_source.sim.model.body_mass[2:] = new_masses.numpy()


'''
def lognormal(train_env_source):
    val=[3.92699082, 2.71433605, 5.0893801]
    mass = torch.tensor(val, dtype=torch.float32)
    mean = torch.log(mass)
    std = torch.ones_like(mass)
    new_masses = torch.exp(torch.normal(mean=mean, std=std))
    train_env_source.sim.model.body_mass[2:] = new_masses.numpy()
'''


def main():

    train_env_source = gym.make('CustomHopper-source-v0')
    train_env_target = gym.make('CustomHopper-target-v0')
    # train_custom=CustomHopper(train_env_source)

    # train_env_source.sim.model.body_mass[1] += 1
    print(train_env_source.sim.model.body_mass[2:])

    test_env_source = make_vec_env("CustomHopper-source-v0")
    test_env_target = make_vec_env("CustomHopper-target-v0")

    total_timesteps = int(1e5)
    model_source = SAC('MlpPolicy', train_env_source, learning_rate=3e-4,
                       buffer_size=int(5e6), batch_size=256, verbose=0)
    model_source.learn(total_timesteps=total_timesteps,
                       callback=CustomCallback(train_env_source))

    model_target = SAC('MlpPolicy', train_env_target, learning_rate=3e-4,
                       buffer_size=int(5e6), batch_size=256, verbose=0)
    model_target.learn(total_timesteps=total_timesteps)

    mean_reward_s_s, _ = evaluate_policy(
        model_source, test_env_source, n_eval_episodes=50)
    mean_reward_s_t, _ = evaluate_policy(
        model_source, test_env_target, n_eval_episodes=50)
    mean_reward_t_t, _ = evaluate_policy(
        model_target, test_env_target, n_eval_episodes=50)

    print(f"Source → Source: {mean_reward_s_s}")
    print(f"Source → Target: {mean_reward_s_t}")
    print(f"Target → Target: {mean_reward_t_t}")

    # Training loop
    '''
    while total_timesteps>0:
        obs = train_env_source.reset()
        uniform(train_env_source,0.1)
        for _ in range(total_timesteps):
                if total_timesteps%1000==0:
                     print(f"timesteps {total_timesteps}")
                action = model_source.predict(obs, deterministic=True)[0]
                obs, reward, done, info = train_env_source.step(action)
                model_source.learn(total_timesteps=1, reset_num_timesteps=False)
                total_timesteps -= 1
                if done:
                    break

    customEnv=SAC('MlpPolicy', train_env_source, learning_rate=3e-4, buffer_size=int(5e6), batch_size=256, verbose=0)
    customEnv.learn(total_timesteps=total_timesteps,cal)
    customEnv
    
    model_target = SAC('MlpPolicy', train_env_target, learning_rate=3e-4, buffer_size=int(5e6), batch_size=256, verbose=0)
    model_target.learn(total_timesteps=total_timesteps)
    
    mean_reward_s_s, _ = evaluate_policy(model_source, test_env_source, n_eval_episodes=50)
    mean_reward_s_t, _ = evaluate_policy(model_source, test_env_target, n_eval_episodes=50)
    mean_reward_t_t, _ = evaluate_policy(model_target, test_env_target, n_eval_episodes=50)

    print(f"Source → Source: {mean_reward_s_s}")
    print(f"Source → Target: {mean_reward_s_t}") 
    print(f"Target → Target: {mean_reward_t_t}")
    '''


if __name__ == '__main__':
    main()
