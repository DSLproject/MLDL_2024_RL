import numpy as np
import matplotlib.pyplot as plt


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def plot_comparison_basic_algoritgms():
    # Load the numpy arrays for each run of REINFORCE
    rewards_reinforce_no = [np.load(
        f'results/REINFORCE/reinforce_rewards_source_noBaseline_{i}.npy') for i in range(1, 6)]

    # Load the numpy arrays for each run of REINFORCE with baseline
    rewards_reinforce_20 = [np.load(
        f'results/REINFORCE/reinforce_rewards_source_20Baseline_{i}.npy') for i in range(1, 6)]

    # Load the numpy arrays for each run of Actor-Critic (AC)
    rewards_ac = [np.load(f'results/AC/rewards_{i}.npy') for i in range(0, 5)]

    # Stack the arrays along a new axis to form (num_runs, num_episodes)
    rewards_reinforce_no = np.stack(rewards_reinforce_no, axis=0)
    rewards_reinforce_20 = np.stack(rewards_reinforce_20, axis=0)
    rewards_ac = np.stack(rewards_ac, axis=0)

    # Assuming all arrays have the same number of episodes
    num_episodes = rewards_reinforce_no.shape[1]
    episodes = np.arange(1, num_episodes + 1)

    # Calculate mean and standard deviation of rewards for each algorithm
    mean_rewards_reinforce_no = np.mean(rewards_reinforce_no, axis=0)
    std_rewards_reinforce_no = np.std(rewards_reinforce_no, axis=0)
    mean_rewards_reinforce_20 = np.mean(rewards_reinforce_20, axis=0)
    std_rewards_reinforce_20 = np.std(rewards_reinforce_20, axis=0)
    mean_rewards_ac = np.mean(rewards_ac, axis=0)
    std_rewards_ac = np.std(rewards_ac, axis=0)

    # Apply moving average with a window size of 100
    window_size = 500

    smoothed_mean_rewards_reinforce_no = moving_average(
        mean_rewards_reinforce_no, window_size)
    smoothed_std_rewards_reinforce_no = moving_average(
        std_rewards_reinforce_no, window_size)
    smoothed_mean_rewards_reinforce_20 = moving_average(
        mean_rewards_reinforce_20, window_size)
    smoothed_std_rewards_reinforce_20 = moving_average(
        std_rewards_reinforce_20, window_size)
    smoothed_mean_rewards_ac = moving_average(mean_rewards_ac, window_size)
    smoothed_std_rewards_ac = moving_average(std_rewards_ac, window_size)

    # Adjust episodes array to match the smoothed data
    episodes = episodes[:len(smoothed_mean_rewards_reinforce_no)]

    # Downsampling factor
    downsample_factor = 10

    # Downsample the data
    episodes = episodes[::downsample_factor]
    smoothed_mean_rewards_reinforce_no = smoothed_mean_rewards_reinforce_no[
        ::downsample_factor]
    smoothed_std_rewards_reinforce_no = smoothed_std_rewards_reinforce_no[::downsample_factor]
    smoothed_mean_rewards_reinforce_20 = smoothed_mean_rewards_reinforce_20[
        ::downsample_factor]
    smoothed_std_rewards_reinforce_20 = smoothed_std_rewards_reinforce_20[::downsample_factor]
    smoothed_mean_rewards_ac = smoothed_mean_rewards_ac[::downsample_factor]
    smoothed_std_rewards_ac = smoothed_std_rewards_ac[::downsample_factor]

    # Plotting
    plt.figure(figsize=(14, 8))

    # Plot for REINFORCE
    plt.plot(episodes, smoothed_mean_rewards_reinforce_no,
             label='REINFORCE', color='green')
    plt.fill_between(episodes, smoothed_mean_rewards_reinforce_no - smoothed_std_rewards_reinforce_no,
                     smoothed_mean_rewards_reinforce_no + smoothed_std_rewards_reinforce_no, color='green', alpha=0.2)

    # Plot for REINFORCE with baseline
    plt.plot(episodes, smoothed_mean_rewards_reinforce_20,
             label='REINFORCE with baseline', color='blue')
    plt.fill_between(episodes, smoothed_mean_rewards_reinforce_20 - smoothed_std_rewards_reinforce_20,
                     smoothed_mean_rewards_reinforce_20 + smoothed_std_rewards_reinforce_20, color='blue', alpha=0.2)

    # Plot for Actor-Critic (AC)
    plt.plot(episodes, smoothed_mean_rewards_ac,
             label='Actor-Critic', color='red')
    plt.fill_between(episodes, smoothed_mean_rewards_ac - smoothed_std_rewards_ac,
                     smoothed_mean_rewards_ac + smoothed_std_rewards_ac, color='red', alpha=0.2)

    plt.xlabel('Episodes', fontsize=22)
    plt.ylabel('Rewards', fontsize=22)
    plt.title('Comparison of basic RL algorithms', fontsize=28)
    plt.legend(loc='upper left', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.grid(True)

    # Save the plot as a PDF file
    plt.savefig('results/plots/comparison_rl_basic_algorithms.pdf', format='pdf')

    # Show the plot
    plt.show()


def plot_comparison_basic_algorithms_sac():
    # Load the numpy arrays for each run of REINFORCE
    rewards_reinforce_no = [np.load(
        f'results/REINFORCE/reinforce_rewards_source_noBaseline_{i}.npy') for i in range(1, 6)]

    # Load the numpy arrays for each run of REINFORCE with baseline
    rewards_reinforce_20 = [np.load(
        f'results/REINFORCE/reinforce_rewards_source_20Baseline_{i}.npy') for i in range(1, 6)]

    # Load the numpy arrays for each run of Actor-Critic (AC)
    rewards_ac = [np.load(f'results/AC/rewards_{i}.npy') for i in range(0, 5)]

    # Load the numpy arrays for each run of Soft Actor-Critic (SAC)
    rewards_sac = [
        np.load(f'results/SAC/sac_rewards_source_{i}.npy') for i in range(1, 6)]

    # Stack the arrays along a new axis to form (num_runs, num_episodes)
    rewards_reinforce_no = np.stack(rewards_reinforce_no, axis=0)
    rewards_reinforce_20 = np.stack(rewards_reinforce_20, axis=0)
    rewards_ac = np.stack(rewards_ac, axis=0)

    # Find the minimum length of SAC arrays
    min_sac_length = min(len(r) for r in rewards_sac)

    # Truncate SAC arrays to the minimum length
    rewards_sac = [r[:min_sac_length] for r in rewards_sac]
    rewards_sac = np.stack(rewards_sac, axis=0)

    # Find the minimum number of episodes among all algorithms
    min_episodes = min(
        rewards_reinforce_no.shape[1], rewards_reinforce_20.shape[1], rewards_ac.shape[1], rewards_sac.shape[1])

    # Truncate all arrays to the minimum number of episodes
    rewards_reinforce_no = rewards_reinforce_no[:, :min_episodes]
    rewards_reinforce_20 = rewards_reinforce_20[:, :min_episodes]
    rewards_ac = rewards_ac[:, :min_episodes]
    rewards_sac = rewards_sac[:, :min_episodes]

    episodes = np.arange(1, min_episodes + 1)

    # Calculate mean and standard deviation of rewards for each algorithm
    mean_rewards_reinforce_no = np.mean(rewards_reinforce_no, axis=0)
    std_rewards_reinforce_no = np.std(rewards_reinforce_no, axis=0)
    mean_rewards_reinforce_20 = np.mean(rewards_reinforce_20, axis=0)
    std_rewards_reinforce_20 = np.std(rewards_reinforce_20, axis=0)
    mean_rewards_ac = np.mean(rewards_ac, axis=0)
    std_rewards_ac = np.std(rewards_ac, axis=0)
    mean_rewards_sac = np.mean(rewards_sac, axis=0)
    std_rewards_sac = np.std(rewards_sac, axis=0)

    # Apply moving average with a window size of 10
    window_size = 10

    smoothed_mean_rewards_reinforce_no = moving_average(
        mean_rewards_reinforce_no, window_size)
    smoothed_std_rewards_reinforce_no = moving_average(
        std_rewards_reinforce_no, window_size)
    smoothed_mean_rewards_reinforce_20 = moving_average(
        mean_rewards_reinforce_20, window_size)
    smoothed_std_rewards_reinforce_20 = moving_average(
        std_rewards_reinforce_20, window_size)
    smoothed_mean_rewards_ac = moving_average(mean_rewards_ac, window_size)
    smoothed_std_rewards_ac = moving_average(std_rewards_ac, window_size)
    smoothed_mean_rewards_sac = moving_average(mean_rewards_sac, window_size)
    smoothed_std_rewards_sac = moving_average(std_rewards_sac, window_size)

    # Adjust episodes array to match the smoothed data
    smoothed_episodes = episodes[:len(smoothed_mean_rewards_reinforce_no)]

    # Plotting
    plt.figure(figsize=(14, 8))

    # Plot for REINFORCE
    plt.plot(smoothed_episodes, smoothed_mean_rewards_reinforce_no,
             label='REINFORCE', color='green')
    plt.fill_between(smoothed_episodes, smoothed_mean_rewards_reinforce_no - smoothed_std_rewards_reinforce_no,
                     smoothed_mean_rewards_reinforce_no + smoothed_std_rewards_reinforce_no, color='green', alpha=0.2)

    # Plot for REINFORCE with baseline
    plt.plot(smoothed_episodes, smoothed_mean_rewards_reinforce_20,
             label='REINFORCE with baseline', color='blue')
    plt.fill_between(smoothed_episodes, smoothed_mean_rewards_reinforce_20 - smoothed_std_rewards_reinforce_20,
                     smoothed_mean_rewards_reinforce_20 + smoothed_std_rewards_reinforce_20, color='blue', alpha=0.2)

    # Plot for Actor-Critic (AC)
    plt.plot(smoothed_episodes, smoothed_mean_rewards_ac,
             label='Actor-Critic', color='red')
    plt.fill_between(smoothed_episodes, smoothed_mean_rewards_ac - smoothed_std_rewards_ac,
                     smoothed_mean_rewards_ac + smoothed_std_rewards_ac, color='red', alpha=0.2)

    # Plot for Soft Actor-Critic (SAC)
    plt.plot(smoothed_episodes, smoothed_mean_rewards_sac,
             label='Soft Actor-Critic', color='orange')
    plt.fill_between(smoothed_episodes, smoothed_mean_rewards_sac - smoothed_std_rewards_sac,
                     smoothed_mean_rewards_sac + smoothed_std_rewards_sac, color='orange', alpha=0.2)

    plt.xlabel('Episodes', fontsize=22)
    plt.ylabel('Rewards', fontsize=22)
    plt.title('Comparison of RL algorithms', fontsize=28)
    plt.legend(loc='upper left', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True)

    # Save the plot as a PDF file
    plt.savefig(
        'results/plots/comparison_rl_basic_algorithms_sac.pdf', format='pdf')

    # Show the plot
    plt.show()


def plot_comparison_sac_udr():
    # Load the numpy arrays for each run of Soft Actor-Critic (SAC) with UDR
    rewards_sac_udr = [np.load(
        f'results/SAC - UDR/rewards_SAC_{i}.npy') for i in [0.0, 0.1, 0.4, 0.7, 1.0]]

    # Find the minimum length of SAC-UDR arrays
    min_sac_udr_length = min(len(r) for r in rewards_sac_udr)

    # Truncate SAC-UDR arrays to the minimum length
    rewards_sac_udr = [r[:min_sac_udr_length] for r in rewards_sac_udr]
    rewards_sac_udr = np.stack(rewards_sac_udr, axis=0)

    episodes = np.arange(1, min_sac_udr_length + 1)

    # Apply moving average with a window size of 10
    window_size = 20
    smoothed_rewards_sac_udr = [moving_average(
        rewards, window_size) for rewards in rewards_sac_udr]

    # Adjust episodes array to match the smoothed data
    smoothed_episodes = episodes[:len(smoothed_rewards_sac_udr[0])]

    # Plotting
    plt.figure(figsize=(14, 8))

    # Plot for Soft Actor-Critic (SAC) without UDR
    plt.plot(smoothed_episodes,
             smoothed_rewards_sac_udr[0], label='SAC', color='orange')
    plt.plot(smoothed_episodes,
             smoothed_rewards_sac_udr[1], label='SAC with UDR [0.1]', color='red')
    plt.plot(smoothed_episodes,
             smoothed_rewards_sac_udr[2], label='SAC with UDR [0.4]', color='green')
    plt.plot(smoothed_episodes,
             smoothed_rewards_sac_udr[3], label='SAC with UDR [0.7]', color='blue')
    plt.plot(smoothed_episodes,
             smoothed_rewards_sac_udr[4], label='SAC with UDR [1.0]', color='violet')

    plt.xlabel('Episodes', fontsize=22)
    plt.ylabel('Rewards', fontsize=22)
    plt.title('Comparison of SAC Algorithms with UDR', fontsize=28)
    plt.legend(loc='upper left', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True)

    # Save the plot as a PDF file
    plt.savefig('results/plots/comparison_sac_udr.pdf', format='pdf')

    # Show the plot
    plt.show()


def plot_SAC_test():

    rewards_sac_test = np.load(
        f'results/SAC/sac_eval_50_rewards_source_target.npy')
    algorithm_names = ['SAC', 'SAC-UDR 0.1',
                       'SAC-UDR 0.4', 'SAC-UDR 0.7', 'SAC-UDR 1.0']

    # Assume rewards_sac_test has multiple columns, each representing different data
    rewards_sac_test = [rewards_sac_test for i in range(5)]

    fig, ax = plt.subplots(figsize=(13, 8))
    ax.boxplot(rewards_sac_test, vert=True, patch_artist=True,
               showmeans=True, meanline=True)

    ax.set_xticklabels(algorithm_names, fontsize=12)
    ax.set_title('Episode Rewards Distribution', fontsize=28)
    ax.set_ylabel('Rewards', fontsize=22)

    # Annotate mean values for each boxplot
    for i, reward_data in enumerate(rewards_sac_test):
        mean_reward = np.mean(reward_data)
        ax.annotate(f'Mean\n{mean_reward:.2f}',
                    xy=(i + 1, mean_reward), xycoords='data',
                    xytext=(55, 0), textcoords='offset points',
                    rotation=0, fontsize=7, color='black',
                    va='bottom', ha='center',
                    bbox=dict(boxstyle='round,pad=0.5', fc='gray', alpha=0.3))

    plt.savefig(
        "results/plots/comparison_sac_source_target_50_test.pdf", format='pdf')

    plt.show()


def plot_comparison_cartpole():
    # Load the numpy arrays for each run of REINFORCE
    rewards_ff = [np.load(f'results/ff/rewards_{i}.npy') for i in range(1, 6)]

    # Load the numpy arrays for each run of Actor-Critic (AC)
    rewards_ac = [
        np.load(f'results/AC_cartpole/rewards_{i}.npy') for i in range(1, 6)]

    # Stack the arrays along a new axis to form (num_runs, num_episodes)
    rewards_ff = np.stack(rewards_ff, axis=0)
    rewards_ac = np.stack(rewards_ac, axis=0)

    # Assuming all arrays have the same number of episodes
    num_episodes = rewards_ff.shape[1]
    episodes = np.arange(1, num_episodes + 1)

    # Calculate mean and standard deviation of rewards for each algorithm
    mean_rewards_reinforce_no = np.mean(rewards_ff, axis=0)
    std_rewards_reinforce_no = np.std(rewards_ff, axis=0)
    mean_rewards_ac = np.mean(rewards_ac, axis=0)
    std_rewards_ac = np.std(rewards_ac, axis=0)

    # Apply moving average with a window size of 100
    window_size = 5

    smoothed_mean_rewards_ff = moving_average(
        mean_rewards_reinforce_no, window_size)
    smoothed_std_rewards_f = moving_average(
        std_rewards_reinforce_no, window_size)
    smoothed_mean_rewards_ac = moving_average(mean_rewards_ac, window_size)
    smoothed_std_rewards_ac = moving_average(std_rewards_ac, window_size)

    # Adjust episodes array to match the smoothed data
    episodes = episodes[:len(smoothed_mean_rewards_ff)]

    # Plotting
    plt.figure(figsize=(14, 8))

    # Plot for Actor-Critic (AC)
    plt.plot(episodes, smoothed_mean_rewards_ac,
             label='Actor-Critic', color='red')
    plt.fill_between(episodes, smoothed_mean_rewards_ac - smoothed_std_rewards_ac,
                     smoothed_mean_rewards_ac + smoothed_std_rewards_ac, color='red', alpha=0.18)

    # Plot for Forward-Forward (FF)
    plt.plot(episodes, smoothed_mean_rewards_ff,
             label='Forward-Forward', color='green')
    plt.fill_between(episodes, smoothed_mean_rewards_ff - smoothed_std_rewards_f,
                     smoothed_mean_rewards_ff + smoothed_std_rewards_f, color='green', alpha=0.18)

    plt.xlabel('Episodes', fontsize=22)
    plt.ylabel('Rewards', fontsize=22)
    plt.title(
        'Comparison of Actor-Critic and Forward-Forward in CartPole', fontsize=28)
    plt.legend(loc='upper left', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True)

    # Save the plot as a PDF file
    plt.savefig('results/plots/comparison_cartpole.pdf', format='pdf')

    # Show the plot
    plt.show()


if __name__ == '__main__':
    plot_comparison_basic_algoritgms()
    plot_comparison_basic_algorithms_sac()
    plot_comparison_sac_udr()
    plot_comparison_cartpole()
