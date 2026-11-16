from IPython.display import display, Image, clear_output
import io
import PIL.Image
import PIL.ImageDraw
import time
import PIL.Image
import numpy as np
from mdp import CartPole, FeaturizedCartPole
import gymnasium as gym
import matplotlib.pyplot as plt

def render_cartpole_env(env, save=False):
    rgb_array = env.render()
    image = PIL.Image.fromarray(rgb_array)
    image = image.resize((image.width // 2, image.height // 2))
    if save:
        image.save('cartpole.png')
    display(image)

def play_cartpole():
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    total_reward = 0  
    state = env.reset()[0]  
    done = False  
    
    rgb_array = env.render()
    image = PIL.Image.fromarray(rgb_array)
    image = image.resize((image.width * 2, image.height * 2))  
    # display(image)
    
    # print(f"Initial State:\n"
    #       f"Cart Position: {state[0]:.2f}\n"
    #       f"Cart Velocity: {state[1]:.2f}\n"
    #       f"Pole Angle: {state[2]:.2f}\n"
    #       f"Pole Angular Velocity: {state[3]:.2f}\n")
    # clear_output(wait=True)
    time.sleep(2)  
    
    while not done:
        action = env.action_space.sample() 
        next_state, reward, done, _, _ = env.step(action)  
        total_reward += reward  
        
        render_cartpole_env(env)
        
        action_str = 'Move Left' if action == 0 else 'Move Right'
        
        # print(f"Action Taken: {action_str}\n"
        #       f"Current State:\n"
        #       f"Cart Position: {next_state[0]:.2f}\n"
        #       f"Cart Velocity: {next_state[1]:.2f}\n"
        #       f"Pole Angle: {next_state[2]:.2f}\n"
        #       f"Pole Angular Velocity: {next_state[3]:.2f}\n"
        #       f"**Reward Received:** {reward}\n"
        #       f"**Total Reward:** {total_reward}\n")
        
        time.sleep(0.15)  
        
        clear_output(wait=True)  
        
        state = next_state  
    
    print(f"Episode Finished\n"
          f"Final State:\n"
          f"Cart Position: {state[0]:.2f}\n"
          f"Cart Velocity: {state[1]:.2f}\n"
          f"Pole Angle: {state[2]:.2f}\n"
          f"Pole Angular Velocity: {state[3]:.2f}\n"
          f"Total Reward Achieved: {total_reward}\n")
    
    env.close()  

def capture_cartpole_frame(env):
    frame = PIL.Image.fromarray(env.render())
    frame.thumbnail((480, 320))
    return frame.copy()


def display_cartpole_gif(frames, episode_over=False):
    """Display captured CartPole frames as an inline GIF."""
    if not frames:
        return

    frames = frames[:500]
    if episode_over and len(frames) < 500:
        frozen_frame = frames[-1].copy()
        draw = PIL.ImageDraw.Draw(frozen_frame)
        draw.rectangle((8, 8, 112, 30), fill="black")
        draw.text((12, 11), "Episode over", fill="white")
        frames.extend([frozen_frame] * (500 - len(frames)))
    else:
        frames.extend([frames[-1]] * (500 - len(frames)))

    gif_buffer = io.BytesIO()
    frames[0].save(
        gif_buffer,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=40,
        loop=0,
        optimize=True,
    )
    display(Image(data=gif_buffer.getvalue(), format="gif"))


def visualize_q_table_policy(env, q_table, discretize, total_steps: int = 500, bins=(6, 12, 6, 12), gamma=0.99):
    obs = env.reset()
    episode_frames = []
    episode_return = 0
    gamma_prev = 1.0
    episode_over = False

    if isinstance(obs, dict):
        obs = obs['observation']

    obs_discretized = discretize(obs, bins)

    for _ in range(min(total_steps, 500)):
        episode_frames.append(capture_cartpole_frame(env))
        action = np.argmax(q_table[obs_discretized])
        obs, reward, episode_over = env.step(action)

        if isinstance(obs, dict):
            obs = obs['observation']

        obs_discretized = discretize(obs, bins)
        episode_return += gamma_prev * reward
        gamma_prev *= gamma

        if episode_over:
            break

    display_cartpole_gif(episode_frames, episode_over=episode_over)
    print("Visualization Ended.")
    return [episode_return]

def plot_rewards(rewards_per_episode):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_per_episode)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.yscale('log')  
    plt.title('Reward Over Time During Q-Learning (Log Scale)')
    plt.grid(True, which="both", ls="--")
    plt.show()
    
def plot_feature_heatmap(cartpole_q_w, feature_names):
    # action_labels = ['Left', 'Right']

    # plt.figure(figsize=(12, 6))
    # sns.heatmap(cartpole_q_w, annot=True, cmap='coolwarm', xticklabels=feature_names, yticklabels=action_labels, fmt=".2f")
    # plt.title('Feature Weights Heatmap')
    # plt.xticks(rotation=45, ha='right')
    # plt.ylabel('Actions')
    # plt.xlabel('Features')
    # plt.show()
    
    action_labels = ['Left', 'Right']
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 6))
    # Create heatmap
    im = ax.imshow(cartpole_q_w, cmap='coolwarm')
    # Add colorbar with size adjustment
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # Set ticks and labels
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_yticks(np.arange(len(action_labels)))
    ax.set_xticklabels(feature_names)
    ax.set_yticklabels(action_labels)
    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # Add text annotations with values
    for i in range(len(action_labels)):
        for j in range(len(feature_names)):
            ax.text(j, i, f"{cartpole_q_w[i, j]:.2f}",
                    ha="center", va="center", color="black")
    # Add labels and title
    ax.set_title('Feature Weights Heatmap')
    ax.set_ylabel('Actions')
    ax.set_xlabel('Features')
    # Adjust layout for better spacing
    fig.tight_layout()
    plt.show()
    
def visualize_q_weight_policy(env, w, total_steps: int = 500):
    obs = env.reset()
    episode_frames = []
    q_vals = (np.expand_dims(obs, axis=0) * w).sum(axis=-1)
    episode_return = 0
    gamma_prev = 1.0
    episode_over = False

    for _ in range(min(total_steps, 500)):
        episode_frames.append(capture_cartpole_frame(env))
        action = np.argmax(q_vals)
        obs, reward, terminated, truncated = env.step(action)
        episode_return += gamma_prev * reward
        gamma_prev *= env.gamma
        q_vals = (np.expand_dims(obs, axis=0) * w).sum(axis=-1)
        episode_over = terminated or truncated

        if episode_over:
            break

    display_cartpole_gif(episode_frames, episode_over=episode_over)
    print("Visualization Ended.")
    return [episode_return]

def visualize_random(env, total_steps: int = 500):
    env.reset()
    episode_frames = []
    episode_over = False

    for _ in range(min(total_steps, 500)):
        episode_frames.append(capture_cartpole_frame(env))
        action = env.action_space.sample()
        _, _, terminated, truncated = env.step(action)
        episode_over = terminated or truncated

        if episode_over:
            break

    display_cartpole_gif(episode_frames, episode_over=episode_over)
    print("Visualization Ended.")
    
def evaluate_policy(env, w, num_episodes: int = 1000, total_steps: int = 1000):
    cumulative_reward_sum = 0
    steps_lasted_sum = 0
    
    for _ in range(num_episodes):
        obs = env.reset()
        cumulative_reward = 0
        terminated = False
        truncated = False
        
        for step in range(total_steps):
            q_vals = (np.expand_dims(obs, axis=0) * w).sum(axis=-1)
            action = np.argmax(q_vals)
            obs, reward, terminated, truncated = env.step(action)
            cumulative_reward += reward
            
            if terminated or truncated:
                break
        
        cumulative_reward_sum += cumulative_reward
        steps_lasted_sum += (step + 1)
    
    average_cumulative_reward = cumulative_reward_sum / num_episodes
    average_steps_lasted = steps_lasted_sum / num_episodes
    
    return average_cumulative_reward, average_steps_lasted

def plot_evaluated_policy(q_learning, unwrapped_cartpole_env, phi):
    num_samples_list = [100, 1000, 1e4, 1e5, 5e5, 1e6]
    alpha = 0.005
    epsilon=0.1


    cumulative_rewards = []
    steps_lasted_list = []

    for num_samples in num_samples_list:
        unwrapped_cartpole_env = gym.make('CartPole-v1')
        cartpole_env = FeaturizedCartPole(unwrapped_cartpole_env, phi)
        
        w, _ = q_learning(cartpole_env, alpha=alpha, epsilon=epsilon, n_samples=num_samples)
        
        average_cumulative_reward, average_steps_lasted = evaluate_policy(cartpole_env, w)
        
        cumulative_rewards.append(average_cumulative_reward)
        steps_lasted_list.append(average_steps_lasted)
        
        print(f"Final result for {num_samples} samples: Avg. Reward = {average_cumulative_reward}, Avg. Steps Lasted = {average_steps_lasted}")

    plt.figure(figsize=(10, 6))
    plt.plot(num_samples_list, cumulative_rewards, marker='o')
    plt.xscale('log')
    plt.xlabel('Number of Samples')
    plt.ylabel('Average Cumulative Reward')
    plt.title('Average Cumulative Reward vs. Number of Training Samples')
    plt.grid(True)
    plt.show()