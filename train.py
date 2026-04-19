import gymnasium as gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from core.custom_env import VastSpaceLander
from core.agent import DQNAgent

def train(n_episodes=2000, max_t=1500, eps_start=1.0, eps_end=0.01, eps_decay=0.995, save_path='models/checkpoint.pth'):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        save_path (str): path to save the model weights
    """
    env = VastSpaceLander()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    agent = DQNAgent(state_size=state_size, action_size=action_size, seed=0, device=device)
    
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    
    # Create models directory if not exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    pbar = tqdm(range(1, n_episodes + 1), desc="Training")
    for i_episode in pbar:
        state, _ = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay * eps) # decrease epsilon
        
        pbar.set_postfix({
            'Avg Score': f'{np.mean(scores_window):.2f}',
            'Epsilon': f'{eps:.2f}'
        })
        
        if i_episode % 100 == 0:
            torch.save(agent.qnetwork_local.state_dict(), save_path)
            
        if np.mean(scores_window) >= 200.0:
            print(f'\nEnvironment solved in {i_episode:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
            torch.save(agent.qnetwork_local.state_dict(), save_path)
            break
            
    return scores

if __name__ == "__main__":
    scores = train()
    
    # Plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('experiments/training_plot.png')
    plt.show()
