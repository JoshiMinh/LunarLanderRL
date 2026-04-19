import os
if "SDL_VIDEODRIVER" not in os.environ:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

import gymnasium as gym
import random
import torch
import numpy as np
import pandas as pd
from collections import deque
from tqdm import tqdm
import time

from core.custom_env import VastSpaceLander
from core.agent import DQNAgent

def train(n_episodes=2000, max_t=2000, eps_start=1.0, eps_end=0.01, eps_decay=0.996, save_path='models/checkpoint.pth', log_path='results/training_log.csv'):
    """
    Cloud-Optimized Deep Q-Learning with CSV logging and headless support.
    """
    env = VastSpaceLander()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    agent = DQNAgent(state_size=state_size, action_size=action_size, seed=0, device=device)
    
    scores = []                        
    scores_window = deque(maxlen=100)  
    eps = eps_start                    
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Initialize Log DataFrame
    history = []

    pbar = tqdm(range(1, n_episodes + 1), desc="Training")
    for i_episode in pbar:
        state, _ = env.reset()
        score = 0
        start_time = time.time()
        
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        
        duration = time.time() - start_time
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)
        
        # Log to list
        history.append({
            'episode': i_episode,
            'score': score,
            'avg_score': np.mean(scores_window),
            'epsilon': eps,
            'duration': duration
        })

        pbar.set_postfix({
            'Avg': f'{np.mean(scores_window):.1f}',
            'Eps': f'{eps:.2f}'
        })
        
        # Periodic Save & Log Dump
        if i_episode % 50 == 0:
            torch.save(agent.qnetwork_local.state_dict(), save_path)
            pd.DataFrame(history).to_csv(log_path, index=False)
            
        if np.mean(scores_window) >= 200.0:
            print(f'\nEnvironment solved in {i_episode:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
            torch.save(agent.qnetwork_local.state_dict(), save_path)
            pd.DataFrame(history).to_csv(log_path, index=False)
            break
            
    return history

if __name__ == "__main__":
    history = train()
    print("Training complete. Log saved to experiments/training_log.csv")
