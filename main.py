import torch
import numpy as np
import time
from core.custom_env import VastSpaceLander
from core.agent import DQNAgent

def run_demo(checkpoint_path='models/checkpoint.pth'):
    """Run a demo of the trained agent."""
    env = VastSpaceLander(render_mode='human')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    agent = DQNAgent(state_size=state_size, action_size=action_size, seed=0, device=device)
    
    # Load weights
    if torch.cuda.is_available():
        agent.qnetwork_local.load_state_dict(torch.load(checkpoint_path))
    else:
        agent.qnetwork_local.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))

    for i in range(5): # Run 5 episodes
        state, _ = env.reset()
        score = 0
        for t in range(500):
            action = agent.act(state, eps=0.0) # Greedy action
            state, reward, terminated, truncated, _ = env.step(action)
            score += reward
            time.sleep(0.02) # Slow down for visibility
            if terminated or truncated:
                break 
        print(f"Episode {i+1} Reward: {score:.2f}")

    env.close()

if __name__ == "__main__":
    import os
    if os.path.exists('models/checkpoint.pth'):
        run_demo()
    else:
        print("Model checkpoint not found. Please train the agent first using train.py.")
