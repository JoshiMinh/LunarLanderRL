import torch
import numpy as np
import time
from core.game import VastSpaceLander
from core.agent import DQNAgent

def run_demo(checkpoint_path='models/checkpoint.pth'):
    """Run a demo of the trained agent."""
    env = VastSpaceLander(render_mode='human')
    env.max_episode_steps = 5000 # Allow full mission duration
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    agent = DQNAgent(state_size=state_size, action_size=action_size, seed=0, device=device)
    
    # Load weights
    if torch.cuda.is_available():
        agent.qnetwork_local.load_state_dict(torch.load(checkpoint_path))
    else:
        agent.qnetwork_local.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))

    for i in range(10): # Run 10 episodes
        state, _ = env.reset()
        score = 0
        for t in range(env.max_episode_steps):
            action = agent.act(state, eps=0.0) # Greedy action
            state, reward, terminated, truncated, info = env.step(action)
            score += reward
            time.sleep(0.01) # Faster demo
            
            if getattr(env, 'user_quit', False):
                print("\n[Q] Pressed - Force Shutting Down...")
                env.close()
                import sys; sys.exit(0)
                
            if terminated or truncated:
                status = info.get('mission_status', 'failed')
                color = "\033[92m" if status == 'success' else "\033[91m"
                reset = "\033[0m"
                print(f"Episode {i+1} | Status: {color}{status.upper()}{reset} | Reward: {score:.2f}")
                break 

    env.close()

if __name__ == "__main__":
    import os
    try:
        if os.path.exists('models/checkpoint.pth'):
            run_demo()
        else:
            print("Model checkpoint not found. Please train the agent first using train.py.")
    except KeyboardInterrupt:
        print("\n[!] Demo interrupted by user. Exiting gracefully...")
