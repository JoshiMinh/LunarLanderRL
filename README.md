# 🚀 LunarLanderRL: Deep Reinforcement Learning for Precision Descent

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-20B2AA?logo=openai&logoColor=white)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced deep reinforcement learning project featuring a custom **VastSpaceLander** environment and a sophisticated **Dueling Double DQN** agent. This project simulates high-gravity atmospheric descent with strict fuel management and precision landing requirements.

---

## 🌌 The VastSpaceLander Environment

Unlike the standard Gymnasium LunarLander, this environment is built for scale and realism.

### 📐 State Vector (9-Dimensional)
The agent perceives a normalized state vector representing its relationship to the landing pad:
1.  **Horizontal Position**: $x$ coordinate relative to the pad.
2.  **Vertical Position**: $y$ coordinate relative to the pad.
3.  **Horizontal Velocity**: Scaled $v_x$.
4.  **Vertical Velocity**: Scaled $v_y$.
5.  **Angle**: Orientation of the lander in radians.
6.  **Angular Velocity**: Speed of rotation.
7.  **Leg 1 Contact**: Binary (0/1) indicator for left leg touchdown.
8.  **Leg 2 Contact**: Binary (0/1) indicator for right leg touchdown.
9.  **Fuel Level**: Remaining fuel percentage (300 units total).

### ⚙️ Physics & Challenges
-   **High Gravity**: Simulation runs at $-20.0$ gravity (double the standard) for faster, more challenging descents.
-   **Fuel Management**: Engines consume fuel rapidly; running out results in a ballistic descent.
-   **Starship Lander**: A custom "Starship" style polygon with unique aerodynamic properties.
-   **Vast Landscape**: A procedurally generated, rugged lunar surface that requires precision to find the pad.

---

## 🧠 Model Architecture

The agent utilizes a **Dueling Double Deep Q-Network (D3QN)** to decouple state valuation from action advantage, significantly improving stability in complex environments.

### Architectural Features:
-   **Dueling Streams**: Splits the network into a **Value** stream (how good is the state?) and an **Advantage** stream (how much better is this action?).
-   **Double DQN**: Prevents the common "overestimation bias" by using the local network for action selection and the target network for evaluation.
-   **Experience Replay**: A buffer of $10^5$ steps to break temporal correlations.
-   **Soft Updates**: Smoothly tracks target weights using $\tau = 0.001$.

```mermaid
graph LR
    S[9D State Input] --> H1[Dense 128 + ReLU]
    H1 --> H2[Dense 128 + ReLU]
    H2 --> V[Value Stream V(s)]
    H2 --> A[Advantage Stream A(s,a)]
    V --> Q[Output: Q(s,a) = V + (A - mean(A))]
    A --> Q
```

---

## 🛠️ Getting Started

### 📦 Installation
```bash
# Install dependencies
pip install gymnasium[box2d] torch matplotlib tqdm pandas pygame
```

### 🏋️ Training the Agent
The training script is optimized for headless server environments but works locally. It supports **automatic resuming**: if `models/checkpoint.pth` is found, the agent loads its weights and continues.

```bash
# Train for the recommended 4000 episodes
python train.py --episodes 4000

# Start from scratch (deletes old weights)
python train.py --reset
```

### 🎮 Watching the Demo
Watch the trained agent perform a 10-episode demonstration:
```bash
python main.py
```

---

## 🤖 CI/CD Automated Training

This repository features a **GitHub Actions** workflow that automates the training process. 

-   **Auto-Resume**: On every push to `main`, the runner pulls the latest weights, trains for additional episodes, and pushes the updated model back to the repo.
-   **Manual Reset**: You can manually trigger the "Automated Training & Sync" workflow from the Actions tab with a `reset` flag to start training from scratch on GitHub's servers.

---

## 📝 License
This project is licensed under the MIT License. Developed for Reinforcement Learning research and demonstration.
