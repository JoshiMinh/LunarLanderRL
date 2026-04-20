# 🚀 LunarLanderRL: Reinforcement Learning (RL) Project

This project focuses on implementing and evaluating the performance of Reinforcement Learning algorithms on the **LunarLander** environment. It is a research-oriented project that bridges standard software engineering constraints with experimental visualization.

---

## 🌎 Origins & Technical Background

### 1. What is Gymnasium?
[Gymnasium](https://gymnasium.farama.org/) is an open-source standard library toolkit used for developing and comparing Reinforcement Learning algorithms. It provides a standardized interface so that an "Agent" (Artificial Intelligence) can interact with an "Environment" (Simulated Environment).

### 2. Why `LunarLander-v2`?
`LunarLander-v2` is a spacecraft landing simulation environment developed by OpenAI.
*   **Legacy**: This is a fine-tuned version (v2) designed to ensure that physics rules and scoring systems are maximally stable.
*   **Technical Contract**: In programming, the ID `LunarLander-v2` is a hardcoded identifier. Keeping this exact ID is strictly required so that the Gymnasium library can properly initialize international standard environment parameters.

---

## 🏗️ Hybrid Architecture (Optimized for Projects)

The project leverages a **Hybrid** model to meet stringent academic structural standards:

1.  **Core Engine (`.py` files)**: Houses the central logic (Modularization). Separating the code into purely distinct Python files keeps the source clean, easy to debug, and demonstrates professional project organization.
2.  **Experimental Sandbox (`.ipynb` files)**: Utilizes Jupyter Notebooks to conduct experiments, plot comparative graphs (2D Graphs), and store training results. This acts as the most critical "visual report" aspect to convince reviewers.
3.  **Live Showcase (UI/Game)**: Integrates immediate active rendering modes to demonstrate the agent performing at real-time speeds, simulating the experience of a complete video game.

---

## 📊 Environment Parameters

| Component | Detailed Characteristic |
| :--- | :--- |
| **Environment** | `LunarLander-v2` (Mandatory identifier) |
| **State Vector (8)** | Coordinates (X, Y), Velocity (X, Y), Angle, Angular Velocity, Leg Contacts. |
| **Action Space (4)** | Discrete: Do Nothing, Fire Left Engine, Fire Main Engine, Fire Right Engine. |
| **Reward Function** | +100 for successful landing, -100 for crashing, minor penalties for fuel usage. |

---

## 🧠 Implemented Algorithms (Models)

The project compares 3 Deep Q-Network architectures:
*   **Vanilla DQN**: The original deep Q-learning network.
*   **Double DQN**: Mitigates the overestimation error by decoupling the action selection network from the evaluation network.
*   **Dueling DQN**: Splits the network into two independent streams: State Value $V(s)$ and Action Advantage $A(s, a)$.

---

## 📁 Proposed Folder Structure

```bash
LunarLanderRL/
├── core/               # Reusable logic components
│   ├── game.py         # Gymnasium environment orchestrator wrapper
│   ├── constants.py    # UI variables and polygon vectors
│   ├── renderer.py     # Canvas graphics and interface handling
│   ├── terrain.py      # Map generation math routines
│   ├── model.py        # Neural Network architectures (PyTorch)
│   ├── agent.py        # DQN, Double DQN, Dueling DQN logic
│   └── memory.py       # Experience Replay queue buffering
├── experiments/        # Experiments & Charts
│   └── comparison.ipynb # 3 Algorithms Comparison & 2D Graph Plotting
├── main.py             # Live demonstration script (Live Render)
├── train.py            # Headless ML agent trainer 
├── models/             # Contains completed weight (.pth) saves
└── README.md           # Detailed Project Documentation
```

---

## 🚀 Deployment Pipeline (Step-by-Step)

### Step 1: Environment Preparation
Install the required toolkits and frameworks:
```bash
pip install gymnasium[box2d] torch matplotlib tqdm pandas pygame
```

### Step 2: Establish Core Logic
Build the `DQN` and `DoubleDQN` classes within the `core/` directory. Always check for graphical compute capability:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Step 3: Train & Analyze
Launch `train.py` continuously, or use Jupyter to handle experiments with isolated variables. Utilize `matplotlib` to graph the Reward and Loss trajectories across historical episodes to determine learning efficacy.

### Step 4: Live Demo Presentation
Launch `main.py` directly to observe the best-performing Agent showcasing its complex atmospheric landing trajectories:
```python
# Launch the demo runner
python main.py
```

---

## 📝 License
This project was implemented for the purposes of academic research and demonstration.
