# 🧠 RLHF Implementation Projects

A curated collection of foundational and advanced Reinforcement Learning (RL) and Reinforcement Learning with Human Feedback (RLHF) implementations. This repository covers:

- 🟩 Direct Preference Optimization (DPO)
- 🟦 Deep Q-Networks (DQN)
- 🟧 Monte Carlo vs Temporal Difference (MC vs TD)
- 🟥 Proximal Policy Optimization (PPO)
- 🟨 RLHF Pipeline
- 🟪 Value Iteration (FrozenLake)

---

## 📁 Project Structure

├── DPO_Code/                       # Direct Preference Optimization
├── DQN_Code/                       # Deep Q-Network
├── MCvsTD/                  # Monte Carlo vs Temporal Difference
├── PPO_Code/                       # Proximal Policy Optimization
├── RLHF_Code/                      # Reinforcement Learning with Human Feedback
├── value_iteration_FrozenLake/ # Value Iteration on FrozenLake
├── README.md
└── requirements.txt

---

## 📌 Module Overviews

### 🔹 DPO (Direct Preference Optimization)
- Implements Direct Preference Optimization, a scalable algorithm for aligning models using human preferences.
- 🚀 Key Features:
  - Preference datasets
  - DPO loss vs traditional RLHF approaches
  - Evaluation metrics for preference alignment

### 🔹 DQN (Deep Q-Network)
- Classic deep reinforcement learning algorithm implemented using PyTorch.
- 🧠 Core Concepts:
  - Experience Replay
  - Target Network updates
  - ε-greedy exploration
- 🧪 Tested Environments: `CartPole-v1`, `LunarLander-v2`

### 🔹 MC vs TD (Monte Carlo vs Temporal Difference)
- A pedagogical comparison of Monte Carlo and TD(0) learning techniques.
- 📊 Focus:
  - FrozenLake-v1 environment
  - Value estimation
  - Convergence speed and stability

### 🔹 PPO (Proximal Policy Optimization)
- Policy gradient method that uses a clipped objective for more stable updates.
- 🔍 Details:
  - Generalized Advantage Estimation (GAE)
  - Continuous action environments
  - OpenAI Gym integration

### 🔹 RLHF (Reinforcement Learning with Human Feedback)
- End-to-end RLHF implementation:
  - Pretraining (optional)
  - Synthetic or manual preference generation
  - Reward model training
  - Fine-tuning using PPO or DPO
- 🤝 Human-in-the-loop Simulation Supported

### 🔹 FrozenLake_ValueIteration
- Tabular method using Value Iteration algorithm on FrozenLake.
- 🧮 Includes:
  - Optimal policy visualization
  - Value convergence tracking
  - Heatmaps and trajectories

---

## 📦 Installation

### 🛠 Requirements

```bash
python>=3.8
torch>=1.10
numpy
matplotlib
gym==0.21
scikit-learn
transformers

