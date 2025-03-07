# MuseDashRL

A Reinforcement Learning agent that learns to play Muse Dash. This project uses computer vision and reinforcement learning to create an AI that can play the rhythm game Muse Dash.

![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-red)
![OpenCV](https://img.shields.io/badge/OpenCV-4.11.0-green)
![Game](https://img.shields.io/badge/Game-MuseDash-pink)

![Status](https://img.shields.io/badge/Status-In_Development-yellow)
![License](https://img.shields.io/badge/License-MIT-blue)

## Overview

This project implements a Deep Q-Network (DQN) and Proximal Policy Optimization(PPO) that learns to play Muse Dash by:
- Capturing game screen to understand the game state in real-time
- Making decisions about which actions to take (up, down, forward, or both)
- Learning from the rewards obtained through gameplay

## Key Components

### Environment (`env.py`)
- Handles game interaction and state management
- Processes screen captures
- Manages action execution
- Tracks game state (score, combo, crash status)

### DQN Agent (`dqn.py`)
- Implements the Deep Q-Learning algorithm
- Neural network for action value prediction
- Experience replay for training stability
- Epsilon-greedy exploration strategy
> network code from [Pytorchasaurus Rex](https://github.com/Lumotheninja/dino-reinforcement-learning/tree/master)

### PPO Agent (`ppo.py`)
- Implements the Proximal Policy Optimization algorithm
- Actor-Critic architecture for policy and value estimation
- Generalized Advantage Estimation (GAE)
- Clipped surrogate objective
- Entropy bonus for exploration
> network code from [Edan Meyer](https://colab.research.google.com/drive/1MsRlEWRAk712AQPmoM9X9E6bNeHULRDb?usp=sharing#scrollTo=RZOgKa5nzG5Y)

### Basic Actions
- UP (F key)
- DOWN (J key) 
- FORWARD (no key)
- BOTH (F+J keys)

## Setup and Usage

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Make sure Muse Dash is installed, then launch the game on 1920*1080 and select a song.
Stop at the difficulty selection screen.

3. Run the training script: ⏳DOING
   ```bash
   python dqn.py
   ```

4. For testing the trained model:
   ```bash
   python test.py
   ```

## License

[LICENSE](./LICENSE)
