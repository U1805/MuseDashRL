# MuseDashRL

A Deep Q-Learning (DQN) agent that learns to play Muse Dash. This project uses computer vision and reinforcement learning to create an AI that can play the rhythm game Muse Dash.

<center>

![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-red)
![OpenCV](https://img.shields.io/badge/OpenCV-4.11.0-green)
![Game](https://img.shields.io/badge/Game-MuseDash-pink)

![Status](https://img.shields.io/badge/Status-In_Development-yellow)
![License](https://img.shields.io/badge/License-MIT-blue)

</center>

## Overview

This project implements a Deep Q-Network (DQN) that learns to play Muse Dash by:
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

### Basic Actions
- UP (F key)
- DOWN (J key) 
- FORWARD (no key)
- BOTH (F+J keys)

### Advanced Actions **⏳TODO**
- HOLD_UP_TAP_DOWN (F key held + J key tap)
- HOLD_DOWN_TAP_UP (J key held + F key tap)
- RAPID_BOTH (F+J keys rapid press)


## Setup and Usage

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Make sure Muse Dash is installed, then launch the game on 1920*1080 and select the song that randomly play.
Stop at the difficulty selection screen.

3. Run the training script:
   ```bash
   python dqn.py
   ```

4. For testing the trained model:
   ```bash
   python test.py
   ```

## License

[LICENSE](./LICENSE)
