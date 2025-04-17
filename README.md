# Jackbot Lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

## Overview

This repository contains a robotics project for training a bipedal robot called Jackbot using reinforcement learning. It features a reward tuning interface and training scripts for both flat and rough terrain locomotion.

**Key Features:**

- Interactive reward weight tuning interface using Streamlit 
- Support for both flat and rough terrain locomotion tasks
- Configurable training parameters and environments
- Real-time training visualization and playback options

## Installation

1. Prerequisites:
- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). We recommend using the conda installation as it simplifies calling Python scripts from the terminal.
- Install streamlit
```bash
pip install streamlit
```

2. Install the package:
```bash
python -m pip install -e source/jackbot_lab
```

3. Verify installation by running:
```bash 
cd scripts/rsl_rl
streamlit run reward_tuner.py
```

## Usage

### Reward Tuning

The reward tuner interface allows you to:

- Adjust weights for different reward components:
  - Velocity tracking
  - Root penalties  
  - Joint penalties
  - Feet rewards
  - Clock-based rewards
  - Gait rewards

- Train models with different configurations:
  - Flat terrain vs rough terrain
  - Number of environments
  - Training iterations
  - Logging options (tensorboard, wandb, neptune)

### Training

Run training with:

```bash
python scripts/rsl_rl/train.py --task=Isaac-Velocity-Flat-Jackbot-v0 --num_envs=4096
```

Available tasks:
- `Isaac-Velocity-Flat-Jackbot-v0`: Flat terrain locomotion
- `Isaac-Velocity-Rough-Jackbot-v0`: Rough terrain locomotion

### Playback

Visualize trained models:

```bash
python scripts/rsl_rl/play.py --task=Isaac-Velocity-Flat-Jackbot-v0 --checkpoint=PATH_TO_CHECKPOINT
```

## Project Structure

- `jackbot_lab/tasks/locomotion/velocity/`: Core locomotion task implementation
  - `config/jackbot/`: Configuration files and reward definitions
  - `mdp/`: MDP components (rewards, events, etc.)
- `jackbot_lab/assets/`: Robot configuration and USD files
- `scripts/rsl_rl/`: Training and visualization scripts
