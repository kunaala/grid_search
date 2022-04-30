# Dynamic programming for Grid Search

## Overview
implements dynammic programming for the Door-Key problems.
<p align="center">
<img src="gif/doorkey.gif" alt="Door-key Problem" width="500"/></br>
</p>

There are 7 test scenes 

| doorkey-5x5-normal |
|:----------------:|
| <img src="imgs/doorkey-5x5-normal.png"> |

| doorkey-6x6-normal   | doorkey-6x6-direct | doorkey-6x6-shortcut |
|:----------------:|:------------------:|:----------------:|
| <img src="imgs/doorkey-6x6-normal.png"> | <img src="imgs/doorkey-6x6-direct.png" > |<img src="imgs/doorkey-6x6-shortcut.png" >|

| doorkey-8x8-normal   | doorkey-8x8-direct | doorkey-8x8-shortcut |
|:----------------:|:------------------:|:----------------:|
| <img src="imgs/doorkey-8x8-normal.png"> | <img src="imgs/doorkey-8x8-direct.png" > |<img src="imgs/doorkey-8x8-shortcut.png" >|

## Installation

- Install [gym-minigrid](https://github.com/maximecb/gym-minigrid)
- Install dependencies
```bash
pip install -r requirements.txt
```

## Instruction

### 1. utils.py
some useful tools in utils.py
- **step()**: Move your agent
- **generate_random_env()**: Generate a random environment for debugging
- **load_env()**: Load the test environments
- **save_env()**: Save the environment for reproducing results
- **plot_env()**: For a quick visualization of your current env, including: agent, key, door, and the goal
- **draw_gif_from_seq()**: Draw and save a gif image from a given action sequence.

