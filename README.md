# Reinforcement Learning, Sim-to-Real techniques and Forward-Forward

**(Developed for the Machine Learning and Deep Learning 2024 course at Politecnico di Torino)**

[Read the paper](paper.pdf)

## About

This repository contains experiments on:

- REINFORCE
- Actor-Critic
- Soft Actor-Critic
- Uniform Domain Randomization
- Forward-Forward

## Project structure

The files in this repository can be divided in:

- `env` folder: code relating the `CustomHopper` environment
- `agent_*.py`: code defining the agents (each policy is deployed by the corresponding `train_` script)
- `train_*.py`: code to test and evaluate the corresponding policy
- `plot_results.py`: make the plots showed in the paper

N.B.: we used the implementation of SAC by Stable-Baselines-3, so there is not an `agent_` file for SAC.

## How to run

1. (recommended) Create a new conda environment with python 3.7
2. Install `requirements.txt`
3. Install MuJoCo 2.1 and the Python Mujoco interface

For the full guide follow the steps provided by the [project template](https://github.com/gabrieletiboni/mldl_2024_template).

## Authors

- Alessandro Arneodo
- Daniele Ercole
- Davide Fassio
