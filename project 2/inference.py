#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2021 mei 10, 14:26:35
@last modified : 2021 mei 10, 23:45:53
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import gym

import ray
import yaml
import argparse
from utils import get_latest_ckpt
from ray.rllib.agents import ppo

parser = argparse.ArgumentParser("Launch the gym environnement from the agent")
parser.add_argument("agent_results_path", type=str)
args = parser.parse_args()

if not os.path.exists(args.agent_results_path) or not os.path.isdir(
    args.agent_results_path
):
    raise Error(f"{args.agent_results_path} does not exist or is not a directory")

env_name = "Humanoid-v2"
ray.init(num_cpus=1, num_gpus=0)

config = yaml.safe_load(open("humanoid-ppo-gae.yaml", "r"))["humanoid-ppo-gae"][
    "config"
]

config = {**ppo.DEFAULT_CONFIG.copy(), **config}


agent = ppo.PPOTrainer(env=env_name, config=config)
agent.restore(get_latest_ckpt(args.agent_results_path))

env = gym.make(env_name)

while True:
    obs = env.reset()
    done = False
    while not done:
        env.render()
        action = agent.compute_action(obs)
        obs, reward, done, info = env.step(action)
