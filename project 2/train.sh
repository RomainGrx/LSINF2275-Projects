#!/bin/bash

case $1 in 
    "ppo")
        rllib train.py -f humanoid-ppo-gae.yaml
        ;;
esac
