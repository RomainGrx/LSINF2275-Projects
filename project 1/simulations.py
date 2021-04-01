import random as rd
from markovDecision import markovDecision
import time
import numpy as np

def result(dice):
    if dice == 0:
        return rd.randint(0, 1)
    elif dice == 1 : 
        return rd.randint(0, 2)
    elif dice == 2 : 
        return rd.randint(0, 3)
    else : 
        raise NotImplementedError()

def trap(case, layout):
    freeze = False
    if layout[case] == 1 : # restart trap
        next_case = 0
    elif layout[case] == 2 : # go back 3 steps
        if case < 3 or case == 10 : 
            next_case = 0
        elif case == 11 or case == 12 : 
            next_case = case - 10
        else : 
            next_case = case - 3
    elif layout[case] == 3 : # freeze
        next_case = case
        freeze = True
    elif layout[case] == 4 : # random teleportation
        next_case == rd.randint(0, 14)
    return next_case, freeze


def play(case, dice_result, layout, circle, dice):
    freeze = False
    # Find the next case
    if dice_result == 0 :                       # same case as before
        next_case = case
    elif case == 2 :                            # junction fast and slow lane -> 1/2 chance to go either way
        FastLane = rd.randint(0, 1)
        if FastLane == 1 : 
            next_case = 9 + dice_result
        else : 
            next_case = case + dice_result
    elif case == 7 and dice_result == 3 :       # jump from 9 to 14 exactly
        next_case = 14
    elif case == 8 and dice_result == 2 :       # jump from 9 to 14 exactly
        next_case = 14
    elif case == 9 and dice_result == 1 :       # jump from 9 to 14 exactly
        next_case = 14
    elif case == 9 or case == 13 :              # circle cases for 9 & 13
        if circle: 
            if dice_result == 2 : 
                next_case = 0 
            elif dice_result == 3 : 
                next_case = 1
        else : 
            next_case = 14
    elif (case == 8 or case == 12) and dice_result == 3 :             # circle cases for 8 & 12
        if circle : 
            next_case = 0
        else : 
            next_case = 14
    else : 
        next_case = case + dice_result
    if layout[next_case] != 0 : #trap
        if dice == 2 : 
            next_case, freeze = trap(next_case, layout)
        elif dice == 1 : 
            TrapActivated = rd.randint(0, 1)
            if TrapActivated == 1 : 
                next_case, freeze = trap(next_case, layout)
    return next_case, freeze

def whatToDo(case, optimal_policy, strategy):
    if strategy == 'optimal':
        return optimal_policy[case]
    elif strategy == 'security_only' : 
        return 0
    elif strategy == 'normal_only' : 
        return 1
    elif strategy == 'risky_only' : 
        return 2
    elif strategy == 'random' : 
        return rd.randint(0, 2)
    # to fill in with all the other strategies


def simulations(layout, circle, N_SIMU, strategy):
    global cost_count
    optimal_dice_to_throw = markovDecision(layout, circle)[1]
    cost = [0]*14
    for case in range(len(cost)):
        cost_to_case = 0
        for i in range(N_SIMU):
            this_case = case
            cost_count = 0.0
            while(this_case != 14) : # destination case
                dice_to_throw = whatToDo(this_case, optimal_dice_to_throw, strategy)
                dice_result = result(dice_to_throw)
                next_case, freeze = play(this_case, dice_result, layout, circle, dice_to_throw)
                if freeze : 
                    cost_count += 1.0
                cost_count += 1.0
                this_case = next_case
            cost_to_case += cost_count # add cost from 1 simulation to total
        cost_to_case = cost_to_case/N_SIMU # compute mean from each simulation
        cost[case] = cost_to_case
    return cost

layout=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
circle = False
N_SIMU = 10000
strategy = 'security_only'
print(markovDecision(layout, circle))
print(simulations(layout, circle, N_SIMU, strategy))
