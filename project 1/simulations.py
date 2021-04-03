import random as rd
from markovDecision import markovDecision
import time
import numpy as np
import re

def result(dice):
    """result.
    Returns the result of the thrown dice
    :param dice: integer representing the type of the thrown dice
                    0 : security dice
                    1 : normal dice
                    2 : risky dice
                
    :return: integer, result of the thrown dice
    """
    if dice == 0:                   # Security Dice 
        return rd.randint(0, 1)
    elif dice == 1 :                # Normal Dice
        return rd.randint(0, 2)
    elif dice == 2 :                # Risky Dice
        return rd.randint(0, 3)
    else : 
        print(dice)
        raise NotImplementedError()

def trap(case, layout):
    """trap.
    Returns the case the player goes to after activating a trap

    :param case:    integer refering to the case the player is on 
    :param layout:  see simulations specification

    :return: integer refering to the case the player goes to after the trap has been activated
    """
    freeze = False
    if layout[case] == 1 :              # Restart Trap
        next_case = 0
    elif layout[case] == 2 :            # Penalty Trap
        if case < 3 or case == 10 : 
            next_case = 0
        elif case == 11 or case == 12 : 
            next_case = case - 10
        else : 
            next_case = case - 3
    elif layout[case] == 3 :            # Prison Trap
        next_case = case
        freeze = True
    elif layout[case] == 4 :            # Gamble Trap
        return rd.randint(0, 14), False
    else : 
        raise NotImplementedError()
    return next_case, freeze


def play(case, dice_result, layout, circle, dice):
    """play.
    Returns the position the player will be after the turn, taking into account the dice throw and the traps.

    :param case:        integer referring to the active case
    :param dice_result: integer representing the result of the thrown dice
    :param layout:      see simulations specification
    :param circle:      see simulations specification
    :param dice:        integer representing the dice that has been thrown

    :return next_case:  integer representing the case the player is on after this turn
    :return freeze:     boolean value, which if True means that the player is frozen for one turn
    """
    freeze = False
    ###   FIND THE NEXT CASE BEFORE TRAPS   ###
    if dice_result == 0 :                                   # No changes
        next_case = case
    elif case == 2 :                                        # Junction FAST and SLOW lane 
        FastLane = rd.randint(0, 1)
        if FastLane == 1 : 
            next_case = 9 + dice_result
        else : 
            next_case = case + dice_result
    elif case == 7 and dice_result == 3 :                   # Jump from 9 to 14 during turn
        next_case = 14
    elif case == 8 and dice_result == 2 :                   # Jump from 9 to 14 during turn
        next_case = 14
    elif (case == 9 or case == 13) and dice_result == 1 :   # Jump from 9 to 14 during turn  
        next_case = 14
    elif case == 9 or case == 13 :                          # Circle cases for 9 & 13
        if circle: 
            if dice_result == 2 : 
                next_case = 0 
            elif dice_result == 3 : 
                next_case = 1
        else : 
            next_case = 14
    elif (case == 8 or case == 12) and dice_result == 3 :   # Circle cases for 8 & 12
        if circle : 
            next_case = 0
        else : 
            next_case = 14
    else :                                                  # General case
        next_case = case + dice_result

    ###   TAKING TRAPS INTO ACCOUNT   ###
    if layout[next_case] != 0 :                                 # There is a Trap on the case the player went on 
        if dice == 2 :                                          # Risky Dice -> all traps are activated
            next_case, freeze = trap(next_case, layout)
        elif dice == 1 :                                        # Normal Dice -> traps are activated half of the time
            TrapActivated = rd.randint(0, 1)
            if TrapActivated == 1 : 
                next_case, freeze = trap(next_case, layout)
    return next_case, freeze

def whatToDo(case, optimal_policy, strategy):
    """whatToDo.
    Returns the number of the dice that will be thrown regarding the strategy used

    :param case:            integer referring to the active case
    :param optimal_policy:  vector of integers containing the optimal policy computed by the MDP process
    :param strategy: a string containing the strategy the simulations will be running
                    'optimal'           the choice of the dice will be the one chosen by de mdp process
                    'security_only'     the choice of the dice will always be the security dice
                    'normal_only'       the choice of the dice will always be the normal dice
                    'risky_only'        the choice of the dice will always be the risky dice
                    ... (to be completed)     

    :return: integer representing the number of the dice to be thrown    
    """
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
    elif strategy.startswith('optimal_with_random_') and strategy[-1].isdigit():
        random_percentile = int(re.search(r'\d+', strategy).group())/100.0
        if rd.random() >= random_percentile : 
            return optimal_policy[case]
        else:
            return rd.randint(0, 2)
    elif strategy.startswith('optimal_with_security_') and strategy[-1].isdigit():
        random_percentile = int(re.search(r'\d+', strategy).group())/100.0
        if rd.random() >= random_percentile : 
            return optimal_policy[case]
        else:
            return 0
    elif strategy.startswith('optimal_with_normal_') and strategy[-1].isdigit():
        random_percentile = int(re.search(r'\d+', strategy).group())/100.0
        if rd.random() >= random_percentile : 
            return optimal_policy[case]
        else:
            return 1
    elif strategy.startswith('optimal_with_risky_') and strategy[-1].isdigit():
        random_percentile = int(re.search(r'\d+', strategy).group())/100.0
        if rd.random() >= random_percentile : 
            return optimal_policy[case]
        else:
            return 2

        

    # to fill in with all the other strategies


def simulations(layout, circle, N_SIMU, strategy):
    """simulations.
    Returns the cost from each case to the destination square, based on empirical testing

    :param layout: a vector of type numpy.ndarray that represents the layout of the game, containing 15 values representing the 15 squares of the Snakes and Ladders game
                    layout[i]  = 0 if it is an ordinary square
                               = 1 if it is a “restart” trap (go back to square 1)
                               = 2 if it is a “penalty” trap (go back 3 steps)
                               = 3 if it is a “prison” trap (skip next turn)
                               = 4 if it is a “gamble” trap (random teleportation)
    :param circle: a boolean variable (type bool), indicating if the player must land exactly on the final, goal, square 15 to win (circle = True) or still wins by overstepping the final square (circle = False).
    :param N_SIMU: integer indicating the number of simulations to be run
    :param strategy: a string containing the strategy the simulations will be running
                    'optimal'           the choice of the dice will be the one chosen by de mdp process
                    'security_only'     the choice of the dice will always be the security dice
                    'normal_only'       the choice of the dice will always be the normal dice
                    'risky_only'        the choice of the dice will always be the risky dice
                    ... (to be completed)

    :return cost: vector containing the cost from each case to the destination, computed as the mean of all simulations run

    """
    # PRELIMINARIES 
    optimal_dice_to_throw = markovDecision(layout, circle)[1]       # Computing optimal policy
    cost = [0]*14                                                   # Init the cost vector
    for case in range(len(cost)):                                   # Iterate on all cases
        cost_to_case = 0                                            
        for i in range(N_SIMU):                                     # Run the process N_SIMU times
            this_case = case
            cost_count = 0.0
            while(this_case != 14) :                                                                # The process ends when the destination square is reached
                dice_to_throw = whatToDo(this_case, optimal_dice_to_throw, strategy)                
                dice_result = result(dice_to_throw)
                next_case, freeze = play(this_case, dice_result, layout, circle, dice_to_throw)
                if freeze :                                                                         # Add extra turn if freeze
                    cost_count += 1.0
                cost_count += 1.0   
                this_case = next_case
            cost_to_case += cost_count                              # Add cost from one simulation to total
        cost_to_case = cost_to_case/N_SIMU                          # Compute mean of all simulations run
        cost[case] = cost_to_case
    return cost



###   TESTING CODE   ###
layout=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
circle = True
N_SIMU = 10000
strategy = 'optimal_with_risky_10'
"""
#UNCOMMENT FOR RANDOM LAYOUT
for i in range(len(layout)) : 
    add_trap = rd.randint(0, 2)
    if add_trap == 1 : 
        layout[i] = rd.randint(1, 4)
"""



print('--------------------------------------')
print('Layout           : ', layout)
mdp_expec, dice_choice = markovDecision(layout, circle)
print('MDP dice choice  : ', dice_choice)
print('MDP expectation  : ', mdp_expec)
print('Empirical values : ', simulations(layout, circle, N_SIMU, strategy))
