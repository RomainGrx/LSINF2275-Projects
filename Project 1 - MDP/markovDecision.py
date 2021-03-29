#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux, Martin Draguet, Arno Gueurts
@date : 2021 Mar 24, 15:13:43
@last modified : 2021 Mar 29, 10:23:38
"""

import random
import numpy as np

SLOW_LANE = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14)
FAST_LANE = (0, 1, 2, 10, 11, 12, 13, 14)


class Trap:
    @staticmethod
    def ordinary(position, on_fast_lane):
        return position, False

    @staticmethod
    def restart(position, on_fast_lane):
        return 0, False

    @staticmethod
    def penalty(position, on_fast_lane):
        idx = FAST_LANE.index(position) if on_fast_lane else SLOW_LANE.index(position)
        next_idx = max(0, idx - 3)
        return FAST_LANE[next_idx] if on_fast_lane else SLOW_LANE[next_idx]

    @staticmethod
    def prison(position, on_fast_lane):
        return position, True

    @staticmethod
    def gamble(position, on_fast_lane):
        return random.randint(0, 14), False

    @staticmethod
    def next_position(idx, position, on_fast_lane):
        """next_position.

        :param idx: the indice of the trap
        :param position: the current position
        :param on_fast_lane: if currently on the fast lane

        :return (next_position, freeze): the next position with trap activated and freeze indicating if the next turn is freezed
        """
        return (Trap.ordinary, Trap.restart, Trap.penalty, Trap.prison, Trap.gamble)[
            idx
        ](position, on_fast_lane)


class Dice:
    N = None
    ACTIVATE_TRAP = None
    TRAP_PROBABILITY = None

    @classmethod
    def get_move(cls):
        """get_move.
        :return: the number of moves to do next
        """
        raise NotImplementedError(
            "Abstract method `get_move` of parent class `Dice` not implemented"
        )


class SecurityDice(Dice):
    """SecurityDice
    The “security” dice which can only take two possible values: 0 or 1. It allows you to move forward by 0 or 1 square, with a probability of 1/2. With that dice, you are invincible, which means that you ignore the presence of traps when playing with the security dice.
    """

    N = 1
    ACTIVATE_TRAP = False
    TRAP_PROBABILITY = None

    @classmethod
    def get_move(cls):
        return random.randint(0, 1)


class NormalDice(Dice):
    """NormalDice
    The “normal” dice, which allows the player to move by 0, 1 or 2 squares with a probability of 1/3. If you land on a trapped square using this dice, you have 50 % chance of triggering the trap.
    """

    N = 2
    ACTIVATE_TRAP = True
    TRAP_PROBABILITY = 0.5

    @classmethod
    def get_move(cls):
        return random.randint(0, 2)


class RiskyDice(Dice):
    """RiskyDice
    The “risky” dice which allows the player to move by 0, 1, 2 or 3 squares with a probability of 1/4. However, when playing with the risky dice, you automatically trigger any trap you land on (100 % trigger probability).
    """

    N = 3
    ACTIVATE_TRAP = True
    TRAP_PROBABILITY = 1.0

    @classmethod
    def get_move(cls):
        return random.randint(0, 3)


class Game:
    START, END = 0, 14

    def __init__(self, layout, circle):
        self._layout = layout
        self._circle = circle
        self._pos = 0
        self._on_fast_lane = False

    def _validate_position(self, position):
        """_validate_position.
        Validate the current position `self._pos` regarding to `self._circle`

        :return validated_position: the validated position
        """
        validated_position = (
            position % (Game.END + 1)
            if self._circle
            else np.clip(position, Game.START, Game.END)
        )
        return validated_position

    def next_position(self, position: int, dice: Dice):
        """next_position.
        Return the next position regarding to the arguement `dice`, the current position and the layout.

        :param position: The current position
        :param dice: An instance of Dice
        """
        freeze = False

        # if we are at the branching sqaure (third square), 50% probability to go on the slow lane and 50% probability to go on the fast lane
        if position == 2:
            self._on_fast_lane = random.choice([True, False])

        # get the number of squares when throwing the dice
        dx = dice.get_move()

        # the next position is computed as usual if we are not at the branching square (third square)
        # if we are at the branching square and `dx` > 0, `dx` is apply on the fast lane
        next_pos = (
            position + dx
            if not (self._on_fast_lane and position == 2) or dx == 0
            else 9 + dx
        )
        # > print(f"next pos :: {next_pos}")
        next_pos = self._validate_position(next_pos)

        # activate trap only if :
        #   - the dice is either a risky or a normal dice
        #   - the `next_pos` is not the start or the end square
        if dice.ACTIVATE_TRAP and next_pos not in (Game.START, Game.END):
            # activate the trap with the probability corresponding to the `dice`
            if random.random() < dice.TRAP_PROBABILITY:
                next_pos, freeze = Trap.next_position(self._layout[next_pos], next_pos)
                next_pos = self._validate_position(next_pos)

        return next_pos, freeze


def markovDecision(layout, circle):
    """markovDecision.
    This function launches the Markov Decision Process algorithm to determine the optimal strategy regarding the choice of the dice in the Snakes and Ladders game, using the “value iteration” method.

    :param layout: a vector of type numpy.ndarray that represents the layout of the game, containing 15 values representing the 15 squares of the Snakes and Ladders game
                   layout[i]   = 0 if it is an ordinary square
                               = 1 if it is a “restart” trap (go back to square 1)
                               = 2 if it is a “penalty” trap (go back 3 steps)
                               = 3 if it is a “prison” trap (skip next turn)
                               = 4 if it is a “gamble” trap (random teleportation)
    :param circle: a boolean variable (type bool), indicating if the player must land exactly on the final, goal, square 15 to win (circle = True) or still wins by overstepping the final square (circle = False).

    :return [Expec, Dice]:
        :return Expec: a vector of type numpy.ndarray containing the expected cost (= number of turns) associated to the 14 squares of the game, excluding the goal square. The vector starts at index 0 (corresponding to square 1) and ends at index 13 (square 14)
        :return Dice: a vector of type numpy.ndarray containing the choice of the dice to use for each of the 14 squares of the game (1 for “security” dice, 2 for “normal” dice and 3 for “risky”), excluding the goal square. Again, the vector starts at index 0 (square 1) and ends at index 13 (square 14).
    """
    raise NotImplementedError()
    # return [Expec, Dice]
