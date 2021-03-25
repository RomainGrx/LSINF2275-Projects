#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux, Martin Draguet, Arno Gueurts
@date : 2021 Mar 24, 15:13:43
@last modified : 2021 Mar 25, 14:52:00
"""

class Trap:
    @staticmethod
    def ordinary(position):
        raise NotImplementedError()

    @staticmethod
    def restart(position):
        raise NotImplementedError()

    @staticmethod
    def penalty(position):
        raise NotImplementedError()

    @staticmethod
    def prison(position):
        raise NotImplementedError()

    @staticmethod
    def gamble(position):
        raise NotImplementedError()

    @staticmethod
    def next_position(idx, position):
        """next_position.

        :param idx: the indice of the trap 
        :param position: the current position 

        :return (next_position, freeze): the next position with trap activated and freeze indicating if the next turn is freezed
        """
        return (Trap.ordinary, Trap.restart, Trap.penalty, Trap.prison, Trap.gamble)[
            idx
        ](position)


class Dice:
    N = None
    ACTIVATE_TRAP = None
    TRAP_PROBABILITY = None

    @classmethod
    def get_move(cls):
        """get_move.
        :return: the number of moves to do next
        """
        raise NotImplementedError("Abstract method `get_move` of parent class `Dice` not implemented")


class SecurityDice(Dice):
    """SecurityDice
    The “security” dice which can only take two possible values: 0 or 1. It allows you to move forward by 0 or 1 square, with a probability of 1/2. With that dice, you are invincible, which means that you ignore the presence of traps when playing with the security dice.
    """

    N = 1
    ACTIVATE_TRAP = False
    TRAP_PROBABILITY = 0.0

    # @classmethod
    # def get_move(cls):
    #     raise NotImplementedError()


class NormalDice(Dice):
    """NormalDice
    The “normal” dice, which allows the player to move by 0, 1 or 2 squares with a probability of 1/3. If you land on a trapped square using this dice, you have 50 % chance of triggering the trap.
    """

    N = 2
    ACTIVATE_TRAP = True
    TRAP_PROBABILITY = 0.5

    @classmethod
    def get_move(cls):
        raise NotImplementedError()


class RiskyDice(Dice):
    """RiskyDice
    The “risky” dice which allows the player to move by 0, 1, 2 or 3 squares with a probability of 1/4. However, when playing with the risky dice, you automatically trigger any trap you land on (100 % trigger probability).
    """

    N = 3
    ACTIVATE_TRAP = True
    TRAP_PROBABILITY = 1.0

    @classmethod
    def get_move(cls):
        raise NotImplementedError()


class Game:
    def __init__(self, layout, circle):
        self._layout = layout
        self._circle = circle
        self._pos = 0

    def _valid_position(self):
        """_valid_position.
        Valid the current position `self._pos` regarding to `self._circle`

        :return: None
        """
        raise NotImplementedError()

    def next_position(self, dice:Dice):
        """next_position.
        Return the next position regarding to the arguement `dice`, the current position and the layout.

        :param dice: An instance of Dice
        """
        raise NotImplementedError()


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
