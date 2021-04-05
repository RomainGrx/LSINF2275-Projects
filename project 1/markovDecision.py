#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux, Martin Draguet, Arno Gueurts
@date : 2021 Mar 24, 15:13:43
@last modified : 2021 Apr 05, 11:37:54
"""

import random
import numpy as np

SLOW_LANE = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14)
FAST_LANE = (0, 1, 2, 10, 11, 12, 13, 14)


class Trap:
    """Trap.
    Trap contains all traps with their behaviors.
    """

    @staticmethod
    def ordinary(position: int):
        """ordinary.
        Stay on the same square without any trap

        :param position: the position for which we want to activate the trap
        """
        return position, False

    @staticmethod
    def restart(position: int):
        """restart.
        Immediately teleports the player back to the first square.

        :param position: the position for which we want to activate the trap
        """
        return 0, False

    @staticmethod
    def penalty(position: int):
        """penalty.
        Immediately teleports the player 3 squares backwards.

        :param position: the position for which we want to activate the trap
        """
        on_fast_lane = 10 <= position <= 13
        idx = FAST_LANE.index(position) if on_fast_lane else SLOW_LANE.index(position)
        next_idx = max(0, idx - 3)
        return FAST_LANE[next_idx] if on_fast_lane else SLOW_LANE[next_idx], False

    @staticmethod
    def prison(position: int):
        """prison.
        The player must wait one turn before playing again.

        :param position: the position for which we want to activate the trap
        """
        return position, True

    @staticmethod
    def gamble(position: int):
        """gamble.
        Randomly teleports the player anywhere on the board1, with equal, uniform, probability.

        :param position: the position for which we want to activate the trap
        """
        return random.randint(0, 14), False

    @staticmethod
    def next_position(trap_idx, position: int):
        """next_position.
        Get the next position for `position` when we activate the trap number `trap_idx`

        :param trap_idx: the indice of the trap
        :param position: the position for which we want to activate the trap

        :return (next_position, freeze): the next position with trap activated and freeze indicating if the next turn is freezed
        """
        assert 0 <= trap_idx <= 4, f"index {trap_idx} not in the bounds [0,4]"
        return (Trap.ordinary, Trap.restart, Trap.penalty, Trap.prison, Trap.gamble)[
            trap_idx
        ](position)


class Dice:
    """Dice.
    The global dice class.
    """

    N = None
    ACTIVATE_TRAP = None
    TRAP_PROBABILITY = None
    MIN_DX, MAX_DX = None, None

    @classmethod
    def get_move(cls):
        """get_move.
        :return: the number of moves to do next
        """
        return random.randint(cls.MIN_DX, cls.MAX_DX)

    @classmethod
    def p(cls):
        """p.
        :return: the probability of throwing a particular move.
        """
        return 1 / (cls.MAX_DX - cls.MIN_DX + 1)


class SecurityDice(Dice):
    """SecurityDice
    The “security” dice which can only take two possible values: 0 or 1. It allows you to move forward by 0 or 1 square, with a probability of 1/2. With that dice, you are invincible, which means that you ignore the presence of traps when playing with the security dice.
    """

    N = 0
    ACTIVATE_TRAP = False
    TRAP_PROBABILITY = 0.0
    MIN_DX, MAX_DX = 0, 1


class NormalDice(Dice):
    """NormalDice
    The “normal” dice, which allows the player to move by 0, 1 or 2 squares with a probability of 1/3. If you land on a trapped square using this dice, you have 50 % chance of triggering the trap.
    """

    N = 1
    ACTIVATE_TRAP = True
    TRAP_PROBABILITY = 0.5
    MIN_DX, MAX_DX = 0, 2


class RiskyDice(Dice):
    """RiskyDice
    The “risky” dice which allows the player to move by 0, 1, 2 or 3 squares with a probability of 1/4. However, when playing with the risky dice, you automatically trigger any trap you land on (100 % trigger probability).
    """

    N = 2
    ACTIVATE_TRAP = True
    TRAP_PROBABILITY = 1.0
    MIN_DX, MAX_DX = 0, 3


class SnakesAndLadders:
    """SnakesAndLadders.
    Use the same structure as gym environments :: https://gym.openai.com/
    """

    START, END = 0, 14
    N_STATE, N_ACTION = 15, 3
    ACTIONS = (SecurityDice, NormalDice, RiskyDice)

    def __init__(self, layout, circle):
        self._layout = layout
        self._circle = circle
        self.reset()

    def reset(self):
        """reset.
        Reset the game.
        """
        self._pos = 0
        self._on_fast_lane = False

    @staticmethod
    def _validate_position(position, circle):
        """_validate_position.
        Validate the current position `self._pos` regarding to `self._circle`

        :return validated_position: the validated position
        """
        validated_position = (
            position % (SnakesAndLadders.END + 1)
            if circle
            else np.clip(position, SnakesAndLadders.START, SnakesAndLadders.END)
        )
        return validated_position

    def seed(self, seed):
        """seed.
        Seed every random process.

        :param seed: the number of the seed.
        """
        random.seed(seed)
        np.random.seed(seed)

    def step(self, dice: Dice):
        """step.
        Return the next position regarding to the arguement `dice`, the current position and the layout.

        :param dice: An instance of Dice
        """
        freeze = False
        self._on_fast_lane = True if 10 <= self._pos <= 13 else False

        # if we are at the branching sqaure (third square), 50% probability to go on the slow lane and 50% probability to go on the fast lane
        if self._pos == 2:
            self._on_fast_lane = random.choice([True, False])

        # get the number of squares when throwing the dice
        dx = dice.get_move()

        # the next position is computed as usual if we are not at the branching square (third square)
        # if we are at the branching square and `dx` > 0, `dx` is apply on the fast lane
        next_pos = (
            self._pos + dx
            if not (self._on_fast_lane and self._pos == 2) or dx == 0
            else 9 + dx
        )
        not_trapped_position = next_pos = SnakesAndLadders._validate_position(
            next_pos, self._circle
        )

        # activate trap only if :
        #   - the dice is either a risky or a normal dice
        #   - the `next_pos` is not the start or the end square
        if dice.ACTIVATE_TRAP and next_pos not in (
            SnakesAndLadders.START,
            SnakesAndLadders.END,
        ):
            # activate the trap with the probability corresponding to the `dice`
            if random.random() < dice.TRAP_PROBABILITY:
                next_pos, freeze = Trap.next_position(self._layout[next_pos], next_pos)
                next_pos = SnakesAndLadders._validate_position(next_pos, self._circle)

        self._pos = next_pos
        done = self._pos == SnakesAndLadders.END
        reward = 0.0
        info = dict(freeze=freeze, not_trapped_position=not_trapped_position)
        return next_pos, reward, done, info


class Strategy:
    """Strategy.
    The global strategy class.
    """

    def __init__(self, layout, circle):
        self._layout = layout
        self._circle = circle
        self._env = SnakesAndLadders(layout, circle)

    def run(self):
        abstract


class ChosenDices(Strategy):
    def __init__(self, layout, circle, dices):
        super().__init__(layout, circle)
        self._dices = dices

    def run(self):
        """run.
        Run the whole strategy.
        """
        done = False
        turn = 0
        while not done:
            dice = random.choice(self._dices)
            _, _, done, info = self._env.step(dice)
            if info["freeze"]:
                turn += 1
            turn += 1
        return turn


class MarkovDecisionProcess(Strategy):
    def __init__(self, layout, circle, theta=1e-9):
        """__init__.

        :param layout: the layout of the map with trap numbers
        :param circle: if the map has a circle path
        :param theta: the theta limit for the max difference of the Q values with the new Q values in the `compute` loop convergence
        """
        super().__init__(layout, circle)
        self.Q = np.random.uniform(
            0, 10, size=(self._env.N_STATE, self._env.N_ACTION)
        )  # The quality matrix containing the expected costs for (state, dice)
        self.Q[-1, :] = 0  # Fix the last square as 0 cost
        # self.Q = np.zeros((self._env.N_STATE, self._env.N_ACTION))
        self._theta = theta

    def policy(self, Q):
        """policy.
        Return the computed policy for the quality matrix `Q` (argmin of the dices)

        :param Q: the quality matrix containing the exepected costs for (state, dice)
        """
        return np.argmin(Q, axis=-1)

    def V(self, Q):
        """V.
        Return the min values per dice (i.e. the best dice for each state).

        :param Q: the quality matrix containing the exepected costs for (state, dice)
        """
        return np.min(Q, axis=-1)

    def next_states(self, state, action):
        """next_states.
        Return all the next states with their positions, if they are freeze and the probability having those particular next states

        :param state: the state for which we want the next states
        :param action: the action chosen for moving (dice)
        """

        def trap_next_states(position):
            """trap_next_states.
            Return all the next states with their positions, if they are freeze and the probability having those particular next states when the trap is activated

            :param position: the position for which we want the next states
            """
            trap = self._layout[position]
            if (
                trap != 4
            ):  # if not the trap 4, the next possible position is a single square
                next_position, freeze = Trap.next_position(trap, position)
                return [(1.0, freeze, next_position)]
            else:  # else, the trap 4 have 15 next states with probability 1/15
                p = 1 / SnakesAndLadders.N_STATE
                return [(p, False, pos) for pos in range(SnakesAndLadders.N_STATE)]

        def get_next_case(case, dice_result, layout, circle):
            if dice_result == 0:  # No changes
                next_case = case
            elif case == 2:  # Junction FAST and SLOW lane
                FastLane = rd.randint(0, 1)
                if FastLane == 1:
                    next_case = 9 + dice_result
                else:
                    next_case = case + dice_result
            elif case == 7 and dice_result == 3:  # Jump from 9 to 14 during turn
                next_case = 14
            elif case == 8 and dice_result == 2:  # Jump from 9 to 14 during turn
                next_case = 14
            elif (
                case == 9 or case == 13
            ) and dice_result == 1:  # Jump from 9 to 14 during turn
                next_case = 14
            elif case == 9 or case == 13:  # Circle cases for 9 & 13
                if circle:
                    if dice_result == 2:
                        next_case = 0
                    elif dice_result == 3:
                        next_case = 1
                else:
                    next_case = 14
            elif (
                case == 8 or case == 12
            ) and dice_result == 3:  # Circle cases for 8 & 12
                if circle:
                    next_case = 0
                else:
                    next_case = 14
            else:  # General case
                next_case = case + dice_result
            return next_case

        def dice_next_states(position, dice):
            """dice_next_states.
            Return all the next states with their positions, if they are freeze and the probability having those particular next states for the particular `dice`

            :param position: the position for which we want the next states
            """
            p = dice.p()
            next_states = []
            for dx in range(dice.MAX_DX + 1):  # We iterate over all possible moves
                if (
                    position == 2
                ):  # If we are at the branching square (3), split in slow and fast lane with probability 0.5
                    next_states += [
                        (0.5 * p, SLOW_LANE[position + dx]),
                        (0.5 * p, FAST_LANE[position + dx]),
                    ]
                else:  # Else it is just a single position with move `dx`
                    # next_pos = SnakesAndLadders._validate_position(
                    #    position + dx, self._circle
                    # )
                    next_pos = get_next_case(position, dx, self._layout, self._circle)
                    next_states.append((p, next_pos))
            return next_states

        dice = self._env.ACTIONS[action]  # Get the dice instance

        for dice_p, dice_state in dice_next_states(state, self._env.ACTIONS[action]):
            yield (
                1 - dice.TRAP_PROBABILITY
            ) * dice_p, False, dice_state  # First, yield the next states not triggering the trap
            if dice.ACTIVATE_TRAP:
                for trap_p, freeze, next_state in trap_next_states(dice_state):
                    yield dice.TRAP_PROBABILITY * dice_p * trap_p, freeze, next_state  # Then, yield the next states when the trap is triggered

    def compute(self, epochs=10000):
        for e in range(epochs):
            Q = np.zeros_like(self.Q)  # the new Q matrix we will compute
            V = self.V(self.Q)  # the current V vector with cost per state
            # Iterate over every state (k) on the board and every dice (a)
            for state in range(self._env.N_STATE - 1):
                for action in range(self._env.N_ACTION):
                    Q[state, action] = 1.0
                    for p, freeze, next_state in self.next_states(state, action):
                        Q[state, action] += p * (
                            V[next_state] + int(freeze)
                        )  # we add the contribution of every next state (k')

            if (
                np.max(np.abs(self.V(self.Q) - self.V(Q))) < self._theta
            ):  # Break if the max difference of the previous Q with the new Q < theta
                break
            self.Q = Q.copy()  # Replace the old Q matrix with the new computed one

        return self.V(Q)[:-1], self.policy(Q)[:-1]


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
    mdp = MarkovDecisionProcess(layout, circle)
    Expec, Dice = mdp.compute()
    return [Expec, Dice]
