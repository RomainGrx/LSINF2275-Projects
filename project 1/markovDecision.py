#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux, Martin Draguet, Arno Gueurts
@date : 2021 Mar 24, 15:13:43
@last modified : 2021 Apr 01, 09:31:47
"""

import random
import numpy as np

SLOW_LANE = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14)
FAST_LANE = (0, 1, 2, 10, 11, 12, 13, 14)


class Trap:
    @staticmethod
    def ordinary(position):
        return position, False

    @staticmethod
    def restart(position):
        return 0, False

    @staticmethod
    def penalty(position):
        on_fast_lane = 10 <= position <= 13
        idx = FAST_LANE.index(position) if on_fast_lane else SLOW_LANE.index(position)
        next_idx = max(0, idx - 3)
        return FAST_LANE[next_idx] if on_fast_lane else SLOW_LANE[next_idx], False

    @staticmethod
    def prison(position):
        return position, True

    @staticmethod
    def gamble(position):
        return random.randint(0, 14), False

    @staticmethod
    def next_position(idx, position):
        """next_position.

        :param idx: the indice of the trap
        :param position: the current position

        :return (next_position, freeze): the next position with trap activated and freeze indicating if the next turn is freezed
        """
        assert 0 <= idx <= 4, f"index {idx} not in the bounds [0,4]"
        return (Trap.ordinary, Trap.restart, Trap.penalty, Trap.prison, Trap.gamble)[
            idx
        ](position)


class Dice:
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
        return 1/(cls.MAX_DX-cls.MIN_DX+1)


class SecurityDice(Dice):
    """SecurityDice
    The “security” dice which can only take two possible values: 0 or 1. It allows you to move forward by 0 or 1 square, with a probability of 1/2. With that dice, you are invincible, which means that you ignore the presence of traps when playing with the security dice.
    """

    N = 0
    ACTIVATE_TRAP = False
    TRAP_PROBABILITY = None
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

    def cost(self, state, action, next_state):
        return 1


class Strategy:
    def __init__(self, layout, circle):
        self._layout = layout
        self._circle = circle
        self._env = SnakesAndLadders(layout, circle)

    def run(self):
        raise NotImplementedError()


class ChosenDices(Strategy):
    def __init__(self, layout, circle, dices):
        super().__init__(layout, circle)
        self._dices = dices

    def run(self):
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
    def __init__(self, layout, circle, theta=1e-5):
        super().__init__(layout, circle)
        self.Q = np.random.uniform(0, 10, size=(self._env.N_STATE, self._env.N_ACTION))
        self.Q[-1, :] = 0
        self._theta = theta

    def policy(self, Q=None):
        Q = Q if Q is not None else self.Q
        return np.argmin(Q, axis=-1)

    def V(self, Q=None):
        Q = Q if Q is not None else self.Q
        return np.min(Q, axis=-1)

    def next_states(self, state, action):
        def trap_next_states(position):
            trap = self._layout[position]
            if trap != 4:
                next_position, freeze = Trap.next_position(trap, position)
                return [(1.0, freeze, next_position)]
            else:
                p = 1 / SnakesAndLadders.END
                return [(p, False, pos) for pos in range(SnakesAndLadders.END)]

        def dice_next_states(position, dice):
            p = dice.p()
            next_states = []
            for dx in range(dice.MAX_DX+1):
                if position == 2:
                    next_states += [
                        (0.5 * p, SLOW_LANE[position] + dx),
                        (0.5 * p, FAST_LANE[position] + dx),
                    ]
                else:
                    next_pos = SnakesAndLadders._validate_position(
                        position + dx, self._circle
                    )
                    next_states.append((p, next_pos))
            return next_states

        for dice_p, dice_state in dice_next_states(state, self._env.ACTIONS[action]):
            for trap_p, freeze, next_state in trap_next_states(dice_state):
                yield dice_p * trap_p, freeze, next_state

    def compute(self, epochs=100):
        for _ in range(epochs):
            Q = np.zeros_like(self.Q)
            previous_Q = self.Q.copy()
            for state in range(self._env.N_STATE - 1):
                for action in range(self._env.N_ACTION):
                    Q[state, action] += 1.0
                    for p, freeze, next_state in self.next_states(state, action):
                        Q[state, action] += p * (self.V()[next_state] + int(freeze))

            self.Q = Q
            if np.max(np.abs(self.V(Q) - self.V(previous_Q))) < self._theta:
                break

        return self.V(), self.policy()


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


if __name__ == "__main__":
    import numpy as np

    layout = np.full(15, 0, dtype=np.uint8)
    circle = False

    mdp = MarkovDecisionProcess(layout, circle)

    for d in range(3):
        print("DICE :: ", d)
        for position in range(15):
            print("POSITION :: ", position)
            for p, freeze, pos in mdp.next_states(position, d):
                print(p, freeze, pos)
