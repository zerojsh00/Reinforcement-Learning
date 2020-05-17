import gym
from gym import spaces
from gym import utils
from gym.utils import seeding
import numpy as np

import logging
logger = logging.getLogger(__name__)


class InventoryEnv(gym.Env, utils.EzPickle):
    """Example 1 (Inventory control with lost sales):
        Consider the problem of day-to-day control of an inventory of a fixed maximum size in the face of uncertain demand:
        Every evening, the decision maker must decide about the quantity to be ordered for the next day.
        In the morning, the ordered quantity arrives with which the inventory is filled up.
        During the day, some stochastic demand is realized, where the demands are independent with a common fixed distribution.
        The goal of the inventory manager is to manage the inventory so as to maximize the present monetary value of the expected total future income.
    """

    def __init__(self, n=100, k=8, c=2, h=6, p=9, lam=8):
        self.n = n
        self.action_space = spaces.Discrete(n) # the number of items ordered in the evening of day t
        self.observation_space = spaces.Discrete(n) # the inventory size
        self.max = n
        self.state = n
        self.k = k # a fixed entry cost k of ordering nonzero items
        self.c = c # a fixed cost c
        self.h = h # a cost of holding an inventory of size x > 0
        self.p = p
        self.lam = lam
        self.done = False

        # Set seed
        self.seed()

        # Start the first round
        self.reset()

    def demand(self):
        return np.random.poisson(self.lam)

    def transition(self, x, a, d):
        m = self.max
        return max(min(x + a, m) - d, 0)

    def reward(self, x, a, y):
        k = self.k
        m = self.max
        c = self.c
        h = self.h
        p = self.p
        r = -k * (a > 0) - c * max(min(x + a, m) - x, 0) - h * x + p * max(min(x + a, m) - y, 0)
        return r

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        done = self.done
        obs = self.state
        demand = self.demand()
        obs2 = self.transition(obs, action, demand)
        self.state = obs2
        reward = self.reward(obs, action, obs2)
        # done = 0
        # if reward > 0:
        #     done = True
        return obs2, reward, done, {}

    def reset(self):
        return self.state
