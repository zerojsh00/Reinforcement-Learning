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

    def __init__(self, n=100, k=100, c=6, h=2, p=10, penalty=100, lam=70):
        self.n = n
        self.action_space = spaces.Discrete(n) # the number of items ordered in the evening of day t
        self.observation_space = spaces.Discrete(n) # the inventory size
        self.max = n
        self.state = 99 # 최초 보유 재고량
        self.k = k # a fixed entry cost k of ordering nonzero items
        self.c = c # a fixed cost c
        self.h = h # a cost of holding an inventory of size x > 0
        self.p = p
        self.lam = lam
        self.done = False
        self.penalty = penalty

        # Set seed
        self.seed()

        # Start the first round
        self.reset()

    def demand(self):
        return np.random.poisson(self.lam)

    def transition(self, x, a, d):
        m = self.max
        if min(x + a, m) - d < 0 :
            lost_sale = min(x + a, m) - d
        else :
            lost_sale = 0
        next_inventory_level = max(min(x + a, m) - d, 0)
        return next_inventory_level, lost_sale

    def reward(self, x, a, y, lost_sale):
        k = self.k
        m = self.max
        c = self.c
        h = self.h
        p = self.p
        penalty = self.penalty
        r = -k * (a > 0) - c * max(min(x + a, m) - x, 0) - h * x + p * max(min(x + a, m) - y, 0) + penalty * lost_sale
        return r

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        done = self.done
        obs = self.state
        demand = self.demand()
        obs2, lost_sale = self.transition(obs, action, demand)
        self.state = obs2
        reward = self.reward(obs, action, obs2, lost_sale)
        # done = 0
        # if reward > 0:
        #     done = True
        return obs2, reward, done, demand

    def reset(self):
        self.state = self.observation_space.sample()
        return self.state
