# ./learners/ContextualMDPs_discrete/Optimal/ContextualOptimalControl.py

import numpy as np
from statisticalrl_learners.MDPs_discrete.Optimal.OptimalControl import Opti_controller
from statisticalrl_environments.MDPs_discrete.utils import categorical_sample

def build_opti(name, env, nS, nA):
    return GlobalOpti_controller(env, nS, nA)

class GlobalOpti_controller(Opti_controller):
    """
    Global optimal controller for contextual environments.

    The controller receives contextual observations of the form (x, s), but
    computes and applies a policy depending only on the MDP state s.
    """

    def __init__(self, env, nS, nA, epsilon=0.001, max_iter=100):
        super().__init__(env, nS, nA, epsilon=epsilon, max_iter=max_iter)
        self.observation = None
        self.x = None
        self.s = None

    def parse_observation(self, observation):
        x, s = observation
        return x, s

    def reset(self, observation):
        self.observation = observation
        self.x, self.s = self.parse_observation(observation)

    def play(self, observation):
        self.observation = observation
        self.x, self.s = self.parse_observation(observation)
        a = categorical_sample([self.policy[self.s, a] for a in range(self.nA)], np.random)
        return a

    def update(self, observation, action, reward, next_observation):
        self.observation = next_observation
        self.x, self.s = self.parse_observation(next_observation)