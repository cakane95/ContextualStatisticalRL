# ./articles/2026_GymAgroCarbon/tests/2026_04_07_GlobalETC4_fullRun_test.py

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))
sys.path.append(PROJECT_ROOT)

# Agents
from learners.ContextualMDPs_discrete.ETC import GlobalETC4
from learners.ContextualMDPs_discrete.ContextualIMED_RL import GlobalIMEDRL, SemiLocalIMEDRL


from learners.ContextualMDPs_discrete.Optimal import ContextualOptimalControl as opt
from statisticalrl_experiments.fullExperiment import runLargeMulticoreExperiment as xp
import environments.register as cW


if __name__ == "__main__":
    env = cW.make("basic-agrocarbon-context")

    nS = env.nS
    nA = env.nA
    nX = env.nX

    skeleton = {
        0: [0, 1, 2, 3],
        1: [0, 1, 3],
        2: [0, 1, 3],
        3: [0, 1, 3],
    }

    agents = [
        (
            GlobalETC4,
            {
                "nS": nS,
                "nA": nA,
                "nX": nX,
                "skeleton": skeleton,
                "gamma": 0.99,
                "epsilon": 1e-6,
                "max_iter": 1000,
                "name": "GlobalETC4",
            },
        ),
         (
            GlobalIMEDRL,
            {
                "nbr_states": nS,
                "nbr_actions": nA,
                "nbr_contexts": nX,
                "skeleton": skeleton,
                "max_iter": 3000,
                "epsilon": 1e-3,
                "max_reward": 1.5,
            },
        ),
        (
        SemiLocalIMEDRL,
        {
            "nbr_states": nS,
            "nbr_actions": nA,
            "nbr_contexts": nX,
            "skeleton": skeleton,
            "max_iter": 3000,
            "epsilon": 1e-3,
            "max_reward": 1.5,
        },
    ),
    ]

    oracle = opt.build_opti(env.name, env, nS, nA)

    root_folder = os.path.join(CURRENT_DIR, "results") + os.sep

    xp(
        env,
        agents,
        oracle,
        timeHorizon=20,
        nbReplicates=10,
        root_folder=root_folder,
    )