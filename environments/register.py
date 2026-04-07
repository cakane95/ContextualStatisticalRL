import gymnasium
from gymnasium.envs.registration import register
import numpy as np
import sys
import scipy.stats as stat

INFINITY = sys.maxsize


def registerContextualMDP(
    nS,
    nA,
    P,
    R,
    isd,
    nX,
    xdist,
    context_is_fixed=True,
    reward_is_contextual=False,
    nameActions=None,
    seed=None,
    max_steps=INFINITY,
    reward_threshold=np.inf,
    name=None,
):
    if name is None:
        name = f"ContextualMDP-S{nS}_A{nA}_X{nX}_s{seed}-v0"

    register(
        id=name,
        entry_point="environments.ContextualMDPs_discrete.contextualMDP:ContextualDiscreteMDP",
        max_episode_steps=max_steps,
        reward_threshold=reward_threshold,
        kwargs={
            "nS": nS,
            "nA": nA,
            "P": P,
            "R": R,
            "isd": isd,
            "nX": nX,
            "xdist": xdist,
            "context_is_fixed": context_is_fixed,
            "reward_is_contextual": reward_is_contextual,
            "nameActions": nameActions,
            "seed": seed,
            "name": name,
        },
    )
    return name


registerContextualStatisticalRLenvironments = {
    "basic-agrocarbon-context": lambda x: registerContextualMDP(
        nS=4,
        nA=4,
        P={
            0: {
                0: [(1, 0, False), (0, 1, False), (0, 2, False), (0, 3, False)],
                1: [(1, 0, False), (0, 1, False), (0, 2, False), (0, 3, False)],
                2: [(0, 0, False), (1, 1, False), (0, 2, False), (0, 3, False)],
                3: [(1, 0, False), (0, 1, False), (0, 2, False), (0, 3, False)],
            },
            1: {
                0: [(0, 0, False), (0, 1, False), (1, 2, False), (0, 3, False)],
                1: [(0, 0, False), (0, 1, False), (1, 2, False), (0, 3, False)],
                2: [(0, 0, False), (0, 1, False), (1, 2, False), (0, 3, False)],
                3: [(0, 0, False), (0, 1, False), (1, 2, False), (0, 3, False)],
            },
            2: {
                0: [(0, 0, False), (0, 1, False), (0, 2, False), (1, 3, False)],
                1: [(0, 0, False), (0, 1, False), (0, 2, False), (1, 3, False)],
                2: [(0, 0, False), (0, 1, False), (0, 2, False), (1, 3, False)],
                3: [(0, 0, False), (0, 1, False), (0, 2, False), (1, 3, False)],
            },
            3: {
                0: [(0, 0, False), (0, 1, False), (0, 2, False), (1, 3, False)],
                1: [(0, 0, False), (0, 1, False), (0, 2, False), (1, 3, False)],
                2: [(0, 0, False), (0, 1, False), (0, 2, False), (1, 3, False)],
                3: [(0, 0, False), (0, 1, False), (0, 2, False), (1, 3, False)],
            },
        },
        R={
            0: {
                0: stat.norm(0.2, 0.1),
                1: stat.norm(0.4, 0.1),
                2: stat.norm(1.2, 0.1),
                3: stat.norm(0.3, 0.1),
            },
            1: {
                0: stat.norm(0.15, 0.1),
                1: stat.norm(0.35, 0.1),
                2: stat.norm(0.7, 0.1),
                3: stat.norm(0.25, 0.1),
            },
            2: {
                0: stat.norm(0.2, 0.1),
                1: stat.norm(0.4, 0.1),
                2: stat.norm(0.7, 0.1),
                3: stat.norm(0.3, 0.1),
            },
            3: {
                0: stat.norm(0.4, 0.1),
                1: stat.norm(0.6, 0.1),
                2: stat.norm(0.7, 0.1),
                3: stat.norm(0.4, 0.1),
            },
        },
        isd=[1.0, 0.0, 0.0, 0.0],
        nX=3,
        xdist=[0.4, 0.4, 0.2],
        context_is_fixed=True,
        reward_is_contextual=False,
        nameActions=["fallow", "fert_fallow", "tree", "baseline"],
        seed=123,
        name="BasicAgroCarbonContextMDP-v0",
    ),
}


def print_envlist():
    print("-" * 30)
    print("List of registered contextual environments:")
    for k in registerContextualStatisticalRLenvironments.keys():
        print("\t" + k)
    print("-" * 30)


def register_env(envName):
    if envName in registerContextualStatisticalRLenvironments.keys():
        regName = registerContextualStatisticalRLenvironments[envName](None)
        if not isinstance(regName, str):
            raise TypeError(f"Registered environment name must be a string, got {type(regName)}")
        print("[REGISTER.INFO] Environment " + envName + " registered as " + regName)
        return regName
    else:
        return envName


def makeWorld(registername):
    return gymnasium.make(registername).unwrapped


def make(envName):
    return makeWorld(register_env(envName))