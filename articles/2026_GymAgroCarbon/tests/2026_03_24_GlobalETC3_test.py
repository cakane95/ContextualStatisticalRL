# ./articles/2026_GymAgroCarbon/tests/2026_03_24_GlobalETC3_test.py

#############################
# Imports
import os
import sys
import scipy.stats as stat
import matplotlib.pyplot as plt
import numpy as np

# Root of the project
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))
sys.path.append(PROJECT_ROOT)

# Environment
from environments.ContextualMDPs_discrete.contextualMDP import ContextualDiscreteMDP

# Learners
from learners.ContextualMDPs_discrete.ETC import GlobalETC3

# Oracle builder
from learners.ContextualMDPs_discrete.Optimal import ContextualOptimalControl as opt

# oneRun
from statisticalrl_experiments.oneRun import oneXpNoRender


#############################
# Instantiate environment
env = ContextualDiscreteMDP(
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
    name="BasicAgroCarbonContextMDP",
)

# Optional: admissible actions A(s)
skeleton = {
    0: [0, 1, 2, 3],
    1: [0, 1, 3],
    2: [0, 1, 3],
    3: [0, 1, 3],
}


#############################
# List learners to be compared
agents = [
    GlobalETC3(
        nS=env.nS,
        nA=env.nA,
        nX=env.nX,
        skeleton=skeleton,
        gamma=0.99,
        epsilon=1e-6,
        max_iter=1000,
        name="GlobalETC3",
    )
]


#############################
# Compute oracle policy
oracle = opt.build_opti(env.name, env, env.nS, env.nA)


#############################
# Run Experiment
root_folder = os.path.join(CURRENT_DIR, "results")
os.makedirs(root_folder, exist_ok=True)

T = 20

for agent in agents:
    print(f"Running {agent.name()}...")
    cummeans = oneXpNoRender(env, agent, T, root_folder)
    print("cummeans =", cummeans)

print("Running oracle...")
cummeans_oracle = oneXpNoRender(env, oracle, T, root_folder)
print("oracle cummeans =", cummeans_oracle)

times = np.arange(1, T + 1)

cummeans_agent = np.array(cummeans)
cummeans_oracle = np.array(cummeans_oracle)
cumregret = cummeans_oracle - cummeans_agent

# 1. Cumulative means
plt.figure(figsize=(8, 5))
plt.plot(times, cummeans_agent, 'o', label="GlobalETC3", color='#377eb8',
         linewidth=2.0, linestyle='--', markevery=max(1, T // 20))
plt.plot(times, cummeans_oracle, 'v', label="Oracle", color='#ff7f00',
         linewidth=2.0, linestyle='--', markevery=max(1, T // 20))
plt.title("BasicAgroCarbonContextMDP")
plt.xlabel("Time steps", fontsize=13, fontname="Arial")
plt.ylabel("Cumulative mean reward", fontsize=13, fontname="Arial")
plt.legend(loc=2)
plt.ticklabel_format(axis='both', useMathText=True, useOffset=True, style='sci', scilimits=(0, 0))
plt.tight_layout()
plt.savefig(os.path.join(root_folder, "cummeans_test.png"))
plt.savefig(os.path.join(root_folder, "cummeans_test.pdf"))
plt.close()

# 2. Cumulative regret
plt.figure(figsize=(8, 5))
plt.plot(times, cumregret, 's', label="GlobalETC3 regret", color='#4daf4a',
         linewidth=2.0, linestyle='--', markevery=max(1, T // 20))
plt.title("BasicAgroCarbonContextMDP")
plt.xlabel("Time steps", fontsize=13, fontname="Arial")
plt.ylabel("Regret Tg*-sum_t r_t", fontsize=13, fontname="Arial")
plt.legend(loc=2)
plt.ticklabel_format(axis='both', useMathText=True, useOffset=True, style='sci', scilimits=(0, 0))
plt.tight_layout()
plt.savefig(os.path.join(root_folder, "cumregret_test.png"))
plt.savefig(os.path.join(root_folder, "cumregret_test.pdf"))
plt.close()