from gymnasium import spaces
from statisticalrl_environments.MDPs_discrete.utils import categorical_sample
from statisticalrl_environments.MDPs_discrete.gymWrapper import DiscreteMDP
import scipy.stats as stat


class ContextualDiscreteMDP(DiscreteMDP):
    """
    Extension of DiscreteMDP with a contextual observation.

    Internal state dynamics still depend on the internal state s.
    The observation returned to the learner is the tuple (x, s), where:
      - x is the discrete context
      - s is the internal MDP state
    """

    def __init__(
        self,
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
        name="ContextualDiscreteMDP",
    ):
        if nameActions is None:
            nameActions = []

        self.nX = nX
        self.xdist = xdist
        self.context_is_fixed = context_is_fixed
        self.reward_is_contextual = reward_is_contextual
        self.x = None

        super().__init__(
            nS=nS,
            nA=nA,
            P=P,
            R=R,
            isd=isd,
            nameActions=nameActions,
            seed=seed,
            name=name,
        )

        # Contextual observation is (context, state)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.nX),
            spaces.Discrete(self.nS),
        ))

    def sample_context(self):
        return categorical_sample(self.xdist, self.np_random)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.x = self.sample_context()
        self.lastaction = None

        return (self.x, self.s), {"mean": 0.0}

    def step(self, a):
        transitions = self.P[self.s][a]

        # Considr context in reward
        if self.reward_is_contextual:
            rewarddis = self.R[self.x][self.s][a]
        else:
            rewarddis = self.R[self.s][a]

        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, next_s, done = transitions[i]


        # Add yield and delta_SOC
        reward = rewarddis.rvs()
        mean_reward = rewarddis.mean()

        self.s = next_s

        if not self.context_is_fixed:
            self.x = self.sample_context()

        self.lastaction = a
        self.lastreward = reward

        return (self.x, self.s), reward, done, False, {"mean": mean_reward}

if __name__ == "__main__":

    env = ContextualDiscreteMDP(
        nS=4,
        nA=4,
        P={
            0 : {
                0 : [(1,0,False),(0,1,False),(0,2,False),(0,3,False)],
                1 : [(1,0,False),(0,1,False),(0,2,False),(0,3,False)],
                2 : [(0,0,False),(1,1,False),(0,2,False),(0,3,False)],
                3 : [(1,0,False),(0,1,False),(0,2,False),(0,3,False)]
                },
            
            1 : {
                0 : [(0,0,False),(0,1,False),(1,2,False),(0,3,False)],
                1 : [(0,0,False),(0,1,False),(1,2,False),(0,3,False)],
                2 : [(0,0,False),(0,1,False),(1,2,False),(0,3,False)],
                3 : [(0,0,False),(0,1,False),(1,2,False),(0,3,False)]
                },
            
            2 : {
                0 : [(0,0,False),(0,1,False),(0,2,False),(1,3,False)],
                1 : [(0,0,False),(0,1,False),(0,2,False),(1,3,False)],
                2 : [(0,0,False),(0,1,False),(0,2,False),(1,3,False)],
                3 : [(0,0,False),(0,1,False),(0,2,False),(1,3,False)]
                },
            
            3 : {
                0 : [(0,0,False),(0,1,False),(0,2,False),(1,3,False)],
                1 : [(0,0,False),(0,1,False),(0,2,False),(1,3,False)],
                2 : [(0,0,False),(0,1,False),(0,2,False),(1,3,False)],
                3 : [(0,0,False),(0,1,False),(0,2,False),(1,3,False)]
                },
        },
        R={
            0 : {
                0 : stat.norm(0.2, 0.1),
                1 : stat.norm(0.4, 0.1),
                2 : stat.norm(1.2, 0.1),
                3 : stat.norm(0.3, 0.1)
                },
            
            1 : {
                0 : stat.norm(0.15, 0.1),
                1 : stat.norm(0.35, 0.1),
                2 : stat.norm(0.7, 0.1),
                3 : stat.norm(0.25, 0.1)
                },
            
            2 : {
                0 : stat.norm(0.2, 0.1),
                1 : stat.norm(0.4, 0.1),
                2 : stat.norm(0.7, 0.1),
                3 : stat.norm(0.3, 0.1)
                },
            
            3 : {
                0 : stat.norm(0.4, 0.1),
                1 : stat.norm(0.6, 0.1),
                2 : stat.norm(0.7, 0.1),
                3 : stat.norm(0.4, 0.1)
                },
        },
        isd=[1.0, 0, 0, 0],
        nX=3,
        xdist=[0.4, 0.4, 0.2],
        context_is_fixed=True,
        reward_is_contextual=False,
        nameActions=["fallow", "fert_fallow", "tree", "baseline"],
        seed=123,
        name="BasicAgroCarbonContextMDP",
    )

    obs, info = env.reset()
    print(obs)