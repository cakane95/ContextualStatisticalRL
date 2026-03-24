import numpy as np
from statisticalrl_environments.MDPs_discrete import utils
from statisticalrl_learners.MDPs_discrete.AgentInterface import Agent

class ContextualAgent(Agent):
    """
    Agent learner for the Contextual MDP (cMDP) setting, where each observation
    is a context-state pair (x, s).

    The `learning_scope` parameter controls how rewards and transitions are learned:
    - "global"    : learning based on s only (context-agnostic).
    - "semi-local": reward learning based on (x, s), transition learning on s.
    - "full-local": learning based on (x, s) for both rewards and transitions.

    Subclasses must implement `play()` and `update()`.
    """
    def __init__(self, nS, nA, nX, learning_scope="global", name="ContextualAgent"):

        super().__init__(nS, nA, name=name)      
        self.nX = nX
        self.learning_scope = learning_scope

        valid_scopes = ["global", "semi-local", "full-local"]
        if self.learning_scope not in valid_scopes:
            raise ValueError(f"learning_scope must be one of {valid_scopes}")

        # Initialisation
        self.observation = None
        self.x = None
        self.s = None
        self.inicontext = None
        self.inistate = None

    def parse_observation(self, observation):
        x, s = observation[0], observation[1]
        return x, s

    def reset(self, observation):
        self.observation = observation
        self.x, self.s = self.parse_observation(observation)
        self.inicontext = self.x
        self.inistate = self.s

    def get_context(self, observation=None):
        if observation is not None:
            return self.parse_observation(observation)[0]
        return self.x

    def get_state(self, observation=None):
        if observation is not None:
            return self.parse_observation(observation)[1]
        return self.s

    def get_reward_key(self, observation=None):
        """Builds the statistical key for rewards according to the learning scope."""
        s = self.get_state(observation)
        if self.learning_scope == "global":
            return s
        else: # semi-local or full-local
            x = self.get_context(observation)
            return (x, s)

    def get_transition_key(self, observation=None):
        """Builds the statistical key for transition kernel according to the learning scope."""
        s = self.get_state(observation)
        if self.learning_scope in ["global", "semi-local"]:
            return s
        else: # full-local
            x = self.get_context(observation)
            return (x, s)

    def uses_context_for_rewards(self):
        return self.learning_scope in ["semi-local", "full-local"]

    def uses_context_for_transitions(self):
        return self.learning_scope == "full-local"

    def play(self, observation):
        raise NotImplementedError("play() must be implemented by subclasses.")

    def update(self, observation, action, reward, next_observation):
        raise NotImplementedError("update() must be implemented by subclasses.")