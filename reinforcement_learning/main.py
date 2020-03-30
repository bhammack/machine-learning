# Use openai gym,
import sys
import numpy as np
import matplotlib.pyplot as plt
#
from gym import utils
from gym.envs.toy_text import discrete
# frozen lake
from frozen_lake import FrozenLakeEnv
from blackjack import BlackjackEnv


# rl method: q learner/learning?


def value_iteration(env, iterations=99999):
    # For a given state, calculate the state-action values for all possible actions from that state.
    # update teh value function of that state iwth the gratest state-action value.
    # Terminate when teh difference between all new state values and old state values is small. 
    # https://medium.com/analytics-vidhya/solving-the-frozenlake-environment-from-openai-gym-using-value-iteration-5a078dffe438
    

    # https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Value%20Iteration%20Solution.ipynb


    # reset the environment
    # for each discount factor
      # run the solver, passing in a convergence check function


    # value iter
    # for each possible next state
      # find the best possible action



    pass

def policy_iteration():
    # https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Policy%20Iteration%20Solution.ipynb

    
    pass

def q_learner():
    pass






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Select an experiment to run')
    parser.add_argument('--frozenlake', action='store_true', help='Use the Frozen Lake gridworld problem')
    parser.add_argument('--blackjack', action='store_true', help='Use the Blackjack problem')
    #
    parser.add_argument('--value', action='store_true', help='Solve using value iteration')
    parser.add_argument('--policy', action='store_true', help='Solve using policy iteration')
    parser.add_argument('--q', action='store_true', help='Solve using a Q-learner')

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()