# Use openai gym,
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
#
import gym
from gym import utils
from gym.envs.toy_text import discrete
# frozen lake
from frozen_lake import FrozenLakeEnv
from blackjack import BlackjackEnv


def get_score(env, policy, episodes=1000):
    """Run the policy on the environment, * episodes."""
    misses = 0
    steps_list = []
    for episode in range(episodes):
        observation = env.reset()
        steps = 0
        while True:
            action = policy[observation]
            observation, reward, done, _ = env.step(action)
            steps += 1
            if done and reward == 1:
                # print('You have got the fucking Frisbee after {} steps'.format(steps))
                steps_list.append(steps)
                break
            elif done and reward == 0:
                # print("You fell in a hole!")
                misses += 1
                break
    print('----------------------------------------------')
    print('You took an average of {:.0f} steps to get the frisbee'.format(np.mean(steps_list)))
    print('And you fell in the hole {:.2f} % of the times'.format((misses/episodes) * 100))
    print('----------------------------------------------')


def get_policy(env, stateValue, lmbda=0.9):
    """Get the policy associated with the value functions computed from value iteration."""
    policy = [0 for i in range(env.nS)]
    for state in range(env.nS):
        action_values = []
        for action in range(env.nA):
            action_value = 0
            for i in range(len(env.P[state][action])):
                prob, next_state, r, _ = env.P[state][action][i]
                action_value += prob * (r + lmbda * stateValue[next_state])
            action_values.append(action_value)
        best_action = np.argmax(np.asarray(action_values))
        policy[state] = best_action
    return policy 


def value_iteration(env, max_iterations=100000, discount=0.9):
    """Performs value iteration on the environment to compute the value function."""
    # For a given state, calculate the state-action values for all possible actions from that state.
    # update teh value function of that state iwth the gratest state-action value.
    # Terminate when teh difference between all new state values and old state values is small. 
    # https://medium.com/analytics-vidhya/solving-the-frozenlake-environment-from-openai-gym-using-value-iteration-5a078dffe438
    # https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Value%20Iteration%20Solution.ipynb
    SMALL = 1e-04
    stateValue = [0 for i in range(env.nS)]
    newStateValue = stateValue.copy()
    for i in range(max_iterations):
        for state in range(env.nS):
            action_values = []      
            for action in range(env.nA):
                state_value = 0
                for i in range(len(env.P[state][action])):
                    prob, next_state, reward, done = env.P[state][action][i]
                    state_action_value = prob * (reward + discount * stateValue[next_state])
                    state_value += state_action_value   
                action_values.append(state_value)      #the value of each action
                best_action = np.argmax(np.asarray(action_values))   # choose the action which gives the maximum value
                newStateValue[state] = action_values[best_action]  #update the value of the state
        if i > 1000: 
            if sum(stateValue) - sum(newStateValue) < SMALL:   # if there is negligible difference break the loop
                break
                print(i)
        else:
            stateValue = newStateValue.copy()
    return stateValue 


def policy_iteration():
    # https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Policy%20Iteration%20Solution.ipynb

    
    pass


def q_learner():
    pass


def main():
    env = None
    if args.frozenlake: env = FrozenLakeEnv(map_name='8x8')
    if args.blackjack: env = BlackjackEnv()
    if args.value:
        x = value_iteration(env)
        print('Value function:', x)
        y = get_policy(env, x)
        print('Policy:', y)
        get_score(env, y)


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
    main()