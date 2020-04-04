# Use openai gym,
import sys
import argparse
import numpy as np
from pdb import set_trace
import matplotlib.pyplot as plt
#
import gym
from gym import utils
from gym.envs.toy_text import discrete
# frozen lake
from frozen_lake import FrozenLakeEnv
from blackjack import BlackjackEnv

# Great description of algorithms:
# https://stackoverflow.com/questions/37370015/what-is-the-difference-between-value-iteration-and-policy-iteration



def get_score(env, policy, episodes=1000):
    """Run the policy on the environment, * episodes."""
    print('Scoring the policy...')
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


def one_step_lookahead(env, discount, state, value_function):
    """Calculate the value of all actions for a given state."""
    # https://gist.github.com/persiyanov/334f64ca14f7405f5b3c7372fecf2857
    action_values = np.zeros(env.nA)
    for action in range(env.nA):
        for prob, next_state, reward, done in env.P[state][action]:
            action_values[action] += prob * (reward + discount * value_function[next_state])
    return action_values



def evaluate_policy(env, policy, discount_factor=0.9, theta=1e-6):
    # https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Policy%20Iteration%20Solution.ipynb
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function
    V = np.zeros(env.nS)
    while True:
        delta = 0
        # For each state, perform a "full backup"
        for s in range(env.nS):
            v = 0
            # Look at the possible next actions
            for a, action_prob in enumerate(policy[s]):
                # For each action, look at the possible next states...
                for  prob, next_state, reward, done in env.P[s][a]:
                    # Calculate the expected value
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
            # How much our value function changed (across any states)
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        # Stop evaluating once our value function change is below a threshold
        if delta < theta:
            break
    return np.array(V)


def policy_iteration(env, max_iterations=100000, discount=0.9):
    # https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Policy%20Iteration%20Solution.ipynb
    # start with a random policy
    # evaluate the policy
    print('Performing policy iteration...')

    # values vector, also known as V
    policy = np.ones([env.nS, env.nA]) / env.nA
    # This algorithm uses a policy as a 2d vector of states and actions, NOT A 1D VECTOR
    while True:

        # Evalute the policy
        value_function = evaluate_policy(env, policy)

        # Improve the policy
        policy_stable = True
        for state in range(env.nS):
            prev_action = np.argmax(policy[state])
            best_action = np.argmax(one_step_lookahead(env, discount, state, value_function))

            # Greedily update the policy
            if prev_action != best_action:
                policy_stable = False
            policy[state] = np.eye(env.nA)[best_action] # what does eye do?
            
        if policy_stable:
            # print('Policy iteration took {} iterations'.format(i))
            return policy


def value_iteration(env):
    """Find the value function for the environment, then extract the policy."""

    def find_value_function(env, max_iterations=100000, discount=0.9):
        """Performs value iteration on the environment to compute the value function."""
        print('Computing the optimal value function...')
        # For a given state, calculate the state-action values for all possible actions from that state.
        # update teh value function of that state iwth the gratest state-action value.
        # Terminate when teh difference between all new state values and old state values is small. 
        # https://medium.com/analytics-vidhya/solving-the-frozenlake-environment-from-openai-gym-using-value-iteration-5a078dffe438
        # https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Value%20Iteration%20Solution.ipynb
        SMALL = 1e-04
        stateValue = [0 for i in range(env.nS)]
        newStateValue = stateValue.copy()
        for i in range(max_iterations):
            for state in range(env.nS): # For every state in the discrete space
                action_values = one_step_lookahead(env, discount, state, stateValue)
                best_action = np.argmax(np.asarray(action_values)) # find the action with the maximum value
                newStateValue[state] = action_values[best_action] # update the state mapping to use this best action

            # TODO: check this section. why am I comparing i to a fixed iteration size?
            if i > 1000: 
                if sum(stateValue) - sum(newStateValue) < SMALL:   # if there is negligible difference break the loop
                    print('Finding value function took {} iterations'.format(i))
                    break
            else:
                stateValue = newStateValue.copy()
        return stateValue 


    def get_policy(env, stateValue, discount=0.9):
        """Get the policy associated with the utilities of the best actions, computed from value iteration."""
        print('Extracting policy from the value function...')
        policy = [0 for i in range(env.nS)]
        for state in range(env.nS):
            action_values = one_step_lookahead(env, discount, state, stateValue)
            best_action = np.argmax(np.asarray(action_values))
            policy[state] = best_action
        return policy


    state_utilities = find_value_function(env)
    policy = get_policy(env, state_utilities)
    return policy


def q_learner():
    pass


def main():
    env = None
    if args.frozenlake: env = FrozenLakeEnv(map_name='8x8')
    if args.blackjack: env = BlackjackEnv()
    if args.value:
        policy = value_iteration(env)
        print(policy)
        get_score(env, policy)
    if args.policy:
        policy = policy_iteration(env)
        # reshape the policy to a 1d array
        policy = np.reshape(np.argmax(policy, axis=1), [env.nS])
        print(policy)
        get_score(env, policy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Select an experiment to run')
    parser.add_argument('--frozenlake', action='store_true', help='Use the Frozen Lake gridworld problem')
    parser.add_argument('--blackjack', action='store_true', help='Use the Blackjack problem')
    #
    parser.add_argument('--value', action='store_true', help='Solve using value iteration')
    parser.add_argument('--policy', action='store_true', help='Solve using policy iteration')
    parser.add_argument('--q', action='store_true', help='Solve using a Q-learner')
    #
    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help()
    main()