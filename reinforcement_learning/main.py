# Use openai gym,
import sys
import argparse
import numpy as np
from pdb import set_trace
import itertools
from collections import defaultdict
import matplotlib.pyplot as plt
import time
#
import gym
from gym import utils
from gym.envs.toy_text import discrete
#
from frozen_lake import FrozenLakeEnv
from toh import TohEnv

# Great description of algorithms:
# https://stackoverflow.com/questions/37370015/what-is-the-difference-between-value-iteration-and-policy-iteration
# http://webgraphviz.com/
# env.nS = env.observation_space.n

def score_frozen_lake(env, policy, episodes=1000):
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


def score_tower_of_hanoi(env, policy, episodes=1000):
    """Run the policy on the environment, * episodes."""
    moves_list = []
    stuck_dict = {}
    # paths = []
    for episode in range(episodes):
        observation = env.reset()
        # path = []
        moves = 0
        # set_trace()
        while True:
            action = policy[observation]
            # old_obs = observation
            # set_trace()
            observation, reward, done, _ = env.step(action)
            moves += 1
            # print(env.state_mapping[observation])
            if done and reward > 0: # reward for final state
               #  print(moves)
                moves_list.append(moves)
                break
            elif done and reward == 0: # agent reached an invalid state and is stuck
            # elif done and reward == 0: # agent reached an invalid state and is stuck
                if observation in stuck_dict:
                    stuck_dict[observation] += 1
                else:
                    stuck_dict[observation] = 1
                break
    print('----------------------------------------------')
    print('The policy used on average {:.0f} moves before terminating'.format(np.mean(moves_list)))
    print('The policy got stuck {:.2f} % of the time'.format(sum(stuck_dict.values()) * 100/episodes, episodes))
    print('----------------------------------------------')


def one_step_lookahead(env, discount, state, value_function):
    """Calculate the value of all actions for a given state."""
    # https://gist.github.com/persiyanov/334f64ca14f7405f5b3c7372fecf2857
    action_values = np.zeros(env.nA)
    for action in range(env.nA):
        for prob, next_state, reward, done in env.P[state][action]:
            action_values[action] += prob * (reward + discount * value_function[next_state])
    return action_values


def evaluate_policy(env, policy, discount, theta):
    # https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Policy%20Iteration%20Solution.ipynb
    # Start with a random (all 0) value function
    V = np.zeros(env.nS)
    i = 0
    while True:
        i += 1
        delta = 0
        # For each state, perform a "full backup"
        for s in range(env.nS):
            v = 0
            # Look at the possible next actions
            for a, action_prob in enumerate(policy[s]):
                # For each action, look at the possible next states...
                for prob, next_state, reward, done in env.P[s][a]:
                    # Calculate the expected value
                    v += action_prob * prob * (reward + discount * V[next_state])
            # How much our value function changed (across any states) over the course of this iteration.
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        # Stop evaluating once our value function change is below a threshold
        if delta < theta:
            break
    print('> policy eval convergence: {} iterations'.format(i))
    return np.array(V)


def policy_iteration(env, discount=0.9, theta=1e-6):
    # https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Policy%20Iteration%20Solution.ipynb
    policy = np.ones([env.nS, env.nA]) / env.nA
    # This algorithm uses a policy as a 2d vector of states and actions, NOT A 1D VECTOR
    i = 0
    start = time.time()
    states_updated = []
    while True:
        i += 1
        updates = 0
        # Evalute the policy
        value_function = evaluate_policy(env, policy, discount, theta)
        # Improve the policy
        policy_stable = True
        for state in range(env.nS):
            prev_action = np.argmax(policy[state])
            best_action = np.argmax(one_step_lookahead(env, discount, state, value_function))
            # Greedily update the policy
            if prev_action != best_action:
                policy_stable = False
                updates += 1
            policy[state] = np.eye(env.nA)[best_action] # what does eye do?
        states_updated.append(updates)
        if policy_stable:
            break
    print('> PI convergence: {} iterations'.format(i))
    print('> duration: {} secs'.format(time.time() - start))
    if args.plot: plt.plot(range(i), states_updated, label='states updated per iteration'), plt.xlabel('Iteration'), plt.ylabel('# states updated'), plt.tight_layout(), plt.show()
    return policy


def compute_policy(env, value_function, discount):
    """Get the policy associated with the utilities of the best actions, computed from value iteration."""
    # print('Extracting policy from the value function...')
    policy = [0 for i in range(env.nS)]
    for state in range(env.nS):
        action_values = one_step_lookahead(env, discount, state, value_function)
        best_action = np.argmax(np.asarray(action_values))
        policy[state] = best_action
    return policy


def compute_value_function(env, discount, theta):
    """Performs value iteration on the environment to compute the value function."""
    # For a given state, calculate the state-action values for all possible actions from that state.
    # update teh value function of that state iwth the gratest state-action value.
    # Terminate when teh difference between all new state values and old state values is small.
    # https://medium.com/analytics-vidhya/solving-the-frozenlake-environment-from-openai-gym-using-value-iteration-5a078dffe438
    # https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Value%20Iteration%20Solution.ipynb
    stateValue = [0 for i in range(env.nS)]
    newStateValue = stateValue.copy()
    i = 0
    deltas = []
    while True: # imply convergence, lol
        i += 1
        delta = 0
        for state in range(env.nS): # For every state in the discrete space
            action_values = one_step_lookahead(env, discount, state, stateValue)
            best_action = np.argmax(np.asarray(action_values)) # find the action with the maximum value
            newStateValue[state] = action_values[best_action] # update the state mapping to use this best action
            delta = max(delta, abs(newStateValue[state] - stateValue[state]))
        # if there is negligible difference, break the loop
        deltas.append(delta)
        if delta < theta:
            break
        else:
            stateValue = newStateValue.copy()
    print('> VI convergence: {} iterations'.format(i))
    if args.plot: plt.plot(range(i), deltas, label='delta per iteration'), plt.xlabel('Iteration'), plt.ylabel('Delta'), plt.tight_layout(), plt.show()
    return stateValue


def value_iteration(env, discount=0.9, theta=1e-6):
    """Find the value function for the environment, then extract the policy."""
    start = time.time()
    state_utilities = compute_value_function(env, discount, theta)
    policy = compute_policy(env, state_utilities, discount)
    print('> duration: {} secs'.format(time.time() - start))
    return policy


def q_learning(env, decay_rate=0.001, learning_rate=0.80, discount=0.90, max_steps=300, total_episodes=15000):
    """
    epsilon - exploration/exploitation rate
    learning_rate - learning rate
    discount - discounted rate
    max_steps = 100 (max moves the agent can make in one attempt/run/exploration)
    """
    # https://medium.com/swlh/introduction-to-reinforcement-learning-coding-q-learning-part-3-9778366a41c0
    # https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/
    # https://deeplizard.com/learn/video/QK_PP_2KgGE
    # https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Q%20learning/FrozenLake/Q%20Learning%20with%20FrozenLake.ipynb

    min_epsilon = 0.01
    max_epsilon = 1.0
    epsilon = 1.0

    # create the empty q table
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    # https://deeplizard.com/learn/video/HGeI30uATws
    rewards = []
    start = time.time()
    terminating_states = {}
    paths = []
    state_visits = [0] * env.observation_space.n
    goal_found = False

    for episode in range(total_episodes):
        if episode % 1000 == 0: print('Episode: {}/{}'.format(episode, total_episodes), end="\r", flush=True)

        state = env.reset()
        done = False
        step = 0
        total_rewards = 0
        path = [state]
        state_visits[state] += 1
        # goal_found = False

        for step in range(max_steps):
            # if step == max_steps - 1: print('Exhausted maximum number of steps/moves!')
            # set_trace()
            # print(env.state_mapping[env.s], end='\r', flush=True)
            # Exploration-exploitation tradeoff. Chose a random action, or the learned action.
            # If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
            # Else doing a random choice --> exploration
            exploit = np.random.uniform(0, 1) > epsilon
            explore = not exploit
            if exploit and False:
                action = np.argmax(Q[state, :]) # take the Q-learned action
            else:
                action = env.action_space.sample() # take a random action

            # We went to this state. Log it

            # step into that action
            new_state, reward, done, info = env.step(action)
            # set_trace()

            # Update the q-table
            predict = Q[state, action]
            target = reward + discount * np.max(Q[new_state, :])
            Q[state, action] = Q[state, action] + learning_rate * (target - predict)

            # Set the new state
            state = new_state

            # Increase our reward earned for this step and log that we visited it
            total_rewards += reward
            path.append(state)
            state_visits[state] += 1

            if done:
                # Log the state where we currently are that caused termination
                if state in terminating_states:
                    terminating_states[state] += 1
                else:
                    terminating_states[state] = 1
                if reward > 0:
                    if goal_found == False:
                        print('Goal state found!\n')
                        goal_found = True
                    # if not goal_found: print('Goal state found!')
                    # goal_found = True
                break

        # Reduce epsilon (because we need less and less exploration)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

        # Log the rewards of all episodes and paths taken
        rewards.append(total_rewards)
        paths.append(path)

    print("> rewards over time: " +  str(sum(rewards)/total_episodes))
    print('> duration: {} secs'.format(time.time() - start))
    if args.plot: plt.plot(range(env.observation_space.n), state_visits, label='state visits'), plt.xlabel('state #'), plt.ylabel('visits'), plt.tight_layout(), plt.show()
    if args.lake:
        print(np.reshape(state_visits, [env.nrow, env.ncol]))
    set_trace()
    return Q


def print_policy(policy):
    """Print the policy in a nice format."""
    if args.lake:
        FROZEN_LAKE_ACTIONS = ['←', '↓', '→', '↑']
        print_list = list(map(lambda i: FROZEN_LAKE_ACTIONS[i], policy))
        # print(np.reshape(print_list, [8, 8]))
        print(print_list)
    if args.tower:
        TOWER_ACTIONS = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
        print_list = list(map(lambda x: '{} -> {}'.format(TOWER_ACTIONS[x][0], TOWER_ACTIONS[x][1]), policy))
        # print(print_list)
        # for i in range(len(print_list)):
        #     print('{} [label={}]'.format(print_list[i], i))
        # print(print_list)
        # print(print_list)


def policy_differences(vi, pi):
    differences = {}
    for i in range(len(vi)):
        if vi[i] != pi[i]:
            if i in differences:
                differences[i] += 1
            else:
                differences[i] = 1
    return differences


def main():
    env = None

    # environment selection
    if args.lake:
        env = FrozenLakeEnv(map_name='8x8')
        if args.size > 0:
            env = FrozenLakeEnv(desc=None, map_name=None, size=args.size)
    if args.tower:
        rings = tuple(range(args.rings - 1, -1, -1))
        print(rings)
        init = (rings, (), ())
        goal = ((), (), rings)
        env = TohEnv(initial_state=init, goal_state=goal, noise=args.noise)

    print('> env number of states: {}'.format(env.nS))
    print('> noise factor: {}'.format(args.noise))

    # solver selection
    discount = args.discount
    print('> discount factor: {}'.format(discount))
    if args.vi:
        vi_policy = value_iteration(env, discount=discount)
        policy = vi_policy
        print_policy(vi_policy)
    if args.pi:
        pi_policy = policy_iteration(env, discount=discount) # reshape to 1d array
        pi_policy = np.reshape(np.argmax(pi_policy, axis=1), [env.nS])
        policy = pi_policy
        print_policy(pi_policy)
    if args.vi and args.pi:
        # Compare the two policies
        diffs = policy_differences(vi_policy, pi_policy)
        print(diffs)
        print('VI and PI policy differences: {}'.format(sum(diffs.values())))
    if args.q:
        Q = q_learning(env, total_episodes=args.episodes)
        # The optimal policy for Q learning is the argmax action with probability 1 - epsilon.
        q_policy = np.reshape(np.argmax(Q, axis=1), [env.nS]).tolist()
        policy = q_policy
        print_policy(q_policy)


    print('Scoring the policy...')
    if args.lake:
        score_frozen_lake(env, policy)
    if args.tower:
        score_tower_of_hanoi(env, policy)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Select an experiment to run')
    parser.add_argument('--lake', action='store_true', help='Use the Frozen Lake gridworld problem')
    parser.add_argument('--tower', action='store_true', help='Use the Tower of Hanoi problem')
    #
    parser.add_argument('--vi', action='store_true', help='Solve using value iteration')
    parser.add_argument('--pi', action='store_true', help='Solve using policy iteration')
    parser.add_argument('--q', action='store_true', help='Solve using a Q-learner')
    #
    parser.add_argument('--plot', action='store_true', help='Create plots')
    parser.add_argument('--discount', type=float, default=0.9, help='Discount/gamma value to use')
    parser.add_argument('--noise', type=float, default=0.1, help='Noise value for Tower of Hanoi problem')
    parser.add_argument('--rings', type=int, default=6, help='Number of rings for Tower of Hanoi problem')
    parser.add_argument('--size', type=int, default=0, help='Size of random puzzle for Frozen Lake problem')
    parser.add_argument('--episodes', type=int, default=100000, help='Number of episodes for q learning')
    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help()
    main()