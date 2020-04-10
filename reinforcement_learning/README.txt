=====
## Reinforcement Learning Analysis
=====

Project code can be found here: https://github.com/bhammack/machine-learning

From the root of this repository, run the main.py script with arguments:
ex: `python reinforcement_learning/main.py --vi --lake`
One MDP learning method must be selected, either `--vi`, `--pi` or `--q`
One MDP problem must be selected, either `--lake` or `--tower`
usage: main.py [-h] [--lake] [--tower] [--vi] [--pi] [--q] [--plot] [--discount DISCOUNT] [--noise NOISE] [--rings RINGS] [--size SIZE]
               [--episodes EPISODES]

Select an experiment to run

optional arguments:
  -h, --help           show this help message and exit
  --lake               Use the Frozen Lake gridworld problem
  --tower              Use the Tower of Hanoi problem
  --vi                 Solve using value iteration
  --pi                 Solve using policy iteration
  --q                  Solve using a Q-learner
  --plot               Create plots
  --discount DISCOUNT  Discount/gamma value to use
  --noise NOISE        Noise value for Tower of Hanoi problem
  --rings RINGS        Number of rings for Tower of Hanoi problem
  --size SIZE          Size of random puzzle for Frozen Lake problem
  --episodes EPISODES  Number of episodes for q learning

Libraries used in this analysis include:
- numpy
- gym
which must all be installed via pip
