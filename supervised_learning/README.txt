=====
## Supervised Learning Analysis
=====

From the root of this repository, run the main.py script with arguments:
ex: `python main.py --digits --svm` or `python main.py --adult --svm`
One data set must be selected, either `--adult` or `--digits` and one or more learners `--svm --bdt --nn`
Data sets are included in their raw form in the data/ directory. Because Digits is a data set from sklearn, there is no raw data to use.
The wine data set was considered for this assignment, but ultimately not used.

usage: main.py [-h] [--dt] [--knn] [--nn] [--svm] [--bdt] [--adult] [--digits]
               [--search]

optional arguments:
  -h, --help  show this help message and exit
  --dt        Run the decision tree classifier experiment
  --knn       Run the k-nearest neighbors experiment
  --nn        Run the artificial neural network experiment
  --svm       Run the support vector machine experiment
  --bdt       Run the boosted decision tree classifier experiment
  --adult     Experiment with the adult data set
  --digits    Experiment with the handwritten digits data set
  --search    Search for the best parameter set


Libraries used in this analysis include:
- sklearn
- numpy
- pandas
which must all be installed via pip

In each Learner child class, an experiment() method is defined. This method contains a few commented out sections of code that can be re-enabled to reproduce results of the analysis.