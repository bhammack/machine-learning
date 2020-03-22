=====
## Unsupervised Learning Analysis
=====

Project code can be found here: https://github.com/bhammack/machine-learning

From the root of this repository, run the main.py script with arguments:
ex: `python unsupervised_learning/main.py --digits --kmeans` or `python main.py --adults --pca`
One data set must be selected, either `--adults` or `--digits` and one or more arguments
Data sets are included in their raw form in the data/ directory. Because Digits is a data set from sklearn, there is no raw data to use.

usage: main.py [-h] [--kmeans] [--em] [--pca] [--ica] [--rca] [--kpca] [--lda]
               [--nn] [--visualize] [--adults] [--digits]

Select an experiment to run

optional arguments:
  -h, --help   show this help message and exit
  --kmeans     Use the k-means clustering algorithm
  --em         Use the expectation maximization algorithm
  --pca        Reduce dimensions using PCA
  --ica        Reduce dimensions using ICA
  --rca        Reduce dimensions using randomized projections
  --kpca       Reduce dimensions using Kernel-PCA
  --lda        Reduce dimensions using LDA
  --nn         Run the neural network on the resultant data
  --visualize  Visualize the clusters
  --adults     Experiment with the U.S. adults data set
  --digits     Experiment with the handwritten digits data set


Libraries used in this analysis include:
- sklearn
- scipy
- numpy
- pandas
which must all be installed via pip
