# Stochastic LBFGS

This repository contains code to reproduce some of the experiments published in the paper
"A Linearly-Convergent Stochastic L-BFGS Algorithm" (see http://arxiv.org/abs/1508.02087).
To run it, you need Julia v0.3 and the following packages installed:

- Regression
- SVM
- Optim
- ArrayViews
- DataStructures
- DataFrames
- ArgParse
- HDF5

The adult dataset can be obtained from
https://archive.ics.uci.edu/ml/machine-learning-databases/adult/

You need to put the adult.data file into the working directory.

Some example parameters can be found in the file `experiments/adult.sh`.
