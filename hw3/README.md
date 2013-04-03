I implemented a decoder loosely based on [Monte Carlo inference and maximization for phrase-based translation](http://www.aclweb.org/anthology/W/W09/W09-1114.pdf) (Arun & al. 2009). The local moves considered are similar: replace a phrase translation, swap two arbitrary phrases (no reordering limit gives a better model score but also strange output), merge two adjacent phrases, split a phrase. Then stochastic search is done considering these possible changes in turn and making random decisions proportional to the likelihood of the change. Simulated annealing is used to allow big changes in the beginning and a more greedy search at the end.

The search was run for 100,000 iterations.

For competition reasons, we also combine the results of several runs.
