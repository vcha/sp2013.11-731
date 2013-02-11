# Implemented solution

I implemented the model described in [Dyer et al. 2013](http://www.ark.cs.cmu.edu/cdyer/fast_valign.pdf) (unpublished), in addition to model 1.

A diagonal prior is imposed on the alignment links, and a mean-field approximation is used in the E step to estimate the posterior translation distribution with a Dirichlet prior.

Dev AER: 0.269823 (Model 1: 0.421544)
