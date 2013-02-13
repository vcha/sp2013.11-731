# Implemented solution

I implemented the model described in [Dyer et al. 2013](http://www.ark.cs.cmu.edu/cdyer/fast_valign.pdf) (unpublished), in addition to model 1.

A diagonal prior is imposed on the alignment links, and a mean-field approximation is used in the E step to estimate the posterior translation distribution with a Dirichlet prior.

Then, I applied several pre-processing steps to improve alignments:
- lowercase the data
- stem German and English with the Snowball stemmer (see stem-corpus.py)
- split the compounds in German using cdec (see csplit.py and uncsplit.py)

Finally, I tuned the hyperparameters of the systems (diagonal tension, null alignment probability and Dirichlet prior strength) on the dev set to minimize AER using the simplex method.

## Results summary (dev corpus)

<table>
<tr><th>Model</th><th>AER</th><th>Δ</th></tr>
<tr><td>Baseline</td><td>0.792556</td><td></td></tr>
<tr><td>Model 1</td><td>0.421544</td><td>0.37</td></tr>
<tr><td>+ diagonal prior</td><td>0.290222</td><td>0.13</td></tr>
<tr><td>+ Dirichlet prior</td><td>0.269823</td><td>0.02</td></tr>
<tr><td>+ compound split</td><td>0.257015</td><td>0.01</td></tr>
<tr><td>+ stem</td><td>0.243009</td><td>0.01</td></tr>
<tr><td>+ tune</td><td>0.233541</td><td>0.01</td></tr>
</table>

## Usage

Run the following command to replicate the experiments:

```bash
cat data/dev-test-train.de-en | python csplit.py | python stem-corpus.py | python modelc.py | python uncsplit.py | ./check | ./grade -n 0
```
