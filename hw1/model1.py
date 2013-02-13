#!/usr/bin/env python
import sys
import logging
import math
from collections import defaultdict
from scipy.special import digamma
from corpus import BiText

n_iter = 5
vb_estimate = True
vb_alpha = (0.01 if vb_estimate else 0)

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logging.info('Loading bitext')
    corpus = BiText(sys.stdin)

    logging.info('Compute possible alignment links')
    possible_alignments = defaultdict(set)
    for f_sentence, e_sentence in corpus:
        for f in f_sentence:
            for e in e_sentence:
                possible_alignments[f].add(e)

    logging.info('Initialize translation table')
    t_f_e = {} # t_f_e[f, e] = p(e|f) = p(e, f)/p(f)
    for f in range(len(corpus.f_voc)):
        u = 1./len(possible_alignments[f])
        for e in possible_alignments[f]:
            t_f_e[f, e] = u

    n_target = sum(len(e_sentence) for _, e_sentence in corpus)
    for it in range(n_iter):
        logging.info('EM iteration %s', it+1)
        log_likelihood = 0
        c_f_e = dict(((f, e), vb_alpha) for (f, e) in t_f_e.iterkeys())
        # E
        for f_sentence, e_sentence in corpus:
            a_prob = 1./len(f_sentence)
            for e in e_sentence:
                t_e = sum(t_f_e[f, e]*a_prob for f in f_sentence)
                log_likelihood += math.log(t_e)
                for f in f_sentence:
                    c_f_e[f, e] += t_f_e[f, e]*a_prob / t_e
        ppl = math.exp(-log_likelihood/n_target)
        logging.info('Previous iteration LL=%.0f ppl=%.3f', log_likelihood, ppl)
        # M
        for f in range(len(corpus.f_voc)):
            c_f = sum(c_f_e[f, e] for e in possible_alignments[f])
            for e in possible_alignments[f]:
                if vb_estimate:
                    t_f_e[f, e] = math.exp(digamma(c_f_e[f, e]) - digamma(c_f))
                else:
                    t_f_e[f, e] = c_f_e[f, e] / c_f

    logging.info('Decode')
    for f_sentence, e_sentence in corpus:
        als = ((max((t_f_e[f, e], j) for j, f in enumerate(f_sentence))[1], i)
                for i, e in enumerate(e_sentence)) # max p(a_i|i)
        als = ('{0}-{1}'.format(j-1, i) for j, i in als if j > 0) # f-e
        print(' '.join(als))

if __name__ == '__main__':
    main()
