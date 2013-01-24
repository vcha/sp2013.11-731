#!/usr/bin/env python
import sys
import logging
import math
from collections import defaultdict
from corpus import BiText
#from scipy.special import digamma

corpus = 'data/dev-test-train.de-en'
n_iter = 5

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
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
        c_f_e = dict(((f, e), 0) for (f, e) in t_f_e.iterkeys())
        # E
        for f_sentence, e_sentence in corpus:
            for e in e_sentence:
                t_e = sum(t_f_e[f, e] for f in f_sentence)
                log_likelihood += math.log(t_e)
                for f in f_sentence:
                    c_f_e[f, e] += t_f_e[f, e] / t_e
        ppl = math.exp(-log_likelihood/n_target)
        logging.info('Previous iteration LL=%.0f ppl=%.3f', log_likelihood, ppl)
        # M
        for f in range(len(corpus.f_voc)):
            c_f = sum(c_f_e[f, e] for e in possible_alignments[f])
            for e in possible_alignments[f]:
                t_f_e[f, e] = c_f_e[f, e]/c_f
                #t_f_e[f, e] = math.exp(digamma(c_f_e[f, e]))/math.exp(digamma(c_f))

    logging.info('Decode')
    for f_sentence, e_sentence in corpus:
        als = ((max((t_f_e[f, e], i) for i, f in enumerate(f_sentence))[1], j)
                for j, e in enumerate(e_sentence))
        als = ('{0}-{1}'.format(i-1, j) for i, j in als if i > 0)
        print(' '.join(als))

    """
    from heapq import nlargest
    for f, f_word in enumerate(corpus.f_voc):
        print '===', f_word, '==='
        for p_f_e, e in nlargest(5, ((t_f_e[f, e], e) for e in possible_alignments[f])):
            print corpus.e_voc[e], p_f_e
        print
    """


if __name__ == '__main__':
    main()
