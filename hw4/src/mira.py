import logging
import argparse
import numpy
import math
import heapq
import pickle
import metrics
from pro import read_candidates

def dot(fv, w):
    return (fv * w.T)[0, 0]

def perceptron_step(w, candidate_list, C):
    y_hat = max(candidate_list, key=lambda c: dot(c.features, w))
    y_plus = max(candidate_list, key=lambda c: dot(c.features, w) + c.score)
    y_minus = max(candidate_list, key=lambda c: dot(c.features, w) - c.score)
    if y_hat != y_plus:
        return w + C * (y_plus.features - y_minus.features)
    return w

def L2(w):
    return math.sqrt(dot(w, w))

REF = '/home/vchahune/projects/sp2013.11-731/hw4/data/dev.ref'

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description='MIRA tuning')
    parser.add_argument('--nbest', help='n-best file', default='/dev/stdin')
    parser.add_argument('--ref', help='reference file', default=REF)
    parser.add_argument('--align', help='alignments file')
    parser.add_argument('--metric', help='metric to use for tuning',
            choices=metrics.METRICS.keys(), default='bleu')
    parser.add_argument('--weights', help='file path to store weights to')
    parser.add_argument('--step', help='perceptron step', type=float, default=0.001)
    args = parser.parse_args()

    logging.info('Reading all candidates and computing features')

    vectorizer, candidates = read_candidates(args.nbest,
            args.ref, args.align, metrics.METRICS[args.metric])

    n_features = len(vectorizer.get_feature_names())
    logging.info('Read %d candidates with %d features', len(candidates), n_features)

    w = numpy.ones((1, n_features)) * 1e-3
    w_avg = numpy.zeros((1, n_features))
    for it in range(10000):
        w_old = w.copy()
        for candidate_list in candidates.itervalues():
            w = perceptron_step(w, candidate_list, args.step)
        logging.info('Weights at iteration %d: %s', it+1, w)
        w_avg += w
        if L2(w - w_old)/L2(w_old) < 1e-2:
            break

    w = w_avg / it

    w /= (L2(w) + 1e-6) # normalize weight vector
    weights = dict(zip(vectorizer.feature_names_, numpy.array(w)[0]))

    score = sum(max(candidate_list, key=lambda c:dot(c.features, w)).score
            for candidate_list in candidates.itervalues())/len(candidates)

    print(u'Final weights: {}'.format(' '.join(u'{}={:.2f}'.format(*kv)
        for kv in heapq.nlargest(50, weights.iteritems(), key=lambda t: t[1]))).encode('utf8'))
    print('Dev score: {}'.format(score))

    if args.weights:
        with open(args.weights, 'w') as weights_file:
            pickle.dump(weights, weights_file)

if __name__ == '__main__':
    main()
