#!/usr/bin/env python
from itertools import izip
import logging
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
import pickle
from azul import sentences
import features

def scores(fn):
    with open(fn) as f:
        for line in f:
            yield int(line)

def training_data(sentence_fn, score_fn):
    for (hyp1, hyp2, ref), score in izip(sentences(sentence_fn), scores(score_fn)):
        f1 = features.extract(ref, hyp1)
        f2 = features.extract(ref, hyp2)
        yield features.diff(f2, f1), score # score = f2 > f1

def main():
    logging.basicConfig(level=logging.INFO)

    logging.info('Extracting features')
    data = training_data('data/train.hyp1-hyp2-ref', 'data/train.gold')
    xx, y = zip(*data)
    vec = DictVectorizer()
    X = vec.fit_transform(xx)
    logging.info('X shape: %s', X.shape)
    logging.info('Features: %s', vec.get_feature_names())

    logging.info('Training linear regression')
    model = Ridge(alpha=0.1)
    model.fit(X, y)

    logging.info('Intercept: %s', model.intercept_)
    logging.info('Weights: %s', dict(zip(vec.get_feature_names(), model.coef_)))

    with open('model.pickle', 'w') as f:
        pickle.dump((vec, model), f)

if __name__ == '__main__':
    main()
