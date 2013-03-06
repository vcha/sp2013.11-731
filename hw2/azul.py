#!/usr/bin/env python
import re
import features
import pickle

entity_re = re.compile("&(#?)(\d{1,5}|\w{1,8});")
# Note: ignores non-ASCII words on purpose
token_re = re.compile('(\'?[a-zA-Z]+\-?[a-zA-Z]*|\d+\.?\,?\d*)')

def tokenize(sentence):
    sentence = sentence.decode('utf8')
    sentence = entity_re.sub('', sentence)
    return token_re.findall(sentence)

def normalize(word):
    return word.lower().replace(',', '.')

def sentences(fn='/dev/stdin'):
    with open(fn) as f:
        for pair in f:
            yield [map(normalize, tokenize(sentence)) for sentence in pair.split(' ||| ')]

def main():
    with open('model.pickle') as f:
        vectorizer, model = pickle.load(f)

    for hyp1, hyp2, ref in sentences():
        f1 = features.extract(ref, hyp1)
        f2 = features.extract(ref, hyp2)
        f = features.diff(f2, f1)
        if f['min_match'] == f['min_match2'] == f['min_match3'] == 0:
            print 0
            continue
        score = model.predict(vectorizer.transform((f,))) # w . (f_2 - f_1)
        if score > 0:
            print 1
        else:
            print -1

if __name__ == '__main__':
    main()
