#!/usr/bin/env python
import sys
import re
from itertools import tee, izip
from collections import Counter

entity_re = re.compile("&(#?)(\d{1,5}|\w{1,8});")
# Note: ignores non-ASCII words on purpose
token_re = re.compile('(\'?[a-zA-Z]+\-?[a-zA-Z]*|\d+\.?\,?\d*)')

def tokenize(sentence):
    sentence = sentence.decode('utf8')
    sentence = entity_re.sub('', sentence)
    return token_re.findall(sentence)

def normalize(word):
    return word.lower().replace(',', '.')

def sentences():
    for pair in sys.stdin:
        yield [map(normalize, tokenize(sentence)) for sentence in pair.split(' ||| ')]

def bigrams(sent):
    x, y = tee(['^']+sent+['$'])
    next(y)
    for t in izip(x, y):
        yield t

def min_match(x, y):
    x, y = list(x), list(y)
    return len(Counter(x) & Counter(y))/((len(x)+len(y))/2.)

def score(ref, hyp):
    return min_match(ref, hyp) + min_match(bigrams(ref), bigrams(hyp))

def main():
    for hyp1, hyp2, ref in sentences():
        delta = score(ref, hyp1) - score(ref, hyp2)
        print (-1 if delta > 0 else 1 if delta < 0 else 0)

if __name__ == '__main__':
    main()
