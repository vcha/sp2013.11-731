import math
from itertools import tee, izip
from collections import Counter
import nltk
import lru

FEATURES = []
def feature(func):
    FEATURES.append(func)
    return func

def bigrams(sent):
    x, y = tee(['^']+sent+['$'])
    next(y)
    for t in izip(x, y):
        yield t

def trigrams(sent):
    x, y, z = tee(['^', '^']+sent+['$', '$'], 3)
    next(y); next(z); next(z)
    for t in izip(x, y, z):
        yield t

# Count match features

@feature
def min_match(x, y):
    x, y = list(x), list(y)
    return len(Counter(x) & Counter(y))/((len(x)+len(y))/2.)

@feature
def min_match2(ref, hyp):
    return min_match(bigrams(ref), bigrams(hyp))

@feature
def min_match3(ref, hyp):
    return min_match(trigrams(ref), trigrams(hyp))

@lru.lru_cache(10)
def pos_tag(sentence):
    return [pos for w, pos in nltk.pos_tag(sentence)]

@feature
def min_match_pos(ref, hyp):
    return min_match(pos_tag(ref), pos_tag(hyp))

@feature
def min_match_pos2(ref, hyp):
    return min_match(bigrams(pos_tag(ref)), bigrams(pos_tag(hyp)))

@feature
def min_match_pos3(ref, hyp):
    return min_match(trigrams(pos_tag(ref)), trigrams(pos_tag(hyp)))

# Binary match features

@feature
def identity(ref, hyp):
    return ref == hyp

@feature
def precision(ref, hyp):
    sref, shyp = set(ref), set(hyp)
    if len(shyp) == 0: return 0
    return len(sref & shyp)/float(len(shyp))

@feature
def recall(ref, hyp):
    sref, shyp = set(ref), set(hyp)
    return len(sref & shyp)/float(len(sref))

@feature
def f1(ref, hyp):
    sref, shyp = set(ref), set(hyp)
    if len(shyp) == 0: return 0
    common = len(sref & shyp)
    if common == 0: return 0
    p = common/float(len(shyp))
    r = common/float(len(sref))
    return 2/(1/p+1/r)

# Length features

@feature
def length_ratio(ref, hyp):
    return len(hyp)/float(len(ref))

@feature
def hyp_len(ref, hyp):
    return len(hyp)

@feature
def hyp_uniq(ref, hyp):
    return len(set(hyp))

# Function/content words

STOP = set(nltk.corpus.stopwords.words('english'))

@feature
def n_function(ref, hyp):
    if len(hyp) == 0: return 0
    return sum(1 for w in hyp if w in STOP)/float(len(hyp))

# Alignment features

import align

@feature
def align_precision(ref, hyp):
    if len(hyp) == 0: return 0
    return len(align.alignment(ref, hyp))/float(len(hyp))

@feature
def align_recall(ref, hyp):
    return len(align.alignment(ref, hyp))/float(len(ref))

@feature
def n_matches(ref, hyp):
    return len(align.alignment(ref, hyp))

@feature
def fragmentation(ref, hyp):
    n_matches = len(align.alignment(ref, hyp))
    if n_matches == 0: return 0
    n_chunks = sum(1 for c in align.chunks(ref, hyp))
    return n_chunks/float(n_matches)

@feature
def max_chunk(ref, hyp):
    n_matches = len(align.alignment(ref, hyp))
    if n_matches == 0: return 0
    return max(len(chunk) for chunk in align.chunks(ref, hyp))/float(n_matches)

@feature
def alignment_score(ref, hyp):
    if len(hyp) == 0: return 0
    n, m = float(len(ref)), float(len(hyp))
    alignment = align.alignment(ref, hyp)
    n_matches = len(alignment)
    if n_matches == 0: return 0
    return sum(abs(i/n-j/m) for i, j, _, _, _ in alignment)/float(n_matches)

@feature
def word_similarity(ref, hyp):
    alignment = align.alignment(ref, hyp)
    n_matches = len(alignment)
    if n_matches == 0: return 0
    return sum(score for _, _, _, _, score in alignment)/float(n_matches)

# Language model

import kenlm
LM = kenlm.LanguageModel('resources/word.klm')
@feature
def lm_score(ref, hyp):
    return math.exp(LM.score(' '.join(hyp)))

@feature
def lm_oovs(ref, hyp):
    return sum(1 for w in hyp if w not in LM)

POS_LM = kenlm.LanguageModel('resources/pos.klm')

@feature
def pos_lm_score(ref, hyp):
    return math.exp(POS_LM.score(' '.join(pos_tag(hyp))))

def extract(ref, hyp):
    fvector = {}
    for fn in FEATURES:
        fv = fn(ref, hyp)
        fvector[fn.func_name] = fv
        fvector['log_'+fn.func_name] = math.log(1+fv)
    return fvector

def diff(f1, f2):
    """ diff(f1, f2) = f1 - f2 """
    return {fn: f1.get(fn, 0) - f2.get(fn, 0) for fn in set(f1.iterkeys()) | set(f2.iterkeys())}
