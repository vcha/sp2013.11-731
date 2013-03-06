from itertools import izip
import lru
import munkres

WORDS = 'resources/words.lst'
EMBEDDINGS = 'resources/embeddings.txt'

dic = {}
with open(WORDS) as f:
    for i, line in enumerate(f):
        dic[line.strip()] = i

embeddings = []
with open(EMBEDDINGS) as f:
    for line in f:
        v = map(float, line.split())
        embeddings.append(v)

def word_sim(w1, w2):
    if w1 == w2: return 1
    if w1 not in dic or w2 not in dic: return 0
    v1, v2 = embeddings[dic[w1]], embeddings[dic[w2]]
    p = sum(x1*x2 for x1, x2 in izip(v1, v2))
    n1 = sum(x1**2 for x1 in v1)
    n2 = sum(x2**2 for x2 in v1)
    return p/(n1*n2)**(0.5)

def align(s1, s2):
    m = [[1-word_sim(w1, w2) for w2 in s2] for w1 in s1]
    for i1, i2 in munkres.Munkres().compute(m):
        yield i1, i2, s1[i1], s2[i2], m[i1][i2]

@lru.lru_cache(10)
def alignment(ref, hyp):
    return list(align(ref, hyp))

def chunks(ref, hyp):
    pi, pj = -1, -1
    chunk = []
    for i, j, wi, wj, score in alignment(ref, hyp):
        if (abs(i-pi) > 1 or abs(j-pj) > 1) and chunk:
            yield chunk
            chunk = []
        chunk.append((i, j, wi, wj, score))
        pi, pj = i, j
    if chunk:
        yield chunk
