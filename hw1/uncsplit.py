import sys
import argparse
from itertools import izip
from collections import defaultdict
from csplit import csplit

def main():
    parser = argparse.ArgumentParser(description='Train alignment models')
    parser.add_argument('corpus', help='original corpus')
    args = parser.parse_args()

    with open(args.corpus) as corpus:
        for sentence, alignments in izip(corpus, sys.stdin):
            de, en = sentence[:-1].decode('utf8').lower().split(' ||| ')
            align = defaultdict(list)
            for pair in alignments.split():
                i, j = map(int, pair.split('-'))
                align[i].append(j)
            def pairs():
                n = 0
                for i, w in enumerate(de.split()):
                    for chunk in csplit(w).split():
                        for j in align[n]:
                            yield (i, j)
                        n += 1
            print(' '.join('{}-{}'.format(*p) for p in pairs()))

if __name__ == '__main__':
    main()
