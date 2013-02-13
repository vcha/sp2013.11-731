#!/usr/bin/env python
import sys
import nltk

def main():
    de_stemmer = nltk.stem.SnowballStemmer('german')
    en_stemmer = nltk.stem.SnowballStemmer('english')
    for sentence in sys.stdin:
        de, en = sentence[:-1].decode('utf8').lower().split(' ||| ')
        de = ' '.join(map(de_stemmer.stem, de.split()))
        en = ' '.join(map(en_stemmer.stem, en.split()))
        print((de+' ||| '+en).encode('utf8'))

if __name__ == '__main__':
    main()
