#!/usr/bin/env python
import argparse
import sys
import logging
import models
from alignment import marginalize, DecodingError

def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Combine several model outputs')
    parser.add_argument('-i', '--input', dest='input', default='data/input',
            help='File containing sentences to translate (default=data/input)')
    parser.add_argument('-t', '--translation-model', dest='tm', default='data/tm',
            help='File containing translation model (default=data/tm)')
    parser.add_argument('-l', '--language-model', dest='lm', default='data/lm',
            help='File containing ARPA-format language model (default=data/lm)')
    parser.add_argument('outputs', nargs='+')
    opts = parser.parse_args()

    tm = models.TM(opts.tm)
    lm = models.LM(opts.lm)

    french_sents = [tuple(line.strip().split()) for line in open(opts.input).readlines()]

    def read_output(output):
        english_sents = [tuple(line.strip().split()) for line in open(output)]
        if (len(french_sents) != len(english_sents)):
            logging.error('ERROR: French and English files are not the same length!')
            sys.exit(1)
        return english_sents

    all_outputs = [read_output(output) for output in opts.outputs]
    logging.info('Combining %d outputs', len(all_outputs))

    total_logprob = 0.0
    for sent_num, (f, es) in enumerate(zip(french_sents, zip(*all_outputs))):
        try:
            logprob, best = max((marginalize(e, f, lm, tm), e) for e in es)
        except DecodingError:
            logging.error('ERROR: COULD NOT ALIGN SENTENCE %d', sent_num)
            sys.exit(1)
        total_logprob += logprob
        print ' '.join(best)

    logging.info('Final score: %.0f', total_logprob)

if __name__ == '__main__':
    main()
