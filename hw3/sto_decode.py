import models
import math
import random
import logging
import sys
import argparse
import multiprocessing as mp

logging.basicConfig(level=logging.INFO, format='%(message)s')

sigmoid = lambda x, alpha: 1/(1+math.exp(-alpha * x))

def tm_score(phrases):
    return sum(phrase.logprob for phrase in phrases)

source_output = lambda phrases: ' '.join('['+' '.join(phrase)+']' for phrase in phrases)
target_output = lambda phrases: ' '.join('['+' '.join(phrase.english)+']' for phrase in phrases)

tm, lm = None, None

def load_models():
    global tm, lm
    tm = models.TM('data/tm')
    lm = models.LM('data/lm')

def translate(input_sentence, n_iter, reordering_limit):
    def lm_score(phrases):
        score = 0
        lm_state = lm.begin()
        for phrase in phrases:
            for word in phrase.english:
                lm_state, word_logprob = lm.score(lm_state, word)
                score += word_logprob
        score += lm.end(lm_state)
        return score

    def replace_moves(i):
        iphrase = source[i]
        ophrase = target[alignment[i]]
        for alternative in tm[iphrase]:
            if alternative == ophrase: continue
            # modify
            replace_apply(i, alternative)
            # score
            tm_delta = alternative.logprob - ophrase.logprob
            yield (i, alternative), tm_delta
            # revert
            replace_apply(i, ophrase)
            
    def replace_apply(i, alternative):
        target[alignment[i]] = alternative

    def merge_moves(i):
        i1, i2 = source[i-1], source[i]
        a1, a2 = alignment[i-1], alignment[i]
        # |a1 - a2| = 1
        a_min = min(a1, a2) # replace
        a_max = max(a1, a2) # remove
        # a_max = a_min + 1
        o1, o2 = target[a_min], target[a_max]
        for alternative in tm.get(i1+i2, []):
            # modify
            merge_apply(i, i1+i2, alternative, a_min)
            # score
            tm_delta = alternative.logprob - o1.logprob - o2.logprob
            yield (i, i1+i2, alternative, a_min), tm_delta
            # revert
            split_apply(i, i1, i2, o1, o2, a1, a2)

    def split_apply(i, i1, i2, o1, o2, a1, a2):
        source.insert(i, i2)
        source[i-1] = i1
        al = min(a1, a2)
        target.insert(al+1, o2)
        target[al] = o1
        for k, a in enumerate(alignment):
            if a >= al+1:
                alignment[k] += 1
        alignment.insert(i, a2)
        alignment[i-1] = a1
            
    def merge_apply(i, src, tgt, al):
        del source[i]
        source[i-1] = src
        del target[al+1]
        target[al] = tgt
        del alignment[i]
        alignment[i-1] = al
        for k, a in enumerate(alignment):
            if a >= al+1:
                alignment[k] -= 1

    def split_moves(i):
        src, tgt = source[i], target[alignment[i]]
        al = alignment[i]
        for k in range(1, len(src)):
            i1, i2 = src[:k], src[k:]
            for o1 in tm.get(i1, []):
                for o2 in tm.get(i2, []):
                    # modify
                    split_apply(i+1, i1, i2, o1, o2, al, al+1)
                    # score
                    tm_delta = o1.logprob + o2.logprob - tgt.logprob
                    yield (i+1, i1, i2, o1, o2, al, al+1), tm_delta
                    # revert
                    merge_apply(i+1, src, tgt, al)

    def swap_moves(i, j):
        # modify
        swap_apply(i, j)
        # score
        yield (i, j), 0
        # revert
        swap_apply(i, j)
        
    def swap_apply(i, j):
        target[alignment[i]], target[alignment[j]] = target[alignment[j]], target[alignment[i]]
        alignment[i], alignment[j] = alignment[j], alignment[i]

    def violates_reordering(i, al):
        d_source_left = sum(len(phrase) for phrase in source[:i])
        d_target_left = sum(len(phrase.english) for phrase in target[:al])
        d_source_right = sum(len(phrase) for phrase in source[i+1:])
        d_target_right = sum(len(phrase.english) for phrase in target[al+1:])
        d = max(abs(d_source_left - d_target_left), abs(d_source_right - d_target_right))
        return (d > reordering_limit)

    def full_score(moves):
        for m, tm_delta in moves:
            yield m, tm_delta, lm_score(target)-score[1]

    def stochastic_strategy(moves, apply_move):
        choice = None
        for m, tm_delta, lm_delta in full_score(moves):
            if sigmoid(tm_delta + lm_delta, alpha) > random.random():
                choice = m, tm_delta, lm_delta
        if choice:
            m, tm_delta, lm_delta = choice
            apply_move(*m)
            score[0] += tm_delta
            score[1] += lm_delta

    # Make initial decoding easy
    for w in input_sentence:
        if not (w,) in tm:
            tm[(w,)] = [models.phrase((w,), -20)]

    source = [(w,) for w in input_sentence]
    target = [max(tm[(w,)], key=lambda phrase: phrase.logprob) for w in input_sentence]
    alignment = [i for i in range(len(input_sentence))]
    score = [tm_score(target), lm_score(target)]

    logging.info(source_output(source))
    logging.info(target_output(target))
    logging.info(' '.join(map(str, alignment)))
    logging.info('Initial score: %s -> %d', score, score[0]+score[1])

    strategy = stochastic_strategy

    history = [((score[:], source[:], target[:], alignment[:]))]

    for it in xrange(n_iter):
        history.append((score[:], source[:], target[:], alignment[:]))
        alpha = 1 - math.exp(-it*10./n_iter)

        # replace
        for i in range(len(source)):
            strategy(replace_moves(i), replace_apply)
        # merge
        i = 1
        while True:
            if i >= len(source): break
            # adjacent target phrases only:
            if abs(alignment[i]-alignment[i-1]) == 1:
                strategy(merge_moves(i), merge_apply)
            i += 1
        # swap
        for i in range(0, len(source)):
            for j in range(0, len(source)):
                if i == j: continue
                if (violates_reordering(i, alignment[j]) or
                        violates_reordering(j, alignment[i])): continue
                strategy(swap_moves(i, j), swap_apply)
        # split
        for i in range(0, len(source)):
            strategy(split_moves(i), split_apply)
        
        if it % (n_iter/100) == 0:
            logging.info('%d | %.2f %s %.2f', it, alpha, target_output(target), score[0]+score[1])

    score, source, target, alignment = max(history, key=lambda t: sum(t[0]))
    logging.info(source_output(source))
    logging.info(target_output(target))
    logging.info(' '.join(map(str, alignment)))
    logging.info('Final score: %s -> %d', score, score[0]+score[1])

    return ' '.join(' '.join(phrase.english) for phrase in target)

def translate_star(args):
    sys.stderr.write('Got sentence!\n')
    return translate(*args)

def main():
    parser = argparse.ArgumentParser(description='Stochastic decoder')
    parser.add_argument('-i', '--n-iter', type=int, required=True)
    parser.add_argument('-r', '--reordering', type=int, default=1000)
    parser.add_argument('-p', '--processes', type=int, default=1)
    args = parser.parse_args()

    logging.info('Starting %d decoders', args.processes)
    pool = mp.Pool(args.processes, load_models)

    tasks = ((sentence.split(), args.n_iter, args.reordering)
            for sentence in sys.stdin)

    for translation in pool.imap(translate_star, tasks):
        print translation
        sys.stdout.flush()

if __name__ == '__main__':
    main()
