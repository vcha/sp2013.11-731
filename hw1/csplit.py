import sys
import cdec

csplit_dir = '/home/vchahune/tools/cdec/compound-split'

features = {'CSplit_BasicFeatures':
            '{0}/de/large_dict.de.gz {0}/de/badlist.de.gz'.format(csplit_dir),
       'CSplit_ReverseCharLM':
            '{0}/de/charlm.rev.5gm.de.lm.gz'.format(csplit_dir)}

decoder = cdec.Decoder(formalism='csplit',
        intersection_strategy='full',
        feature_function=features)
decoder.read_weights(csplit_dir+'/de/weights.trained')

csplit = lambda word: word if len(word) < 6 else decoder.translate(word).viterbi()[2:]

def main():
    for sentence in sys.stdin:
        de, en = sentence[:-1].decode('utf8').lower().split(' ||| ')
        print((' '.join(csplit(w) for w in de.split())+' ||| '+en).encode('utf8'))

if __name__ == '__main__':
    main()
