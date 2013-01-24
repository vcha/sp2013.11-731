__N__ = 0
__NULL__ = '__NULL__'

class Vocabulary:
    def __init__(self):
        self.word2id = {__NULL__: __N__}
        self.id2word = [__NULL__]

    def __getitem__(self, word):
        if isinstance(word, int):
            assert word >= 0
            return self.id2word[word]
        if word not in self.word2id:
            self.word2id[word] = len(self)
            self.id2word.append(word)
        return self.word2id[word]

    def __iter__(self):
        return iter(self.id2word)

    def __len__(self):
        return len(self.id2word)

def read_bitext(stream, f_voc, e_voc):
    for line in stream:
        f_sentence, e_sentence = line.lower().split(' ||| ')
        f_words = [__N__] + [f_voc[f] for f in f_sentence.split()] 
        e_words = [e_voc[e] for e in e_sentence.split()]
        yield f_words, e_words

class BiText:
    def __init__(self, stream):
        self.f_voc = Vocabulary()
        self.e_voc = Vocabulary()
        self.segments = list(read_bitext(stream, self.f_voc, self.e_voc))

    def __iter__(self):
        return iter(self.segments)
