import os, sys
from time import time
import numpy as np
import random

def ngrams(sentence, n):
    """
    Returns:
        list: a list of lists of words corresponding to the ngrams in the sentence.
    """
    return [sentence[i:i+n] for i in range(len(sentence)-n+1)]

class GloveEmbedding(object):
    """
    Reference: http://nlp.stanford.edu/projects/glove
    """

    def __init__(self, default='zero'):

        self.default = default
        self.dim = 300

        start = time()

        self.dict = self.load_emb('embeddings/glove/glove.840B.300d.txt')

        print('Pre-trained Glove embeddings loaded in {} seconds!'.format(time()-start))

    def load_emb(self, filename):

        emb_dict = {}
        with open(filename) as fid:
            for line in fid:
                items = line.rstrip().split(' ')
                word = items[0]
                embd = [float(x) for x in items[1:]]
                assert(len(embd) == self.dim)
                emb_dict[word] = embd

        return emb_dict

    def emb(self, word, oov_default = 'zero'):

        embd = self.dict.get(word)
        if embd == None:
            if oov_default == 'zero':
                embd = [0.0 for i in range(self.dim)]
            else:
                np.random.seed(123)
                embd = [random.uniform(-0.1, 0.1) for i in range(self.dim)]

        return embd

class KazumaCharEmbedding(object):
    """
    Reference: http://www.logos.t.u-tokyo.ac.jp/~hassy/publications/arxiv2016jmt/
    """

    def __init__(self):

        self.dim = 100

        start = time()

        self.dict = self.load_emb('embeddings/kazuma/charNgram.txt')

        print('Pre-trained character Ngram embeddings loaded in {} seconds!'.format(time()-start))

    def load_emb(self, filename):

        emb_dict = {}
        with open(filename) as fid:
            for line in fid:
                items = line.rstrip().split(' ')
                word = items[0]
                embd = [float(x) for x in items[1:]]
                assert(len(embd) == self.dim)
                emb_dict[word] = embd

        return emb_dict

    def emb(self, word, oov_default='zero'):

        chars = ['#BEGIN#'] + list(word) + ['#END#']
        if oov_default == 'zero':
            embs = np.zeros(self.dim, dtype=np.float32)
        else:
            np.random.seed(123)
            embs = np.random.uniform(-0.1, 0.1, self.dim)
        match = {}
        for i in [2, 3, 4]:
            grams = ngrams(chars, i)
            for g in grams:
                g = '{}gram-{}'.format(i, ''.join(g))
                e = self.dict.get(g)
                if e is not None:
                    match[g] = np.array(e, np.float32)
        if match:
            embs = sum(match.values()) / len(match)
        return embs.tolist()

if __name__ == '__main__':

    #g = GloveEmbedding()
    k = KazumaCharEmbedding()
    for w in ['canada', 'vancouver', 'toronto']:
        start = time()
        print('embedding {}'.format(w))
        print(k.emb(w))
        print('took {}s'.format(time() - start))