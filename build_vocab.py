import pickle
import argparse
import os
import re
from collections import Counter


class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def parse_code(string):
    return re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?|[^\W\d]+|\d+|[\W]", string)

def build_vocab(path, threshold,debug):
    """Build a simple vocabulary wrapper."""
    counter = Counter()
    i = 0
    for root, dirs, files in os.walk(os.path.abspath(path)):
        for file in files:
            if file.endswith(".txt"):
                with open(os.path.join(root,file), 'r') as f:
                    code = str(f.read())
                    tokens = parse_code(code)
                    if debug == True:
                        print(tokens)
                    counter.update(tokens)
                i = i+ 1

                if i % 1000 == 0:
                    print("[%d] Tokenized the captions." %(i))


    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Creates a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Adds the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)

    print(counter)
    return vocab

def main(args):
    vocab = build_vocab(path=args.data_path, threshold=args.threshold,debug=args.debug)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: %d" %len(vocab))
    print("Saved the vocabulary wrapper to '%s'" %vocab_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, 
                        default='./data/', 
                        help='path for train annotation file,must be full path')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl', 
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=1, 
                        help='minimum word count threshold')
    parser.add_argument('-d','--debug', action='store_true', help='debug mode')
    args = parser.parse_args()
    main(args)
