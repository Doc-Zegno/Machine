import sys

import nltk
import nltk.corpus as corpus

from model_syntax_joint import Model
from syntax import FormExtractor, WordExtractor, TagExtractor
from sentences import TagsFromSentences
from charcnn import get_num_zeroes


def transform(sents):
    for sent in sents:
        yield nltk.chunk.tree2conlltags(sent)


def load_from_conllu(file_path):
    sents = []
    sent = []

    for line in open(file_path):
        # Skip blank lines
        if len(line) <= 1:
            continue

        # Skip comments
        if line[0] == '#':
            continue

        # Tokenize and get rid of ellipsis enhancement
        line = line.split()
        if line[0].find(".") != -1:
            continue

        # Reset sublist
        if line[0] == "1":
            if sent:
                sents.append(sent)
            sent = []

        head = int(line[6])

        tag = line[3]
        form = line[1]
        word = line[2]
        rel = line[7]

        triple = ((form, word, tag), rel, head)
        sent.append(triple)

    sents.append(sent)
    return sents


if __name__ == "__main__":
    NUM_SENTS = 1000
    EMBEDDING_SIZE = 128
    TAG_SIZE = 32
    NUM_EPOCHS = 3
    CONTEXT_SIZE = 5

    if len(sys.argv) < 3:
        raise RuntimeError("too few arguments: paths to train and test data expected")

    train_path = sys.argv[1]
    test_path = sys.argv[2]

    sents = load_from_conllu(train_path)
    # sents = sents[:1000]
    tests = load_from_conllu(test_path)
    # tests = tests[:1000]
    print(sents[1])
    print(tests[1])

    model = Model("syntagrus_ud_full", sents, WordExtractor(sents), FormExtractor(sents), EMBEDDING_SIZE,
                  TagExtractor(sents), TAG_SIZE,
                  context_size=CONTEXT_SIZE,
                  lrs=(0.25, 1.0, 0.05), epochs_per_decrease=NUM_EPOCHS + 1, lr_decrease_factor=0.1)

    model.train(sents, NUM_EPOCHS, machines=["POS Tagging", "Chunking", "Parsing"])
    print("num zeroes:", get_num_zeroes())
    model.test(tests)
