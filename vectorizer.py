import gensim
import os
import torch

import nltk.corpus as corpus

from chargram import CharGramSentences, CharGram
from sentences import LowercaseSentences


# Vectorizer.
# Consists of word-level embedding machine and a character-level one
class Vectorizer:
    def __init__(self, word_sents, char_sents, name, embedding_size, filler,
                 force_train=False, ce_enabled=True, tf_enabled=True):
        # The following parameters help to distinguish models
        self.name = name
        self.embedding_size = embedding_size

        # Filler is a functor which provides embedding vector
        # when a word or its substring cannot be found in a dictionary
        self.filler = filler

        # Character embedding dramatically increases the precision of taggers
        # but it's not always necessary
        self.ce_enabled = ce_enabled

        self.tf_enabled = tf_enabled

        # Grammer is used to transform a word into a list of its
        # character n-grams
        self.grammer = CharGram()

        # Used models
        self.word_model = None
        self.char_model = None

        # CUDA flag
        self.is_cuda_available = torch.cuda.is_available()

        # Model misses.
        # Can be used for debugging and as an aggregate estimation
        # of vocabulary completeness
        self.num_word_misses = 0
        self.num_char_misses = 0
        self.num_words = 0
        self.num_grams = 0

        # Trained models should be stored
        word_model_path = "internal/{}_{}_word_model.w2v".format(name, embedding_size)
        char_model_path = "internal/{}_{}_char_model.w2v".format(name, embedding_size)

        # Very
        # Important
        # Decision
        train_word_model = False
        train_char_model = False

        if force_train:
            train_char_model = True
            train_word_model = True
        else:
            if not os.path.exists(word_model_path):
                train_word_model = True
            if not os.path.exists(char_model_path):
                train_char_model = True

        # Word model
        if train_word_model:
            # Train model
            self.log("Training word model...")
            iterator = LowercaseSentences(word_sents)
            # iterator = word_sents
            self.word_model = gensim.models.Word2Vec(iterator, size=embedding_size,
                                                     sg=1, workers=4, min_count=3)
            self.word_model.save(word_model_path)

        else:
            # Load
            self.log("Loading existing word model...")
            self.word_model = gensim.models.Word2Vec.load(word_model_path)
        self.log("Done")

        # Character model
        if not ce_enabled:
            return

        if train_char_model:
            # Train model
            self.log("Training character model...")
            iterator = CharGramSentences(char_sents)
            self.char_model = gensim.models.Word2Vec(iterator, size=embedding_size,
                                                     sg=1, workers=4, min_count=3)
            self.char_model.save(char_model_path)

        else:
            # Load
            self.log("Loading existing character model...")
            self.char_model = gensim.models.Word2Vec.load(char_model_path)
        self.log("Done")

    def get_vector_size(self):
        size = self.embedding_size
        if self.ce_enabled:
            size *= 2
        if self.tf_enabled:
            size += 2
        return size

    def reset_counters(self):
        self.num_grams = 0
        self.num_words = 0
        self.num_word_misses = 0
        self.num_char_misses = 0

    def __call__(self, form, word, context=None):
        # Word level embedding
        self.num_words += 1
        try:
            output = self.word_model.wv[word.lower()]
            word_embedding = torch.FloatTensor(output)
        except KeyError:
            self.num_word_misses += 1
            if context is None:
                word_embedding = self.filler()
            else:
                # Deduce embedding from context
                embeds = torch.zeros(self.embedding_size)
                if self.is_cuda_available:
                    embeds = embeds.cuda()
                for item in context:
                    try:
                        item = self.word_model.wv[item.lower()]
                        item = torch.FloatTensor(item)
                    except KeyError:
                        item = self.filler()
                    if self.is_cuda_available:
                        item = item.cuda()
                    embeds += item
                word_embedding = embeds / max(len(context), 1)

        if self.is_cuda_available:
            word_embedding = word_embedding.cuda()

        # Title feature
        if self.tf_enabled:
            title_embedding = [1.0, 0.0] if form.istitle() else [0.0, 1.0]
            if self.is_cuda_available:
                title_embedding = torch.cuda.FloatTensor(title_embedding)
            else:
                title_embedding = torch.FloatTensor(title_embedding)
            word_embedding = torch.cat((title_embedding, word_embedding), dim=0)

        if not self.ce_enabled:
            return word_embedding

        # Character level embedding
        grams = self.grammer(form)
        # unique_grams = set(grams)
        unique_grams = grams
        char_embedding = torch.zeros(self.embedding_size)
        if self.is_cuda_available:
            char_embedding = char_embedding.cuda()
        for gram in unique_grams:
            try:
                embedding = torch.FloatTensor(self.char_model.wv[gram])
            except KeyError:
                self.num_char_misses += 1
                embedding = self.filler()
            if self.is_cuda_available:
                embedding = embedding.cuda()
            char_embedding += embedding

        # Take an average
        num_uniques = len(unique_grams)
        if num_uniques == 0:
            num_uniques = 1
        char_embedding /= num_uniques

        self.num_grams += num_uniques

        # Return concatenation
        return torch.cat((word_embedding, char_embedding), dim=0)

    def log(self, message):
        print("Vectorizer [{}]: {}".format(self.name, message))


if __name__ == "__main__":
    TEST_RANGE = 2
    NAME = "BrownTest"
    EMBEDDING_SIZE = 4

    sents = corpus.brown.sents()[:TEST_RANGE]
    vectorizer = Vectorizer(word_sents=sents, char_sents=sents, name=NAME,
                            embedding_size=EMBEDDING_SIZE, force_train=True)

    i = 0
    for sent in sents:
        print("sentence #{}".format(i))
        i += 1

        for word in sent:
            word = word.lower()
            print("word:", word)
            print("embedding:")
            print(vectorizer(word))
            print()

    print("Done")
