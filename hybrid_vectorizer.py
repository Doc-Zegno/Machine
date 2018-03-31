import gensim
import os
import torch
from torch.autograd import Variable

import nltk.corpus as corpus

from chargram import CharGramSentences, CharGram
from sentences import LowercaseSentences
from charcnn import CharCNN


# Vectorizer.
# Consists of word-level embedding machine and a character-level one
class HybridVectorizer:
    def __init__(self, word_sents, name, embedding_size, filler,
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

        # CharCNN
        self.cnn = CharCNN(16, embedding_size, 5)

        # Model misses.
        # Can be used for debugging and as an aggregate estimation
        # of vocabulary completeness
        self.num_word_misses = 0
        self.num_char_misses = 0
        self.num_words = 0
        self.num_grams = 0

        # Trained models should be stored
        word_model_path = "internal/{}_{}_word_model.w2v".format(name, embedding_size)

        # Very
        # Important
        # Decision
        train_word_model = False

        if force_train:
            train_word_model = True
        else:
            if not os.path.exists(word_model_path):
                train_word_model = True

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

    def __call__(self, form, word):
        # Word level embedding
        self.num_words += 1
        try:
            output = self.word_model.wv[form.lower()]
            word_embedding = torch.FloatTensor(output)
        except KeyError:
            self.num_word_misses += 1
            word_embedding = self.filler()

        if self.is_cuda_available:
            word_embedding = word_embedding.cuda()
        word_embedding = Variable(word_embedding)

        # Title feature
        if self.tf_enabled:
            title_embedding = [1.0, 0.0] if form.istitle() else [0.0, 1.0]
            if self.is_cuda_available:
                title_embedding = torch.cuda.FloatTensor(title_embedding)
            else:
                title_embedding = torch.FloatTensor(title_embedding)
        title_embedding = Variable(title_embedding)

        # Character level embedding
        char_embedding = self.cnn(form)

        # Return concatenation
        return torch.cat((title_embedding, word_embedding, char_embedding), dim=0)

    def log(self, message):
        print("HVectorizer [{}]: {}".format(self.name, message))
