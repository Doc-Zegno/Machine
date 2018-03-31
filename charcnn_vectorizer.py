import os

from charcnn import CharCNN

import torch
from torch.autograd import Variable


class CharCNNVectorizer:
    def __init__(self, name, embedding_size):
        self.name = name
        self.embedding_size = embedding_size
        self.is_cuda_available = torch.cuda.is_available()
        self.cnn = CharCNN(16, embedding_size, 3)
        self.log("Done")

    def get_vector_size(self):
        return self.embedding_size + 2

    def reset_counters(self):
        pass

    def __call__(self, form, word):
        # TF
        if form.istitle():
            tf = [1.0, 0.0]
        else:
            tf = [0.0, 1.0]

        tf = torch.FloatTensor(tf)
        if self.is_cuda_available:
            tf = tf.cuda()
        tf = Variable(tf)

        # Char embedding
        char_embedding = self.cnn(form)
        return torch.cat((tf, char_embedding))

    def log(self, message):
        print("CVectorizer [{}]: {}".format(self.name, message))
