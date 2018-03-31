import os

import fastText
from fastText import load_model

import torch


# Dummy.
# Makes use of pre-trained model
class FastTextVectorizer:
    def __init__(self, name, embedding_size, path2model):
        if not os.path.exists(path2model):
            raise RuntimeError("invalid model path")

        self.name = name
        self.embedding_size = embedding_size
        self.is_cuda_available = torch.cuda.is_available()
        self.log("Loading model from a file...")
        self.model = load_model(path2model)
        self.log("Done")

    def get_vector_size(self):
        return self.embedding_size

    def reset_counters(self):
        pass

    def __call__(self, form, word):
        embedding = torch.from_numpy(self.model.get_word_vector(form))
        if self.is_cuda_available:
            embedding = embedding.cuda()
        return embedding

    def log(self, message):
        print("FVectorizer [{}]: {}".format(self.name, message))
