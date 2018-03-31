# Created specially for syntagrus
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


OOV = 0
BOS = 1
EOS = 2
ALPHABET_SIZE = 110


num_zeroes = 0


class CharCNN(nn.Module):
    def __init__(self, char_size, output_size, kernel_width=3):
        super(CharCNN, self).__init__()
        self.char_size = char_size
        self.output_size = output_size
        self.is_cuda_available = torch.cuda.is_available()

        self.char_embedding = nn.Embedding(ALPHABET_SIZE, char_size)
        self.conv = nn.Conv1d(1, output_size, kernel_width)
        self.reflection = nn.ReflectionPad2d((1, 1, 0, 0))

        if self.is_cuda_available:
            self.char_embedding = self.char_embedding.cuda()
            self.conv = self.conv.cuda()
            self.reflection = self.reflection.cuda()

    def forward(self, form):
        # Word to Tensor
        indices = word2index(form.lower())
        indices = torch.LongTensor(indices)
        indices = indices.view((1, -1))
        if self.is_cuda_available:
            indices = indices.cuda()
        indices = Variable(indices)

        # Extract features
        # embeddings are 1 x WORD_LEN x CHAR_SIZE
        embeddings = self.char_embedding(indices)

        # Conv1d expects:
        # BATCH_SIZE x INPUT_CHANNELS x LEN
        # i.e.:
        # 1 x 1 x (WORD_LEN * CHAR_SIZE)
        embeddings = embeddings.view((1, 1, -1))
        # embeddings = self.reflection(embeddings)
        features = self.conv(embeddings)

        # Features are 1 x OUTPUT_SIZE x LEN
        features = features.view((self.output_size, -1))

        # Max pooling
        features, argmax = torch.max(features, dim=1)

        # Average pooling
        # features = torch.mean(features, dim=1)

        return features


def get_num_zeroes():
    return num_zeroes


def word2index(word):
    # Always starts with Begin-of-Sequence
    ords = [BOS]
    for ch in word:
        index = char2index(ch)
        if index == 0:
            global num_zeroes
            num_zeroes += 1
        ords.append(index)
    ords.append(EOS)
    return ords


def char2index(ch):
    ch = ord(ch)
    if ch <= 64:
        # OOV
        if ch < 33:
            return 0
        else:
            return ch - 30

    elif ch <= 126:
        # OOV
        if ch < 91:
            return 0
        else:
            return ch - 56

    elif ch == 176:
        return 71

    elif ch == 183:
        return 72

    elif ch <= 1105:
        # OOV
        if ch < 1072:
            return 0
        else:
            return ch - 999

    elif ch == 8230:
        return 107

    elif ch == 8364:
        return 108

    elif ch == 8470:
        return 109

    else:
        return 0
