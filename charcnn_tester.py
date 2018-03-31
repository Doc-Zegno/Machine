import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from model_tester import load_from_conllu
from charcnn import CharCNN


class CharCNNTrainer(nn.Module):
    def __init__(self, char_size, output_size, kernel_width, vocab_size):
        super(CharCNNTrainer, self).__init__()
        self.cnn = CharCNN(char_size, output_size, kernel_width)
        self.linear = nn.Linear(output_size, vocab_size)
        if torch.cuda.is_available():
            self.linear = self.linear.cuda()

    def forward(self, words):
        # Linear expects (N, *, in_features)
        embeddings = []
        for word in words:
            embeddings.append(self.cnn(word))
        embeddings = torch.stack(embeddings)

        # Embeddings are SEQ_LEN x EMBEDDING_SIZE
        output = self.linear(embeddings)
        return F.log_softmax(output, dim=1)


if __name__ == "__main__":
    sents = load_from_conllu("data/ru_syntagrus-ud-train.conllu")
    sents = sents[:5000]
    words = []
    for sent in sents:
        for dep, rel, head in sent:
            form, word, tag = dep
            words.append(form.lower())

    vocabulary = set(words)
    vocab_size = len(vocabulary)
    word2index = {w: i for i, w in enumerate(vocabulary)}

    # Optimize encoder
    encoder = CharCNNTrainer(16, 128, 3, vocab_size)
    optimizer = optim.SGD(encoder.parameters(), lr=0.025)
    loss_function = nn.NLLLoss()

    NUM_EPOCHS = 3
    sents_per_dot = len(sents) // 10

    is_cuda_available = torch.cuda.is_available()
    for epoch in range(NUM_EPOCHS):
        current_sent = 0
        loss = 0
        next_dot = sents_per_dot

        print("epoch #{}: [".format(epoch), end="", flush=True)
        for sent in sents:
            words = [dep[0].lower() for dep, rel, head in sent]
            targets = [word2index[word] for word in words]
            targets = torch.LongTensor(targets)
            if is_cuda_available:
                targets = targets.cuda()
            targets = Variable(targets)

            optimizer.zero_grad()
            output = encoder(words)
            current_loss = loss_function(output, targets)
            loss += current_loss.data[0]
            current_loss.backward()
            optimizer.step()

            current_sent += 1
            if current_sent == next_dot:
                next_dot += sents_per_dot
                print("%", end="", flush=True)

        print("], AEL =", loss / len(sents))
