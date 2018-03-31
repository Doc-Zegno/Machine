import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


# POS Machine
# Core part -- bi-LSTM
class Tagger(nn.Module):
    def __init__(self, input_size, output_size,
                 machine="LSTM", hidden_size=None, clip_grad=None, compression_size=None):
        super(Tagger, self).__init__()

        # Cache parameters
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = input_size if hidden_size is None else hidden_size
        self.clip_grad = clip_grad

        # Define used machines
        self.compressor = None
        if compression_size is not None:
            self.compressor = nn.Linear(compression_size, input_size)

        kwargs = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "bidirectional": True,
            "bias": True
        }
        if machine == "LSTM":
            self.unit = nn.LSTM(**kwargs)
        elif machine == "GRU":
            self.unit = nn.GRU(**kwargs)
        else:
            raise RuntimeError("unknown machine name: {}".format(machine))

        # self.unit = nn.GRU(input_size=input_size, hidden_size=self.hidden_size, bidirectional=True)
        self.hidden2tags = nn.Linear(self.hidden_size * 2, output_size)

        # Transfer to CUDA
        if torch.cuda.is_available():
            if self.compressor is not None:
                self.compressor = self.compressor.cuda()
            self.unit = self.unit.cuda()
            self.hidden2tags = self.hidden2tags.cuda()

        # Store hidden state for the last input sequence
        self.last_output = None

    def reset(self):
        pass

    @staticmethod
    def clip(var, minimum, maximum):
        var.register_hook(lambda g: g.clamp(minimum, maximum))

    def forward(self, embed):
        """Taking word embedding as input, return log probabilities of its POS tag."""
        if self.compressor is not None:
            embed = F.sigmoid(self.compressor(embed))

        output, hidden = self.unit(embed)

        # Adding ReLU
        output = F.relu(output)

        # Store last hidden state (i.e. output)
        self.last_output = output

        # Clip gradient
        if self.clip_grad is not None:
            self.clip(output, -self.clip_grad, self.clip_grad)

        # 'output' is (seq_len, 1, 2 * hidden_size).
        # Reshape it to (seq_len, 2 * hidden_size).
        # Convert it to tags
        seq_len = output.size()[0]
        output = output.view((seq_len, -1))
        tags = self.hidden2tags(output)

        # Apply ReLU
        # tags = F.relu(tags)

        # Return log probabilities
        return F.log_softmax(tags, dim=1)
