import torch


class ZeroFiller:
    def __init__(self, size):
        self.size = size

    def __call__(self, is_word=True):
        return torch.zeros(self.size)


class UniformFiller:
    def __init__(self, size, minimum=-1.0, maximum=1.0):
        self.size = size
        self.minimum = minimum
        self.maximum = maximum

    def __call__(self, is_word=True):
        return torch.rand(self.size) * (self.maximum - self.minimum) - self.minimum


class NormalFiller:
    def __init__(self, size, mu=0, sigma=1):
        self.size = size
        self.mu = mu
        self.sigma = sigma

    def __call__(self, is_word=True):
        return torch.randn(self.size) * self.sigma + self.mu
