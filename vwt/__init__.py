import abc
import torch

from typing import Optional


class AbstractWordTokenizer(metaclass=abc.ABCMeta):

    def __init__(self):
        self.embeddings = None
        self.top_k = None
        self.thresh = None
        self.vocab = None
        self.pad_token = None

    @abc.abstractmethod
    def forward(self, pixel_values, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def tokenize(self, pixel_values, embeddings, top_k, thresh, vocab, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def pretokenize(self, pixel_values, patch_size, **kwargs):
        raise NotImplementedError


class WordTokenizer(AbstractWordTokenizer):

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self, pixel_values, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def tokenize(self, pixel_values, embeddings, top_k, thresh, vocab, **kwargs):
        raise NotImplementedError

    def pretokenize(self, pixel_values, patch_size, **kwargs):
        patches = pixel_values.unfold(1, 3, 3).unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        return patches.reshape(patches.size(0), -1, 3 * patch_size * patch_size)
