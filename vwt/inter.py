import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from typing import Optional
from vwt import WordTokenizer


def wrap_model(model, thresh):
    for name, module in model.named_children():

        if len(list(module.children())) > 0:
            wrap_model(module, thresh)

        if name == 'embeddings':
            setattr(model, name, InterImageTokenizer(module, thresh))
            module.training = model.training


def buffered_arange(end):
    if not hasattr(buffered_arange, 'buf'):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        buffered_arange.buf = torch.tensor([], dtype=torch.long, device=device)

    if end > buffered_arange.buf.numel():
        buffered_arange.buf.resize_(end)
        torch.arange(end, out=buffered_arange.buf)

    return buffered_arange.buf[:end]


class InterImageTokenizer(nn.Module, WordTokenizer):

    def __init__(self, embeddings, thresh):
        super().__init__()
        self.embeddings = embeddings
        self.thresh = thresh

        try:
            self.patch_embeddings = self.embeddings.patch_embeddings
        except:
            self.patch_embedding = self.embeddings.patch_embedding

        self.pad_token = nn.Parameter(torch.zeros(1, 1, embeddings.config.hidden_size))

    def forward(self, pixel_values, **kwargs):
        embeddings = self.embeddings.forward(pixel_values)
        return self.tokenize(pixel_values, embeddings, self.thresh, self.vocab)

    def tokenize(self, pixel_values, embeddings, thresh, vocab, **kwargs):
        patches = self.pretokenize(pixel_values, self.patch_size)
        scores = 1 - torch.matmul(F.normalize(patches, p=2, dim=-1), vocab)

        scores, labels = scores.min(-1)
        masks = scores > thresh

        labels[masks] = buffered_arange(masks.sum()) + self.vocab.size(-1)
        labels, indices = labels.sort()

        for i, label in enumerate(labels):
            _, labels[i] = torch.unique_consecutive(label, return_inverse=True)

        labels = labels.gather(1, indices.argsort())
        batch = self.mean_by_label(embeddings[:, 1:, :], labels)

        batch = torch.cat((embeddings[:, :1, :], batch), dim=1)
        batch.labels = labels

        if batch.size(0) > 1:

            # replace the masked visual tokens by pad_tokens
            bool_masked_pos = batch.isnan().all(-1)
            batch[bool_masked_pos] = self.pad_token

            # expand attention_mask
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            batch.attention_mask = self._expand_mask(~bool_masked_pos, batch.dtype)

        return batch

    def mean_by_label(self, embeddings, labels, **kwargs):
        labels = labels[..., None].expand_as(embeddings)
        mean = torch.full_like(embeddings, torch.nan, device=embeddings.device)
        mean = mean.scatter_reduce(1, labels, embeddings, reduce='mean', include_self=False)
        return mean[:, :labels.max() + 1, :]

    @staticmethod
    def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
        """
        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
        """
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

        inverted_mask = 1.0 - expanded_mask

        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

    def learn_words(self, data, patch_size, vocab_size, **kwargs):
        self.patch_size = patch_size
        model = MiniBatchKMeans(n_clusters=vocab_size, n_init='auto', **kwargs)

        for batch in tqdm(data):
            if isinstance(batch, dict):
                batch = batch['pixel_values']

            patches = self.pretokenize(batch, self.patch_size)

            patches = patches.reshape(-1, patches.size(-1))
            model.partial_fit(patches.detach().numpy())

        self.vocab = torch.from_numpy(model.cluster_centers_)

    def save_pretrained(self, save_directory, **kwargs):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        torch.save(self.vocab, os.path.join(save_directory, 'vocab.pt'))

    def load_words(self, data, **kwargs):
        self.vocab = torch.load(data)
        if torch.cuda.is_available():
            self.vocab = self.vocab.cuda()

        self.vocab = F.normalize(self.vocab, p=2, dim=-1).T.unsqueeze(0)
        self.patch_size = int(math.sqrt(self.vocab.size(1) // 3))