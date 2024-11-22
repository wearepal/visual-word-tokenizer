import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from typing import Optional


# Defined functions
def wrap_model(model, top_k=None, thresh=0.0, rand=False):
    for name, module in model.named_children():

        if len(list(module.children())) > 0:
            wrap_model(module, top_k, thresh, rand)

        if name == 'embeddings':
            setattr(model, name, VisualWordTokenizer(module, top_k, thresh, rand))
            module.training = model.training


# Adapted from https://github.com/facebookresearch/fairseq/blob/129d8594ccdc6644be84dc249e16489e049f4bfd/fairseq/utils.py#L265
def buffered_arange(end):
    if not hasattr(buffered_arange, 'buf'):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        buffered_arange.buf = torch.tensor([], dtype=torch.long, device=device)

    if end > buffered_arange.buf.numel():
        buffered_arange.buf.resize_(end)
        torch.arange(end, out=buffered_arange.buf)

    return buffered_arange.buf[:end]


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


# Defined classes
class VisualWordTokenizer(nn.Module):

    def __init__(self, embeddings, top_k, thresh, rand):
        super().__init__()
        self.embeddings = embeddings
        self.patch_size = self.embeddings.patch_size

        self.top_k = top_k
        self.thresh = thresh

        self.vocab = None
        self.rand = rand

        try:
            self.patch_embeddings = self.embeddings.patch_embeddings
        except:
            self.patch_embedding = self.embeddings.patch_embedding

        self.pad_token = nn.Parameter(torch.zeros(1, 1, embeddings.config.hidden_size))

    def forward(self, pixel_values, **kwargs):
        embeddings = self.embeddings.forward(pixel_values)
        return self.tokenize(pixel_values, embeddings, self.top_k, self.thresh, self.vocab)

    def tokenize(self, pixel_values, embeddings, top_k, thresh, vocab, **kwargs):
        if not hasattr(self, 'criterion'):
            if top_k:
                patches = self.pretokenize(pixel_values, self.patch_size)
                scores = patches.var(dim=-1)

                _, indices = torch.topk(scores, top_k, largest=True)
                indices, _ = indices.sort()

                indices = indices.unsqueeze(-1).expand(-1, -1, embeddings.size(-1))
                batch = torch.gather(embeddings[:, 1:, :], 1, indices)

                batch = torch.cat((embeddings[:, :1, :], batch), dim=1)
                batch.labels = indices

            elif thresh:
                raise NotImplementedError

        else:
            if self.criterion == 'image':
                patches = self.pretokenize(pixel_values, self.patch_size)

            elif self.criterion == 'embed':
                try:
                    patches = self.patch_embeddings.forward(pixel_values)
                except:
                    patches = self.patch_embedding.forward(pixel_values)

                patches = patches.flatten(2).transpose(1, 2)

            scores = 1 - torch.matmul(F.normalize(patches, p=2, dim=-1), vocab)
            if self.rand:
                scores = torch.rand_like(scores)
                scores = scores.uniform_(0, 2)

            if top_k:
                raise NotImplementedError

            elif thresh:
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
                    batch.attention_mask = _expand_mask(~bool_masked_pos, batch.dtype)

        return batch

    # Adapted from https://discuss.pytorch.org/t/how-to-extract-patches-from-an-image/79923/4
    def pretokenize(self, pixel_values, patch_size, **kwargs):
        patches = pixel_values.unfold(1, 3, 3).unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        return patches.reshape(patches.size(0), -1, 3 * patch_size * patch_size)

    def mean_by_label(self, embeddings, labels, **kwargs):
        labels = labels[..., None].expand_as(embeddings)
        mean = torch.full_like(embeddings, torch.nan, device=embeddings.device)
        mean = mean.scatter_reduce(1, labels, embeddings, reduce='mean', include_self=False)
        return mean[:, :labels.max() + 1, :]

    def learn_words(self, data, vocab_size, criterion, **kwargs):
        self.criterion = criterion
        model = MiniBatchKMeans(n_clusters=vocab_size, n_init='auto', **kwargs)

        for batch in tqdm(data):
            if isinstance(batch, dict):
                batch = batch['pixel_values']

            if self.criterion == 'image':
                patches = self.pretokenize(batch, self.patch_size)

            elif self.criterion == 'embed':
                try:
                    patches = self.patch_embeddings.forward(batch)
                except:
                    patches = self.patch_embedding.forward(batch)

                patches = patches.flatten(2).transpose(1, 2)

            patches = patches.reshape(-1, patches.size(-1))
            model.partial_fit(patches.detach().numpy())

        self.vocab = torch.from_numpy(model.cluster_centers_)

    def save_pretrained(self, save_directory, **kwargs):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        torch.save(self.vocab, os.path.join(save_directory, 'vocab.pt'))

    def load_words(self, data, criterion, **kwargs):
        self.criterion = criterion
        self.vocab = torch.load(data)
        print(f'\nLoaded vocabulary from {data}.\n')

        if torch.cuda.is_available():
            self.vocab = self.vocab.cuda()

        self.vocab = F.normalize(self.vocab, p=2, dim=-1).T.unsqueeze(0)
