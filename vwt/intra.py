import torch
import torch.nn as nn

from vwt import WordTokenizer


def wrap_model(model, top_k):
    for name, module in model.named_children():

        if len(list(module.children())) > 0:
            wrap_model(module, top_k)

        if name == 'embeddings':
            setattr(model, name, IntraImageTokenizer(module, top_k))
            module.training = model.training


class IntraImageTokenizer(nn.Module, WordTokenizer):

    def __init__(self, embeddings, top_k):
        super().__init__()
        self.embeddings = embeddings
        self.top_k = top_k

        try:
            self.patch_embeddings = self.embeddings.patch_embeddings
        except:
            self.patch_embedding = self.embeddings.patch_embedding

    def forward(self, pixel_values, **kwargs):
        embeddings = self.embeddings.forward(pixel_values)
        return self.tokenize(pixel_values, embeddings, self.top_k)

    def tokenize(self, pixel_values, embeddings, top_k, **kwargs):
        patches = self.pretokenize(pixel_values, self.patch_size)
        scores = patches.var(dim=-1)

        _, indices = torch.topk(scores, top_k, largest=True)
        indices, _ = indices.sort()

        indices = indices.unsqueeze(-1).expand(-1, -1, embeddings.size(-1))
        batch = torch.gather(embeddings[:, 1:, :], 1, indices)

        batch = torch.cat((embeddings[:, :1, :], batch), dim=1)
        batch.labels = indices

        return batch
