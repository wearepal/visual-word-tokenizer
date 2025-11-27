# Visual-Word Tokenizer: Beyond Fixed Sets of Tokens in Vision Transformers

The repository wraps the code for the paper titled [**Visual-Word Tokenizer: Beyond Fixed Sets of Tokens in Vision Transformers**](https://openreview.net/forum?id=YYOS1FHYG3) on TMLR, into a ready to use library for your own application.

The [tmlr](https://github.com/wearepal/visual-word-tokenizer/tree/tmlr) branch contains the original code for the paper.  

**Authors:** Leonidas Gee, Wing Yan Li, Viktoriia Sharmanska, Novi Quadrianto

**Affiliations:** University of Sussex, University of Surrey, Basque Center for Applied Mathematics, Monash University (Indonesia)

## Installation
```
git clone https://github.com/wearepal/visual-word-tokenizer.git
```

## Usage

### Intra-image Approach
Note that the `top_k` flag corresponds to retaining the top-K most heterogenous patches in the image. In our paper, we set `top_k` to the (total number of patches - number of patches to drop).

```python
from transformers import AutoModel

from vwt.intra import wrap_model


model = AutoModel.from_pretrained('openai/clip-vit-base-patch16')

# initializing an intra-image tokenizer
wrap_model(model.vision_model, top_k=98)

# deploy the model for inference on your downstream task...

```

### Inter-image Approach
Note that the inter-image approach requires attention masking to work. The encoder implementation of the transformer on HuggingFace already possesses an `attention_mask` flag that is unused for the vision transformer. Add the following line to the input of `self.encoder`:

```python
attention_mask=getattr(hidden_states, 'attention_mask', None),

```

Please refer to *modeling_clip.py* and *modeling_blip.py* in the examples folder for more clarity.

```python
from examples.modeling_clip import CLIPModel

from vwt.inter import wrap_model


model = CLIPModel.from_pretrained('openai/clip-vit-base-patch16')

# load your pre-processing dataset here...
pre_process_data =  # a dataloader of image tensors

# initializing an intra-image tokenizer
wrap_model(model.vision_model, thresh=0.1)

vwt = model.vision_model.embeddings 
vwt.learn_words(
    pre_process_data,
    vocab_size=1000, # number of visual words
    batch_size=1024 # batch size for clustering
)
# deploy the model for inference on your downstream task...

# saving the visual word vocabulary
vwt.save_pretrained('pre_process_data')

# reusing the visual word vocabulary
new_model = AutoModel.from_pretrained('openai/clip-vit-base-patch16')
wrap_model(new_model.vision_model, thresh=0.1)

new_vwt = model.vision_model.embeddings 
new_vwt.load_words('pre_process_data/vocab.pt')

```

You may also load the pre-processed visual words from HuggingFace. We provide the ImageNet-1K vocabulary with sizes of 100, 1000, and 10000.

```python
from huggingface_hub import snapshot_download

from examples.modeling_clip import CLIPModel
from vwt.inter import wrap_model


model = CLIPModel.from_pretrained('openai/clip-vit-base-patch16')

# initializing an intra-image tokenizer
wrap_model(model.vision_model, thresh=0.1)

vwt = model.vision_model.embeddings 

# downloading the visual word vocabulary
snapshot_download(repo_id='LeonidasY/inter-image-imgnet-100', local_dir='tokenizer')
# snapshot_download(repo_id='LeonidasY/inter-image-imgnet-1000', local_dir='tokenizer')
# snapshot_download(repo_id='LeonidasY/inter-image-imgnet-10000', local_dir='tokenizer')

# loading the visual word vocabulary
vwt.load_words('tokenizer/vocab.pt')

# deploy the model for inference on your downstream task...

```

## Citation
```
@article{
gee2025visualword,
title={Visual-Word Tokenizer: Beyond Fixed Sets of Tokens in Vision Transformers},
author={Leonidas Gee and Wing Yan Li and Viktoriia Sharmanska and Novi Quadrianto},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025},
url={https://openreview.net/forum?id=YYOS1FHYG3},
note={}
}
```
