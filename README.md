# Efficient Online Inference of Vision Transformers by Training-Free Tokenization

The repository wraps the code for the paper titled [**Efficient Online Inference of Vision Transformers by Training-Free Tokenization**]() on Arxiv, into a ready to use library for your own application.

The [arxiv](https://github.com/wearepal/visual-word-tokenizer/tree/arxiv) branch contains the original code for the paper.  

**Authors:** Leonidas Gee, Wing Yan Li, Viktoriia Sharmanska, Novi Quadrianto

**Affiliations:** University of Sussex, University of Surrey, Basque Center for Applied Mathematics, Monash University (Indonesia)

## Installation
```
git clone https://github.com/wearepal/visual-word-tokenizer.git
```

## Usage

### Intra-image Approach
```python
from transformers import AutoModel

from vwt.intra import wrap_model


model = AutoModel.from_pretrained('openai/clip-vit-base-patch16')

# initializing an intra-image tokenizer
wrap_model(model.vision_model, top_k=98)

# deploy the model for inference on your downstream task...

```

### Inter-image Approach
```python
from transformers import AutoModel

from vwt.inter import wrap_model


model = AutoModel.from_pretrained('openai/clip-vit-base-patch16')

# load your pre-processing dataset here...
pre_process_data = ['A list of images', '...']  # dummy data

# initializing an intra-image tokenizer
wrap_model(model.vision_model, thresh=0.1)

vwt = model.vision_model.embeddings 
vwt.learn_words(
    split,
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

## Citation
