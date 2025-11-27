# Visual-Word Tokenizer: Beyond Fixed Sets of Tokens in Vision Transformers

Official code for the paper titled [**Visual-Word Tokenizer: Beyond Fixed Sets of Tokens in Vision Transformers**](https://openreview.net/pdf?id=YYOS1FHYG3) on TMLR.

> The cost of deploying vision transformers increasingly represents a barrier to wider industrial adoption. Existing compression techniques require additional end-to-end fine-tuning or incur a significant drawback to energy efficiency, making them ill-suited for online (real-time) inference, where a prediction is made on any new input as it comes in. We introduce the Visual-Word Tokenizer (VWT), a training-free method for reducing energy costs while retaining performance. The VWT groups visual subwords (image patches) that are frequently used into visual words, while infrequent ones remain intact. To do so, intra-image or inter-image statistics are leveraged to identify similar visual concepts for sequence compression. Experimentally, we demonstrate a reduction in energy consumed of up to 47%. Comparative approaches of 8-bit quantization and token merging can lead to significantly increased energy costs (up to 500% or more). Our results indicate that VWTs are well-suited for efficient online inference with a marginal compromise on performance. The experimental code for our paper is also made publicly available.

## Usage

The code requires `python=3.10`.

```bash
cat requirements.txt | xargs -n 1 pip install
```

### Datasets

The CelebA and COCO (2014) datasets have to be downloaded manually and placed in the correct directory.

- **CelebA:** Download the dataset from [here](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) and place it in '/data/celeba/'.

- **COCO:** Download the dataset from [here](https://cocodataset.org/#download) and place it in '/data/coco/'.

### Running

If using the inter-image approach, the visual words have to be built first using `process.py`.

```bash
CUDA_VISIBLE_DEVICES=0 python process.py --data imgnet --space image --vocab 1000 --model clip
```

Subsequently, you may run the following scripts:

- `clip.py` and `blip.py` to measure performance.

- `len.py` and `eff.py` to measure efficiency.

- `match.py` and `word.py` for visualization.

Please refer to each individual script for the required and optional flags.

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
