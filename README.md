# Efficient Online Inference of Vision Transformers by Training-Free Tokenization

Official code for the paper titled [**Efficient Online Inference of Vision Transformers by Training-Free Tokenization**](https://arxiv.org/abs/2411.15397) on Arxiv.

> The cost of deploying vision transformers increasingly represents a barrier to wider industrial adoption. Existing compression requires additional end-to-end fine-tuning or incurs a significant drawback to runtime, thus making them ill-suited for online inference. We introduce the **Visual Word Tokenizer** (VWT), a training-free method for reducing energy costs while retaining performance and runtime. The VWT groups patches (visual subwords) that are frequently used into visual words while infrequent ones remain intact. To do so, intra-image or inter-image statistics are leveraged to identify similar visual concepts for compression. Experimentally, we demonstrate a reduction in wattage of up to 19% with only a 20% increase in runtime at most. Comparative approaches of 8-bit quantization and token merging achieve a lower or similar energy efficiency but exact a higher toll on runtime (up to $2\times$ or more). Our results indicate that VWTs are well-suited for efficient online inference with a marginal compromise on performance.

## Usage

The code requires `python=3.8`.

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
@misc{gee2024efficientonlineinferencevision,
      title={Efficient Online Inference of Vision Transformers by Training-Free Tokenization}, 
      author={Leonidas Gee and Wing Yan Li and Viktoriia Sharmanska and Novi Quadrianto},
      year={2024},
      eprint={2411.15397},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.15397}, 
}
```
