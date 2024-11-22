import argparse
import numpy as np
import os
import pandas as pd
import pynvml
import sys
import torch

from accelerate import init_empty_weights
from accelerate.utils import load_and_quantize_model
from accelerate.utils import BnbQuantizationConfig
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader
from transformers import set_seed
from tqdm import tqdm

sys.path.append(os.path.join('utils'))

from utils.confounder_utils import prepare_confounder_data
from utils.modeling_clip import CLIPModel
from utils.modeling_blip import BlipForConditionalGeneration
from utils.vwt import wrap_model


def main():

  # Define the arguments
  parser = argparse.ArgumentParser()

  parser.add_argument(
    '--data',
    type=str,
    choices=['cub', 'celeba', 'meta', 'open_common', 'open_rare', 'coco', 'nocaps'],
    required=True,
    help='The dataset to use.'
  )

  group = parser.add_mutually_exclusive_group()

  group.add_argument(
    '--quant',
    type=str,
    choices=['True', 'False'],
    help='Whether to quantize the model.'
  )
  group.add_argument(
    '--tome',
    type=int,
    help='The reduction size via ToMe.'
  )
  group.add_argument(
    '--vocab',
    type=int,
    choices=[100, 1000, 10000],
    help='The vocabulary size to use.'
  )

  parser.add_argument(
    '--imgnet',
    type=str,
    choices=['True', 'False'],
    help='Whether to use ImageNet-1K visual words.'
  )
  parser.add_argument(
    '--top_k',
    type=float,
    help='The top_k to use.'
  )
  parser.add_argument(
    '--thresh',
    type=float,
    help='The threshold to use.'
  )

  parser.add_argument(
    '--batch',
    type=int,
    required=True,
    help='The batch size to use.'
  )
  parser.add_argument(
    '--metric',
    type=str,
    choices=['watt', 'time'],
    required=True,
    help='The metric to measure.'
  )

  args = parser.parse_args()

  # Set the hyperparameters
  DATA = args.data

  QUANT = args.quant == 'True'
  TOME = args.tome
  VOCAB = args.vocab

  IMGNET = args.imgnet == 'True'
  TOP_K = args.top_k
  THRESH = args.thresh

  BATCH = args.batch
  METRIC = args.metric

  # Define the variables
  if DATA in ['cub', 'celeba', 'meta', 'open_common', 'open_rare']: 
    model_name = f'openai/clip-vit-base-patch16'

  else:
    model_name = 'Salesforce/blip-image-captioning-base'

  if TOME:
    folder = os.path.join(f'r{TOME}', f'b{BATCH}')

  elif not VOCAB and TOP_K:
    folder = os.path.join(f'd{TOP_K}', f'b{BATCH}')

  elif VOCAB and THRESH:
    folder = os.path.join(f'v{VOCAB}' + ('-imgnet' if IMGNET else ''), f't{THRESH}', f'b{BATCH}')

  else:
    folder = os.path.join('base', f'b{BATCH}')

  save_path = os.path.join('logs', DATA, folder)


  # Run the experiments
  for i in ([0] if DATA in ['open_common', 'open_rare', 'coco', 'nocaps'] else range(1, 4)):

    # Set the output directory
    print(f'\nSeed {i}:\n')

    output_dir = os.path.join(
      save_path,
      '' if i == 0 else f'seed_{i}',
      'quant' if QUANT else ''
    )

    # Load the dataset
    set_seed(i)

    dataset = prepare_confounder_data(DATA, shuffle=True)

    # Load the model
    set_seed(0)

    model = BlipForConditionalGeneration if DATA in ['coco', 'nocaps'] else CLIPModel
    if QUANT:
      with init_empty_weights():
        model = model.from_pretrained(model_name)

    else:
      model = model.from_pretrained(model_name)

    if TOME:
      for layer in model.vision_model.encoder.layers:
        layer.r += TOME

    elif TOP_K:
      num_patches = (model.config.vision_config.image_size // model.config.vision_config.patch_size) ** 2
      wrap_model(model.vision_model, top_k=int(num_patches * (1 - TOP_K)))
      vwt = model.vision_model.embeddings

    elif THRESH:
      wrap_model(model.vision_model, thresh=THRESH)
      vwt = model.vision_model.embeddings

    if VOCAB:
      tokenizer_dir = os.path.join(
        'tokenizers',
        'imgnet' if IMGNET else DATA,
        f'v{VOCAB}',
        '' if (IMGNET or i == 0) else f'seed_{i}',
        'vocab.pt'
      )
      vwt.load_words(tokenizer_dir, criterion='image')

    if QUANT:
      model = load_and_quantize_model(
        model, 
        weights_location=snapshot_download(repo_id=model_name), 
        bnb_quantization_config=BnbQuantizationConfig(load_in_8bit=True),
        device_map='auto'
      )

    model = model.vision_model


    # Compute the statistics
    df = pd.DataFrame()

    image_size = model.config.image_size
    dtype = torch.half if QUANT else torch.float
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    split = DataLoader(
      dataset['val_data' if DATA == 'nocaps' else 'test_data'], 
      batch_size=BATCH if BATCH else 1, 
      num_workers=4
    )
    if not QUANT:
      model = model.to(device)

    # Warm-up the model
    for _ in range(50):
      _ = model(torch.rand((1, 3, image_size, image_size), dtype=dtype, device=device))

    torch.cuda.synchronize()

    if METRIC == 'watt':
      pynvml.nvmlInit()
      gpu_index = int(os.environ['CUDA_VISIBLE_DEVICES'])
      handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

    metric = []
    for batch in tqdm(split):

      with torch.no_grad():

        if METRIC == 'watt':
          _ = model(batch['pixel_values'].to(device, dtype=dtype))

          torch.cuda.synchronize()
          metric += [pynvml.nvmlDeviceGetPowerUsage(handle) / 1000] * batch['pixel_values'].size(0)

        # Adapted from https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964/2
        elif METRIC == 'time':

          start = torch.cuda.Event(enable_timing=True)
          end = torch.cuda.Event(enable_timing=True)

          start.record()
          _ = model(batch['pixel_values'].to(device, dtype=dtype))
          end.record()

          torch.cuda.synchronize()
          metric += [start.elapsed_time(end)] * batch['pixel_values'].size(0)

    if METRIC == 'watt':
      pynvml.nvmlShutdown()

    df['average'] = np.array(metric)

    if DATA in ['cub', 'celeba', 'meta']:

      groups = split.dataset.get_group_array()
      for group_idx in np.unique(groups):

        submetric = np.empty(len(df['average']))
        submetric[:] = np.nan

        mask = (groups == group_idx)
        submetric[mask] = df['average'][mask]

        df[f'subgroup_{group_idx}'] = submetric

    # Save the statistics
    df = df.describe()
    print('\n', df, '\n')

    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    df.to_csv(os.path.join(output_dir, f'{METRIC}.csv'), index=False)


if __name__ == '__main__':
  main()