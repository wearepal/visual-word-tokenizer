import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import torch

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

  parser.add_argument(
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
    '--thresh',
    type=float,
    help='The threshold to use.'
  )
  parser.add_argument(
    '--space',
    type=str,
    choices=['image', 'embed'],
    help='The space of the visual words to use.'
  )
  parser.add_argument(
    '--rand',
    type=str,
    choices=['True', 'False'],
    help='Whether to randomize the scores.'
  )

  parser.add_argument(
    '--metric',
    type=str,
    choices=['len', 'freq'],
    required=True,
    help='The metric to measure.'
  )

  args = parser.parse_args()

  # Set the hyperparameters
  DATA = args.data

  VOCAB = args.vocab

  IMGNET = args.imgnet == 'True'
  THRESH = args.thresh
  SPACE = args.space
  RAND = args.rand == 'True'

  BATCH = 1
  METRIC = args.metric

  # Define the variables
  if DATA in ['cub', 'celeba', 'meta', 'open_common', 'open_rare']: 
    model_name = f'openai/clip-vit-base-patch16'
    MODEL = 'clip'

  else:
    model_name = 'Salesforce/blip-image-captioning-base'
    MODEL = 'blip'

  folder = os.path.join(f'v{VOCAB}' + ('-imgnet' if IMGNET else ''), f't{THRESH}', f'b{BATCH}')

  save_path = os.path.join('logs', DATA, folder)


  # Run the experiments
  for i in ([0] if DATA in ['open_common', 'open_rare', 'coco', 'nocaps'] else range(1, 4)):

    if METRIC == 'freq' and i > 1:
      continue

    # Set the output directory
    print(f'\nSeed {i}:\n')

    output_dir = os.path.join(
      save_path,
      '' if i == 0 else f'seed_{i}',
      MODEL if SPACE == 'embed' else '',
      'rand' if RAND else ''
    )

    # Load the dataset
    set_seed(i)

    dataset = prepare_confounder_data(DATA, shuffle=True)

    # Load the model
    set_seed(0)

    model = BlipForConditionalGeneration if DATA in ['coco', 'nocaps'] else CLIPModel
    model = model.from_pretrained(model_name)

    wrap_model(model.vision_model, thresh=THRESH, rand=RAND)
    vwt = model.vision_model.embeddings

    tokenizer_dir = os.path.join(
      'tokenizers',
      'imgnet' if IMGNET else DATA,
      f'v{VOCAB}',
      '' if (IMGNET or i == 0) else f'seed_{i}',
      MODEL if SPACE == 'embed' else '',
      'vocab.pt'
    )
    vwt.load_words(tokenizer_dir, criterion=SPACE)


    # Compute the statistics
    df = pd.DataFrame()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    split = DataLoader(
      dataset['val_data' if DATA == 'nocaps' else 'test_data'], 
      batch_size=BATCH if BATCH else 1, 
      num_workers=4
    )
    model = model.vision_model.to(device)

    metric = [] if METRIC == 'len' else torch.zeros(VOCAB).to(device)
    for batch in tqdm(split):

      with torch.no_grad():

        if METRIC == 'len':
          outputs = model(batch['pixel_values'].to(device))
          metric += [outputs['last_hidden_state'].size(1)] * batch['pixel_values'].size(0)
        
        else:
          outputs = vwt(batch['pixel_values'].to(device))
          labels = outputs.labels.squeeze(0)
          labels = labels[labels < VOCAB]
          labels, counts = torch.unique(labels, return_counts=True)
          metric[[labels]] += counts

    if METRIC == 'len':
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
    if METRIC == 'len':
      df = df.describe()
      print('\n', df, '\n')

      if not os.path.exists(output_dir):
        os.makedirs(output_dir)
      df.to_csv(os.path.join(output_dir, 'len.csv'), index=False)

    else:
      metric = metric / metric.sum()
      plt.bar(range(VOCAB), metric.cpu().numpy())
      plt.xticks(list(range(0, VOCAB, VOCAB // 10)) + [VOCAB - 1])

      plt.title(f'$T_{{inter}}^{{{VOCAB}}}$')
      plt.xlabel('Visual Word')
      plt.ylabel('Probability')
      plt.tight_layout()

      if not os.path.exists(output_dir):
        os.makedirs(output_dir)
      plt.savefig(os.path.join(output_dir, 'freq.png'))
      plt.close()


if __name__ == '__main__':
  main()