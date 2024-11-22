import argparse
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
import torch

from sklearn.metrics import pairwise_distances_argmin_min
from torch.utils.data import DataLoader
from transformers import set_seed
from tqdm import tqdm

sys.path.append(os.path.join('utils'))

from utils.confounder_utils import prepare_confounder_data

matplotlib.use('agg')
plt.rcParams.update({'font.size': 14})


def main():

  # Define the arguments
  parser = argparse.ArgumentParser()

  parser.add_argument(
    '--vocab',
    type=int,
    choices=[100, 1000, 10000],
    required=True,
    help='The vocabulary size to use.'
  )

  args = parser.parse_args()

  # Set the hyperparameters
  DATA = 'imgnet'
  VOCAB = args.vocab

  # Define the variables
  folder = f'v{VOCAB}' if VOCAB else 'base'

  save_path = os.path.join('logs', DATA, folder)


  # Run the experiments
  i = 0

  # Set the output directory
  print(f'\nSeed {i}:\n')

  output_dir = os.path.join(save_path, 'word')

  # Load the dataset
  set_seed(i)

  dataset = prepare_confounder_data(DATA, shuffle=True)

  # Load the model
  set_seed(0)

  tokenizer_dir = os.path.join(
    'tokenizers',
    'imgnet',
    f'v{VOCAB}',
    'vocab.pt'
  )
  vocab = torch.load(tokenizer_dir)

  # Generate the visualizations
  split = DataLoader(
    dataset['train_data'], 
    batch_size=1, 
    num_workers=4
  )

  MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
  STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])

  check = {}
  for j, batch in enumerate(tqdm(split)):

    with torch.no_grad():
      patches = batch['pixel_values'].unfold(1, 3, 3).unfold(2, 16, 16).unfold(3, 16, 16)
      patches = patches.reshape(patches.size(0), -1, 3 * 16 * 16)

      patches = patches.reshape(-1, patches.size(-1))
      labels, _ = pairwise_distances_argmin_min(
        patches.detach().numpy(), 
        vocab.detach().numpy(),
        metric='euclidean'
      )

      for patch, label in zip(patches, labels):

        if label not in check:
          check[label] = True
          word = vocab[label]

          word = word.reshape(3, 16, 16)
          word = word * STD[:, None, None] + MEAN[:, None, None]
          word = word.permute(1, 2, 0)
          word = word.detach().cpu().numpy()

          plt.axis('off')
          plt.tight_layout()
          plt.imshow(word)

          if not os.path.exists(output_dir):
            os.makedirs(output_dir)
          plt.savefig(os.path.join(output_dir, f'word_{label}.png'))
          plt.close()

        patch = patch.reshape(3, 16, 16)
        patch = patch * STD[:, None, None] + MEAN[:, None, None]
        patch = patch.permute(1, 2, 0)
        patch = patch.detach().cpu().numpy()

        plt.axis('off')
        plt.tight_layout()
        plt.imshow(patch)

        if not os.path.exists(output_dir):
          os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, f'image_{j}.png'))
        plt.close()


if __name__ == '__main__':
  main()
