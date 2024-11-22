import argparse
import os
import sys

from torch.utils.data import DataLoader
from transformers import set_seed

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
    choices=['cub', 'celeba', 'meta', 'coco', 'imgnet'],
    required=True,
    help='The dataset to use.'
  )
  parser.add_argument(
    '--space',
    type=str,
    choices=['image', 'embed'],
    required=True,
    help='Whether to use the image or embedding visual words.'
  )
  parser.add_argument(
    '--vocab',
    type=int,
    choices=[100, 1000, 10000],
    required=True,
    help='The vocabulary size to use.'
  )

  parser.add_argument(
    '--model',
    type=str,
    choices=['clip', 'blip'],
    required=True,
    help='The model to use.'
  )

  parser.add_argument(
    '--ovr',
    type=str,
    choices=['True', 'False'],
    help='Whether to overwrite the existing vocabulary.'
  )

  args = parser.parse_args()

  # Set the hyperparameters
  DATA = args.data
  SPACE = args.space
  VOCAB = args.vocab

  MODEL = args.model

  OVR = args.ovr == 'True'

  # Check the hyperparameters
  if DATA in ['cub', 'celeba', 'meta']:
    assert MODEL == 'clip'

  if DATA == 'coco':
    assert MODEL == 'blip'

  # Define the variables
  if MODEL == 'clip':
    model_name = f'openai/clip-vit-base-patch16'

  elif MODEL == 'blip':
    model_name = 'Salesforce/blip-image-captioning-base'

  folder = f'v{VOCAB}' if VOCAB else 'base'

  save_path = os.path.join('tokenizers', DATA)
  save_path = os.path.join(save_path, folder)

  # Define the configuation
  batch_size = 1024


  # Run the experiments
  for i in ([0] if DATA in ['coco', 'imgnet'] else range(1, 4)):

    # Set the output directory
    print(f'\nSeed {i}:\n')

    output_dir = os.path.join(
      save_path,
      '' if i == 0 else f'seed_{i}',
      MODEL if SPACE == 'embed' else ''
    )

    # Skip if tokenizer exists
    if not OVR and os.path.exists(os.path.join(output_dir, 'vocab.pt')):
      print('Tokenizer already exists.\n')
      continue

    # Load the dataset
    set_seed(i)

    dataset = prepare_confounder_data(DATA, shuffle=True)

    # Load the model
    set_seed(0)

    if MODEL == 'clip':
      model = CLIPModel.from_pretrained(model_name)

    elif MODEL == 'blip':
      model = BlipForConditionalGeneration.from_pretrained(model_name)

    # Load the tokenizer
    wrap_model(model.vision_model)
    vwt = model.vision_model.embeddings

    # Learn the visual words
    split = DataLoader(
      dataset['train_data'], 
      batch_size=batch_size, 
      num_workers=4
    )

    vwt.learn_words(
      split,
      vocab_size=VOCAB,
      space=SPACE,
      batch_size=batch_size
    )

    # Save the tokenizer
    vwt.save_pretrained(output_dir)


if __name__ == '__main__':
  main()
