import argparse
import os
import sys

from accelerate import init_empty_weights
from accelerate.utils import load_and_quantize_model
from accelerate.utils import BnbQuantizationConfig
from huggingface_hub import snapshot_download
from transformers import set_seed

sys.path.append(os.path.join('utils'))

from utils import dec
from utils.coco_dataset import coco_caption_eval
from utils.confounder_utils import prepare_confounder_data
from utils.modeling_blip import BlipForConditionalGeneration
from utils.vwt import wrap_model

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def main():

  # Define the arguments
  parser = argparse.ArgumentParser()

  parser.add_argument(
    '--data',
    type=str,
    choices=['coco', 'nocaps'],
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
    '--rand',
    type=str,
    choices=['True', 'False'],
    help='Whether to randomize the scores.'
  )

  parser.add_argument(
    '--penalty',
    default=1.0,
    type=float,
    help='The length penalty to use.'
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
  RAND = args.rand == 'True'

  BATCH = 1
  PENALTY = args.penalty

  # Define the variables
  model_name = 'Salesforce/blip-image-captioning-base'

  if TOME:
    folder = os.path.join(f'r{TOME}', f'b{BATCH}')

  elif not VOCAB and TOP_K:
    folder = os.path.join(f'd{TOP_K}', f'b{BATCH}')

  elif VOCAB and THRESH:
    folder = os.path.join(f'v{VOCAB}' + ('-imgnet' if IMGNET else ''), f't{THRESH}', f'b{BATCH}')

  else:
    folder = os.path.join('base', f'b{BATCH}')

  folder = os.path.join(folder, f'p{PENALTY}' if PENALTY else '')

  save_path = os.path.join('logs', DATA, folder)

  # Define the configuation
  config = dict()
  config['batch_size'] = BATCH
  config['num_workers'] = 4


  # Run the experiments
  i = 0

  # Set the output directory
  print(f'\nSeed {i}:\n')

  config['output_dir'] = os.path.join(
    save_path,
      '' if i == 0 else f'seed_{i}',
    'quant' if QUANT else '',
    'rand' if RAND else ''
  )

  # Load the dataset
  set_seed(i)

  dataset = prepare_confounder_data(DATA)

  # Load the model
  set_seed(0)

  if QUANT:
    with init_empty_weights():
      model = BlipForConditionalGeneration.from_pretrained(model_name)

  else:
    model = BlipForConditionalGeneration.from_pretrained(model_name)

  if TOME:
    for layer in model.vision_model.encoder.layers:
      layer.r += TOME

  elif TOP_K:
    num_patches = (model.config.vision_config.image_size // model.config.vision_config.patch_size) ** 2
    wrap_model(model.vision_model, top_k=int(num_patches * (1 - TOP_K)))
    vwt = model.vision_model.embeddings

  elif THRESH:
    wrap_model(model.vision_model, thresh=THRESH, rand=RAND)
    vwt = model.vision_model.embeddings

  if VOCAB:
    tokenizer_dir = os.path.join(
      'tokenizers',
      'imgnet' if IMGNET else 'coco',
      f'v{VOCAB}',
      'vocab.pt'
    )
    vwt.load_words(tokenizer_dir, criterion='image')

  # Evaluate the model
  if QUANT:
    model = load_and_quantize_model(
      model, 
      weights_location=snapshot_download(model_name), 
      bnb_quantization_config=BnbQuantizationConfig(load_in_8bit=True), 
      device_map='auto'
    )

  result_file = dec.test_model(
    model, 
    config, 
    dataset['val_data' if DATA == 'nocaps' else 'test_data'], 
    length_penalty=PENALTY
  )

  # Save the results
  if DATA == 'coco':
    coco_caption_eval(os.path.join('data', args.data), result_file, 'test')


if __name__ == '__main__':
  main()
