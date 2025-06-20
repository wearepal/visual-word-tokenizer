import argparse
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from transformers import set_seed
from tqdm import tqdm

sys.path.append(os.path.join('utils'))

from utils.confounder_utils import prepare_confounder_data
from utils.modeling_clip import CLIPModel
from utils.modeling_blip import BlipForConditionalGeneration
from utils.vis import make_visualization
from utils.vwt import wrap_model

matplotlib.use('agg')
plt.rcParams.update({'font.size': 14})


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

  args = parser.parse_args()

  # Set the hyperparameters
  DATA = args.data

  VOCAB = args.vocab

  IMGNET = args.imgnet == 'True'
  TOP_K = args.top_k
  THRESH = args.thresh
  RAND = args.rand == 'True'

  BATCH = 1

  # Define the variables
  if DATA in ['cub', 'celeba', 'meta', 'open_common', 'open_rare']: 
    model_name = f'openai/clip-vit-base-patch16'

  else:
    model_name = 'Salesforce/blip-image-captioning-base'

  if not VOCAB and TOP_K:
    folder = os.path.join(f'd{TOP_K}', f'b{BATCH}')

  elif VOCAB and THRESH:
    folder = os.path.join(f'v{VOCAB}' + ('-imgnet' if IMGNET else ''), f't{THRESH}', f'b{BATCH}')

  else:
    folder = os.path.join('base', f'b{BATCH}')

  save_path = os.path.join('logs', DATA, folder)


  # Run the experiments
  for i in ([0] if DATA in ['coco', 'nocaps'] else [1]):

    # Set the output directory
    print(f'\nSeed {i}:\n')

    output_dir = os.path.join(
      save_path, 
      '' if i == 0 else f'seed_{i}',
      'match'
    )

    # Load the dataset
    set_seed(i)

    dataset = prepare_confounder_data(DATA, shuffle=True)

    # Load the model
    set_seed(0)

    model = BlipForConditionalGeneration if DATA in ['coco', 'nocaps'] else CLIPModel
    model = model.from_pretrained(model_name)

    if TOP_K:
      num_patches = (model.config.vision_config.image_size // model.config.vision_config.patch_size) ** 2
      wrap_model(model.vision_model, top_k=int(num_patches * (1 - TOP_K)), rand=RAND)
      vwt = model.vision_model.embeddings

    elif THRESH:
      wrap_model(model.vision_model, thresh=THRESH, rand=RAND)
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


    # Generate the visualizations
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    split = DataLoader(
      dataset['val_data' if DATA == 'nocaps' else 'test_data'], 
      batch_size=1, 
      num_workers=4
    )
    model = model.vision_model.to(device)

    MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
    STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])

    for j, batch in enumerate(tqdm(split)):

      if j >= 300:
        break

      with torch.no_grad():
        image = split.dataset[j]['pixel_values']
        image = image * STD[:, None, None] + MEAN[:, None, None]
        image = image.permute(1, 2, 0)
        image = image.detach().cpu().numpy()

        h, w, _ = image.shape
        ph = h // 16
        pw = w // 16

        if TOP_K:
          outputs = vwt(batch['pixel_values'].to(device))
          labels = outputs.labels.squeeze()

          plt.title(f'Patches: {len(labels)}')
          plt.axis('off')
          plt.tight_layout()

          mask = torch.zeros(batch['pixel_values'].size(0), ph * pw)
          mask[:, labels] = 1

          mask = mask.float().view(1, 1, ph, pw)
          mask = F.interpolate(mask, size=(h, w), mode="nearest")

          mask = mask.view(h, w, 1).numpy()
          plt.imshow(image * mask)

        elif VOCAB:
          outputs = vwt(batch['pixel_values'].to(device))
          labels = outputs.labels.squeeze()

          plt.title(f'Patches: {len(labels.unique())}')
          plt.axis('off')
          plt.tight_layout()
          plt.imshow(image)

          image = make_visualization(image, labels, VOCAB, 16)
          plt.imshow(image, alpha=0.4)

        else:
          plt.title(f'Patches: {ph * pw}')
          plt.axis('off')
          plt.tight_layout()
          plt.imshow(image)

        if not os.path.exists(output_dir):
          os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, f'image_{j}.png'))

        plt.close()


if __name__ == '__main__':
  main()
