import gdown
import logging
import numpy as np
import os
import pandas as pd
import tarfile
import torchvision.transforms as transforms

from pathlib import Path

from confounder_dataset import ConfounderDataset


# Defined functions
def get_transform(train, augment_data, target_resolution):

    scale = 256.0 / 224.0
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    assert target_resolution is not None

    if (not train) or (not augment_data):

        # Resizes the image to a slightly larger square then crops the center.
        transform = transforms.Compose([
            transforms.Resize((
                int(target_resolution[0] * scale),
                int(target_resolution[1] * scale),
            )),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            normalize,
        ])

    else:

        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=2,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    return transform


# Adapated from https://github.com/YyzHarry/SubpopBench/blob/main/subpopbench/scripts/download.py
def download_and_extract(url, dst, remove=True):
    gdown.download(url, dst, quiet=False)
    tar = tarfile.open(dst, 'r:gz')
    tar.extractall(os.path.dirname(dst))
    tar.close()
    if remove:
        os.remove(dst)


def generate_metadata_metashift(data_path, test_pct=0.25, val_pct=0.1):
    logging.info('Generating metadata for MetaShift...')
    dirs = {
        'train/cat/cat(indoor)': [1, 1],
        'train/dog/dog(outdoor)': [0, 0],
        'test/cat/cat(outdoor)': [1, 0],
        'test/dog/dog(indoor)': [0, 1]
    }

    all_data = []
    for dir in dirs:
        folder_path = os.path.join(data_path, dir)
        y = dirs[dir][0]
        g = dirs[dir][1]
        for img_path in Path(folder_path).glob('*.jpg'):
            all_data.append({
                'img_filename': os.path.join(dir, os.path.split(img_path)[1]),
                'y': y,
                'place': g
            })
    df = pd.DataFrame(all_data)

    rng = np.random.RandomState(42)

    test_idxs = rng.choice(np.arange(len(df)), size=int(len(df) * test_pct), replace=False)
    val_idxs = rng.choice(np.setdiff1d(np.arange(len(df)), test_idxs), size=int(len(df) * val_pct), replace=False)

    split_array = np.zeros((len(df), 1))
    split_array[val_idxs] = 1
    split_array[test_idxs] = 2

    df['split'] = split_array.astype(int)
    df.to_csv(os.path.join(data_path, 'metadata.csv'), index=False)


# Defined classes
class MetaDataset(ConfounderDataset):

    def __init__(self, root_dir, target_name, confounder_names, augment_data=False, target_resolution=None):
        
        self.root_dir = root_dir
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.augment_data = augment_data
        self.target_resolution = target_resolution

        # Prepare the data
        if not os.path.exists(os.path.join(self.root_dir, 'metadata.csv')):
            os.makedirs(self.root_dir)
            head = os.path.split(self.root_dir)[0]

            download_and_extract(
                'https://www.dropbox.com/s/a7k65rlj4ownyr2/metashift.tar.gz?dl=1',
                os.path.join(head, 'metashift.tar.gz'),
                remove=True
            )
            os.rename(os.path.join(head, 'MetaShift-Cat-Dog-indoor-outdoor'), self.root_dir)

            generate_metadata_metashift(self.root_dir)

        # Read in metadata
        self.metadata_df = pd.read_csv(os.path.join(self.root_dir, 'metadata.csv'))

        # Get the y values
        self.y_array = self.metadata_df['y'].values
        self.n_classes = 2

        # We only support one confounder for CUB for now
        assert len(self.confounder_names) == 1
        self.confounder_array = self.metadata_df[self.confounder_names[0]].values
        self.n_confounders = 1

        # Map to groups
        self.n_groups = pow(2, 2)
        assert self.n_groups == 4
        self.group_array = (self.y_array * (self.n_groups / 2) + self.confounder_array).astype('int')

        # Extract filenames and splits
        self.filename_array = self.metadata_df['img_filename'].values
        self.split_array = self.metadata_df['split'].values
        self.split_dict = {'train': 0, 'val': 1, 'test': 2}

        # Set transform
        self.features_mat = None
        self.train_transform = get_transform(train=True, augment_data=augment_data, target_resolution=self.target_resolution)
        self.eval_transform = get_transform(train=False, augment_data=augment_data, target_resolution=self.target_resolution)

        # Define the labels
        self.labels = ['dog', 'cat']
        self.labels = [f'a photo of a {label}' for label in self.labels]

    def __getitem__(self, idx):

        x, y, g, idx = super().__getitem__(idx)
        return {'pixel_values': x, 'labels': y}
