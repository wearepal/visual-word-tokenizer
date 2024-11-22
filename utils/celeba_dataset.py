import numpy as np
import os
import pandas as pd
import torchvision.transforms as transforms

from confounder_dataset import ConfounderDataset


# Defined functions
def get_transform(train, augment_data, target_resolution):

    orig_w = 178
    orig_h = 218
    orig_min_dim = min(orig_w, orig_h)

    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    if target_resolution is None:
        target_resolution = (orig_w, orig_h)

    if (not train) or (not augment_data):
        
        transform = transforms.Compose([
            transforms.CenterCrop(orig_min_dim),
            transforms.Resize(target_resolution),
            transforms.ToTensor(),
            normalize,
        ])

    else:

        # Orig aspect ratio is 0.81, so we don't squish it in that direction any more
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(1.0, 1.3333333333333333),
                interpolation=2,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    return transform


# Defined classes
class CelebADataset(ConfounderDataset):
    
    """
    CelebA dataset (already cropped and centered).
    Note: idx and filenames are off by one.
    """
    
    def __init__(self, root_dir, target_name, confounder_names, augment_data=False, target_resolution=None):
        
        self.root_dir = root_dir
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.augment_data = augment_data
        self.target_resolution = target_resolution

        # Read in attributes        
        self.attrs_df = pd.read_csv(os.path.join(self.root_dir, 'list_attr_celeba.csv'))

        # Split out filenames and attribute names
        self.filename_array = self.attrs_df['image_id'].values
        self.attrs_df = self.attrs_df.drop(labels='image_id', axis='columns')
        self.attr_names = self.attrs_df.columns.copy()

        # Then cast attributes to numpy array and set them to 0 and 1
        # (originally, they're -1 and 1)
        self.attrs_df = self.attrs_df.values
        self.attrs_df[self.attrs_df == -1] = 0

        # Get the y values
        target_idx = self.attr_idx(self.target_name)
        self.y_array = self.attrs_df[:, target_idx]
        self.n_classes = 2

        # Map the confounder attributes to a number 0,...,2^|confounder_idx|-1
        self.confounder_idx = [self.attr_idx(a) for a in self.confounder_names]
        self.n_confounders = len(self.confounder_idx)
        confounders = self.attrs_df[:, self.confounder_idx]
        self.confounder_array = np.matmul(
            confounders.astype(int), np.power(2, np.arange(len(self.confounder_idx)))
        )

        # Map to groups
        self.n_groups = self.n_classes * pow(2, len(self.confounder_idx))
        self.group_array = (self.y_array * (self.n_groups / 2) + self.confounder_array).astype('int')

        # Read in train/val/test splits
        self.split_df = pd.read_csv(os.path.join(self.root_dir, 'list_eval_partition.csv'))
        self.split_array = self.split_df['partition'].values
        self.split_dict = {'train': 0, 'val': 1, 'test': 2}

        self.features_mat = None
        self.train_transform = get_transform(train=True, augment_data=augment_data, target_resolution=self.target_resolution)
        self.eval_transform = get_transform(train=False, augment_data=augment_data, target_resolution=self.target_resolution)

        # Update the root dir
        self.root_dir = os.path.join(self.root_dir, 'img_align_celeba', 'img_align_celeba')

        # Define the labels
        self.labels = ['non-blond', 'blond']
        self.labels = [f'a photo of a {label}' for label in self.labels]

    def attr_idx(self, attr_name):

        return self.attr_names.get_loc(attr_name)

    def __getitem__(self, idx):

        x, y, g, idx = super().__getitem__(idx)
        return {'pixel_values': x, 'labels': y}
