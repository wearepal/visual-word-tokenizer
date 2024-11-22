import gdown
import os
import pandas as pd
import tarfile
import torchvision.transforms as transforms

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


def download_and_extract(url, dst, remove=True):
    gdown.download(url, dst, quiet=False)
    tar = tarfile.open(dst, 'r:gz')
    tar.extractall(os.path.dirname(dst))
    tar.close()
    if remove:
        os.remove(dst)


# Defined classes
class CUBDataset(ConfounderDataset):

    """
    CUB dataset (already cropped and centered).
    NOTE: metadata_df is one-indexed.
    """

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
                'https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz',
                os.path.join(head, 'waterbird_complete95_forest2water2.tar.gz'),
                remove=True
            )
            os.rename(os.path.join(head, 'waterbird_complete95_forest2water2'), self.root_dir)

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
        self.labels = ['landbird', 'waterbird']
        self.labels = [f'a photo of a {label}' for label in self.labels]

    def __getitem__(self, idx):

        x, y, g, idx = super().__getitem__(idx)
        return {'pixel_values': x, 'labels': y}
