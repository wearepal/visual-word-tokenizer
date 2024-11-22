import os
import fiftyone as fo
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset


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


# Defined classes
class OpenDataset(Dataset):

    def __init__(self, root_dir, split, type, augment_data=False, target_resolution=None):

        self.root_dir = root_dir
        self.augment_data = augment_data
        self.target_resolution = target_resolution

        # Customize where zoo datasets are downloaded
        fo.config.dataset_zoo_dir = self.root_dir
        if not os.path.exists(os.path.join(self.root_dir, 'open-images-v6')):
            _ = fo.zoo.load_zoo_dataset('open-images-v6', split=split)

        # Read in metadata
        self.annot_file = os.path.join(self.root_dir, type, 'annots.txt')
        with open(self.annot_file, 'r', encoding='utf-8') as f:
            self.imglist = [
                os.path.join(
                    self.root_dir, 
                    'open-images-v6', 
                    split,
                    'data', 
                    f"{line.strip().split('/')[-1].split(',')[0]}.jpg") for line in f
            ]

        self.transform = get_transform(
            train=split=='train', 
            augment_data=augment_data, 
            target_resolution=self.target_resolution
        )

        # Define the labels
        with open(os.path.join(self.root_dir, type, 'taglist.txt'), 'r', encoding='utf-8') as f:
            self.labels = [f'a photo of a {line.strip()}' for line in f]

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, idx):

        x = Image.open(self.imglist[idx]).convert('RGB')
        x = self.transform(x)
        return {'pixel_values': x, 'labels': self.labels}
