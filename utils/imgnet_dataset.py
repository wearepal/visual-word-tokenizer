import torchvision.transforms as transforms

from datasets import load_dataset
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
class IMGNETDataset(Dataset):

    def __init__(self, root_dir, split, augment_data=False, target_resolution=None):

        self.root_dir = root_dir
        self.augment_data = augment_data
        self.target_resolution = target_resolution

        data = load_dataset(
            'ILSVRC/imagenet-1k', 
            token=None, 
            cache_dir=self.root_dir
        )
        self.features_mat = data[split]

        self.transform = get_transform(
            train=split=='train', 
            augment_data=augment_data, 
            target_resolution=self.target_resolution
        )

    def __len__(self):
        return len(self.features_mat)

    def __getitem__(self, idx):

        sample = self.features_mat[idx]
        y = sample['label']

        x = sample['image'].convert('RGB')
        x = self.transform(x)

        return {'pixel_values': x, 'labels': y}
