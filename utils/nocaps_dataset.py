import json
import os

from datasets import load_dataset
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from torchvision.transforms.functional import InterpolationMode

from randaugment import RandomAugment


# Defined functions
def get_transform(train, augment_data, target_resolution, min_scale=0.5):

    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    assert target_resolution is not None

    if (not train) or (not augment_data):

        transform = transforms.Compose([
            transforms.Resize(target_resolution, interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])

    else:

        transform = transforms.Compose([                        
            transforms.RandomResizedCrop(target_resolution, scale=(min_scale, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2, 5, isPIL=True, augs=[
                'Identity', 'AutoContrast', 'Brightness', 'Sharpness', 'Equalize', 'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate'
            ]),     
            transforms.ToTensor(),
            normalize,
        ]) 

    return transform


# Defined classes
class nocaps_eval(Dataset):

    def __init__(self, root_dir, split, augment_data=False, target_resolution=None):   
        urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/nocaps_val.json',
                'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/nocaps_test.json'}
        filenames = {'val':'nocaps_val.json','test':'nocaps_test.json'}

        download_url(urls[split], root_dir)

        data = load_dataset(
            'HuggingFaceM4/NoCaps', 
            cache_dir=root_dir
        )
        self.features_mat = data['validation' if split == 'val' else split]

        self.annotation = json.load(open(os.path.join(root_dir, filenames[split]),'r'))
        self.transform = get_transform(
            train=(split=='train'), 
            augment_data=augment_data, 
            target_resolution=target_resolution
        )
        self.image_root = root_dir

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        ann = self.annotation[index]

        sample = self.features_mat[index]
        image = sample['image'].convert('RGB')
        image = self.transform(image)

        return {'pixel_values': image, 'image_id': int(ann['img_id'])}
