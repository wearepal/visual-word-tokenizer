import json
import os
import re

from PIL import Image
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from torchvision.transforms.functional import InterpolationMode

from randaugment import RandomAugment


# Defined functions
def coco_caption_eval(coco_gt_root, results_file, split):
    urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val_gt.json',
            'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json'}
    filenames = {'val':'coco_karpathy_val_gt.json','test':'coco_karpathy_test_gt.json'}    
    
    download_url(urls[split], coco_gt_root)
    annotation_file = os.path.join(coco_gt_root, filenames[split])

    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    path = os.path.split(results_file)[0]
    with open(os.path.join(path, 'results.txt'), 'w') as f:

        for metric, score in coco_eval.eval.items():
            print(f'{metric}: {score:.3f}', file=f)


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
class coco_karpathy_train(Dataset):

    def __init__(self, root_dir, split, augment_data=False, target_resolution=None, max_words=30, prompt=''):        
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''
        url = 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_train.json'
        filename = 'coco_karpathy_train.json'

        download_url(url, root_dir)

        self.annotation = json.load(open(os.path.join(root_dir, filename), 'r'))
        self.transform = get_transform(
            train=(split=='train'), 
            augment_data=augment_data, 
            target_resolution=target_resolution
        )
        self.image_root = root_dir
        self.max_words = max_words      
        self.prompt = prompt

        self.img_ids = {}  
        n = 0
        for ann in self.annotation:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.image_root,ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        caption = self.prompt + self.pre_caption(ann['caption'], self.max_words)

        return {'pixel_values': image, 'caption': caption, 'image_id': self.img_ids[ann['image_id']]}

    def pre_caption(self, caption, max_words=50):
        caption = re.sub(
            r"([.!\"()*#:;~])",       
            ' ',
            caption.lower(),
        )
        caption = re.sub(
            r"\s{2,}",
            ' ',
            caption,
        )
        caption = caption.rstrip('\n') 
        caption = caption.strip(' ')

        #truncate caption
        caption_words = caption.split(' ')
        if len(caption_words) > max_words:
            caption = ' '.join(caption_words[:max_words])

        return caption


class coco_karpathy_caption_eval(Dataset):

    def __init__(self, root_dir, split, augment_data=False, target_resolution=None):  
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json',
                'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json'}
        filenames = {'val':'coco_karpathy_val.json','test':'coco_karpathy_test.json'}

        download_url(urls[split], root_dir)

        self.annotation = json.load(open(os.path.join(root_dir, filenames[split]), 'r'))
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

        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        img_id = ann['image'].split('/')[-1].strip('.jpg').split('_')[-1]

        return {'pixel_values': image, 'image_id': int(img_id)}
