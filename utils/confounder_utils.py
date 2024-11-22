import os
import torch

from torch.utils.data import Dataset


# Defined functions
def prepare_confounder_data(dataset, shuffle=False, **kwargs):

    if dataset in ['open_common', 'open_rare']:
        path = os.path.join('data', 'openimages')
    else:
        path = os.path.join('data', dataset)

    if dataset == 'cub':

        from cub_dataset import CUBDataset
        full_data = CUBDataset(
            root_dir=path,
            target_name='waterbird_complete95',
            confounder_names=['place'],
            augment_data=False,
            target_resolution=(224, 224)
        )

    elif dataset == 'celeba':

        from celeba_dataset import CelebADataset
        full_data = CelebADataset(
            root_dir=path,
            target_name='Blond_Hair',
            confounder_names=['Male'],
            augment_data=False,
            target_resolution=(224, 224)
        )

    elif dataset == 'meta':

        from meta_dataset import MetaDataset
        full_data = MetaDataset(
            root_dir=path,
            target_name='dog_cat',
            confounder_names=['place'],
            augment_data=False,
            target_resolution=(224, 224)
        )

    splits = ['train', 'val', 'test']

    data = {}
    for split in splits:

        if dataset == 'imgnet':

            from imgnet_dataset import IMGNETDataset
            data[f'{split}_data'] = IMGNETDataset(
                root_dir=path,
                split='validation' if split == 'val' else split,
                augment_data=False,
                target_resolution=(224, 224)
            )

        elif dataset in ['open_common', 'open_rare']:

            from open_dataset import OpenDataset
            if split != 'test':
                continue

            data[f'{split}_data'] = OpenDataset(
                root_dir=path,
                split=split,
                type=dataset.split('_')[1],
                augment_data=False,
                target_resolution=(224, 224)
            )

        elif dataset == 'coco':

            from coco_dataset import coco_karpathy_train, coco_karpathy_caption_eval
            data[f'{split}_data'] = (coco_karpathy_train if split == 'train' else coco_karpathy_caption_eval)(
                root_dir=path,
                split=split,
                augment_data=False,
                target_resolution=(384, 384)
            )
            if split == 'train':
                data[f'{split}_data'].prompt = 'a picture of '

        elif dataset == 'nocaps':

            from nocaps_dataset import nocaps_eval
            if split == 'train':
                continue

            data[f'{split}_data'] = nocaps_eval(
                root_dir=path,
                split=split,
                augment_data=False,
                target_resolution=(384, 384)
            )

        else:
            subsets = full_data.get_splits(splits, train_frac=1.0, shuffle=shuffle)

            data[f'{split}_data'] = DRODataset(
                subsets[split],
                process_item_fn=None,
                n_groups=full_data.n_groups,
                n_classes=full_data.n_classes,
                group_str_fn=full_data.group_str
            )

    return data


# Defined classes
class DRODataset(Dataset):

    def __init__(self, dataset, process_item_fn, n_groups, n_classes, group_str_fn):

        self.dataset = dataset
        self.process_item = process_item_fn
        self.n_groups = n_groups
        self.n_classes = n_classes
        self.group_str = group_str_fn

        group_array = self.get_group_array()
        y_array = self.get_label_array()

        self._group_array = torch.LongTensor(group_array)
        self._y_array = torch.LongTensor(y_array)

        self._group_counts = ((torch.arange(self.n_groups).unsqueeze(1) == self._group_array).sum(1).float())
        self._y_counts = (torch.arange(self.n_classes).unsqueeze(1) == self._y_array).sum(1).float()

        self.labels = self.dataset.labels

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, idx):

        if self.process_item is None:
            return self.dataset[idx]
        else:
            return self.process_item(self.dataset[idx])

    def input_size(self):

        for x, y, g, _ in self:
            return x.size()

    def get_label_array(self):

        if self.process_item is None:
            return self.dataset.get_label_array()
        else:
            raise NotImplementedError

    def get_group_array(self):

        if self.process_item is None:
            return self.dataset.get_group_array()
        else:
            raise NotImplementedError

    def class_counts(self):

        return self._y_counts

    def group_counts(self):

        return self._group_counts
