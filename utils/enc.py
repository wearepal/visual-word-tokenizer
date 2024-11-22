import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from transformers import AutoProcessor
from tqdm import tqdm
from typing import TextIO

from loss import LossComputer
from metrics import get_mAP
from open_dataset import OpenDataset


# Defined functions
def test_model(model, args, test_data):
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.eval()
        model = model.to(device)
    except:
        pass

    dataset = DataLoader(test_data, batch_size=args['batch_size'], num_workers=args['num_workers'])

    processor = AutoProcessor.from_pretrained(model.name_or_path)
    label_tokens = processor(
        text=test_data.labels,
        padding=True,
        images=None,
        return_tensors='pt'
    )

    label_tokens = label_tokens.to(device)
    label_emb = model.get_text_features(**label_tokens)

    preds = []
    for sample in tqdm(dataset):

        with torch.no_grad():
            image = sample['pixel_values'].to(device)

            try:
                img_emb = model.get_image_features(image)
            except:
                img_emb = model.get_image_features(image.half())

            scores = torch.matmul(
                F.normalize(img_emb, p=2, dim=-1), 
                F.normalize(label_emb, p=2, dim=-1).T
            )
            preds.append(scores.detach().cpu().numpy())

    preds = np.concatenate(preds)
    preds = np.float32(preds)

    # Save the predictions
    path = args['output_dir']
    if not os.path.exists(path):
        os.makedirs(path)

    output = pd.DataFrame({'Prediction': preds.argmax(axis=1)})
    if not isinstance(test_data, OpenDataset):
        labels, groups = test_data.get_label_array(), test_data.get_group_array()
        output['Label'], output['Group'] = labels, groups

    output.to_csv(os.path.join(path, 'predictions.csv'), index=False)

    # Save the results
    if isinstance(test_data, OpenDataset):
        def print_write(f: TextIO, s: str):
            print(s)
            f.write(s + '\n')

        mAP, _ = get_mAP(preds, test_data.annot_file, test_data.labels)
        with open(os.path.join(path, 'results.txt'), 'w', encoding='utf-8') as f:
            print_write(f, f'mAP: {mAP*100.0}')

    else:
        loss_computer = LossComputer(test_data, nn.CrossEntropyLoss(reduction='none'))
        loss_computer.loss(
            torch.from_numpy(preds).to(device), 
            torch.from_numpy(labels).to(device), 
            torch.from_numpy(groups).to(device)
        )
        loss_computer.log_stats(path)
