import json
import os
import torch

from torch.utils.data import DataLoader
from transformers import AutoProcessor
from tqdm import tqdm


# Defined functions
def test_model(model, args, test_data, **kwargs):
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.eval()
        model = model.to(device)
    except:
        pass

    dataset = DataLoader(test_data, batch_size=args['batch_size'], num_workers=args['num_workers'])

    processor = AutoProcessor.from_pretrained(model.name_or_path)
    config = {
        'max_length': 40 if kwargs.get('length_penalty', 1.0) > 1.0 else 20,
        'min_length': 5,
        'do_sample': False,
        'num_beams': 3,
        **kwargs
    }

    result = []
    for sample in tqdm(dataset): 

        with torch.no_grad():
            image = sample['pixel_values'].to(device)
            image_id = sample['image_id']

            try:
                captions = model.generate(image, **config)
            except:
                captions = model.generate(image.half(), **config)

            for caption, img_id in zip(captions, image_id):

                caption = processor.decode(caption, skip_special_tokens=True)
                result.append({'image_id': img_id.item(), 'caption': caption})

    # Save the predictions
    path = args['output_dir']
    if not os.path.exists(path):
        os.makedirs(path)

    result_file = os.path.join(path, 'predictions.json')

    result_new = []
    id_list = []
    for res in result:

        if res['image_id'] not in id_list:
            id_list.append(res['image_id'])
            result_new.append(res)         

    json.dump(result_new, open(result_file, 'w'))

    return result_file
