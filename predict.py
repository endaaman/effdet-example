import os
import re
import argparse

from tqdm import tqdm
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from effdet import EfficientDet, DetBenchPredict, get_efficientdet_config

from datasets import CircleDataset

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--checkpoint', action='store_true')
parser.add_argument('-b', '--batch-size', type=int, default=24)
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict_images(bench, images):
    imgs = [i.resize((size, size)) for i in imgs]

    bs = self.args.batch_size
    outputs = []
    start = 0
    t = tqdm(range(0, len(imgs), bs))
    for start in t:
        batch = imgs[start:start + bs]

        tt = torch.stack([pil_to_tensor(i) for i in batch]).to(self.device)
        with torch.no_grad():
            output_tensor = bench(tt)
        outputs.append(output_tensor.detach().cpu())
        t.set_description(f'{start} ~ {start + bs} / {len(imgs)}')
        t.refresh()
    outputs = torch.cat(outputs).type(torch.long)

    results = []
    print(outputs.shape)
    for img, bboxes in zip(imgs, outputs):
        best_bboxes = []
        for i in LABEL_TO_STR.keys():
            m = bboxes[bboxes[:, 5] == i]
            if len(m) > 0:
                best_bboxes.append(m[0])
            else:
                print('missing {LABEL_TO_STR[i]}')

        draw = ImageDraw.Draw(img)
        for i, result in enumerate(best_bboxes):
            bbox = result[:4]
            label = result[5].item()
            draw.text((bbox[0], bbox[1]), LABEL_TO_STR[label], font=self.font, fill='yellow')
            draw.rectangle(((bbox[0], bbox[1]), (bbox[2], bbox[3])), outline='yellow', width=1)
        results.append(img)
    return results

# effdetはyxyxで受け取る
dataset = CircleDataset(use_yxyx=True)
loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=args.workers,
)

checkpoint = torch.load(args.checkpoint)
network = checkpoint['args'].network

cfg = get_efficientdet_config(f'tf_efficientdet_{network}')
cfg.num_classes = 1
model = EfficientDet(cfg)
bench = DetBenchPredict(model).to(device)

