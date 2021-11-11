import os
import re
import argparse

from tqdm import tqdm
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from effdet import EfficientDet, DetBenchTrain, get_efficientdet_config

from datasets import CircleDataset


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-b', '--batch-size', type=int, default=24)
parser.add_argument('--workers', type=int, default=os.cpu_count()//2)
parser.add_argument('-n', '--network', default='d0', type=str, choices=[f'd{i}' for i in range(8)])
parser.add_argument('-e', '--epoch', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.01)
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# モデル準備
cfg = get_efficientdet_config(f'tf_efficientdet_{args.network}')
# 識別する対象は一種類
cfg.num_classes = 1
model = EfficientDet(cfg)
bench = DetBenchTrain(model).to(device)

# effdetはyxyxで受け取る。image_sizeはいずれのモデルの正方形なのでscalerで保持
dataset = CircleDataset(use_yxyx=True, image_size=cfg.image_size[0])
loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=args.workers,
)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True)

print('Starting training')
for epoch in range(1, args.epoch + 1):
    header = f'[{epoch}/{args.epoch}] '

    lr = optimizer.param_groups[0]['lr']
    print(f'{header}Starting lr={lr:.7f}')

    train_metrics = {
        'loss': [],
    }
    t = tqdm(loader, leave=False)
    for (inputs, targets) in t:
        inputs = inputs.to(device)
        targets['bbox'] = targets['bbox'].to(device)
        targets['cls'] = targets['cls'].to(device)
        optimizer.zero_grad()
        losses = bench(inputs, targets)
        loss = losses['loss']
        loss.backward()
        optimizer.step()
        iter_metrics = {
            'loss': float(loss.item()),
        }
        message = ' '.join([f'{k}:{v:.4f}' for k, v in iter_metrics.items()])
        t.set_description(f'{header}{message}')
        t.refresh()
        for k, v in iter_metrics.items():
            train_metrics[k].append(v)
    train_message = ' '.join([f'{k}:{np.mean(v):.4f}' for k, v in train_metrics.items()])
    print(f'{header}Train: {train_message}')

    #* save checkpoint
    if epoch % 10 == 0:
        state = {
            'epoch': epoch,
            'args': args,
            'image_size': cfg.image_size[0],
             # multi GPUは考慮しない
            'state_dict': model.state_dict(),
        }
        checkpoint_dir = f'checkpoints/{self.args.network}'
        os.makedirs(checkpoint_dir, exist_ok=True)
        # checkpoints/d1/20.pth みたいな形式で保存
        checkpoint_path = os.path.join(checkpoint_dir, f'{epoch}.pth')
        torch.save(state, checkpoint_path)
        print(f'{header}Saved "{checkpoint_path}"')

    scheduler.step(train_metrics['loss'])
    print()

if __name__ == '__main__':
    main()
