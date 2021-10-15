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

class Main:
    def __init__(self):
        self.parser = argparse.ArgumentParser(add_help=False)
        self.common_parser = argparse.ArgumentParser(add_help=False)
        self.arg_common(self.common_parser)
        subparsers = self.parser.add_subparsers()
        targets = [k[4:] for k in dir(self) if re.match(r'^run_.+', k)]
        for target in targets:
            subparser = subparsers.add_parser(target, parents=[self.common_parser])
            arg_func = getattr(self, 'arg_' + target, None)
            if callable(arg_func):
                arg_func(subparser)
            subparser.set_defaults(target=target)
        self.args = self.parser.parse_args()
        self.use_gpu = not self.args.cpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')

    def run(self):
        if not hasattr(self.args, 'target'):
            self.parser.print_help()
            exit(0)
        run_func = getattr(self, f'run_{self.args.target}', None)
        if not callable(run_func):
            print(f'invalid func name: {run_func}')
        run_func()

    def create_model(self, network):
        cfg = get_efficientdet_config(f'tf_efficientdet_{network}')
        cfg.num_classes = 1
        return EfficientDet(cfg)

    def save_checkpoint(self, model, epoch):
        state = {
            'epoch': epoch,
            'args': self.args,
            'state_dict': get_state_dict(model),
        }
        checkpoint_dir = f'weights/{self.args.network}'
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f'{epoch}.pth')
        torch.save(state, checkpoint_path)
        return weights_path

    def arg_common(self, parser):
        parser.add_argument('--cpu', action='store_true')
        parser.add_argument('-b', '--batch-size', type=int, default=24)
        parser.add_argument('--workers', type=int, default=os.cpu_count()//2)

    def arg_train(self, parser):
        parser.add_argument('-n', '--network', default='d0', type=str, choices=[f'd{i}' for i in range(8)])
        parser.add_argument('-e', '--epoch', type=int, default=50)
        parser.add_argument('--lr', type=float, default=0.01)

    def run_train(self):
        model = self.create_model(self.args.network)
        bench = DetBenchTrain(model).to(self.device)

        dataset = CircleDataset(use_yxyx=True)
        loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
        )

        optimizer = optim.Adam(model.parameters(), lr=self.args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True)

        print('Starting training')
        for epoch in range(1, self.args.epoch + 1):
            header = f'[{epoch}/{self.args.epoch}] '

            # lr = scheduler.get_last_lr()[0]
            lr = optimizer.param_groups[0]['lr']
            print(f'{header}Starting lr={lr:.7f}')

            metrics = {
                'loss': [],
            }
            t = tqdm(loader, leave=False)
            for (inputs, targets) in t:
                inputs = inputs.to(self.device)
                targets['bbox'] = targets['bbox'].to(self.device)
                targets['cls'] = targets['cls'].to(self.device)
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
                    metrics[k].append(v)
            train_metrics = {k: np.mean(v) for k, v in metrics.items()}
            train_message = ' '.join([f'{k}:{v:.4f}' for k, v in train_metrics.items()])
            print(f'{header}Train: {train_message}')

            #* save weights
            if epoch % 10 == 0:
                weights_path = self.save_checkpoint(bench.model, epoch)
                print(f'{header}Saved "{weights_path}"')

            scheduler.step(train_metrics['loss'])
            print()

    def arg_predict(self, parser):
        pass

    def run_predict(self):
        print('train')

if __name__ == '__main__':
    Main().run()
