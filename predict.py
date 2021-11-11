import os
import re
import argparse

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import numpy as np
import torch
from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader
from effdet import EfficientDet, DetBenchPredict, get_efficientdet_config

from datasets import CircleDataset

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-c', '--checkpoint', type=str)
parser.add_argument('-s', '--src', type=str, required=True)
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.checkpoint:
    checkpoint = torch.load(args.checkpoint)
    network = checkpoint['args'].network
else:
    print('using default weights')
    network = 'd0'

# モデル準備
cfg = get_efficientdet_config(f'tf_efficientdet_{network}')
cfg.num_classes = 1
# cfg.soft_nms = True
model = EfficientDet(cfg).eval()
bench = DetBenchPredict(model).to(device)

# 入力データ準備
img = Image.open(args.src)
original_size = (img.width, img.height)
transform = transforms.Compose([
    transforms.ToTensor(),
])
input_tensor = transform(img.resize(cfg.image_size))
# 先頭にバッチのインデックスをつける
input_tensor = input_tensor[None, :] # CHW -> BCHW

# モデルに入力
with torch.no_grad():
    output_tensor = bench(input_tensor.to(device))

# 出力形式は [[x0, y0, x1, y1, confidence, cls]] となっている
# DetBenchPredictが内部でnmsなのよしなに済ませてくれてくれる
output_tensor = output_tensor.detach().cpu().type(torch.long)

# 一枚だけ入力しているので最初だけ取得
bboxes = output_tensor[0]


# フォントは各自適当なものを使うこと
font = ImageFont.truetype('/usr/share/fonts/ubuntu/Ubuntu-R.ttf', size=16)

draw = ImageDraw.Draw(img)
scale = np.array(original_size) / np.array(cfg.image_size) # [w, h]
# [x0, y0, x1, y1] に掛けやすい形に変形
scale = np.tile(scale, 2) # [w, h, w, h]
for bbox in bboxes:
    label = bbox[5].item()
    # 元の画像サイズにスケールし直して四捨五入
    bbox = np.rint(bbox[:4].numpy() * scale).astype(np.int64)
    draw.text((bbox[0], bbox[1]), f'{label}', font=font, fill='yellow')
    draw.rectangle(((bbox[0], bbox[1]), (bbox[2], bbox[3])), outline='yellow', width=1)

os.makedirs('out', exist_ok=True)
p = os.path.join('out', os.path.basename(args.src))
img.save(p)
print(f'wrote {p}')
