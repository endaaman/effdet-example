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

from datasets import CircleDataset, draw_bbox

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-c', '--checkpoint', type=str)
parser.add_argument('-s', '--src', type=str, required=True)
parser.add_argument('-d', '--dest', type=str)
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.checkpoint:
    checkpoint = torch.load(args.checkpoint)
    network = checkpoint['args'].network
    image_size = checkpoint['image_size']
else:
    print('using default weights')
    network = 'd0'
    image_size = 512

# モデル準備
cfg = get_efficientdet_config(f'tf_efficientdet_{network}')
cfg.num_classes = 1
# top-5のみ
cfg.max_det_per_image = 5
# cfg.soft_nms = True
model = EfficientDet(cfg).eval()
bench = DetBenchPredict(model).to(device)

# 入力データ準備
img = Image.open(args.src)
original_size = (img.width, img.height)
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])
input_tensor = transform(img)
# 先頭にバッチのインデックスをつける
input_tensor = input_tensor[None, :] # CHW -> BCHW

# モデルに入力
with torch.no_grad():
    #  バッチサイズ==1なので先頭だけ持ってくる
    output_tensor = bench(input_tensor.to(device)).detach().cpu()[0]

# DetBenchPredictが内部でnmsなのよしなに済ませており、
# 出力形式は [[x0, y0, x1, y1, confidence, cls]] となっている

# 検出されたbbox
bboxes = output_tensor[:, :4].type(torch.long)
# 各bboxのconfidence
confs = output_tensor[:, 4]
# 各bboxのラベル(今回ラベルは一つなので無視)
# labels = output_tensor[5].type(torch.long)

# フォントは各自適当なものを使うこと
font = ImageFont.truetype('/usr/share/fonts/ubuntu/Ubuntu-R.ttf', size=16)

# original_size は int[2] のタプルなので *2 で int[4] になる
# [x0, y0, x1, y1] に掛けやすい形に変形しておく
scale = np.array(original_size * 2) / image_size # [w, h, w, h]
for bbox, conf in zip(bboxes, confs):
    # 元の画像サイズにスケールし直して四捨五入
    rect = np.rint(bbox[:4].numpy() * scale).astype(np.int64)
    draw_bbox(img, rect, text=f'{conf:.2f}', font=font)

if args.dest:
    dest = args.dest
else:
    os.makedirs('out', exist_ok=True)
    dest = os.path.join('out', os.path.basename(args.src))
img.save(dest)
print(f'wrote {dest}')
