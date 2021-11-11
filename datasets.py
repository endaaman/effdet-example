import torch
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

np.random.seed(42)


def generate_dummy_pair(
        image_size=512,
        target_radius=(0, 128),
        bg_color=(255, 0, 0),
        fg_color=(0, 0, 0)):
    img = Image.new('RGB', (image_size, image_size), bg_color)

    r = np.random.randint(*target_radius)
    x = np.random.randint(0, 512)
    y = np.random.randint(0, 512)

    rect = (x - r,
            y - r,
            x + r,
            y + r)
    draw = ImageDraw.Draw(img)
    draw.ellipse(rect, fill=fg_color)

    return img, rect


class CircleDataset(Dataset):
    def __init__(self, use_yxyx=True, image_size=512, bg=(0, 0, 0), fg=(255, 0, 0), normalized=True):
        self.use_yxyx = use_yxyx
        self.bg = bg
        self.fg = fg
        self.image_size = image_size

        # 適当なaugmentaion
        self.albu = A.Compose([
            A.RandomResizedCrop(width=self.image_size, height=self.image_size, scale=[0.8, 1.0]),
            A.GaussNoise(p=0.2),
            A.OneOf([
                A.MotionBlur(p=.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=5, p=0.5),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.Emboss(),
                A.RandomBrightnessContrast(),
            ], p=0.3),
            A.HueSaturationValue(p=0.3),
            # 可視化するとき正規化されるとnoisyなのでトグれるようにする
            # A.Normalize(mean=[0.2, 0.1, 0.1], std=[0.2, 0.1, 0.1]) if normalized else None,
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    def __len__(self):
        return 1000 # 1epochあたりの枚数。自動生成なので適当

    def __getitem__(self, idx):
        img, rect = generate_dummy_pair()
        # shapeはbox_count x box_coords (N x 4)。円は常に一つなので、今回は画像一枚に対して(1 x 4)

        # rect は (left, top, right, bottom) の tuple
        bboxes = np.array([
            # albumentationsにはPASCAL VOC形式の[[left, top, right, bottom]]をピクセル単位で入力する
            rect,
        ])

        labels = np.array([
            # 検出対象はid>=1である必要あり。0はラベルなしとして無視される。
            1,
        ])

        result = self.albu(
            image=np.array(img),
            bboxes=bboxes,
            labels=labels,
        )
        x = result['image']
        bboxes = torch.FloatTensor(result['bboxes'])
        labels = torch.FloatTensor(result['labels'])

        # albumentationsのrandom cropでbboxが範囲外に出るとラベルのサイズがなくなるのでdummyデータで埋めておく
        # 複数のbboxを扱う場合は、足りない要素数分emptyなbboxとclsで補う処理が必要
        if bboxes.shape[0] == 0:
            bboxes = torch.zeros([1, 4], dtype=bboxes.dtype)
        if labels.shape[0] < 1:
            labels = torch.zeros([1, 1], dtype=labels.dtype)

        # effdetはデフォルトではyxyxで受け取るので、インデックスを入れ替える
        if self.use_yxyx:
            bboxes = bboxes[:, [1, 0, 3, 2]]

        # effdetのtargetは以下の形式
        y = {
            'bbox': bboxes,
            'cls': labels,
        }
        return x, y

if __name__ == '__main__':
    # draw_bounding_boxesはxyxy形式
    ds = CircleDataset(use_yxyx=False, normalized=False)
    for (x, y) in ds:
        to_pil_image(x).save(f'example_x.png')
        t = draw_bounding_boxes(image=x, boxes=y['bbox'], labels=[str(v.item()) for v in y['cls']])
        img = to_pil_image(t)
        img.save(f'example_xy.png')
        break
