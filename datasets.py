import torch
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

np.random.seed(42)


class CircleDataset(Dataset):
    def __init__(self, use_yxyx=True, image_size=512, bg=(0, 0, 0), fg=(255, 0, 0)):
        self.use_yxyx = use_yxyx
        self.bg = bg
        self.fg = fg
        self.image_size = image_size

        # 適当なaugmentaion
        self.albu = A.Compose([
            A.RandomResizedCrop(width=self.image_size, height=self.image_size, scale=[0.7, 1.0]),
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
            A.Normalize(mean=[0.2, 0.1, 0.1], std=[0.2, 0.1, 0.1]),
            ToTensorV2(),
            # label_fieldsで与えたkwargsがラベルとして扱われる
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    def __len__(self):
        return 1000 # 1epochあたりの枚数。適当に決めた

    def __getitem__(self, idx):
        img = Image.new('RGB', (512, 512), self.bg)

        size = np.random.randint(0, 256)
        left = np.random.randint(0, 256) # left
        top = np.random.randint(0, 256) # top

        right = left + size
        bottom = top + size
        draw = ImageDraw.Draw(img)
        draw.ellipse((left, top, right, bottom), fill=self.fg)

        # shapeはbox_count x box_coords (N x 4)。円は常に一つなので、今回は画像一枚に対して(1 x 4)
        bboxes = np.array([
            # albumentationsのために最初はPASCAL VOC形式の[x0, y0, x1, y1]をピクセル単位で
            [left, top, right, bottom,],
        ])

        labels = np.array([
            # 0はラベルなしで無視される。検出対象は1<=である必要ある
            1,
        ])

        result = self.albu(
            image=np.array(img),
            bboxes=bboxes,
            labels=labels,
        )
        # まとめてndarrayにキャスト
        x = result['image']
        bboxes = torch.FloatTensor(result['bboxes'])
        labels = torch.FloatTensor(result['labels'])

        # albumentationsのrandom cropでbboxが範囲外に出るとラベルのサイズがなくなるのでゼロ埋めしておく
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


def tensor_to_pil(tensor):
    a = tensor.min()
    b = tensor.max()
    img = (tensor - a) / (b - a)
    return to_pil_image(img)

if __name__ == '__main__':
    # draw_bounding_boxesはxyxy形式
    ds = CircleDataset(use_yxyx=False)
    for (x, y) in ds:
        tensor_to_pil(x).save(f'example_x.png')
        t = draw_bounding_boxes(image=x, boxes=y['bbox'], labels=[str(v.item()) for v in y['cls']])
        img = tensor_to_pil(t)
        img.save(f'example_xy.png')
        break