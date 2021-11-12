import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

np.random.seed(42)


def draw_bbox(img, rect, text=None, font=None, color='yellow'):
    ''' draw_bbox
        img: Image
        rect: list|tuple|np.ndarray|torch.Tensor
    '''
    rect = [int(v) for v in rect]
    draw = ImageDraw.Draw(img)
    draw.rectangle(((rect[0], rect[1]), (rect[2], rect[3])), outline=color, width=1)
    if text:
        if not font:
            font = ImageFont.truetype('/usr/share/fonts/ubuntu/Ubuntu-R.ttf', size=16)
        draw.text((rect[0], rect[1]), str(text), font=font, fill=color)

def generate_dummy_pair(image_size):
    bg = (0, 0, 0)
    fg = (255, 0, 0)
    img = Image.new('RGB', (image_size, image_size), bg)

    size = np.random.randint(10, image_size//2)
    left = np.random.randint(0, image_size//2)
    top = np.random.randint(0, image_size//2)

    right = left + size
    bottom = top + size
    rect = (left, top, right, bottom)
    draw = ImageDraw.Draw(img)
    draw.ellipse(rect, fill=fg)
    return img, rect

class CircleDataset(Dataset):
    def __init__(self, image_size=512, use_yxyx=True, normalized=True):
        self.use_yxyx = use_yxyx
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
            A.Normalize(mean=[0.2, 0.1, 0.1], std=[0.2, 0.1, 0.1]) if normalized else None,
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    def __len__(self):
        return 200 # 1epochあたりの枚数。自動生成なので適当

    def __getitem__(self, idx):
        # img = Image.new('RGB', (512, 512), self.bg)
        # size = np.random.randint(1, 256)
        # left = np.random.randint(0, 256)
        # top = np.random.randint(0, 256)
        # right = left + size
        # bottom = top + size
        # draw = ImageDraw.Draw(img)

        img, rect = generate_dummy_pair(self.image_size)

        # shapeはbox_count x box_coords (N x 4)。円は常に一つなので、今回は画像一枚に対して(1 x 4)
        bboxes = np.array([
            # albumentationsにはASCAL VOC形式の[x0, y0, x1, y1]をピクセル単位で入力する
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

        # albumentationsのrandom cropでbboxが範囲外に出るとラベルのサイズがなくなるのでゼロ埋めしておく
        # 複数のbboxを扱う場合は、足りない要素数分emptyなbboxとclsで補う処理が必要
        if bboxes.shape[0] == 0:
            bboxes = torch.zeros([1, 4], dtype=bboxes.dtype)
        if labels.shape[0] < 1:
            labels = torch.zeros([1], dtype=labels.dtype)

        # effdetはデフォルトではyxyxで受け取るので、インデックスを入れ替える
        if self.use_yxyx:
            bboxes = bboxes[:, [1, 0, 3, 2]]

        assert bboxes.shape == (1, 4)
        assert labels.shape == (1, )

        # effdetのtargetは以下の形式
        y = {
            'bbox': bboxes,
            'cls': labels,
        }
        return x, y

if __name__ == '__main__':
    img, rect = generate_dummy_pair(512)
    img.save(f'out/example.png')
    draw_bbox(img, rect)
    img.save(f'out/example_gt.png')
