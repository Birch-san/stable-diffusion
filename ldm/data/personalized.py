import os
import numpy as np
import PIL
from PIL import Image
from PIL.Image import Resampling
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path

# import random

template_fumo = 'photo of {} plush doll'

imagenet_templates_smallest = [
    'a photo of a {}',
]

imagenet_templates_small = [
    'a photo of a {}',
    'a rendering of a {}',
    'a cropped photo of the {}',
    'the photo of a {}',
    'a photo of a clean {}',
    'a photo of a dirty {}',
    'a dark photo of the {}',
    'a photo of my {}',
    'a photo of the cool {}',
    'a close-up photo of a {}',
    'a bright photo of the {}',
    'a cropped photo of a {}',
    'a photo of the {}',
    'a good photo of the {}',
    'a photo of one {}',
    'a close-up photo of the {}',
    'a rendition of the {}',
    'a photo of the clean {}',
    'a rendition of a {}',
    'a photo of a nice {}',
    'a good photo of a {}',
    'a photo of the nice {}',
    'a photo of the small {}',
    'a photo of the weird {}',
    'a photo of the large {}',
    'a photo of a cool {}',
    'a photo of a small {}',
]

imagenet_dual_templates_small = [
    'a photo of a {} with {}',
    'a rendering of a {} with {}',
    'a cropped photo of the {} with {}',
    'the photo of a {} with {}',
    'a photo of a clean {} with {}',
    'a photo of a dirty {} with {}',
    'a dark photo of the {} with {}',
    'a photo of my {} with {}',
    'a photo of the cool {} with {}',
    'a close-up photo of a {} with {}',
    'a bright photo of the {} with {}',
    'a cropped photo of a {} with {}',
    'a photo of the {} with {}',
    'a good photo of the {} with {}',
    'a photo of one {} with {}',
    'a close-up photo of the {} with {}',
    'a rendition of the {} with {}',
    'a photo of the clean {} with {}',
    'a rendition of a {} with {}',
    'a photo of a nice {} with {}',
    'a good photo of a {} with {}',
    'a photo of the nice {} with {}',
    'a photo of the small {} with {}',
    'a photo of the weird {} with {}',
    'a photo of the large {} with {}',
    'a photo of a cool {} with {}',
    'a photo of a small {} with {}',
]

per_img_token_list = [
    'א',
    'ב',
    'ג',
    'ד',
    'ה',
    'ו',
    'ז',
    'ח',
    'ט',
    'י',
    'כ',
    'ל',
    'מ',
    'נ',
    'ס',
    'ע',
    'פ',
    'צ',
    'ק',
    'ר',
    'ש',
    'ת',
]


class PersonalizedBase(Dataset):
    def __init__(
        self,
        data_root,
        size=None,
        repeats=100,
        # interpolation='bicubic',
        interpolation='lanczos',
        flip_p=0.5,
        set='train',
        placeholder_token='*',
        per_image_tokens=False,
        center_crop=False,
        mixing_prob=0.25,
        coarse_class_text=None,
    ):

        self.data_root = data_root

        self.image_paths = [
            os.path.join(self.data_root, file_path)
            for file_path in os.listdir(self.data_root) if file_path.endswith('.png') or file_path.endswith('.jpg')
        ]

        # self._length = len(self.image_paths)
        self.num_images = len(self.image_paths)
        self._length = self.num_images

        self.placeholder_token = placeholder_token

        self.per_image_tokens = per_image_tokens
        self.center_crop = center_crop
        self.mixing_prob = mixing_prob

        self.coarse_class_text = coarse_class_text

        if per_image_tokens:
            assert self.num_images < len(
                per_img_token_list
            ), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

        if set == 'train':
            self._length = self.num_images * repeats

        self.size = size
        self.interpolation = {
            # linear was deprecated
            # https://pillow.readthedocs.io/en/stable/reference/Image.html#resampling-filters
            'linear': Resampling.BILINEAR,
            'bilinear': Resampling.BILINEAR,
            'bicubic': Resampling.BICUBIC,
            'lanczos': Resampling.LANCZOS,
        }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image_path: str = self.image_paths[i % self.num_images]
        stem: str = Path(image_path).stem
        image = Image.open(image_path)

        if not image.mode == 'RGB':
            image = image.convert('RGB')

        placeholder_string = self.placeholder_token
        if self.coarse_class_text:
            placeholder_string = (
                f'{self.coarse_class_text} {placeholder_string}'
            )

        # if self.per_image_tokens and np.random.uniform() < self.mixing_prob:
        #     text = random.choice(imagenet_dual_templates_small).format(
        #         placeholder_string, per_img_token_list[i % self.num_images]
        #     )
        # else:
        #     text = random.choice(imagenet_templates_small).format(
        #         placeholder_string
        #     )

        # text = template_fumo.format(placeholder_string)
        match stem:
            case 'nagisa':
                text = 'photo of Itaru Hinoue nagisa furukawa clannad, clannad jacket cosplay {} plush doll with brown hair, brown eyes, chibi smiling, holding brown briefcase, next to dango'.format(placeholder_string)
            case 'teto':
                text = 'photo of vocaloid kasane teto {} plush doll with brown hair, brown eyes, chibi smiling'.format(placeholder_string)
            case 'korone':
                text = 'photo of anime girl {} plush doll with yellow jacket, white dress, brown hair, brown eyes, hairclip, uwu face'.format(placeholder_string)
            case 'kudo1':
                text = 'photo of kud wafter noumi little busters na-ga {} plush doll with yellow jacket, white dress, brown hair, brown eyes, chibi smiling'.format(placeholder_string)
            case 'patchouli':
                text = 'photo of patchouli touhou {} plush doll chibi unhappy'.format(placeholder_string)
            case _:
                assert False

        example['caption'] = text

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = (
                img.shape[0],
                img.shape[1],
            )
            img = img[
                (h - crop) // 2 : (h + crop) // 2,
                (w - crop) // 2 : (w + crop) // 2,
            ]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize(
                (self.size, self.size), resample=self.interpolation
            )

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example['image'] = (image / 127.5 - 1.0).astype(np.float32)
        return example
