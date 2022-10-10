import os
import numpy as np
import PIL
from PIL import Image
from PIL.Image import Resampling
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from typing import List
from random import sample, randrange, random

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
            # placeholder_string = (
            #     f'{self.coarse_class_text} {placeholder_string}'
            # )
            placeholder_string = (
                f'{placeholder_string} {self.coarse_class_text}'
            )

        # if self.per_image_tokens and np.random.uniform() < self.mixing_prob:
        #     text = random.choice(imagenet_dual_templates_small).format(
        #         placeholder_string, per_img_token_list[i % self.num_images]
        #     )
        # else:
        #     text = random.choice(imagenet_templates_small).format(
        #         placeholder_string
        #     )

        def describe_placeholder() -> str:
            if random() < 0.3:
                return self.placeholder_token
            return placeholder_string


        def describe_subject(character: str) -> str:
            placeholder: str = describe_placeholder()
            if random() < 0.3:
                return f"photo of {placeholder}"
            return f"photo of {character} {placeholder}"

        def make_prompt(character: str, general_labels: List[str], sitting=True, on_floor=True) -> str:
            even_more_labels = [*general_labels, 'one girl']
            if sitting:
                even_more_labels.append('sitting')
            if on_floor:
                even_more_labels.append('on floor')
            subject: str = describe_subject(character)
            label_count = randrange(0, len(even_more_labels))
            if label_count == 0:
                return subject
            labels = sample(even_more_labels, label_count)
            joined = ', '.join(labels)
            return f"photo of {character} {placeholder_string}, {joined}"

        match stem:
            case 'koishi':
                text = make_prompt('komeiji koishi', ['green hair', 'black footwear', 'medium hair', 'blue eyes', 'yellow jacket', 'green skirt' 'hat', 'black headwear', 'smile', 'touhou project'])
            case 'flandre':
                text = make_prompt('flandre scarlet', ['fang', 'red footwear', 'slit pupils', 'medium hair', 'blonde hair', 'red eyes', 'red dress', 'mob cap', 'smile', 'short sleeves', 'yellow ascot', 'touhou project'])
            case 'sanae':
                text = make_prompt('kochiya sanae', ['green hair', 'blue footwear', 'long hair', 'green eyes', 'white dress', 'blue skirt', 'frog hair ornament', 'snake hair ornament', 'smile', 'standing', 'touhou project'])
            case 'sanaestand':
                text = make_prompt('kochiya sanae', ['green hair', 'blue footwear', 'long hair', 'green eyes', 'white dress', 'blue skirt', 'frog hair ornament', 'snake hair ornament', 'smile', 'touhou project'], sitting=False)
            case 'tenshi':
                text = make_prompt('hinanawi tenshi', ['blue hair', 'brown footwear', 'slit pupils', 'very long hair', 'red eyes', 'white dress', 'blue skirt', 'hat', 'black headwear', 'smile', 'touhou project'])
            case 'youmu':
                text = make_prompt('konpaku youmu', ['silver hair', 'black footwear', 'medium hair', 'slit pupils', 'green eyes', 'green dress', 'sleeveless dress', 'white sleeves', 'black ribbon', 'hair ribbon', 'unhappy', 'touhou project'])
            case 'yuyuko':
                text = make_prompt('saigyouji yuyuko', ['pink hair', 'black footwear', 'medium hair', 'pink eyes', 'wide sleeves', 'long sleeves', 'blue dress', 'mob cap', 'touhou project'])
            case 'nagisa':
                text = make_prompt('furukawa nagisa', ['brown hair', 'brown footwear', 'medium hair', 'brown eyes', 'smile', 'school briefcase', 'blue skirt', 'yellow jacket', 'antenna hair', 'dango', 'clannad'])
            case 'teto':
                text = make_prompt('kasane teto', ['pink hair', 'red footwear', 'red eyes', 'medium hair', 'detached sleeves', 'twin drills', 'drill hair', 'grey dress', 'smile', 'vocaloid'])
            case 'korone':
                text = make_prompt('inugami korone', ['yellow jacket', 'blue footwear', 'long hair', 'white dress', 'brown hair', 'brown eyes', 'on chair', 'hairclip', 'uwu', 'hololive'], on_floor=False)
            case 'kudo':
                text = make_prompt('kudryavka noumi', ['fang', 'black footwear', 'very long hair', 'white hat', 'white cape', 'silver hair', 'grey skirt', 'blue eyes', 'smile', 'little busters!'])
            case 'patchouli':
                text = make_prompt('patchouli knowledge', ['mob cap', 'pink footwear', 'long hair', 'slit pupils', 'striped dress', 'pink dress', 'purple hair', 'ribbons in hair', 'unhappy', 'touhou project'])
            case _:
                text = f"photo of {placeholder_string}"

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
