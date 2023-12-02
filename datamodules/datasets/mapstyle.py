import os
from typing import Optional
from collections.abc import Callable
import random
import numpy as np
import json

import torch
from PIL import Image
from torchvision.datasets import VisionDataset
from datamodules.datasets.coco import CocoCaptions
from datamodules.tokenizers import TokenizerUtils
from torch.utils.data import ConcatDataset
from datamodules.datasets.dataclass import Item


class MapStyleDataset(VisionDataset, TokenizerUtils):
    names = {"coco2014"}
    splits = {"train", "val", "test", "test_dev", "overfit"}
    locs = {"local"}

    def __init__(
        self,
        name: str,
        loc: str,
        split: str,
        transform: Optional[Callable] = None,
        tokenizer_type: Optional[str] = None,
        bpe_pdrop: Optional[float] = None,
        text_ctx: int = 77,
        gt_text: bool = False,
        is_test: bool = False,
        **ignore_kwargs,
    ):
        assert name in self.names, f"{name} is not in {self.names}"
        assert split in self.splits, f"{split} is not in {self.splits}"
        assert loc in self.locs, f"{loc} is not in {self.locs}"

        if name.startswith("coco"):
            super().__init__('/ssd0/data/coco', transform=transform)

        self.name = name
        self.split = split
        self.gt_text = gt_text
        self.build_tokenizer(tokenizer_type, text_ctx, lowercase=True, dropout=bpe_pdrop)

        self.items = []
        if name.startswith("coco"):
            data_list = []
            if split == "overfit":
                data_list.append(CocoCaptions(root=f'{self.root}/images/train2014', annFile=f'{self.root}/annotations/captions_{split}2014.json'))
            elif split == "test":
                data_list.append(CocoCaptions(root=f'{self.root}/images/val2014', annFile=f'{self.root}/annotations/dataset_coco_test.json'))
            else:
                data_list.append(CocoCaptions(root=f'{self.root}/images/{split}2014',
                                              annFile=f'{self.root}/annotations/captions_{split}2014.json'))
            self.items = ConcatDataset(data_list)

        self.custom_len = None

    def set_custom_length(self, l):
        assert len(self.items) >= l
        self.custom_len = l

    def __len__(self):
        if self.custom_len is not None:
            return self.custom_len
        if self.split == 'val':
            return 5000
        return len(self.items)

    def __getitem__(self, item: int):
        if self.name.startswith("coco"):
            imgpath, img, gt_txt = self.items[item]

            if len(gt_txt) > 5:
                gt_txt = gt_txt[:5]
            elif len(gt_txt) < 5:
                gt_txt.append(gt_txt[:(5 - len(gt_txt))])

            if self.transform:
                img = self.transform(img)

            # text = ' '.join(text)  # text is a list of sentences. Concat them.
            if self.split == "train":
                rnd_txt = random.randint(0, len(gt_txt)-1)
                txt = gt_txt[rnd_txt]
            else:
                txt = gt_txt[0]

            txt_item = self.get_input(txt, pre_proc=self.pre_caption)
            domain = None

        item = Item(imgpath=imgpath, img=img, txt=txt_item.txt, txt_mask=txt_item.txt_mask, txt_pos_id=txt_item.pos_id, gt_txt=gt_txt, domain=domain)
        return item

    def set_epoch(self, epoch):
        self.epoch = epoch
