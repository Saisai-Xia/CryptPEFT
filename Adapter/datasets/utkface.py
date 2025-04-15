#refrence from https://github.com/ziqi-zhang/TEESlice-artifact/blob/main/membership-inference/demoloader/dataloader.py
import os
import torch
import pandas
import torchvision
torch.manual_seed(0)
import torch.nn as nn
import PIL.Image as Image
import torchvision.transforms as transforms
import numpy as np

from typing import Any, Callable, List, Optional, Union, Tuple

class UTKFaceDataset(torch.utils.data.Dataset):
    def __init__(self, root, attr: Union[List[str], str] = "gender", transform=None, target_transform=None)-> None:
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.files = os.listdir(root+'/UTKFace/')
        if isinstance(attr, list):
            self.attr = attr
        else:
            self.attr = [attr]

        self.lines = []
        for txt_file in self.files:
            image_name = txt_file.split('jpg ')[0]
            attrs = image_name.split('_')
            if len(attrs) < 4 or int(attrs[2]) >= 4:
                continue
            if attrs[1] == "":
                continue
            self.lines.append(image_name)


    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index:int)-> Tuple[Any, Any]:
        
        attrs = self.lines[index].split('_')
        # print(self.lines[index], attrs, attrs[0], attrs[1], attrs[2])

        age = int(attrs[0])
        gender = int(attrs[1])
        race = int(attrs[2])

        image_path = os.path.join(self.root+'/UTKFace', self.lines[index]).rstrip()

        image = Image.open(image_path).convert('RGB')

        target: Any = []
        for t in self.attr:
            if t == "age":
                target.append(age)
            elif t == "gender":
                target.append(gender)
            elif t == "race":
                target.append(race)
            
            else:
                raise ValueError("Target type \"{}\" is not recognized.".format(t))

        if self.transform:
            image = self.transform(image)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return image, target