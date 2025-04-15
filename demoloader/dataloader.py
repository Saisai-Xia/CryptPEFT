import os
import torch

torch.manual_seed(0)

import torchvision.transforms as transforms

from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import timm
from easydict import EasyDict
from timm.models.layers import trunc_normal_
from Adapter.util.crop import RandomResizedCrop
from Adapter.models import vision_transformer


def get_dataset(args):
    import torchvision.datasets as datasets
    _mean = IMAGENET_DEFAULT_MEAN
    _std = IMAGENET_DEFAULT_STD
    transform_train = transforms.Compose([
        RandomResizedCrop(224, interpolation=3),
        transforms.RandomHorizontalFlip(),
        #transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_mean, std=_std)])
    transform_val = transforms.Compose([
        transforms.Resize(224, interpolation=3),
        transforms.CenterCrop(224),
        #transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_mean, std=_std)])
    

    if args.dataset == 'imagenet':
        dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
        dataset_val = datasets.ImageFolder(os.path.join(args.data_path, 'val'), transform=transform_val)
        args.nb_classes = 1000
    elif args.dataset == 'cifar100':
        dataset_train = datasets.CIFAR100(os.path.join(args.data_path, 'cifar100'), transform=transform_train, train=True, download=True)
        dataset_val = datasets.CIFAR100(os.path.join(args.data_path, 'cifar100'), transform=transform_val, train=False, download=True)
        args.nb_classes = 100
    elif args.dataset == 'cifar10':
        dataset_train = datasets.CIFAR10(os.path.join(args.data_path, 'cifar10'), transform=transform_train, train=True, download=True)
        dataset_val = datasets.CIFAR10(os.path.join(args.data_path, 'cifar10'), transform=transform_val, train=False, download=True)
        args.nb_classes = 10
    elif args.dataset == 'stl10':
        dataset_train = datasets.stl10.STL10(os.path.join(args.data_path, 'stl10'), split='train', transform=transform_train, download=True)  
        dataset_val = datasets.stl10.STL10(os.path.join(args.data_path, 'stl10'), split='test', transform=transform_val, download=True)  
        args.nb_classes = 10
    elif args.dataset == 'utkface':
        from Adapter.datasets.utkface import UTKFaceDataset
        attr = "race"
        transform_train = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)])
        dataset = UTKFaceDataset(root=os.path.join(args.data_path, 'utkface'), attr=attr, transform=transform_train)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        dataset_train, dataset_val = torch.utils.data.random_split(dataset, [train_size, test_size])
        args.nb_classes = 4
    elif args.dataset == 'flowers102':
        from Adapter.datasets.flowers102 import Flowers102
        dataset_train = Flowers102(os.path.join(args.data_path, 'flowers102'), split='train', transform=transform_train, download=True)
        dataset_val = Flowers102(os.path.join(args.data_path, 'flowers102'), split='test', transform=transform_val, download=True)
        args.nb_classes = 102
    elif args.dataset == 'svhn':
        from torchvision.datasets import SVHN
        dataset_train = SVHN(os.path.join(args.data_path, 'svhn'), split='train', transform=transform_train, download=True)
        dataset_val = SVHN(os.path.join(args.data_path, 'svhn'), split='test', transform=transform_val, download=True)
        args.nb_classes = 10
    elif args.dataset == 'food101':
        from Adapter.datasets.food101 import Food101
        dataset_train = Food101(os.path.join(args.data_path, 'food101'), split='train', transform=transform_train, download=True)
        dataset_val = Food101(os.path.join(args.data_path, 'food101'), split='test', transform=transform_val, download=True)
        args.nb_classes = 101
    elif args.dataset == 'caltech101':
        transform_train = transforms.Compose([
        # RandomResizedCrop(224, interpolation=3),
        # transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_mean, std=_std)])
        dataset = datasets.Caltech101(os.path.join(args.data_path, 'caltech101'), transform=transform_train, download=True)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        dataset_train, dataset_val = torch.utils.data.random_split(dataset, [train_size, test_size])
        args.nb_classes = 101
    elif args.dataset == 'sun397':
        dataset = datasets.SUN397(root=os.path.join(args.data_path, 'sun397'), transform=transform_train, download=True)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        dataset_train, dataset_val = torch.utils.data.random_split(dataset, [train_size, test_size])
        args.nb_classes = 397
    else:
        raise ValueError(args.dataset)

    return dataset_train, dataset_val



def get_CryptPEFT_model(args):
    tuning_config = EasyDict(
    adapt_on = args.adapt_on,
    adapter_scaler = args.adapter_scaler,
    bottleneck = args.rank,
    adapter_type = args.adapter_type,
    fulltune = args.fulltune,
    num_repeat_blk = args.num_repeat_blk,
    first_layer = args.first_layer,
    num_head = args.num_head,
    approx = args.approx,
    adapter_arch = args.adapter_arch,
    )
    if args.model == "Vit_B_16":
        model = vision_transformer.Vit_B_16(pretrained=True, num_classes = args.nb_classes, config = tuning_config)
    else:
        raise NotImplementedError("Model not implemented")

    if(tuning_config.adapt_on and not args.fulltune):
        model.freeze_backbone_only() # freeze backbone, use adapter only
    if(args.finetune_layer is not None and not tuning_config.adapt_on):
        model.freeze_layers(finetune_layer_num=args.finetune_layer)

    return model
