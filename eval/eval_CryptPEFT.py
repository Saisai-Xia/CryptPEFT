from math import ceil
from tqdm import tqdm
import torch
import argparse
from pathlib import Path
from demoloader.dataloader import *

#logger
from log_utils import Logger


import gol
gol._init()
gol.set_value("debug", False)

def get_args_parser():
    parser = argparse.ArgumentParser('CRYPTPEFT: Parameter-Efficient Fine-Tuning for Privacy-Preserving Neural Network Inferenc', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=20, type=int)

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')


    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (absolute lr)')
    
    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--finetune_layer', default=None, type=int,
                        help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=100, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')


    # custom configs
    parser.add_argument('--dataset', default='cifar100', choices=['cifar100', 'flowers102', 'svhn', 'food101', 'cifar10'])

    # CryptPEFT related parameters
    parser.add_argument('--adapt_on', default=False, action='store_true', help='whether activate Adapter')
    parser.add_argument('--rank', default=120, type=int, help='rank of adapter')
    parser.add_argument('--num_repeat_blk', default=1, type=int, help='number of base block in adapter')
    parser.add_argument('--adapter_type', default="liner", type=str, help='type of adapter')
    parser.add_argument('--adapter_scaler', default="0.1", type=str, help='adapter scaler')
    parser.add_argument('--fulltune', default=False, action='store_true', help='full finetune model')


    parser.add_argument('--first_layer', default=6, type=int, help='Start using outputs for transfer learning.')
    parser.add_argument('--num_head', default=12, type=int)

    parser.add_argument('--layer_id', default=None, type=int)

    parser.add_argument('--adapter_arch', default="CryptPEFT", type=str, choices=['lora', 'CryptPEFT', 'adaptformer'], help='adapter architecture')

    parser.add_argument('--option', default="reproduce", type=str, help='what you wanna do? -> [reproduce, utility_first, efficiency_first, OWC_PEFT, TWC_PEFT]')

    return parser

def train_and_test_model(model, trainloader, testloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(tqdm(trainloader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    avg_loss_train = total_loss / len(trainloader)
    accuracy_train = 100. * correct / total
    print(f"Epoch[{epoch}] | train accuracy {accuracy_train} | train loss {avg_loss_train}")

    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(testloader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    avg_loss_test = total_loss / len(testloader)
    accuracy_test = 100. * correct / total

    print(f"Epoch[{epoch}] | test accuracy {accuracy_test} | test loss {avg_loss_test}")

    return accuracy_train, avg_loss_train, accuracy_test, avg_loss_test

def evaluate(model, testloader, criterion, device, epoch):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(testloader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    avg_loss_test = total_loss / len(testloader)
    accuracy_test = 100. * correct / total

    print(f"Epoch[{epoch}] | test accuracy {accuracy_test} | test loss {avg_loss_test}")

    return accuracy_test, avg_loss_test


def test_CryptPEFT(args, device):

    #prepare data
    trainset, testset = get_dataset(args)
    
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=True, num_workers=4)

    #prepare model
    model = get_CryptPEFT_model(args=args) #args.ffn_adapt = True
    #print(model.adapter)

    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone_n_parameters = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    if args.option == "utility_first":
        param_limit = 0.05
    elif args.option == "efficiency_first":
        param_limit = 0.02

    if args.option in ["utility_first", "efficiency_first"]:
        if n_parameters > param_limit * backbone_n_parameters:
            return 0.0, n_parameters / 1.e6

    print('number of params (M): %.2f' % (n_parameters / 1.e6))


    criterion = torch.nn.CrossEntropyLoss()


    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=5e-4, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(args.epochs):
        _, _, accuracy_test, _ = train_and_test_model(model, trainloader, testloader, criterion, optimizer, device, epoch)
        scheduler.step()
    

    return accuracy_test, n_parameters / 1.e6



def main(args):
    print(torch.cuda.is_available())
    DEVICE = torch.device(args.device)


    if args.option == "utility_first":
        logger = Logger(log2file=True if args.log_dir is not None else False, mode=f"s_auto_search_"+args.dataset, path=args.log_dir)
        H = [1,2,4,6,10,12]
        R = [60, 120, 180, 240, 300]
        S = [6,4,2]
        res = {}
        max_acc = 0.0
        for num_head in H:
            for rank in R:
                for s in S:
                    args.rank = rank
                    args.first_layer = 12-s
                    args.num_head = num_head
                    print(f"now-> num_head{num_head} rank{rank} scope{s}")
                    acc, n_param = test_CryptPEFT(args=args, device=DEVICE)
                    if(acc > max_acc):
                        max_acc = acc
                        res['acc'] = max_acc
                        res['n_param'] = n_param
                        res['num_head'] = num_head
                        res['rank'] = rank
                        res['scope'] = s
        res['dataset'] = args.dataset

        print("auto search finish")
        for key, value in res.items():
                logger.add_line(f"{key}: {value}")
    elif args.option == "efficiency_first":
        logger = Logger(log2file=True if args.log_dir is not None else False, mode=f"s_auto_search_"+args.dataset, path=args.log_dir)
        H = [1, 2]
        R = [60, 120]
        S = [1, 2]
        res = {}
        max_acc = 0.0
        for num_head in H:
            for rank in R:
                for s in S:
                    args.rank = rank
                    args.first_layer = 12-s
                    args.num_head = num_head
                    print(f"now-> num_head{num_head} rank{rank} scope{s}")
                    acc, n_param = test_CryptPEFT(args=args, device=DEVICE)
                    if(acc > max_acc):
                        max_acc = acc
                        res['acc'] = max_acc
                        res['n_param'] = n_param
                        res['num_head'] = num_head
                        res['rank'] = rank
                        res['scope'] = s
        res['dataset'] = args.dataset

        print("auto search finish")
        for key, value in res.items():
                logger.add_line(f"{key}: {value}")
    elif args.option == "OWC_PEFT":
        logger = Logger(log2file=True if args.log_dir is not None else False, mode=f"OWC_PEFT_{args.adapter_arch}_"+args.dataset, path=args.log_dir)
        args.first_layer = 0
        acc, n_param = test_CryptPEFT(args=args, device=DEVICE)
        logger.add_line(f"acc:{acc}")
        logger.add_line(f"n_param:{n_param}")
    elif args.option == "TWC_PEFT":
        logger = Logger(log2file=True if args.log_dir is not None else False, mode=f"TWC_PEFT_{args.adapter_type}_"+args.dataset, path=args.log_dir)
        acc, n_param = test_CryptPEFT(args=args, device=DEVICE)
        logger.add_line(f"acc:{acc}")
        logger.add_line(f"n_param:{n_param}")
    elif args.option == "reproduce":
        method = "utility_first" # utility first or efficiency first
        logger = Logger(log2file=True if args.log_dir is not None else False, mode=f"{args.option}_{method}_{args.dataset}", path=args.log_dir)
        if method == "utility_first":
            if args.dataset == "cifar10":
                adapter_struct = [12,240,4]
            elif args.dataset == "cifar100":
                adapter_struct = [1,180,2]
            elif args.dataset == "flowers102":
                adapter_struct = [12,180,4]
            elif args.dataset == "svhn":
                adapter_struct = [10,180,6]
            elif args.dataset == "food101":
                adapter_struct = [10,240,4]
        elif method == "efficiency_first":
            if args.dataset == "cifar10":
                adapter_struct = [1,120,2]
            elif args.dataset == "cifar100":
                adapter_struct = [1,60,2]
            elif args.dataset == "flowers102":
                adapter_struct = [1,120,2]
            elif args.dataset == "svhn":
                adapter_struct = [2,60,2]
            elif args.dataset == "food101":
                adapter_struct = [2,120,2]
        args.num_head = adapter_struct[0]
        args.rank = adapter_struct[1]
        args.first_layer = 12 - adapter_struct[2]

        acc, n_param = test_CryptPEFT(args=args, device=DEVICE)
        logger.add_line(f"acc:{acc}")
        logger.add_line(f"n_param:{n_param}")
    elif args.option == "SFT":
        num_layer = 1 #Last layer->1    Last 2 layer->2
        args.finetune_layer = num_layer
        args.adapt_on = False
        logger = Logger(log2file=True if args.log_dir is not None else False, mode=f"{args.option}_{num_layer}_{args.dataset}", path=args.log_dir)
        acc, n_param = test_CryptPEFT(args=args, device=DEVICE)
        logger.add_line(f"acc:{acc}")
        logger.add_line(f"n_param:{n_param}")
    elif args.option == "adapter_place":
        args.adapter_type = "single_adapter"
        layer_id = [0,1,2,3,4,5,6,7,8,9,10,11]
        for id in layer_id:
            args.layer_id = id
            logger = Logger(log2file=True if args.log_dir is not None else False, mode=f"{args.option}_{id}_{args.dataset}", path=args.log_dir)
            acc, n_param = test_CryptPEFT(args=args, device=DEVICE)
            logger.add_line(f"acc:{acc}")
            logger.add_line(f"n_param:{n_param}")
    elif args.option == "scaler":
        scaler = [0.1, 0.5, 1.0, 2.0, 4.0, 6.0]
        ######our paper used######
        args.rank = 120
        args.first_layer = 2
        args.num_head = 2
        for scale in scaler:
            args.adapter_scaler = scale
            logger = Logger(log2file=True if args.log_dir is not None else False, mode=f"{args.option}_{scale}_{args.dataset}", path=args.log_dir)
            acc, n_param = test_CryptPEFT(args=args, device=DEVICE)
            logger.add_line(f"acc:{acc}")
            logger.add_line(f"n_param:{n_param}")
    elif args.option == "test":
        logger = Logger(log2file=True if args.log_dir is not None else False, mode=f"{args.option}_{args.adapter_type}_{args.dataset}", path=args.log_dir)
        acc, n_param = test_CryptPEFT(args=args, device=DEVICE)
        logger.add_line(f"acc:{acc}")
        logger.add_line(f"n_param:{n_param}")




    
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
