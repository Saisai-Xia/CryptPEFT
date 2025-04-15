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
    parser.add_argument('--dataset', default='imagenet', choices=['imagenet', 'cifar100', 'flowers102', 'svhn', 'food101', 'caltech101', 'sun397', 'cifar10'])

    # CryptPEFT related parameters
    parser.add_argument('--adapt_on', default=False, action='store_true', help='whether activate Adapter')
    parser.add_argument('--rank', default=120, type=int, help='rank of adapter')
    parser.add_argument('--num_repeat_blk', default=1, type=int, help='number of base block in adapter')
    parser.add_argument('--adapter_type', default="liner", type=str, help='type of adapter')
    parser.add_argument('--adapter_scaler', default="0.1", type=str, help='adapter scaler')
    parser.add_argument('--fulltune', default=False, action='store_true', help='full finetune model')

    parser.add_argument('--approx', default=None, type=int, help='whether use approx')
    parser.add_argument('--first_layer', default=6, type=int, help='Start using outputs for transfer learning.')
    parser.add_argument('--num_head', default=12, type=int)

    parser.add_argument('--adapter_arch', default="CryptPEFT", type=str, choices=['lora', 'CryptPEFT', 'adaptformer'], help='adapter architecture')

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

    if n_parameters > 0.05*backbone_n_parameters:
        return 0.0, n_parameters / 1.e6

    print('number of params (M): %.2f' % (n_parameters / 1.e6))


    criterion = torch.nn.CrossEntropyLoss()


    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=5e-4, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(args.epochs):
        _, _, accuracy_test, _ = train_and_test_model(model, trainloader, testloader, criterion, optimizer, device, epoch)
        scheduler.step()
    
    # logger.add_line(f"acc:{accuracy_test}")

    return accuracy_test, n_parameters / 1.e6



def main(args):
    print(torch.cuda.is_available())
    DEVICE = torch.device(args.device)
    search = False
    approx = True
    baseline = False
    old_adapter_arch = False
    if search:
        logger = Logger(log2file=True if args.log_dir is not None else False, mode=f"s_auto_search_"+args.dataset, path=args.log_dir)
        num_heads = [1,2,4,6,10,12]
        bottlenecks = [60, 120, 180, 240, 300]
        first_layers = [6,8,10]
        res = {}
        max_acc = 0.0
        for num_head in num_heads:
            for bottleneck in bottlenecks:
                for first_layer in first_layers:
                    args.rank = bottleneck
                    args.first_layer = first_layer
                    args.num_head = num_head
                    print(f"now-> num_head{num_head} bottleneck{bottleneck} first_layer{first_layer}")
                    acc, n_param = test_CryptPEFT(args=args, device=DEVICE)
                    if(acc > max_acc):
                        max_acc = acc
                        res['acc'] = max_acc
                        res['n_param'] = n_param
                        res['num_head'] = num_head
                        res['bottleneck'] = bottleneck
                        res['first_layer'] = first_layer
        res['dataset'] = args.dataset

        print("auto search finish")
        for key, value in res.items():
                logger.add_line(f"{key}: {value}")
    elif approx: # use efficiency-first configuration
        logger = Logger(log2file=True if args.log_dir is not None else False, mode=f"s_approx_"+args.dataset, path=args.log_dir)
        ViT_B_layers = 12
        if args.dataset == 'cifar100':
            param = [1, 120, 2] # h, r, s
            args.rank = param[1]
            args.first_layer = ViT_B_layers - param[2]
            args.num_head = param[0]
            acc, n_param = test_CryptPEFT(args=args, device=DEVICE)
            logger.add_line(f"acc:{acc}")
            logger.add_line(f"n_param:{n_param}")
            logger.add_line(f"num_head:{args.num_head}")
            logger.add_line(f"rank:{args.rank}")
            logger.add_line(f"first_layer:{args.first_layer}")
            logger.add_line(f"approx_p:{args.approx}")

        elif args.dataset == 'food101':
            param = [2, 120, 2] # h, r, s
            args.rank = param[1]
            args.first_layer = ViT_B_layers - param[2]
            args.num_head = param[0]
            acc, n_param = test_CryptPEFT(args=args, device=DEVICE)
            logger.add_line(f"acc:{acc}")
            logger.add_line(f"n_param:{n_param}")
            logger.add_line(f"num_head:{args.num_head}")
            logger.add_line(f"rank:{args.rank}")
            logger.add_line(f"first_layer:{args.first_layer}")
            logger.add_line(f"approx_p:{args.approx}")
        
        elif args.dataset == 'svhn':
            param = [2, 120, 2] # h, r, s
            args.rank = param[1]
            args.first_layer = ViT_B_layers - param[2]
            args.num_head = param[0]
            acc, n_param = test_CryptPEFT(args=args, device=DEVICE)
            logger.add_line(f"acc:{acc}")
            logger.add_line(f"n_param:{n_param}")
            logger.add_line(f"num_head:{args.num_head}")
            logger.add_line(f"rank:{args.rank}")
            logger.add_line(f"first_layer:{args.first_layer}")
            logger.add_line(f"approx_p:{args.approx}")

    elif old_adapter_arch: # lora and adaptformer
        logger = Logger(log2file=True if args.log_dir is not None else False, mode=f"s_arch_{args.adapter_arch}_"+args.dataset, path=args.log_dir)
        s = 2 # for efficiency priority
        ViT_B_layers = 12
        args.first_layer = ViT_B_layers - s
        acc, n_param = test_CryptPEFT(args=args, device=DEVICE)
        logger.add_line(f"acc:{acc}")
        logger.add_line(f"n_param:{n_param}")


    
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
