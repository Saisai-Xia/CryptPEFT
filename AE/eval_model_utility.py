from math import ceil
from tqdm import tqdm
import torch
import argparse
from pathlib import Path
from demoloader.dataloader import *

#logger
from log_utils import Logger
import time
from eval.controller import Controller

import numpy as np
import random
# ptimized adapter structures obtained by NAS -> [H,R,S]
Utility_first = {
    "cifar10":[12,240,4],
    "cifar100":[1,180,2],
    "food101":[10,240,4],
    "svhn":[10,180,6],
    "flowers102":[12,180,4]
}

Efficiency_first = {
    "cifar10":[1,120,2],
    "cifar100":[1,60,2],
    "food101":[2,120,2],
    "svhn":[2,60,2],
    "flowers102":[1,120,2]
}



def get_args_parser():
    parser = argparse.ArgumentParser('CRYPTPEFT: Efficient and Private Neural Network Inference via Parameter-Efficient Fine-Tuning', add_help=False)
    
    parser.add_argument('--batch_size', default=50, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    
    parser.add_argument('--epochs', default=20, type=int)

    # Model parameters
    parser.add_argument('--model', default='Vit_B_16', type=str, metavar='MODEL',
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
    parser.add_argument('--data_path', default='Adapter/experiments/dataset', type=str,
                        help='dataset path')
    
    parser.add_argument('--nb_classes', default=100, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    
    parser.add_argument('--log_dir', default='CRYPTPEFT_NDSS_AE/eval_result', type=str,
                        help='path where to save log')
    
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')


    # custom configs
    parser.add_argument('--dataset', default='cifar100', choices=['cifar100', 'flowers102', 'svhn', 'food101', 'cifar10'])

    # CryptPEFT related parameters
    parser.add_argument('--adapt_on', default=False, action='store_true', help='whether activate Adapter')
    parser.add_argument('--rank', default=120, type=int, help='rank of adapter')
    parser.add_argument('--num_repeat_blk', default=1, type=int, help='number of base block in adapter')
    parser.add_argument('--adapter_type', default="CryptPEFT", type=str, help='type of adapter')
    parser.add_argument('--adapter_scaler', default="0.5", type=str, help='adapter scaler')
    parser.add_argument('--fulltune', default=False, action='store_true', help='full finetune model')


    parser.add_argument('--first_layer', default=6, type=int, help='Start using outputs for transfer learning.')
    parser.add_argument('--num_head', default=12, type=int)

    parser.add_argument('--layer_id', default=None, type=int)

    parser.add_argument('--adapter_arch', default="CryptPEFT", type=str, choices=['lora', 'CryptPEFT', 'adaptformer'], help='adapter architecture')

    parser.add_argument('--resume', default=False, action='store_true', help='Load the adapter weights from a checkpoint or retrain them')

    parser.add_argument('--eval_method', default='CRYPTPEFT_Efficiency_first', type=str, choices=['CRYPTPEFT_Efficiency_first', 'CRYPTPEFT_Utility_first', 'PEFT_LoRA', 'PEFT_AdaptFormer', 'SFT_Last_Layer', 'SFT_Last_2_Layers', 'search'])


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

def evaluate(model, testloader, criterion, device):
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

    print(f"test accuracy {accuracy_test} | test loss {avg_loss_test}")

    return accuracy_test, avg_loss_test

def get_CryptPEFT_checkpoint(args, device):

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

    if args.eval_method in ['PEFT_LoRA', 'PEFT_AdaptFormer']:
        n_parameters = sum(p.numel() for p in model.parameters())
    
    print('number of private params (M): %.2f' % (n_parameters / 1.e6))


    criterion = torch.nn.CrossEntropyLoss()


    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=5e-4, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(args.epochs):
        _, _, accuracy_test, _ = train_and_test_model(model, trainloader, testloader, criterion, optimizer, device, epoch)
        scheduler.step()
    
    if args.eval_method in ['CRYPTPEFT_Efficiency_first', 'CRYPTPEFT_Utility_first']:
        torch.save({
                    'heads': model.heads.state_dict(),
                    'ln': model.encoder.ln.state_dict(),
                    'adapters': model.encoder.adapters.state_dict(),
                }, f'CRYPTPEFT_NDSS_AE/checkpoints/checkpoint_{args.eval_method}_{args.dataset}.pth')
    else:
        torch.save({
                    'model': model.state_dict(),
                }, f'CRYPTPEFT_NDSS_AE/checkpoints/checkpoint_{args.eval_method}_{args.dataset}.pth')

    return accuracy_test, n_parameters / 1.e6

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

    if args.eval_method in ['PEFT_LoRA', 'PEFT_AdaptFormer']:
        n_parameters = sum(p.numel() for p in model.parameters())
    
    print('number of private params (M): %.2f' % (n_parameters / 1.e6))


    criterion = torch.nn.CrossEntropyLoss()


    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=5e-4, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    if not args.resume:
        for epoch in range(args.epochs):
            _, _, accuracy_test, _ = train_and_test_model(model, trainloader, testloader, criterion, optimizer, device, epoch)
            scheduler.step()
    else:
        file = f'CRYPTPEFT_NDSS_AE/checkpoints/checkpoint_{args.eval_method}_{args.dataset}.pth'
        if os.path.exists(file):
            checkpoint = torch.load(f'CRYPTPEFT_NDSS_AE/checkpoints/checkpoint_{args.eval_method}_{args.dataset}.pth', map_location='cpu')
            if args.eval_method in ['CRYPTPEFT_Efficiency_first', 'CRYPTPEFT_Utility_first']:
                model.heads.load_state_dict(checkpoint['heads'])
                model.encoder.ln.load_state_dict(checkpoint['ln'])
                model.encoder.adapters.load_state_dict(checkpoint['adapters'])
            else:
                model.load_state_dict(checkpoint['model'])
            accuracy_test, _ = evaluate(model,testloader, criterion, device) 
        else:
            print("can not load checkpoint, start traing model")
            for epoch in range(args.epochs):
                _, _, accuracy_test, _ = train_and_test_model(model, trainloader, testloader, criterion, optimizer, device, epoch)
                scheduler.step()
    return accuracy_test, n_parameters / 1.e6

def latency(h,r,s):
    return  (0.03828*h + 0.00465*r + 0.48139)*s + 0.3212

def search_adapter(args):
    DEVICE = torch.device(args.device)
    H = [1,2,4,6,10,12]
    R = [60, 120, 180, 240, 300]
    S = [1,2,3,4,5,6]
    latency_target = 2.7
    logger = Logger(log2file=True if args.log_dir is not None else False, mode=f"CryptPEFT_auto_search_"+args.dataset, path=args.log_dir)
    num_choices_list = [6, 5]  
    controller = Controller(num_choices_list)
    optimizer = torch.optim.Adam(controller.parameters(), lr=0.001)
    res = {}
    s = 1
    width = 4
    utility = 0.0
    utility_target = 1.1*{'cifar10': 97.51, 'cifar100': 87.41, 'flowers102': 90.63, 'svhn': 91.81, 'food101': 85.56}[args.dataset]
    start_time = time.time()
    while s <= max(S):
        stop_cnt = 0
        for step in range(len(H) * len(R)):
            if stop_cnt >= 3:
                break
            actions, log_probs = controller()
            args.rank = R[actions[1]]
            args.first_layer = 12-s
            args.num_head = H[actions[0]]
            if latency(args.num_head, args.rank, s) > latency(H[0],R[0],s+width) or latency(args.num_head, args.rank, s) > latency_target:
                reward = 0
                stop_cnt += 1
                acc = 0
            else:
                acc, n_param = test_CryptPEFT(args=args, device=DEVICE)
                if acc > utility:
                    stop_cnt = 0
                    utility = acc
                    h_best = args.num_head
                    s_best = s
                    r_best = args.rank
                    res['acc'] = utility
                    res['n_param'] = n_param
                    res['num_head'] = h_best
                    res['rank'] = r_best
                    res['scope'] = s_best
                else:
                    stop_cnt += 1
                if utility > utility_target:
                    res['acc'] = utility
                    res['n_param'] = n_param
                    res['num_head'] = h_best
                    res['rank'] = r_best
                    res['scope'] = s_best
                    break
                reward = acc / 100.0

            loss = -sum(log_probs) * reward

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logger.add_line(f"Step {s}->{step+1}: Actions->(h,r,s) = {[H[actions[0]], R[actions[1]], s]}, Reward = {reward:.4f}, stop_cnt = {stop_cnt}, acc = {acc}")
            #logger.add_line(f"Step {s}->{step+1}: Actions->(h,r,s) = {[H[actions[0]], R[actions[1]], s]}, Reward = {reward:.4f}")

        if utility > utility_target:
            break
        s += 1

    end_time = time.time()
    print(f"Search time: {(end_time - start_time)//60} minutes")
    res['dataset'] = args.dataset
    res['search_time'] = (end_time - start_time) // 60

    print("auto search finish")
    for key, value in res.items():
            logger.add_line(f"{key}: {value}")


def main(args):
    DEVICE = torch.device(args.device)
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if args.eval_method in ['CRYPTPEFT_Efficiency_first', 'CRYPTPEFT_Utility_first']:
        args.adapt_on = True
        args.adapter_type = "CryptPEFT"
        args.adapter_arch = "CryptPEFT"
        adapter_struct = Efficiency_first[args.dataset] if args.eval_method == 'CRYPTPEFT_Efficiency_first' else Utility_first[args.dataset]
        args.num_head = adapter_struct[0]
        args.rank = adapter_struct[1]
        args.first_layer = 12 - adapter_struct[2]
    elif args.eval_method in ['PEFT_LoRA', 'PEFT_AdaptFormer']:
        args.adapt_on = True
        args.adapter_type = "lora" if args.eval_method == 'PEFT_LoRA' else "adaptformer"
    elif args.eval_method in ['SFT_Last_Layer', 'SFT_Last_2_Layers']:
        args.adapt_on = False
        args.finetune_layer = 1 if args.eval_method == 'SFT_Last_Layer' else 2
    elif args.eval_method == "search":
        args.adapt_on = True
        args.adapter_type = "CryptPEFT"
        args.adapter_arch = "CryptPEFT"
        search_adapter(args)
    
    if args.eval_method != "search":
        logger = Logger(log2file=True if args.log_dir is not None else False, mode=f"{args.eval_method}_{args.dataset}", path=args.log_dir)
        acc, n_param = test_CryptPEFT(args,DEVICE)
        logger.add_line(f"========= eval result for {args.eval_method} on {args.dataset} =========")
        logger.add_line(f"acc:{acc}")
        logger.add_line(f"n_param:{n_param}")



if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)