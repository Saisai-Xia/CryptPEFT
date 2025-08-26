from easydict import EasyDict
from benchmark.models import Adjuster, VisionTransformer, transfer_scope_baseline, MPCViT, BaseAdapter
import crypten.communicator as comm
import crypten
import crypten.nn
from benchmark.utils.multiprocess_launcher import MultiProcessLauncher
from benchmark.utils.multimachine_launcher import MultiMachineLauncher
import torch
import time
import torchvision
import argparse
from log_utils import Logger
import gc

from crypten.config import cfg
cfg.communicator.verbose = True
crypten.debug.configure_logging()
import os

def get_args_parser():
    parser = argparse.ArgumentParser(description="CrypTen Cifar Training")
    parser.add_argument(
        "--world_size",
        type=int,
        default=2,
        help="The number of parties to launch. Each party acts as its own process",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="The rank of the current party. Each party acts as its own process",
    )
    parser.add_argument(
        "--master_address",
        type=str,
        default = "127.0.0.1",
        help="master IP Address",
    )
    parser.add_argument(
        "--master_port",
        type=int,
        default=29557,
        help="master port",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="NCCL",
        help="backend for torhc.distributed, 'NCCL' or 'gloo'.",
    )
    #for benchmark
    parser.add_argument("--method", type=str, default = "lora", help="benchmark method",choices=["lora", "adaptformer", "SFT", "CryptPEFT", "mpcvit", "base_adapter"]) 
    parser.add_argument("--atten_method", type=str, default="CryptPEFT")
    parser.add_argument("--dataset", type=str, default = "cifar100")
    parser.add_argument("--batch_size", type=int, default = 64)
    parser.add_argument("--transfer_scope", type=int, default = 1)
    parser.add_argument("--ablation", default=False, action='store_true')

    parser.add_argument("--net", type=str, default="none")
    parser.add_argument("--mode", type=str, default="none")

    return parser

def set_config(args):

    if args.method == "mpcvit":
        if args.dataset == "cifar100":
            alpha_list = [[1, 1, 1, 1],
                            [1, 1, 1, 1],
                            [1, 1, 1, 1],
                            [1, 0, 1, 1],
                            [1, 1, 1, 1],
                            [0, 0, 0, 0],
                            [0, 0, 0, 1]] # get from https://github.com/PKU-SEC-Lab/mpcvit vit_7_4_32 for cifar100
            num_classes = 100
        elif args.dataset == "cifar10":
            alpha_list = [[1, 0, 0, 0],
                            [1, 1, 1, 1],
                            [1, 1, 1, 1],
                            [1, 1, 1, 1],
                            [0, 1, 1, 1],
                            [0, 1, 0, 1],
                            [0, 1, 0, 1]]
            num_classes = 10
        config = EasyDict(
        dim =256,
        alpha_list = alpha_list,
        image_size = 32,
        patch_size = 4,
        num_encoderblk = 7,
        mlp_dim = 256*4,
        mlp_drop = 0.0,
        layer_norm_eps = 1e-6,
        num_heads = 4,
        attn_drop = 0.0,
        proj_drop = 0.0,
        use_approx = False,
        encoder_drop = 0.0,
        num_classes = num_classes,
        batch_size = args.batch_size,
        )
    else:
        config = EasyDict(
        dim = 768,
        image_size = 224,
        patch_size = 16,
        num_encoderblk = 12,
        mlp_dim = 768*4,
        mlp_drop = 0.0,
        layer_norm_eps = 1e-6,
        num_heads = 12,
        attn_drop = 0.0,
        proj_drop = 0.0,
        use_approx = True,
        encoder_drop = 0.0,
        bottleneck = 120,
        adjuster_drop = 0.1,
        adjuster_scale = 4.0,
        adjuster_n_blks = 1,
        num_classes = 100,
        transfer_scope = 1,
        use_PEFT = None,
        batch_size = args.batch_size,
        )

    
    if args.net == "LAN":
        cifar100_CryptPEFT_adapter_set = {'h': 1, 'r': 240, 's': 1}
        food101_CryptPEFT_adapter_set = {'h': 6, 'r': 180, 's': 1}
        svhn_CryptPEFT_adapter_set = {'h': 12, 'r': 60, 's': 1}
        cifar10_CryptPEFT_adapter_set = {'h': 4, 'r': 120, 's': 1}
        flowers102_CryptPEFT_adapter_set = {'h': 1, 'r': 120, 's': 1}
    else:# WAN or others
        cifar100_CryptPEFT_adapter_set = {'h': 1, 'r': 300, 's': 1}
        food101_CryptPEFT_adapter_set = {'h': 4, 'r': 180, 's': 1}
        svhn_CryptPEFT_adapter_set = {'h': 12, 'r': 300, 's': 1}
        cifar10_CryptPEFT_adapter_set = {'h': 2, 'r': 120, 's': 2}
        flowers102_CryptPEFT_adapter_set = {'h': 1, 'r': 180, 's': 1}

    if args.dataset == "cifar100":
        config.num_classes = 100
    elif args.dataset == "food101":
        config.num_classes = 101
    elif args.dataset == "svhn":
        config.num_classes = 10
    elif args.dataset == "cifar10":
        config.num_classes = 10
    elif args.dataset == "flowers102":
        config.num_classes = 102
    
    if args.method == "lora":
        config.use_PEFT = "lora"
    elif args.method == "adaptformer":
        config.use_PEFT = "adaptformer"
    elif args.method == "SFT":
        config.transfer_scope = args.transfer_scope
    elif args.method == "CryptPEFT":
        if args.dataset == "cifar100":
            config.num_heads = cifar100_CryptPEFT_adapter_set['h']
            config.bottleneck = cifar100_CryptPEFT_adapter_set['r']
            config.transfer_scope = cifar100_CryptPEFT_adapter_set['s']
        elif args.dataset == "food101":
            config.num_heads = food101_CryptPEFT_adapter_set['h']
            config.bottleneck = food101_CryptPEFT_adapter_set['r']
            config.transfer_scope = food101_CryptPEFT_adapter_set['s']
        elif args.dataset == "svhn":
            config.num_heads = svhn_CryptPEFT_adapter_set['h']
            config.bottleneck = svhn_CryptPEFT_adapter_set['r']
            config.transfer_scope = svhn_CryptPEFT_adapter_set['s']
        elif args.dataset == "cifar10":
            config.num_heads = cifar10_CryptPEFT_adapter_set['h']
            config.bottleneck = cifar10_CryptPEFT_adapter_set['r']
            config.transfer_scope = cifar10_CryptPEFT_adapter_set['s']
        elif args.dataset == "flowers102":
            config.num_heads = flowers102_CryptPEFT_adapter_set['h']
            config.bottleneck = flowers102_CryptPEFT_adapter_set['r']
            config.transfer_scope = flowers102_CryptPEFT_adapter_set['s']
    elif args.method == "base_adapter":
        config.num_heads = 2
        config.bottleneck = 120

    config["dataset"] = args.dataset
    config["method"] = args.method
    config["batch_size"] = args.batch_size
    config["atten_method"] = args.atten_method
    config["ablation"] = args.ablation
    config["net"] = args.net
    config["mode"] = args.mode
    return config



def run_test(args):
    Alice = 0
    Server = 1
    crypten.init()
    rank = comm.get().get_rank()

    if rank == Alice:
        path = "AE/eval_private_inference_result"
        if not os.path.exists(path):
            os.makedirs(path)
        if args.mode == "ablation_LinAtten":
            logger = Logger(log2file=True, mode=f"{args.mode}_{args.net}_{args.atten_method}_{args.dataset}", path=path)
        else:
            logger = Logger(log2file=True, mode=f"{args.mode}_{args.net}_{args.dataset}", path=path)

    if args.method in ["lora", "adaptformer", "mpcvit"]:
        #get input size
        test_full_ViT = True
    else:
        test_full_ViT = False

    if not test_full_ViT:
        B = args.batch_size
        N = (224//16)**2 + 1
        C = 768
        input_size = [B, N, C]
    else:
        if args.method == "mpcvit":
            B = args.batch_size
            C = 3
            H = 32
            W = 32
            input_size = [B, C, H, W]
        else:
            B = args.batch_size
            C = 3
            H = 224
            W = 224
            input_size = [B, C, H, W]


    plaintext_input_size = [args.batch_size, 3, 224, 224]
    plaintext_model = torchvision.models.vit_b_16()
    plaintext_input = torch.empty(plaintext_input_size)

    dummy_input = torch.empty(input_size)

    if args.method in ["lora", "adaptformer"]:
        dummy_model = VisionTransformer(args)
        dummy_model = dummy_model.encrypt(src=Server)
        dummy_input = crypten.cryptensor(dummy_input, src=Alice)
    elif args.method == "SFT":
        dummy_model = transfer_scope_baseline(args=args)
        dummy_model = dummy_model.encrypt(src=Server)
        dummy_input = crypten.cryptensor(dummy_input, src=Alice)
    elif args.method == "CryptPEFT":
        dummy_model = Adjuster(args)
        dummy_model = dummy_model.encrypt(src=Server)
        dummy_input = crypten.cryptensor(dummy_input, src=Alice)
    elif args.method == "mpcvit":
        dummy_model = MPCViT(args)
        dummy_model = dummy_model.encrypt(src=Server)
        dummy_input = crypten.cryptensor(dummy_input, src=Alice)
    elif args.method == "base_adapter":
        dummy_model = BaseAdapter(args)
        dummy_model = dummy_model.encrypt(src=Server)
        dummy_input = crypten.cryptensor(dummy_input, src=Alice)

    use_GPU = False
    if use_GPU:
        plaintext_input = plaintext_input.to(f"cuda:{rank}")
        plaintext_model = plaintext_model.to(f"cuda:{rank}")
        dummy_input.cuda(rank)
        dummy_model.cuda(rank)
    repeat = 10
    plaintext_model.eval()
    dummy_model.eval()
    with torch.no_grad():
        plaintext_time = 0
        #plaintext time
        if args.method in ["SFT", "CryptPEFT"]:
            time_s = time.time()
            for i in range(repeat):
                output = plaintext_model(plaintext_input)
            time_e = time.time()
            plaintext_time = time_e - time_s
            
            del plaintext_model 
            del plaintext_input
            gc.collect()
            torch.cuda.empty_cache()

        comm.get().reset_communication_stats()
        time_s = time.time()
        for i in range(repeat):
            output = dummy_model(dummy_input)
        time_e = time.time()

    cost = {}

    cost["total_time"] = (time_e - time_s + plaintext_time) / args.batch_size / repeat
    cost["comm_round"] = comm.get().get_communication_stats()["rounds"] / repeat
    cost["comm_cost"] = (comm.get().get_communication_stats()["bytes"] / (1024*1024*1024)) / args.batch_size / repeat #B -> GB
    cost["comm_time"] = comm.get().get_communication_stats()["time"] / args.batch_size / repeat
    
    if rank == Alice:
        for key, value in cost.items():
            logger.add_line(f"{key}: {value}")

    print(f"finish, and print communication stats of {repeat} executions.")
    comm.get().print_communication_stats()

def main():
    parser = get_args_parser()
    param = parser.parse_args()
    args = set_config(param)
    #launcher = MultiProcessLauncher(2, run_test, args)
    launcher = MultiMachineLauncher(param.world_size, param.rank, param.master_address, param.master_port, run_test, args)
    launcher.start()
    launcher.join()
    launcher.terminate()

if __name__ == '__main__':
    main()



