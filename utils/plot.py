import re
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--fig_name", type=str, default="fig8a")
    parser.add_argument("--output_dir", type=str, default="AE/eval_private_inference_result")

    return parser





def plot_fig9a(args):

    files = [
        "ablation_LinAtten_1gbit_MPCFormer_cifar100",
        "ablation_LinAtten_1gbit_MPCViT_cifar100",
        "ablation_LinAtten_1gbit_SHAFT_cifar100",
        "ablation_LinAtten_1gbit_CryptPEFT_cifar100"
    ]

    
    comm_rounds = []
    folder_path = args.output_dir
    
    for file in files:
        for filename in os.listdir(folder_path):
            if filename.startswith(file):
                file = filename
                break
        filepath = os.path.join(folder_path, file)
        
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith("comm_round:"):
                    value = float(line.strip().split(":")[1])
                    comm_rounds.append(value)
                    break


    labels = [f.split('_')[3] for f in files]  


    plt.figure(figsize=(8, 5))
    plt.bar(labels, comm_rounds, color=('#2a607e','#2a607e','#2a607e',"#e15759"),width=0.2)
    plt.ylabel("Comm. round")
    plt.ylim(0, max(comm_rounds) * 1.2)

    # for i, v in enumerate(comm_rounds):
    #     plt.text(i, v + 1, f"{v:.1f}", ha='center', va='bottom')

    plt.tight_layout()

    output_dir = args.output_dir
    output_file = "fig9a.png"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, output_file)

    plt.savefig(save_path)

    plt.close()

def plot_fig9b(args):

    files = [
        "ablation_LinAtten_1gbit_MPCFormer_cifar100",
        "ablation_LinAtten_1gbit_MPCViT_cifar100",
        "ablation_LinAtten_1gbit_SHAFT_cifar100",
        "ablation_LinAtten_1gbit_CryptPEFT_cifar100"
    ]


    comm_costs = []
    folder_path = args.output_dir
    for file in files:
        for filename in os.listdir(folder_path):
            if filename.startswith(file):
                file = filename
                break
        filepath = os.path.join(folder_path, file)
        
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith("comm_cost:"):
                    value = float(line.strip().split(":")[1])
                    comm_costs.append(value)
                    break


    labels = [f.split('_')[3] for f in files]  


    plt.figure(figsize=(8, 5))
    plt.bar(labels, comm_costs, color=('#2a607e','#2a607e','#2a607e',"#e15759"), width=0.2)
    plt.ylabel("Comm. (GB)")
    plt.ylim(0, max(comm_costs) * 1.2)


    # for i, v in enumerate(comm_costs):
    #     plt.text(i, v + 1, f"{v:.2f}", ha='center', va='bottom')

    plt.tight_layout()

    output_dir = args.output_dir
    output_file = "fig9b.png"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, output_file)

    plt.savefig(save_path)

    plt.close()

def plot_fig10a(args):

    folder_path = args.output_dir  

    save_path = os.path.join(folder_path,"fig10a.png")  

    bandwidths = ["100mbit", "500mbit", "1gbit", "5gbit"]
    models = ["MPCFormer", "MPCViT", "SHAFT", "CryptPEFT"]

    prefixes = {}
    for bw in bandwidths:
        for model in models:
            prefixes[(bw, model)] = f"ablation_LinAtten_{bw}_{model}_cifar100"

    data = {model: {} for model in models}


    

    for (bw, model), prefix in prefixes.items():
        for filename in os.listdir(folder_path):
            filepath = os.path.join(folder_path, filename)
            if not os.path.isfile(filepath):
                continue
            if filename.startswith(prefix):
                with open(filepath, "r") as f:
                    content = f.read()
                    for line in content.splitlines():
                        if line.startswith("comm_time:"):
                            comm_time = float(line.split(":")[1].strip())
                            data[model][bw] = comm_time
                            break
                break


    comm_time_matrix = []
    for model in models:
        comm_time_matrix.append([data[model].get(bw, 0) for bw in bandwidths])

    comm_time_matrix = np.array(comm_time_matrix)  


    num_models = len(models)
    num_bw = len(bandwidths)
    bar_width = 0.2
    index = np.arange(num_bw)


    colors = ["#2a607e", "#7ba0b7", "#deebf7", "#e15759"]

    plt.figure(figsize=(10, 6))

    for i, model in enumerate(models):
        plt.bar(index + i * bar_width, comm_time_matrix[i], bar_width, label=model, color=colors[i])

    
    plt.ylabel("Comm. time (s)", fontsize=12)
    plt.xticks(index + bar_width * (num_models / 2 - 0.5), bandwidths)
    plt.legend(fontsize=10)

    plt.tight_layout()


    plt.savefig(save_path)
    plt.close()

def plot_fig10b(args):

    folder_path = args.output_dir  

    save_path = os.path.join(folder_path,"fig10b.png")  

    bandwidths = ["100mbit", "500mbit", "1gbit", "5gbit"]
    models = ["MPCFormer", "MPCViT", "SHAFT", "CryptPEFT"]

    prefixes = {}
    for bw in bandwidths:
        for model in models:
            prefixes[(bw, model)] = f"ablation_LinAtten_{bw}_{model}_cifar100"

    data = {model: {} for model in models}


    for (bw, model), prefix in prefixes.items():
        for filename in os.listdir(folder_path):
            filepath = os.path.join(folder_path, filename)
            if not os.path.isfile(filepath):
                continue
            if filename.startswith(prefix):
                with open(filepath, "r") as f:
                    content = f.read()
                    for line in content.splitlines():
                        if line.startswith("total_time:"):
                            comm_time = float(line.split(":")[1].strip())
                            data[model][bw] = comm_time
                            break
                break


    comm_time_matrix = []
    for model in models:
        comm_time_matrix.append([data[model].get(bw, 0) for bw in bandwidths])

    comm_time_matrix = np.array(comm_time_matrix)  


    num_models = len(models)
    num_bw = len(bandwidths)
    bar_width = 0.2
    index = np.arange(num_bw)

    colors = ["#2a607e", "#7ba0b7", "#deebf7", "#e15759"]

    plt.figure(figsize=(10, 6))

    for i, model in enumerate(models):
        data = comm_time_matrix[i]
        plt.bar(index + i * bar_width, data, bar_width, label=model, color=colors[i])

    
    plt.ylabel("Total time (s)", fontsize=12)
    plt.xticks(index + bar_width * (num_models / 2 - 0.5), bandwidths)
    plt.legend(fontsize=10)

    plt.tight_layout()


    plt.savefig(save_path)
    plt.close()

def main(args):
    plot_fig9a(args)
    plot_fig9b(args)
    plot_fig10a(args)
    plot_fig10b(args)

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)