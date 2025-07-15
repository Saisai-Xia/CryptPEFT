# Installation

The source code can be downloaded [here](https://anonymous.4open.science/r/CryptPEFT-8405).

Run:
```bash
conda env create -f environment.yml
conda activate adapter310

git clone https://github.com/andeskyl/SHAFT
cd SHAFT
pip install .
cd ..
```
# Notes

The main evaluation scripts and programs for NDSS-2026-AE are located in the AE folder. Inside this folder, you will find the following contents:
```bash
checkpoints [folder]    Stores the weights of the adapters found during the search
eval_private_inference_result [folder]    Stores the evaluation results for private inference
eval_result [folder]    Stores the evaluation results for model utility
search_result [folder]    Stores the results of the search algorithm
ablation_LinAtten_PI.sh [script]    Reports the results in Fig. 8 and Fig. 9
eval_CRYPTPEFT_PI.sh [script]    Reports the results in Table VI for CRYPTPEFT
eval_model_utility.py [code]    Code for the experiments in Table V
eval_model_utility.sh [script]    Reports the results in Table V
eval_MPCViT_PI.sh [script]    Reports the results in Table VIII
eval_PEFT_AdaptFormer_PI.sh [script]    Reports the results in Table VI for the PEFT_AdaptFormer baseline
eval_PEFT_LoRA_PI.sh [script]    Reports the results in Table VI for the PEFT_LoRA baseline
eval_SFT_Last_2_Layers_PI.sh [script]    Reports the results in Table VI for the SFT_Last_2_Layers baseline
eval_SFT_Last_PI.sh [script]    Reports the results in Table VI for the SFT_Last baseline
README.md
search.sh [script]    Script to run the search algorithm
```
# Running Experiments

## 1. Reproduce Table V
According to the relevant notice from AE, we use only the `CPU` by default for evaluation. If your device has a GPU, you can modify the `device` field in the eval_model_utility.sh script to `cuda:0` or another GPU device to accelerate the experiment.

Run (make sure you are in the project root directory `CryptPEFT`):

```bash
bash AE/eval_model_utility.sh
```

In this script, we include the following five datasets: `cifar10`, `cifar100`, `flowers102`, `svhn`, `food101`, and two different configurations for CryptPEFT: `CRYPTPEFT_Efficiency_first` and `CRYPTPEFT_Utility_first`. For both configurations and each dataset, we provide pre-trained weights, so running this script will complete quickly.

For baselines, there are four options: `PEFT_LoRA`, `PEFT_AdaptFormer`, `SFT_Last_Layer`, `SFT_Last_2_Layers`. You can modify the *eval_method* field in `eval_model_utility.sh` to run different baseline experiments. Note that we do not provide pre-trained weights for baselines. Reproducing each baseline requires about one GPU hour of training to obtain the results.

The experimental data will be saved in the folder `AE/eval_result`.

## 2. Reproduce Table VI and Table VII

This experiment evaluates the private inference latency of CryptPEFT and various baselines under different network environments. We simulate two common settings: a wide-area network (WAN, 400 Mbps, 4 ms latency) and a local-area network (LAN, 1 Gbps, 0.5 ms latency). To closely reflect real-world scenarios, we strongly recommend using the Traffic Control (TC) tool on Linux to simulate network environments. Directly using analytical formulas to convert communication time without actual network simulation often leads to large discrepancies.

### 2.0. Notes

(1): Other baseline experiments can be executed following a similar procedure. We only provide necessary instructions for reproducing the key results.

(2): Private inference latency consists of both communication time and computation time. If the simulated network environment does not match, the communication time may fluctuate. By default, the communication environment is better than LAN or WAN, so without simulation, communication time will be lower than expected. Additionally, different CPU computational capabilities will affect computation time. We used an Intel Xeon Silver 4310 CPU with 12 cores and 24 threads.

(3): Regardless of the network environment or CPU capabilities, the number of communication rounds and communication volume in private inference remain unchanged. If you are unable to simulate the network environment or if your CPU specifications differ significantly from the Intel Xeon Silver 4310, you can skip the simulation and directly follow Section 2.2 to perform private inference. In this case, replace LAN with Default in the script.Pay special attention to the data for Comm. (GB) and Comm. round in Table VII of the paper:
| Metrics | SFT | CryptPEFT | Improvements |
| :---: | :---: | :---: | :---: |
| Comm. (GB) | 1.55 | 0.03 | 51.67x |
| Comm. round | 77 | 55 | 1.40x |

### 2.1. Simulate LAN environment

Open a terminal on Linux (avoid using VSCode terminal as it may cause issues), and connect to the remote server via SSH (if not running locally). Designate this terminal as a dedicated terminal for the simulation network. 

run:
```bash
sudo tc qdisc add dev lo root netem rate 1gbit delay 0.5ms
```
### 2.2. Run private inference

(Here you can use VSCode to execute any scripts or code.)

We provide five scripts: `eval_CRYPTPEFT_PI.sh`, `eval_PEFT_AdaptFormer_PI.sh`, `eval_PEFT_LoRA_PI.sh`, `eval_SFT_Last_2_Layers_PI.sh`, and `eval_SFT_Last_PI.sh`, corresponding to CryptPEFT and four baseline experiments respectively.

To reproduce CryptPEFT, open two terminals $0 and $1, and run:
```bash
$0: bash AE/eval_CRYPTPEFT_PI.sh 0 LAN
$1: bash AE/eval_CRYPTPEFT_PI.sh 1 LAN
```
To reproduce a baseline, e.g., SFT-Last-Layer (It can also be replaced with other baseline experiment scripts), open two terminals $0 and $1, and run:
```bash
$0: bash AE/eval_SFT_Last_PI.sh 0 LAN
$1: bash AE/eval_SFT_Last_PI.sh 1 LAN
```
### 2.3. Simulate WAN environment

Open the dedicated terminal mentioned in [2.1](#21-simulate-lan-environment)

run:
```bash
sudo tc qdisc del dev lo root
sudo tc qdisc add dev lo root netem rate 400mbit delay 4ms
```
### 2.4. Run private inference

To reproduce CryptPEFT, open two terminals $0 and $1, and run:
```bash
$0: bash AE/eval_CRYPTPEFT_PI.sh 0 WAN
$1: bash AE/eval_CRYPTPEFT_PI.sh 1 WAN
```
To reproduce a baseline, e.g., SFT-Last-Layer, open two terminals $0 and $1, and run:
```bash
$0: bash AE/eval_SFT_Last_PI.sh 0 WAN
$1: bash AE/eval_SFT_Last_PI.sh 1 WAN
```
### 2.5. Reset network
Open the dedicated terminal mentioned in [2.1](#21-simulate-lan-environment)

run:

```bash
sudo tc qdisc del dev lo root
```
### 2.6. Results

The experimental data will be saved in the folder `AE/eval_private_inference_result`, with filenames starting with *eval_CryptPEFT* or *eval_Last_layer*.

## 3. Reproduce Table VIII

### 3.0. Notes

Table VIII summarizes the overall data. Utility data for MPCViT on cifar10 and cifar100 are taken from the original MPCViT paper. The utility and private inference latency data for CryptPEFT on cifar10 and cifar100 can be obtained from the above experiments (presented in Table V and Table VI of our paper). This experiment mainly reproduces the private inference latency of MPCViT (for fairness, we implemented MPCViT on the same system).

### 3.1. Simulate network environment

This part follows exactly Sections [2.1](#21-simulate-lan-environment), [2.3](#23-simulate-wan-environment), and [2.5](#25-reset-network).

### 3.2. Run private inference

To reproduce MPCViT, open two terminals $0 and $1.

If you are in LAN environment, run:

```bash
$0: bash AE/eval_MPCViT_PI.sh 0 LAN
$1: bash AE/eval_MPCViT_PI.sh 1 LAN
```

If you are in WAN environment, run:

```bash
$0: bash AE/eval_MPCViT_PI.sh 0 WAN
$1: bash AE/eval_MPCViT_PI.sh 1 WAN
```

If you are in Default environment (results may vary significantly from the paper and are not recommended), run:

```bash
$0: bash AE/eval_MPCViT_PI.sh 0 Default
$1: bash AE/eval_MPCViT_PI.sh 1 Default
```

### 3.3. Reset network

Don't forget to restore the network environment.

run:

```bash
sudo tc qdisc del dev lo root
```

### 3.4. Results

The experimental data will be saved in the folder `AE/eval_private_inference_result`, with filenames starting with *eval_MPCViT*.

## 4. Reproduce Fig. 8 and Fig. 9

This experiment evaluates the private inference efficiency of LinAtten under different network bandwidths, and compares it with common attention mechanisms. Similar to previous experiments, the inference time will vary on different devices depending on CPU cores, threads, and network simulation. However, the communication volume and number of rounds shown in Fig. 8 will remain consistent. If you cannot simulate different bandwidths, you can focus on reproducing Fig. 8.

### 4.0. Adjust network bandwidth

For all bandwidth settings, we fix the latency at 4 ms. Taking 5 Gbps as an example:

First, ensure you are in the default network state by running the following command to remove previous settings:
```bash
sudo tc qdisc del dev lo root
```
You will see:
```
Error: Cannot delete qdisc with handle of zero.
```
Then run:
```bash
sudo tc qdisc add dev lo root netem rate 5gbit delay 4ms
```

### 4.1. Run private inference

For 5 Gbps, open two terminals $0 and $1, and run:

```bash
$0: bash ablation_LinAtten_PI.sh 0 5G
$1: bash ablation_LinAtten_PI.sh 1 5G
```

### 4.2. Reset network

run:
```bash
sudo tc qdisc del dev lo root
```

### 4.3. Results

The experimental data will be saved in the folder `AE/eval_private_inference_result`, with filenames starting with *ablation_LinAtten*.

## 5. Adapter search based on NAS

In response to the reviewers’ feedback, we plan to include an ablation study to highlight the effectiveness of our proposed NAS strategy, which was not part of the original submission. As such, we understand that this component may not be considered during the AE process.

This part corresponds to our updated code version, the related data are not yet included in the original submission. We will update the explanation on how to reproduce the paper’s results later. For now, we only provide the command to run the search algorithm.

Make sure you are in the project root directory CryptPEFT.

Run:
```bash
bash AE/search.sh
```
The above search process uses the flowers102 dataset, which is relatively small. With an RTX 4090 GPU, the search can be completed within one hour. If you have sufficient time, you may consider replacing it with a larger-scale dataset. Note that if you do not have a GPU and instead use an 8-core CPU, it will take approximately 20 hours.