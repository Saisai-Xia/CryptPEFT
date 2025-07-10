## 安装

**(NDSS-2026-AE可以跳过这一步，我们给的远程运行实例已配好环境)**

源代码下载链接在[这里](https://anonymous.4open.science/r/CryptPEFT-8405)

run:

```bash
cd CryptPEFT
conda env create -f environment.yml
conda activate adapter310
cd ..

git clone https://github.com/andeskyl/SHAFT
cd SHAFT
pip install .
cd ../CryptPEFT
```

## 说明

NDSS-2026-AE的主要评估脚步和程序在AE文件夹，在AE文件夹中，你会看到下面这些内容

```bash
checkpints [folder] 存放我们搜索到的Adapter的权重
eval_private_inference_result [folder] 存放评估隐私推理效率的结果
eval_result [folder] 存放评估模型utility的结果
ablation_LinAtten_PI.sh [script] 报告Fig. 8和Fig. 9的实验结果
eval_CRYPTPEFT_PI.sh [script] 报告Table VI的CRYPTPEFT实验结果
eval_model_utility.py [code] Table V的实验代码
eval_model_utility.sh [script] 报告Table V的实验结果
eval_MPCViT_PI.sh [script] 报告Table VIII的实验结果
eval_PEFT_AdaptFormer_PI.sh [script] 报告Table VI的baseline实验结果
eval_PEFT_LoRA_PI.sh [script] 报告Table VI的baseline实验结果
eval_SFT_Last_2_Layers_PI.sh [script] 报告Table VI的baseline实验结果
eval_SFT_Last_PI.sh [script] 报告Table VI的baseline实验结果
```

## 运行实验

**(NDSS-2026-AE可以直接从这里开始)**

### 1.复现Table V

run (确保你处于项目根目录`CryptPEFT`下):

```bash
bash AE/eval_model_utility.sh
```

上述脚本我们设置了`cifar10, cifar100, flowers102, svhn, food101`这个五个数据集，以及CryptPEFT的两种不同配置`CRYPTPEFT_Efficiency_first`和`CRYPTPEFT_Utility_first`，针对**CryptPEFT**的两种不同配置以及不同的数据集，我们提供了预训练好的权重文件，因此执行上述脚本会很快执行完毕

baseline有四种方案`PEFT_LoRA, PEFT_AdaptFormer, SFT_Last_Layer, SFT_Last_2_Layers`，可以通过修改eval_model_utility.sh文件中的*eval_method*字段来实现不同的baseline实验，注意，我们没有为baseline方案提供预训练好的权重文件，复现baseline时，每次实验大约需要1小时的时间来训练并得到最终数据.

上述实验数据保存在文件夹`AE\eval_result`下

### 2.复现Table VI and Table VII

该实验模拟不同的网络环境下，**CryptPEFT**和各种baseline方法的隐私推理延迟，我们模拟了两种常见的网络环境，即广域网WAN(400 Mbps, 4 ms), 局域网WAN(1 Gbps, 0.5 ms)，为了贴近真实情况，我们强烈建议你使用Traffic Control (TC) tool on Linux模拟网络环境(如果在不模拟网络环境的情况下直接使用公式换算不同网络环境下的通信时间，往往和真实情况差距较大).

#### 2.0. 注意

1、其余baseline实验可以参考以下流程执行，我们仅给出了复现实验数据必要的部分。

2、隐私推理延迟包含通信时间和计算时间，如果通信环境不匹配，会导致通信时间有波动，默认的通信环境要优于LAN和WAN，因此，如果不模拟通信环境的话，得到的实验数据中的通信时间会小于预期; 如果实验设备cpu计算能力不同，会影响计算时间，我们使用的是an Intel Xeon Silver 4310 CPU, 有12个核心, 24个线程

3、无论通信环境和CPU计算能力如何，隐私推理中的通信轮数和通信量是不变的，在复现下列实验时，如果你受限于某些原因而无法模拟网络环境或CPU规格和Intel Xeon Silver 4310 CPU有较大差距，您可以跳过模拟网络环境，直接参考2.2来执行隐私推理，并把2.2脚本中的`LAN`替换为`Default`

并重点对照论文中Table VII关于**Comm. (GB)** 和 **Comm. round**的数据：

| Metrics | SFT | CryptPEFT | Improvements |
| :---: | :---: | :---: | :---: |
| Comm. (GB) | 1.55 | 0.03 | 51.67x |
| Comm. round | 77 | 55 | 1.40x |

#### 2.1. 模拟LAN环境

在linux上打开一个终端（不要使用vscode等，我们发现这有可能导致出错），使用ssh连接远程服务器（如果你不是在本地执行实验），run:

```bash
sudo tc qdisc add dev lo root netem rate 1gbit delay 0.5ms
```

#### 2.2. 执行隐私推理

（这里你可以使用vscode执行任意脚步、代码等）
我们提供了五个脚本`eval_CRYPTPEFT_PI.sh`, `eval_PEFT_AdaptFormer_PI.sh`, `eval_PEFT_LoRA_PI.sh`, `eval_SFT_Last_2_Layers_PI.sh`, `eval_SFT_Last_PI.sh`，分别对应执行**CryptPEFT**和四个baseline实验.

**复现CryptPEFT**，打开两个终端 $0 和 $1，run:

```bash
$0: bash AE/eval_CRYPTPEFT_PI.sh 0 LAN
$1: bash AE/eval_CRYPTPEFT_PI.sh 1 LAN
```

**复现baseline，如SFT-Last-Layer**, 打开两个终端 $0 和 $1，run:

```bash
$0: bash AE/eval_SFT_Last_PI.sh 0 LAN
$1: bash AE/eval_SFT_Last_PI.sh 1 LAN
```

#### 2.3. 模拟WAN环境

在linux上打开一个终端（不要使用vscode等，我们发现这有可能导致出错），使用ssh连接远程服务器（如果你不是在本地执行实验），run:

```bash
sudo tc qdisc del dev lo root
sudo tc qdisc add dev lo root netem rate 400mbit delay 4ms
```

#### 2.4. 执行隐私推理

**复现CryptPEFT**，打开两个终端 $0 和 $1，run:

```bash
$0: bash AE/eval_CRYPTPEFT_PI.sh 0 WAN
$1: bash AE/eval_CRYPTPEFT_PI.sh 1 WAN
```

**复现baseline，如SFT-Last-Layer**, 打开两个终端 $0 和 $1，run:

```bash
$0: bash AE/eval_SFT_Last_PI.sh 0 WAN
$1: bash AE/eval_SFT_Last_PI.sh 1 WAN
```

#### 2.5. 恢复网络

run

```bash
sudo tc qdisc del dev lo root
```

#### 2.6. 结果

上述实验数据保存在文件夹`AE\eval_private_inference_result`下,文件名以'WAN'或'LAN'开头，后面跟着‘CryptPEFT’或者‘SFT’

### 3.复现Table VIII

#### 3.0. 说明

**Table VIII**是一个综合性的数据，MPCViT在`cifar10` 和 `cifar100`上的utility数据，我们是从MPCViT论文中获得的，CryptPEFT在`cifar10` 和 `cifar100`上的utility数据和隐私推理延迟数据，都能够通过上面两个实验获得，或者说，具体数据都已展示在论文的**Table V** 和 **Table VI**中了，本次实验主要是复现MPCViT的隐私推理延迟(为保证公平，我们将MPCViT实现在了和我们相同的系统上)

#### 3.1. 模拟网络环境

这一部分可以完全参照章节[2.1](#21-模拟lan环境), [2.3](#23-模拟wan环境)和[2.5](#25-恢复网络)

#### 3.2. 执行隐私推理

**复现MPCViT**，打开两个终端 $0 和 $1，

如果你处于LAN环境 run:

```bash
$0: bash AE/eval_MPCViT_PI.sh 0 LAN
$1: bash AE/eval_MPCViT_PI.sh 1 LAN
```

如果你处于WAN环境 run:

```bash
$0: bash AE/eval_MPCViT_PI.sh 0 WAN
$1: bash AE/eval_MPCViT_PI.sh 1 WAN
```

如果你处于Default环境(实验数据会和论文中有较大波动，我们不建议这么做) run:

```bash
$0: bash AE/eval_MPCViT_PI.sh 0 Default
$1: bash AE/eval_MPCViT_PI.sh 1 Default
```

#### 3.3. 恢复网络

run

```bash
sudo tc qdisc del dev lo root
```

#### 3.4. 结果

上述实验数据保存在文件夹`AE\eval_private_inference_result`下,文件名以'WAN'或'LAN'开头，后面跟着‘MPCViT’

### 4.复现Fig. 8 and Fig. 9

这个实验来测试在不同网络带宽下，LinAtten的隐私推理效率，并和常见的几种注意力机制进行比较，和上文所述类似，隐私推理时间这一维度的测试，在不同计算能力的设备下，会有变化，主要取决于CPU核心数和线程数以及是否成功模拟不同网络带宽。但是Fig. 8展示的通信量和通信轮数不会出现明显变化，如果你无法模拟带宽来进行实验，可以重点关注是否能成功复现Fig. 8

#### 4.0. 调整网络带宽

所有不同带宽的情况下，我们都固定网络延迟为4 ms，下面以5Gbps带宽为例

建议先确保已经处于默认网络，执行下面的命令确认删除了之前的网络设置

```bash
sudo tc qdisc del dev lo root
```

按照预期，你会得到下面的反馈

```bash
Error: Cannot delete qdisc with handle of zero.
```

then run:

```bash
sudo tc qdisc add dev lo root netem rate 5gbit delay 4ms
```

#### 4.1. 执行隐私推理

以5Gbps带宽为例, 打开两个终端 $0 和 $1，
run

```bash
$0: bash ablation_LinAtten_PI.sh 0 5G
$1: bash ablation_LinAtten_PI.sh 1 5G
```

#### 4.2. 恢复网络

run

```bash
sudo tc qdisc del dev lo root
```

#### 4.3. 结果

上述实验数据保存在文件夹`AE\eval_private_inference_result`下,文件名以'5G'开头，后面跟着‘MPCViT’, 'SHAFT', 'MPCFormer' or 'CryptPEFT'
