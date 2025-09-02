# CryptPEFT Evaluation

This repository provides scripts for evaluating CryptPEFT, including utility assessment and privacy inference efficiency.

## Environment Setup

**Create Conda Virtual Environment:**

To set up the required environment, you can use the provided `environment.yml` file. Run the following command to create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate adapter310

#install SHAFT
git clone https://github.com/andeskyl/SHAFT
cd SHAFT
pip install .
```

## Evaluating CryptPEFT Utility

To evaluate the utility of CryptPEFT, run the following command:

```
bash /eval/eval_CryptPEFT.sh
```

Configuration:

You may need to modify the configuration in eval_CryptPEFT.sh as needed. For detailed parameter explanations, refer to the get_args_parser() function inside eval_CryptPEFT.py.

The eval_CryptPEFT.sh script takes several configuration options that control the evaluation process. These can be adjusted to suit your experimental setup. For a complete list of parameters and their descriptions, check out the get_args_parser() function in eval_CryptPEFT.py.
## Evaluating CryptPEFT Privacy-preserving Inference Efficiency

We used SHAFT in our evaluation, ensuring the use of the CrypTen library provided by [SHAFT](https://github.com/andeskyl/SHAFT).
Requirements:

    For CPU Evaluation: Ensure sufficient memory is available.
    For GPU Evaluation: Ensure at least two GPUs are available for evaluation.
    Ensure that the LAN or WAN network environment simulation is completed before the evaluation.


Running the Benchmark:

To run the privacy inference efficiency benchmark, you will need to use two terminals.

Terminal 0: Run the following command to start the first evaluation:

    bash benchmark/secure_inference.sh 0

Terminal 1: Run the following command to start the second evaluation:

    bash benchmark/secure_inference.sh 1

Configuration:

You can modify the secure_inference.sh script as needed. For additional configurations, refer to the secure_inference.py file.

# License
This project is under the MIT license.