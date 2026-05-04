# TMIA


This repository is the official implementation of the paper "Timing is Everything: Inferring Temporal Membership in Continually Updated Deep Learning Models".

## Installation

You need to make sure there are GPU hardware and the CUDA Driver Version: 12.2， CUDA Toolkit Version: 11.6

You can install all requirements with:

```bash
conda create --name TMIA python=3.10
conda activate TMIA
pip install -r requirements.txt
```

## Code Structure of PGR
* **attack**: provide the implementation of our method and baselines. 
* **data**: provide data downloading scripts and raw data loader to process original data.
* **training**: provide the implementation of continuous updating.
* **model**: provide the DNN and LM models.
* **save**: The output of the continuously updated model at each timestamp.
* **utils**: privacy budget of DP calculation and DPSGD model training tools.


## Demo Experiments
You can run our method and baseline directly with the following default parameters (For example in Table 2, 3, 4 and 5):

To obtain attack results on other datasets and network architectures,
you only need to change the `--dataset_name` and `--net_name` arguments.

```bash
python main.py --attack_method TMIA  --U_method continuous_update_finetune --dataset_name sst5 --net_name pythia70m  --lr 0.00001 --proportion_of_group_unlearn 0.01  --device cuda:0
python main.py --attack_method TW_MIA --U_method continuous_update_finetune --dataset_name sst5 --net_name pythia70m --proportion_of_group_unlearn 0.01 --device cuda:0
python main.py --attack_method UW_MIA --U_method continuous_update_finetune --dataset_name sst5 --net_name pythia70m --proportion_of_group_unlearn 0.01 --device cuda:0
```
For each of the above commands, the console will report the evaluation results of timestamp-level and interval-level attacks in the following format:

```bash
Timestamp-level evaluation (T-Pre, T-Rec, T-F1)
======================================================================
T-Pre: {xx}
T-Rec: {xx}
T-F1: {xx}
======================================================================
Interval-level evaluation (I-Pre, I-Rec, I-F1)
======================================================================
I-Pre: {xx}
I-Rec: {xx}
I-F1: {xx}
======================================================================
```

Due to the substantial computational cost of training continuously
updated models, we provide pre-generated continuous update outputs
for two settings:
(1) a Pythia-70M model on the SST-5 dataset, and
(2) a SimpleCNN model on the CIFAR-10 dataset as above.

These outputs can be used directly to reproduce the reported attack
results without re-running the full continuous update training process.
For other model architectures and datasets, users can generate the
corresponding continuous update outputs by running the commands
listed below.

```bash
for LM model training:
python main.py --U_method continuous_update_finetune --dataset_name sst5 --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.01  --num_epochs 10 --device cuda:1

for CNN model training:
python main.py --U_method continuous_update_finetune --dataset_name cifar10 --net_name resnet18  --trials 1  --batch_size 256 --lr 0.001 --proportion_of_group_unlearn 0.01  --num_epochs 50 --device cuda:1
```

To obtain output on other datasets and network architectures,
you only need to change the `--dataset_name` and `--net_name` arguments.