# test dataset - finetune
#net_name pythia70m
python main.py --U_method continuous_update_finetune --dataset_name sst5 --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.01  --num_epochs 10 --device cuda:1
python main.py --U_method continuous_update_finetune --dataset_name news20 --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.01  --num_epochs 10 --device cuda:1
python main.py --U_method continuous_update_finetune --dataset_name mnli --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.005  --num_epochs 10 --device cuda:1

#net_name roberta
python main.py --U_method continuous_update_finetune --dataset_name sst5 --net_name roberta  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.01  --num_epochs 10 --device cuda:1
python main.py --U_method continuous_update_finetune --dataset_name news20 --net_name roberta  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.01  --num_epochs 10 --device cuda:1
python main.py --U_method continuous_update_finetune --dataset_name mnli --net_name roberta  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.005  --num_epochs 10 --device cuda:1

#net_name gpt2
python main.py --U_method continuous_update_finetune --dataset_name sst5 --net_name gpt2  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.01  --num_epochs 10 --device cuda:1
python main.py --U_method continuous_update_finetune --dataset_name news20 --net_name gpt2  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.01  --num_epochs 10 --device cuda:1
python main.py --U_method continuous_update_finetune --dataset_name mnli --net_name gpt2  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.005  --num_epochs 10 --device cuda:1

#net_name T5
python main.py --U_method continuous_update_finetune --dataset_name sst5 --net_name T5  --trials 1  --batch_size 32 --lr 0.0001 --proportion_of_group_unlearn 0.01  --num_epochs 10 --device cuda:1
python main.py --U_method continuous_update_finetune --dataset_name news20 --net_name T5  --trials 1  --batch_size 32 --lr 0.0001 --proportion_of_group_unlearn 0.01  --num_epochs 10 --device cuda:1
python main.py --U_method continuous_update_finetune --dataset_name mnli --net_name T5  --trials 1  --batch_size 32 --lr 0.0001 --proportion_of_group_unlearn 0.005  --num_epochs 10 --device cuda:1

# image dataset - finetune
#resnet18
python main.py --U_method continuous_update_finetune --dataset_name cifar100 --net_name resnet18  --trials 1  --batch_size 256 --lr 0.001 --proportion_of_group_unlearn 0.01  --num_epochs 50 --device cuda:1
python main.py --U_method continuous_update_finetune --dataset_name cinic10 --net_name resnet18  --trials 1  --batch_size 256 --lr 0.001 --proportion_of_group_unlearn 0.001  --num_epochs 50 --device cuda:1
python main.py --U_method continuous_update_finetune --dataset_name cifar10 --net_name resnet18  --trials 1  --batch_size 256 --lr 0.001 --proportion_of_group_unlearn 0.01  --num_epochs 50 --device cuda:1
python main.py --U_method continuous_update_finetune --dataset_name svhn --net_name resnet18  --trials 1  --batch_size 256 --lr 0.001 --proportion_of_group_unlearn 0.01  --num_epochs 50 --device cuda:1

#simplecnn
python main.py --U_method continuous_update_finetune --dataset_name cifar100 --net_name simple_cnn  --trials 1  --batch_size 256 --lr 0.001 --proportion_of_group_unlearn 0.01  --num_epochs 50 --device cuda:1
python main.py --U_method continuous_update_finetune --dataset_name cinic10 --net_name simple_cnn  --trials 1  --batch_size 256 --lr 0.001 --proportion_of_group_unlearn 0.001  --num_epochs 50 --device cuda:1
python main.py --U_method continuous_update_finetune --dataset_name cifar10 --net_name simple_cnn  --trials 1  --batch_size 256 --lr 0.001 --proportion_of_group_unlearn 0.01  --num_epochs 50 --device cuda:1
python main.py --U_method continuous_update_finetune --dataset_name svhn --net_name simple_cnn  --trials 1  --batch_size 256 --lr 0.001 --proportion_of_group_unlearn 0.01  --num_epochs 50 --device cuda:1

#mobilenet
python main.py --U_method continuous_update_finetune --dataset_name cifar100 --net_name mobilenet  --trials 1  --batch_size 256 --lr 0.001 --proportion_of_group_unlearn 0.01  --num_epochs 50 --device cuda:2
python main.py --U_method continuous_update_finetune --dataset_name cinic10 --net_name mobilenet  --trials 1  --batch_size 256 --lr 0.001 --proportion_of_group_unlearn 0.001  --num_epochs 50 --device cuda:2
python main.py --U_method continuous_update_finetune --dataset_name cifar10 --net_name mobilenet  --trials 1  --batch_size 256 --lr 0.001 --proportion_of_group_unlearn 0.01  --num_epochs 50 --device cuda:2
python main.py --U_method continuous_update_finetune --dataset_name svhn --net_name mobilenet  --trials 1  --batch_size 256 --lr 0.001 --proportion_of_group_unlearn 0.01  --num_epochs 50 --device cuda:2

# multiply
python main.py --U_method continuous_update_finetune_multiple_update --dataset_name sst5 --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.01  --num_epochs 10 --device cuda:1
python main.py --U_method continuous_update_finetune_multiple_update --dataset_name news20 --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.01  --num_epochs 10 --device cuda:1
python main.py --U_method continuous_update_finetune_multiple_update --dataset_name mnli --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.005  --num_epochs 10 --device cuda:2

python main.py --U_method continuous_update_finetune_multiple_update --dataset_name cinic10 --net_name simple_cnn  --trials 1  --batch_size 256 --lr 0.001 --proportion_of_group_unlearn 0.005  --num_epochs 50 --device cuda:1
python main.py --U_method continuous_update_finetune_multiple_update --dataset_name cifar10 --net_name simple_cnn  --trials 1  --batch_size 256 --lr 0.001 --proportion_of_group_unlearn 0.01  --num_epochs 50 --device cuda:1
python main.py --U_method continuous_update_finetune_multiple_update --dataset_name svhn --net_name simple_cnn  --trials 1  --batch_size 256 --lr 0.001 --proportion_of_group_unlearn 0.01  --num_epochs 50 --device cuda:1


# different unlearning method
python main.py --U_method continuous_update_finetune_GA --dataset_name sst5 --net_name pythia70m  --trials 3  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.01  --num_epochs 10 --device cuda:1
python main.py --U_method continuous_update_finetune_NPO --dataset_name sst5 --net_name pythia70m  --trials 3  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.01  --num_epochs 10 --device cuda:1

python main.py --U_method continuous_update_finetune_GA --dataset_name cifar10 --net_name simple_cnn  --trials 1  --batch_size 256 --lr 0.001 --proportion_of_group_unlearn 0.01  --num_epochs 50 --device cuda:1
python main.py --U_method continuous_update_finetune_scrub --dataset_name cifar10 --net_name simple_cnn  --trials 1  --batch_size 256 --lr 0.001 --proportion_of_group_unlearn 0.01  --num_epochs 50 --device cuda:2


#LLM
python main.py --U_method continuous_update_finetune_LLM --dataset_name squad --net_name llama3b  --trials 1 --proportion_of_group_unlearn 0.01  --device cuda:0
python main.py --U_method continuous_update_finetune_LLM --dataset_name simpleqa --net_name llama3b  --trials 1 --proportion_of_group_unlearn 0.01  --device cuda:0

python main.py --U_method continuous_update_finetune_LLM --dataset_name squad --net_name llama8b  --trials 1 --proportion_of_group_unlearn 0.01  --device cuda:0
python main.py --U_method continuous_update_finetune_LLM --dataset_name simpleqa --net_name llama8b  --trials 1 --proportion_of_group_unlearn 0.01  --device cuda:0

#denfense
#drop out
python main.py --U_method continuous_update_finetune --dataset_name sst5 --net_name pythia70m_dropout  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.01  --num_epochs 10 --device cuda:1
python main.py --U_method continuous_update_finetune --dataset_name cifar10 --net_name simple_cnn_dropout  --trials 1  --batch_size 256 --lr 0.001 --proportion_of_group_unlearn 0.01  --num_epochs 50 --device cuda:1

#dp
python main.py --eps 30 --U_method continuous_update_finetune_dp --dataset_name sst5 --net_name pythia70m  --trials 1  --batch_size 128 --lr 0.0001 --proportion_of_group_unlearn 0.01  --num_epochs 10 --device cuda:1
python main.py --eps 10 --U_method continuous_update_finetune_dp --dataset_name sst5 --net_name pythia70m  --trials 1  --batch_size 128 --lr 0.0001 --proportion_of_group_unlearn 0.01  --num_epochs 10 --device cuda:2

python main.py --eps 30 --U_method continuous_update_finetune_dp --dataset_name cifar10 --net_name simple_cnn  --trials 1   --batch_size 512 --lr 0.001 --proportion_of_group_unlearn 0.01  --num_epochs 50 --device cuda:3
python main.py --eps 10 --U_method continuous_update_finetune_dp --dataset_name cifar10 --net_name simple_cnn  --trials 1  --batch_size 512 --lr 0.001 --proportion_of_group_unlearn 0.01  --num_epochs 50 --device cuda:3


#difference  update set
python main.py --U_method continuous_update_finetune --dataset_name sst5 --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.05  --num_epochs 10 --device cuda:1
python main.py --U_method continuous_update_finetune --dataset_name sst5 --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.1  --num_epochs 10 --device cuda:1
python main.py --U_method continuous_update_finetune --dataset_name sst5 --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.005  --num_epochs 10 --device cuda:1
python main.py --U_method continuous_update_finetune --dataset_name sst5 --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.001  --num_epochs 10 --device cuda:1


python main.py --U_method continuous_update_finetune --dataset_name news20 --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.05  --num_epochs 10 --device cuda:1
python main.py --U_method continuous_update_finetune --dataset_name news20 --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.1  --num_epochs 10 --device cuda:1
python main.py --U_method continuous_update_finetune --dataset_name news20 --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.005  --num_epochs 10 --device cuda:1
python main.py --U_method continuous_update_finetune --dataset_name news20 --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.001  --num_epochs 10 --device cuda:1


python main.py --U_method continuous_update_finetune --dataset_name mnli --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.05  --num_epochs 10 --device cuda:1
python main.py --U_method continuous_update_finetune --dataset_name mnli --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.1  --num_epochs 10 --device cuda:1
python main.py --U_method continuous_update_finetune --dataset_name mnli --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.01  --num_epochs 10 --device cuda:1
python main.py --U_method continuous_update_finetune --dataset_name mnli --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.001  --num_epochs 10 --device cuda:1


python main.py --U_method continuous_update_finetune --dataset_name cifar10 --net_name simple_cnn  --trials 1  --batch_size 256 --lr 0.001 --proportion_of_group_unlearn 0.05  --num_epochs 50 --device cuda:1
python main.py --U_method continuous_update_finetune --dataset_name cifar10 --net_name simple_cnn  --trials 1  --batch_size 256 --lr 0.001 --proportion_of_group_unlearn 0.1  --num_epochs 50 --device cuda:1
python main.py --U_method continuous_update_finetune --dataset_name cifar10 --net_name simple_cnn  --trials 1  --batch_size 256 --lr 0.001 --proportion_of_group_unlearn 0.05  --num_epochs 50 --device cuda:1
python main.py --U_method continuous_update_finetune --dataset_name cifar10 --net_name simple_cnn  --trials 1  --batch_size 256 --lr 0.001 --proportion_of_group_unlearn 0.001  --num_epochs 50 --device cuda:1

python main.py --U_method continuous_update_finetune --dataset_name cinic10 --net_name simple_cnn  --trials 1  --batch_size 256 --lr 0.001 --proportion_of_group_unlearn 0.05  --num_epochs 50 --device cuda:1
python main.py --U_method continuous_update_finetune --dataset_name cinic10 --net_name simple_cnn  --trials 1  --batch_size 256 --lr 0.001 --proportion_of_group_unlearn 0.1  --num_epochs 50 --device cuda:1
python main.py --U_method continuous_update_finetune --dataset_name cinic10 --net_name simple_cnn  --trials 1  --batch_size 256 --lr 0.001 --proportion_of_group_unlearn 0.005  --num_epochs 50 --device cuda:1
python main.py --U_method continuous_update_finetune --dataset_name cinic10 --net_name simple_cnn  --trials 1  --batch_size 256 --lr 0.001 --proportion_of_group_unlearn 0.01  --num_epochs 50 --device cuda:1

python main.py --U_method continuous_update_finetune --dataset_name svhn --net_name simple_cnn  --trials 1  --batch_size 256 --lr 0.001 --proportion_of_group_unlearn 0.05  --num_epochs 50 --device cuda:1
python main.py --U_method continuous_update_finetune --dataset_name svhn --net_name simple_cnn  --trials 1  --batch_size 256 --lr 0.001 --proportion_of_group_unlearn 0.1  --num_epochs 50 --device cuda:1
python main.py --U_method continuous_update_finetune --dataset_name svhn --net_name simple_cnn  --trials 1  --batch_size 256 --lr 0.001 --proportion_of_group_unlearn 0.005  --num_epochs 50 --device cuda:1
python main.py --U_method continuous_update_finetune --dataset_name svhn --net_name simple_cnn  --trials 1  --batch_size 256 --lr 0.001 --proportion_of_group_unlearn 0.01  --num_epochs 50 --device cuda:1

# difference interval and insert time
python main.py --U_method continuous_update_finetune_multiple_update --dataset_name sst5 --net_name pythia70m  --trials 3  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.01  --num_epochs 10 --device cuda:0
python main.py --U_method continuous_update_finetune_multiple_update --dataset_name news20 --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.01  --num_epochs 10 --device cuda:0
python main.py --U_method continuous_update_finetune_multiple_update --dataset_name cifar10 --net_name simple_cnn  --trials 1  --batch_size 256 --lr 0.001 --proportion_of_group_unlearn 0.01  --num_epochs 50 --device cuda:0

#CNN
