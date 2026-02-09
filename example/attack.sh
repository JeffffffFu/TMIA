# Baseline MIA（train=test 同一套参数）
# python main.py --attack_method TW_MIA --U_method continuous_update_finetune --dataset_name sst5 --net_name pythia70m --proportion_of_group_unlearn 0.01 --device cuda:0
# python main.py --attack_method UW_MIA --U_method continuous_update_finetune --dataset_name sst5 --net_name pythia70m --proportion_of_group_unlearn 0.01 --device cuda:0

python main.py  --attack_method TMIA  --window_size 5 --U_method continuous_update_finetune --dataset_name sst5 --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.01  --num_epochs 10 --device cuda:3
python main.py  --attack_method TMIA  --window_size 5  --U_method continuous_update_finetune --dataset_name news20 --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.01  --num_epochs 10 --device cuda:3
python main.py  --attack_method TMIA  --window_size 5 --U_method continuous_update_finetune --dataset_name mnli --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.005  --num_epochs 10 --device cuda:3


python main.py  --attack_method TMIA  --window_size 5 --U_method continuous_update_finetune --dataset_name sst5 --net_name gpt2  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.01  --num_epochs 10 --device cuda:3
python main.py  --attack_method TMIA  --window_size 5  --U_method continuous_update_finetune --dataset_name news20 --net_name gpt2  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.01  --num_epochs 10 --device cuda:3
python main.py  --attack_method TMIA  --window_size 5 --U_method continuous_update_finetune --dataset_name mnli --net_name gpt2  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.005  --num_epochs 10 --device cuda:3


python main.py --attack_method TMIA --window_size 5 --U_method continuous_update_finetune --dataset_name sst5 --net_name T5  --trials 3  --batch_size 32 --lr 0.0001 --proportion_of_group_unlearn 0.01  --num_epochs 10 --device cuda:1
python main.py --attack_method TMIA --window_size 5 --U_method continuous_update_finetune --dataset_name news20 --net_name T5  --trials 1  --batch_size 32 --lr 0.0001 --proportion_of_group_unlearn 0.01  --num_epochs 10 --device cuda:1
python main.py --attack_method TMIA --window_size 5 --U_method continuous_update_finetune --dataset_name mnli --net_name T5  --trials 1  --batch_size 32 --lr 0.0001 --proportion_of_group_unlearn 0.005  --num_epochs 10 --device cuda:1

#python main.py  --attack_method TMIA  --window_size 5 --U_method continuous_update_finetune --dataset_name sst5 --net_name roberta  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.01  --num_epochs 10 --device cuda:3
#python main.py  --attack_method TMIA  --window_size 5 --U_method continuous_update_finetune --dataset_name news20 --net_name roberta  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.01  --num_epochs 10 --device cuda:3
#python main.py  --attack_method TMIA  --window_size 5 --U_method continuous_update_finetune --dataset_name mnli --net_name roberta  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.005  --num_epochs 10 --device cuda:3

#image
#resnet18
python main.py --attack_method TMIA --window_size 5 --U_method continuous_update_finetune --dataset_name svhn --net_name resnet18  --trials 1   --lr 0.001 --proportion_of_group_unlearn 0.01  --num_epochs 50 --device cuda:1
python main.py --attack_method TMIA --window_size 5 --U_method continuous_update_finetune --dataset_name cinic10 --net_name resnet18  --trials 1   --lr 0.001 --proportion_of_group_unlearn 0.001  --num_epochs 50 --device cuda:1
python main.py --attack_method TMIA --window_size 5 --U_method continuous_update_finetune --dataset_name cifar10 --net_name resnet18  --trials 1  --lr 0.001 --proportion_of_group_unlearn 0.01  --num_epochs 50 --device cuda:1

#simplecnn
python main.py --attack_method TMIA --window_size 5 --U_method continuous_update_finetune --dataset_name svhn --net_name simple_cnn  --trials 1  --lr 0.001 --proportion_of_group_unlearn 0.01  --num_epochs 50 --device cuda:1
python main.py --attack_method TMIA --window_size 5 --U_method continuous_update_finetune --dataset_name cinic10 --net_name simple_cnn  --trials 1  --lr 0.001 --proportion_of_group_unlearn 0.001  --num_epochs 50 --device cuda:1
python main.py --attack_method TMIA --window_size 5 --U_method continuous_update_finetune --dataset_name cifar10 --net_name simple_cnn  --trials 1   --lr 0.001 --proportion_of_group_unlearn 0.01  --num_epochs 50 --device cuda:1

#mobilenet
python main.py --attack_method TMIA --window_size 5 --U_method continuous_update_finetune --dataset_name svhn --net_name mobilenet  --trials 1  --lr 0.001 --proportion_of_group_unlearn 0.01  --num_epochs 50 --device cuda:1
python main.py --attack_method TMIA --window_size 5 --U_method continuous_update_finetune --dataset_name cinic10 --net_name mobilenet  --trials 1  --lr 0.001 --proportion_of_group_unlearn 0.001  --num_epochs 50 --device cuda:1
python main.py --attack_method TMIA --window_size 5 --U_method continuous_update_finetune --dataset_name cifar10 --net_name mobilenet  --trials 1   --lr 0.001 --proportion_of_group_unlearn 0.01  --num_epochs 50 --device cuda:1

#LLM
python main.py --attack_method TMIA --window_size 5 --U_method continuous_update_finetune_LLM --dataset_name squad --net_name llama3b  --trials 1 --proportion_of_group_unlearn 0.01  --device cuda:0
python main.py --attack_method TMIA --window_size 5 --U_method continuous_update_finetune_LLM --dataset_name simpleqa --net_name llama3b  --trials 1 --proportion_of_group_unlearn 0.01  --device cuda:0

python main.py --attack_method TMIA --window_size 5 --U_method continuous_update_finetune_LLM --dataset_name squad --net_name llama8b  --trials 1 --proportion_of_group_unlearn 0.01  --device cuda:0
python main.py --attack_method TMIA --window_size 5 --U_method continuous_update_finetune_LLM --dataset_name simpleqa --net_name llama8b  --trials 1 --proportion_of_group_unlearn 0.01  --device cuda:0


#parameter-window size
python main.py  --attack_method TMIA  --window_size 3 --U_method continuous_update_finetune --dataset_name sst5 --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.01   --device cuda:3
python main.py  --attack_method TMIA  --window_size 3  --U_method continuous_update_finetune --dataset_name news20 --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.01  --device cuda:3
python main.py  --attack_method TMIA  --window_size 3  --U_method continuous_update_finetune --dataset_name mnli --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.005  --device cuda:3
python main.py  --attack_method TMIA  --window_size 3  --U_method continuous_update_finetune --dataset_name cifar10 --net_name simple_cnn  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.01  --device cuda:3
python main.py  --attack_method TMIA  --window_size 3  --U_method continuous_update_finetune --dataset_name svhn --net_name simple_cnn  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.01  --device cuda:3
python main.py  --attack_method TMIA  --window_size 3  --U_method continuous_update_finetune --dataset_name cinic10 --net_name simple_cnn  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.001  --device cuda:3

python main.py  --attack_method TMIA  --window_size 7 --U_method continuous_update_finetune --dataset_name sst5 --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.01   --device cuda:3
python main.py  --attack_method TMIA  --window_size 7  --U_method continuous_update_finetune --dataset_name news20 --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.01  --device cuda:3
python main.py  --attack_method TMIA  --window_size 7  --U_method continuous_update_finetune --dataset_name mnli --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.005  --device cuda:3
python main.py  --attack_method TMIA  --window_size 7  --U_method continuous_update_finetune --dataset_name cifar10 --net_name simple_cnn  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.01  --device cuda:3
python main.py  --attack_method TMIA  --window_size 7  --U_method continuous_update_finetune --dataset_name svhn --net_name simple_cnn  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.01  --device cuda:3
python main.py  --attack_method TMIA  --window_size 7  --U_method continuous_update_finetune --dataset_name cinic10 --net_name simple_cnn  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.001  --device cuda:3

python main.py  --attack_method TMIA  --window_size 9 --U_method continuous_update_finetune --dataset_name sst5 --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.01   --device cuda:3
python main.py  --attack_method TMIA  --window_size 9  --U_method continuous_update_finetune --dataset_name news20 --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.01  --device cuda:3
python main.py  --attack_method TMIA  --window_size 9  --U_method continuous_update_finetune --dataset_name mnli --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.005  --device cuda:3
python main.py  --attack_method TMIA  --window_size 9  --U_method continuous_update_finetune --dataset_name cifar10 --net_name simple_cnn  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.01  --device cuda:3
python main.py  --attack_method TMIA  --window_size 9  --U_method continuous_update_finetune --dataset_name svhn --net_name simple_cnn  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.01  --device cuda:3
python main.py  --attack_method TMIA  --window_size 9  --U_method continuous_update_finetune --dataset_name cinic10 --net_name simple_cnn  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.001  --device cuda:3

python main.py  --attack_method TMIA  --window_size 11 --U_method continuous_update_finetune --dataset_name sst5 --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.01   --device cuda:3
python main.py  --attack_method TMIA  --window_size 11  --U_method continuous_update_finetune --dataset_name news20 --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.01  --device cuda:3
python main.py  --attack_method TMIA  --window_size 11  --U_method continuous_update_finetune --dataset_name mnli --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.005  --device cuda:3
python main.py  --attack_method TMIA  --window_size 11  --U_method continuous_update_finetune --dataset_name cifar10 --net_name simple_cnn  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.01  --device cuda:3
python main.py  --attack_method TMIA  --window_size 11  --U_method continuous_update_finetune --dataset_name svhn --net_name simple_cnn  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.01  --device cuda:3
python main.py  --attack_method TMIA  --window_size 11  --U_method continuous_update_finetune --dataset_name cinic10 --net_name simple_cnn  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.001  --device cuda:3

#parameter-update size
python main.py  --attack_method TMIA  --window_size 5 --U_method continuous_update_finetune --dataset_name sst5 --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.001   --device cuda:3
python main.py  --attack_method TMIA  --window_size 5  --U_method continuous_update_finetune --dataset_name news20 --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.001   --device cuda:3
python main.py  --attack_method TMIA  --window_size 5  --U_method continuous_update_finetune --dataset_name mnli --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.001   --device cuda:3
python main.py  --attack_method TMIA  --window_size 5  --U_method continuous_update_finetune --dataset_name cifar10 --net_name simple_cnn  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.001   --device cuda:3
python main.py  --attack_method TMIA  --window_size 5  --U_method continuous_update_finetune --dataset_name svhn --net_name simple_cnn  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.001   --device cuda:3
python main.py  --attack_method TMIA  --window_size 5  --U_method continuous_update_finetune --dataset_name cinic10 --net_name simple_cnn  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.001   --device cuda:3

python main.py  --attack_method TMIA  --window_size 5  --U_method continuous_update_finetune --dataset_name sst5 --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.005   --device cuda:3
python main.py  --attack_method TMIA  --window_size 5  --U_method continuous_update_finetune --dataset_name news20 --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.005   --device cuda:3
python main.py  --attack_method TMIA  --window_size 5  --U_method continuous_update_finetune --dataset_name mnli --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.005   --device cuda:3
python main.py  --attack_method TMIA  --window_size 5  --U_method continuous_update_finetune --dataset_name cifar10 --net_name simple_cnn  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.005   --device cuda:3
python main.py  --attack_method TMIA  --window_size 5  --U_method continuous_update_finetune --dataset_name svhn --net_name simple_cnn  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.005   --device cuda:3
python main.py  --attack_method TMIA  --window_size 5  --U_method continuous_update_finetune --dataset_name cinic10 --net_name simple_cnn  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.005   --device cuda:3


python main.py  --attack_method TMIA  --window_size 5 --U_method continuous_update_finetune --dataset_name sst5 --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.01   --device cuda:3
python main.py  --attack_method TMIA  --window_size 5  --U_method continuous_update_finetune --dataset_name news20 --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.01   --device cuda:3
python main.py  --attack_method TMIA  --window_size 5  --U_method continuous_update_finetune --dataset_name mnli --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.01   --device cuda:3
python main.py  --attack_method TMIA  --window_size 5  --U_method continuous_update_finetune --dataset_name cifar10 --net_name simple_cnn  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.01   --device cuda:3
python main.py  --attack_method TMIA  --window_size 5  --U_method continuous_update_finetune --dataset_name svhn --net_name simple_cnn  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.01   --device cuda:3
python main.py  --attack_method TMIA  --window_size 5  --U_method continuous_update_finetune --dataset_name cinic10 --net_name simple_cnn  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.01   --device cuda:3


python main.py  --attack_method TMIA  --window_size 5 --U_method continuous_update_finetune --dataset_name sst5 --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.05   --device cuda:3
python main.py  --attack_method TMIA  --window_size 5  --U_method continuous_update_finetune --dataset_name news20 --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.05   --device cuda:3
python main.py  --attack_method TMIA  --window_size 5  --U_method continuous_update_finetune --dataset_name mnli --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.05   --device cuda:3
python main.py  --attack_method TMIA  --window_size 5  --U_method continuous_update_finetune --dataset_name cifar10 --net_name simple_cnn  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.05   --device cuda:3
python main.py  --attack_method TMIA  --window_size 5  --U_method continuous_update_finetune --dataset_name svhn --net_name simple_cnn  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.05   --device cuda:3
python main.py  --attack_method TMIA  --window_size 5  --U_method continuous_update_finetune --dataset_name cinic10 --net_name simple_cnn  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.05   --device cuda:3

python main.py  --attack_method TMIA  --window_size 5 --U_method continuous_update_finetune --dataset_name sst5 --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.1   --device cuda:3
python main.py  --attack_method TMIA  --window_size 5  --U_method continuous_update_finetune --dataset_name news20 --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.1   --device cuda:3
python main.py  --attack_method TMIA  --window_size 5  --U_method continuous_update_finetune --dataset_name mnli --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.1   --device cuda:3
python main.py  --attack_method TMIA  --window_size 5  --U_method continuous_update_finetune --dataset_name cifar10 --net_name simple_cnn  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.1   --device cuda:3
python main.py  --attack_method TMIA  --window_size 5  --U_method continuous_update_finetune --dataset_name svhn --net_name simple_cnn  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.1   --device cuda:3
python main.py  --attack_method TMIA  --window_size 5  --U_method continuous_update_finetune --dataset_name cinic10 --net_name simple_cnn  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.1   --device cuda:3


# multiply
python main.py  --attack_method TMIA  --window_size 5 --U_method continuous_update_finetune_multiple_update --dataset_name sst5 --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.01  --device cuda:3
python main.py  --attack_method TMIA  --window_size 5  --U_method continuous_update_finetune_multiple_update --dataset_name news20 --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.01 --device cuda:3
python main.py  --attack_method TMIA  --window_size 5  --U_method continuous_update_finetune_multiple_update --dataset_name mnli --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.005 --device cuda:3

python main.py  --attack_method TMIA  --window_size 5  --U_method continuous_update_finetune_multiple_update --dataset_name cifar10 --net_name simple_cnn  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.01 --device cuda:3
python main.py  --attack_method TMIA  --window_size 5  --U_method continuous_update_finetune_multiple_update --dataset_name svhn --net_name simple_cnn  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.01 --device cuda:3
python main.py  --attack_method TMIA  --window_size 5  --U_method continuous_update_finetune_multiple_update --dataset_name cinic10 --net_name simple_cnn  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.001 --device cuda:3

#differential unlearning
python main.py  --attack_method TMIA  --window_size 5 --U_method continuous_update_finetune_GA --dataset_name sst5 --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.01   --device cuda:3
python main.py  --attack_method TMIA  --window_size 5 --U_method continuous_update_finetune_NPO --dataset_name sst5 --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.01  --device cuda:3

python main.py  --attack_method TMIA  --window_size 5 --U_method continuous_update_finetune_GA --dataset_name cifar10 --net_name simple_cnn  --trials 1  --proportion_of_group_unlearn 0.01  --device cuda:3
python main.py  --attack_method TMIA  --window_size 5 --U_method continuous_update_finetune_scrub --dataset_name cifar10 --net_name simple_cnn  --trials 1  --proportion_of_group_unlearn 0.01  --device cuda:3


python main.py  --attack_method TMIA  --window_size 9 --U_method continuous_update_finetune_GA --dataset_name sst5 --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.01   --device cuda:3
python main.py  --attack_method TMIA  --window_size 9 --U_method continuous_update_finetune_NPO --dataset_name sst5 --net_name pythia70m  --trials 1  --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.01  --device cuda:3

python main.py  --attack_method TMIA  --window_size 9 --U_method continuous_update_finetune_GA --dataset_name cifar10 --net_name simple_cnn  --trials 1  --proportion_of_group_unlearn 0.01  --device cuda:3
python main.py  --attack_method TMIA  --window_size 9 --U_method continuous_update_finetune_scrub --dataset_name cifar10 --net_name simple_cnn  --trials 1  --proportion_of_group_unlearn 0.01  --device cuda:3

#Transfer
#for membership timestamp metric-text and image
python main.py  --attack_method Transfer_dataset --window_size 5 --device cuda:3
python main.py  --attack_method Transfer_model --window_size 5 --device cuda:3
python main.py  --attack_method Transfer_unlearning  --window_size 5 --device cuda:3


#for membership interval metric-text
python main.py  --attack_method Transfer_dataset --window_size 9 --device cuda:3
python main.py  --attack_method Transfer_model --window_size 9 --device cuda:3
python main.py  --attack_method Transfer_unlearning  --window_size 9 --device cuda:3


#for membership interval metric-image
python main.py  --attack_method Transfer_dataset --window_size 19 --device cuda:3
python main.py  --attack_method Transfer_model --window_size 19 --device cuda:3
python main.py  --attack_method Transfer_unlearning  --window_size 19 --device cuda:3


# defense-dropout
python main.py  --attack_method TMIA  --window_size 5 --U_method continuous_update_finetune --dataset_name sst5 --net_name pythia70m_dropout  --trials 1 --proportion_of_group_unlearn 0.01  --num_epochs 10 --device cuda:3
python main.py  --attack_method TMIA --window_size 5 --U_method continuous_update_finetune --dataset_name cifar10 --net_name simple_cnn_dropout  --trials 1  --proportion_of_group_unlearn 0.01  --num_epochs 50 --device cuda:1

# defense-label only, need to set the confidence_mode='label_only' of function load_data_and_compute_confidence() of get_training_data.py
python main.py  --attack_method TMIA  --window_size 5 --U_method continuous_update_finetune --dataset_name sst5 --net_name pythia70m  --trials 1 --proportion_of_group_unlearn 0.01  --num_epochs 10 --device cuda:3
python main.py  --attack_method TMIA --window_size 5 --U_method continuous_update_finetune --dataset_name cifar10 --net_name simple_cnn  --trials 1  --proportion_of_group_unlearn 0.01  --num_epochs 50 --device cuda:1

#defense-dp
python main.py --attack_method TMIA --window_size 5 --U_method continuous_update_finetune_dp --dataset_name sst5 --net_name pythia70m  --trials 1  --batch_size 128 --lr 0.0001 --proportion_of_group_unlearn 0.01  --num_epochs 20 --device cuda:3
python main.py --attack_method TMIA --window_size 5 --U_method continuous_update_finetune_dp --dataset_name cifar10 --net_name simple_cnn  --trials 1  --batch_size 512 --lr 0.001 --proportion_of_group_unlearn 0.01  --num_epochs 50 --device cuda:0



#interval level
# defense-dropout
python main.py  --attack_method TMIA  --window_size 9 --U_method continuous_update_finetune --dataset_name sst5 --net_name pythia70m_dropout  --trials 1 --proportion_of_group_unlearn 0.01  --num_epochs 10 --device cuda:3
python main.py  --attack_method TMIA --window_size 19 --U_method continuous_update_finetune --dataset_name cifar10 --net_name simple_cnn_dropout  --trials 1  --proportion_of_group_unlearn 0.01  --num_epochs 50 --device cuda:1

# defense-label only, need to set the confidence_mode='label_only' of function load_data_and_compute_confidence() of get_training_data.py
python main.py  --attack_method TMIA  --window_size 9 --U_method continuous_update_finetune --dataset_name sst5 --net_name pythia70m  --trials 1 --proportion_of_group_unlearn 0.01  --num_epochs 10 --device cuda:3
python main.py  --attack_method TMIA --window_size 19 --U_method continuous_update_finetune --dataset_name cifar10 --net_name simple_cnn  --trials 1  --proportion_of_group_unlearn 0.01  --num_epochs 50 --device cuda:1

#defense-dp
python main.py --attack_method TMIA --window_size 9 --U_method continuous_update_finetune_dp --dataset_name sst5 --net_name pythia70m  --trials 1  --batch_size 128 --lr 0.0001 --proportion_of_group_unlearn 0.01  --num_epochs 20 --device cuda:3
python main.py --attack_method TMIA --window_size 19 --U_method continuous_update_finetune_dp --dataset_name cifar10 --net_name simple_cnn  --trials 1  --batch_size 512 --lr 0.001 --proportion_of_group_unlearn 0.01  --num_epochs 50 --device cuda:0

