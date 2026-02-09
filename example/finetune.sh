#python main.py --U_method continuous_update_finetune --dataset_name sst5 --net_name pythia70m --batch_size 32 --lr 0.00001  --trials 3 --proportion_of_group_unlearn 0.01 --num_epochs 10  --device cuda:0
#python main.py --U_method continuous_update_finetune --dataset_name news20 --net_name pythia70m --batch_size 32 --lr 0.00001  --trials 3 --proportion_of_group_unlearn 0.01 --num_epochs 10  --device cuda:0
#python main.py --U_method continuous_update_finetune --dataset_name mnli --net_name pythia70m --batch_size 32 --lr 0.00001  --trials 3 --proportion_of_group_unlearn 0.005 --num_epochs 10  --device cuda:0

#python main.py --U_method continuous_update_finetune --dataset_name sst5 --net_name roberta --batch_size 32 --lr 0.00001  --trials 3 --proportion_of_group_unlearn 0.01 --num_epochs 10  --device cuda:0
#python main.py --U_method continuous_update_finetune --dataset_name news20 --net_name roberta --batch_size 32 --lr 0.00001  --trials 3 --proportion_of_group_unlearn 0.01 --num_epochs 10  --device cuda:0
#python main.py --U_method continuous_update_finetune --dataset_name mnli --net_name roberta --batch_size 32 --lr 0.00001  --trials 3 --proportion_of_group_unlearn 0.005 --num_epochs 10  --device cuda:0


#python main.py --U_method continuous_update_finetune --dataset_name sst5 --net_name opt13b --batch_size 32 --lr 0.00001  --trials 3 --proportion_of_group_unlearn 0.01 --num_epochs 10  --device cuda:0
#python main.py --U_method continuous_update_finetune --dataset_name news20 --net_name opt13b --batch_size 32 --lr 0.00001  --trials 3 --proportion_of_group_unlearn 0.01 --num_epochs 10  --device cuda:0
#python main.py --U_method continuous_update_finetune --dataset_name mnli --net_name opt13b --batch_size 32 --lr 0.00001  --trials 3 --proportion_of_group_unlearn 0.005 --num_epochs 10  --device cuda:0


#python main.py --U_method continuous_update_finetune --dataset_name sst5 --net_name gpt2 --batch_size 32 --lr 0.00001  --trials 3 --proportion_of_group_unlearn 0.01 --num_epochs 10  --device cuda:1
#python main.py --U_method continuous_update_finetune --dataset_name news20 --net_name gpt2 --batch_size 32 --lr 0.00001  --trials 3 --proportion_of_group_unlearn 0.01 --num_epochs 10  --device cuda:0
#python main.py --U_method continuous_update_finetune --dataset_name mnli --net_name gpt2 --batch_size 32 --lr 0.00001  --trials 3 --proportion_of_group_unlearn 0.005 --num_epochs 10  --device cuda:0
python main.py --U_method continuous_update_finetune --dataset_name mnli --net_name gpt2 --batch_size 32 --lr 0.00001  --trials 1 --proportion_of_group_unlearn 0.005 --num_epochs 10  --device cuda:3


# mutiple insert and remove
#python main.py --U_method continuous_update_finetune_mutiple_update --dataset_name sst5 --net_name pythia70m --batch_size 32 --lr 0.00001  --trials 1 --proportion_of_group_unlearn 0.01 --num_epochs 10  --device cuda:0
#python main.py --U_method continuous_update_finetune_mutiple_update --dataset_name news20 --net_name pythia70m --batch_size 32 --lr 0.00001  --trials 1 --proportion_of_group_unlearn 0.01 --num_epochs 10  --device cuda:0
