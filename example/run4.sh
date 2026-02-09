#python main.py --attack_method LSTM --window_size 9 --U_method continuous_update_finetune --dataset_name svhn --net_name resnet18  --trials 1   --lr 0.001 --proportion_of_group_unlearn 0.01  --num_epochs 50 --device cuda:1
#python main.py --attack_method LSTM --window_size 9 --U_method continuous_update_finetune --dataset_name cinic10 --net_name resnet18  --trials 1   --lr 0.001 --proportion_of_group_unlearn 0.001  --num_epochs 50 --device cuda:1
#python main.py --attack_method LSTM --window_size 9 --U_method continuous_update_finetune --dataset_name cifar10 --net_name resnet18  --trials 1  --lr 0.001 --proportion_of_group_unlearn 0.01  --num_epochs 50 --device cuda:1
#
##simplecnn
#python main.py --attack_method LSTM --window_size 9 --U_method continuous_update_finetune --dataset_name svhn --net_name simple_cnn  --trials 1  --lr 0.001 --proportion_of_group_unlearn 0.01  --num_epochs 50 --device cuda:1
#python main.py --attack_method LSTM --window_size 9 --U_method continuous_update_finetune --dataset_name cinic10 --net_name simple_cnn  --trials 1  --lr 0.001 --proportion_of_group_unlearn 0.001  --num_epochs 50 --device cuda:1
#python main.py --attack_method LSTM --window_size 9 --U_method continuous_update_finetune --dataset_name cifar10 --net_name simple_cnn  --trials 1   --lr 0.001 --proportion_of_group_unlearn 0.01  --num_epochs 50 --device cuda:1
#
##mobilenet
#python main.py --attack_method LSTM --window_size 9 --U_method continuous_update_finetune --dataset_name svhn --net_name mobilenet  --trials 1  --lr 0.001 --proportion_of_group_unlearn 0.01  --num_epochs 50 --device cuda:1
#python main.py --attack_method LSTM --window_size 9 --U_method continuous_update_finetune --dataset_name cinic10 --net_name mobilenet  --trials 1  --lr 0.001 --proportion_of_group_unlearn 0.001  --num_epochs 50 --device cuda:1
#python main.py --attack_method LSTM --window_size 9 --U_method continuous_update_finetune --dataset_name cifar10 --net_name mobilenet  --trials 1   --lr 0.001 --proportion_of_group_unlearn 0.01  --num_epochs 50 --device cuda:1
python main.py --attack_method LSTM --window_size 9 --U_method continuous_update_finetune --dataset_name cifar10 --net_name resnet18  --trials 1  --lr 0.001 --proportion_of_group_unlearn 0.01  --num_epochs 50 --device cuda:1
