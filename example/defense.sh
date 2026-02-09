#label only don't need train

#dropout , keep shadow
#python main.py --pre_train both --U_method retrain --dataset_name cifar10 --net_name simple_cnn_dropout --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:0
#python main.py --pre_train both --U_method retrain --dataset_name cinic10 --net_name simple_cnn_dropout --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3  --observations 5  --device cuda:0
#python main.py --pre_train both --U_method retrain --dataset_name sst5 --net_name pythia70m_dropout --num_epochs 10 --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.02 --trials 3 --observations 10  --device cuda:2


#dp, keep shadow eps=2.0,5.0
#python main.py --pre_train both --U_method retrain_dp --sigma 1.5 --dataset_name cifar10 --net_name simple_cnn --num_epochs 50 --batch_size 1024 --trials 1 --observations 5  --device cuda:0
#python main.py --pre_train both --U_method retrain_dp --sigma 2.0 --dataset_name cifar10 --net_name simple_cnn --num_epochs 50 --batch_size 1024 --trials 1 --observations 5  --device cuda:0

#python main.py --pre_train both --U_method retrain_dp --sigma 1.0 --dataset_name cinic10 --net_name simple_cnn --num_epochs 50 --batch_size 1024 --trials 1 --observations 5  --device cuda:0
#python main.py --pre_train both --U_method retrain_dp --sigma 2.0 --dataset_name cinic10 --net_name simple_cnn --num_epochs 50 --batch_size 1024 --trials 1 --observations 5  --device cuda:0


#python main.py --pre_train both --U_method retrain_dp --sigma 0.91 --dataset_name sst5 --net_name pythia70m --num_epochs 10 --batch_size 128 --lr 0.00001 --proportion_of_group_unlearn 0.02 --trials 1 --device cuda:0
#python main.py --pre_train both --U_method retrain_dp --sigma 1.49 --dataset_name sst5 --net_name pythia70m --num_epochs 10 --batch_size 128 --lr 0.00001 --proportion_of_group_unlearn 0.02 --trials 1  --device cuda:0

#python main.py --attack_method TC_MIA --U_method retrain_dp --sigma 0.91 --dataset_name sst5 --net_name pythia70m --num_epochs 10 --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.02 --trials 1 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain_dp --sigma 1.49 --dataset_name sst5 --net_name pythia70m --num_epochs 10 --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.02 --trials 1  --device cuda:0
python main.py --attack_method TC_MIA --U_method retrain --dataset_name sst5 --net_name pythia70m --num_epochs 10 --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.02 --trials 3 --observations 10  --device cuda:2
