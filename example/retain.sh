
# for target training, need to modify main.py
#python main.py  --attack_method Double_Attack --U_method all --dataset_name cifar10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method Double_Attack --U_method all --dataset_name cifar100 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method Double_Attack --U_method all --dataset_name cinic10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method Double_Attack --U_method all --dataset_name tinyimagenet --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#


# for attack, need to modify __init__.py
python main.py  --attack_method Double_Attack --U_method retrain --dataset_name cifar10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
python main.py --attack_method Double_Attack --U_method retrain --dataset_name cifar100 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
python main.py --attack_method Double_Attack --U_method retrain --dataset_name cinic10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
python main.py --attack_method Double_Attack --U_method retrain --dataset_name tinyimagenet --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0

python main.py  --attack_method Double_Attack --U_method GA --dataset_name cifar10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
python main.py --attack_method Double_Attack --U_method GA --dataset_name cifar100 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
python main.py --attack_method Double_Attack --U_method GA --dataset_name cinic10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
python main.py --attack_method Double_Attack --U_method GA --dataset_name tinyimagenet --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0

python main.py --attack_method Double_Attack --U_method scrub --dataset_name cifar10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
python main.py --attack_method Double_Attack --U_method scrub --dataset_name cifar100 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
python main.py --attack_method Double_Attack --U_method scrub --dataset_name cinic10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
python main.py --attack_method Double_Attack --U_method scrub --dataset_name tinyimagenet --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0

python main.py  --attack_method Double_Attack --U_method sparsity --dataset_name cifar10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
python main.py --attack_method Double_Attack --U_method sparsity --dataset_name cifar100 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
python main.py --attack_method Double_Attack --U_method sparsity --dataset_name cinic10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
python main.py --attack_method Double_Attack --U_method sparsity --dataset_name tinyimagenet --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0

