import argparse

def parameter_parser():
    parser = argparse.ArgumentParser()

    ######################### general parameters ################################
    parser.add_argument('--dataset_name', type=str, default='cifar10',
                        choices=[ 'cifar10', 'stl10', 'cifar100','svhn','celebA','tinyimagenet','cinic10','sst5','news20','snli','mnli','mrpc','imdb','rte','ag_news','squad','simpleqa'])
    parser.add_argument('--device', type=str, default='cuda:3',
                        help="Choose the  device")
    parser.add_argument('--flag', type=str, default='none',
                        help="differential confidence samples")
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--random', type=int, default=42)
    ######################### target model related parameters ################################
    parser.add_argument('--net_name', type=str, default='resnet18',
                        choices=[ 'simple_cnn', 'resnet18', 'resnet20','vgg','resnet18_dp','resnet50','mobilenet', 'densenet','simple_cnn_dropout','pythia70m','pythia70m_dropout','roberta','opt13b','gpt2','T5','llama3b','llama8b'])
    # parser.add_argument('--attack_model', type=str, default='DT',
    #                     choices=['DT', 'MLP', 'LR', 'RF'])
    parser.add_argument('--U_method', type=str, default='None',
                        choices=['retrain', 'sisa','GA','sparsity','IF','fisher','scrub','sisa','retrain_dp','certified','NegGrad','NPO','None','continuous_unlearn_retrain','continuous_update_finetune_NPO','continuous_update_finetune_GA','all','continuous_update_finetune','continuous_update_finetune_dp','continuous_update_finetune_multiple_update','continuous_update_finetune_LLM','continuous_update_finetune_sparsity','continuous_update_finetune_scrub'])
    parser.add_argument('--retrain', type=int, default=0)
    parser.add_argument('--pre_train', type=str, default='both',choices=['both','target', 'shadow'])
    parser.add_argument('--attack_method', type=str, default='None',
                        choices=['TMIA','Transfer_dataset','Transfer_model','Transfer_unlearning','TW_MIA','UW_MIA','None'])
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--optim', type=str, default="Adam",
                        choices=['Adam', 'SGD'])
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate (default: .1)", )
    parser.add_argument("--dropout_rate", type=float, default=0.1,
                        help="dropout rate for pythia70m_dropout model (default: 0.1)", )
    ######################### attack related parameters ################################
    parser.add_argument('--trials', type=int, default=3,
                        help="number of trials")
    parser.add_argument('--window_size', type=int, default=5)
    parser.add_argument('--observations', type=int, default=5,
                        help="number of observations")
    parser.add_argument('--base_num_class', type=int, default=3,
                        help="number of class for baseline: 2 or 3")
    parser.add_argument('--proportion_of_group_unlearn', type=float, default=0.02,
                        help=">=1 mean the exact number of unlearn")
    parser.add_argument('--number_of_shadow_unlearned_model', type=int, default=1,
                        help="1 means number of shadow unlearned model equal to number of shadow original model")
    parser.add_argument('--size_of_shadow_training', type=float, default=-1,
                        help="-1 means using all shadow dataset for training")
    # For DPSGD
    parser.add_argument('--sigma', type=float, default=0.0,
                        help="noise of DPSGD")
    parser.add_argument('--eps', type=float, default=1.0,
                        help="privacy budget of DPSGD")
    parser.add_argument('--C', type=float, default=1.0,
                        help="C of model parameters (for DPSGD)")


    args = vars(parser.parse_args())
    #args = parser.parse_args()

    return args