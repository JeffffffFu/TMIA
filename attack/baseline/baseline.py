import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from .utils import parse_list_arg, save_eval_result_json
except ImportError:
    from utils import parse_list_arg, save_eval_result_json

BASELINE_DEFAULTS = {
    'trial': 0,
    'data_root': None,
    'num_target_per_pattern': 333,
    'num_epochs': 100,
    'learning_rate': 0.0001,
    'hidden_dim': 32,
    'checkpoint_dir': './mia_checkpoints',
    'seed': 42,
    'num_trials': 3,
    'device': 'cuda:0',
    'debug': False,
    'normalize_ppl': True,
    'num_shadow_samples': 500,
    'num_target_samples': 1000,
    'num_insert': 1,
    'num_remove': 1,
    'test_same_as_train': True,
    'enable_k_tolerance_eval': False,
}


def run_baseline_attack(global_args: dict):
    data_root = os.path.join(project_root, 'save')
    baseline_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(project_root, 'mia_checkpoints')
    result_root = os.path.join(project_root, 'result')
    defaults = dict(BASELINE_DEFAULTS)
    defaults['data_root'] = data_root
    defaults['checkpoint_dir'] = checkpoint_dir
    defaults['result_root'] = result_root

    keys_from_main = ('attack_method', 'U_method', 'dataset_name', 'net_name',
                      'proportion_of_group_unlearn', 'device')
    merged = dict(defaults)
    for k in keys_from_main:
        if k in global_args and global_args[k] is not None:
            merged[k] = global_args[k]

    merged['use_ppl'] = '_LLM' in (merged.get('U_method') or '')

    attack = (merged.get('attack_method') or '').strip()
    if attack == 'TW_MIA':
        internal_method = 'mia_threshold'
    elif attack == 'UW_MIA':
        internal_method = 'delta'
    else:
        raise ValueError(f"Unsupported attack_method: {attack}. Supported: TW_MIA, UW_MIA")

    net_name = merged.get('net_name') or 'pythia70m'
    dataset_name = merged.get('dataset_name') or 'sst5'
    merged['train_net_names'] = merged['test_net_names'] = net_name
    merged['train_dataset_names'] = merged['test_dataset_names'] = dataset_name

    net_list = parse_list_arg(net_name)
    dataset_list = parse_list_arg(dataset_name)
    if not net_list:
        net_list = [net_name]
    if not dataset_list:
        dataset_list = [dataset_name]
    is_batch = len(net_list) > 1 or len(dataset_list) > 1

    if internal_method == 'mia_threshold':
        if is_batch:
            from .mia import run_batch_mia_threshold_attack
            run_batch_mia_threshold_attack(
                args=merged,
                data_root=data_root,
                checkpoint_dir=merged['checkpoint_dir'],
                result_root=merged['result_root'],
                device=merged['device'],
                num_target_per_pattern=merged['num_target_per_pattern'],
                seed=merged['seed'],
                debug=merged['debug'],
            )
        else:
            from .mia import run_mia_threshold_attack
            args = {
                'U_method': merged['U_method'],
                'net_name': net_list[0],
                'dataset_name': dataset_list[0],
                'proportion_of_group_unlearn': merged['proportion_of_group_unlearn'],
                'trial': merged['trial'],
                'use_ppl': merged['use_ppl'],
                'normalize_ppl': merged['normalize_ppl'],
                'attack_method': 'mia_threshold',
                'enable_k_tolerance_eval': merged.get('enable_k_tolerance_eval', False),
            }
            results = run_mia_threshold_attack(
                args=args,
                data_root=data_root,
                checkpoint_dir=checkpoint_dir,
                device=merged['device'],
                num_target_per_pattern=merged['num_target_per_pattern'],
                num_trials=merged['num_trials'],
                seed=merged['seed'],
                debug=merged['debug'],
            )
            evaluation = results.get('evaluation', {})
            save_eval_result_json(
                merged['result_root'],
                merged['U_method'],
                net_list[0],
                dataset_list[0],
                merged.get('proportion_of_group_unlearn'),
                attack,
                {
                    'precision': evaluation.get('precision', 0.0),
                    'recall': evaluation.get('recall', 0.0),
                    'f1': evaluation.get('f1', 0.0),
                    'I_Pre': evaluation.get('I_Pre', 0.0),
                    'I_Rec': evaluation.get('I_Rec', 0.0),
                    'I_F1': evaluation.get('I_F1', 0.0),
                },
            )
    else:
        assert internal_method == 'delta'
        if is_batch:
            from .delta_attack import run_batch_delta_attack
            run_batch_delta_attack(
                args=merged,
                data_root=data_root,
                result_root=merged['result_root'],
                num_shadow_samples=merged['num_shadow_samples'],
                num_target_samples=merged['num_target_samples'],
                num_insert=merged['num_insert'],
                num_remove=merged['num_remove'],
                print_config=True,
                num_trials=merged['num_trials'],
                use_ppl=merged['use_ppl'],
                enable_k_tolerance_eval=merged.get('enable_k_tolerance_eval', False),
            )
        else:
            from .delta_attack import run_delta_attack
            shadow_args = {
                'U_method': merged['U_method'],
                'net_name': net_list[0],
                'dataset_name': dataset_list[0],
                'proportion_of_group_unlearn': merged['proportion_of_group_unlearn'],
                'trial': merged['trial'],
                'shadow_or_target': 'shadow',
            }
            target_args = {
                'U_method': merged['U_method'],
                'net_name': net_list[0],
                'dataset_name': dataset_list[0],
                'proportion_of_group_unlearn': merged['proportion_of_group_unlearn'],
                'trial': merged['trial'],
                'shadow_or_target': 'target',
            }
            results = run_delta_attack(
                shadow_args=shadow_args,
                target_args=target_args,
                num_shadow_samples=merged['num_shadow_samples'],
                num_target_samples=merged['num_target_samples'],
                num_insert=merged['num_insert'],
                num_remove=merged['num_remove'],
                shadow_data_root=data_root,
                target_data_root=data_root,
                random_seed=merged['seed'],
                num_trials=merged['num_trials'],
                use_ppl=merged['use_ppl'],
                enable_k_tolerance_eval=merged.get('enable_k_tolerance_eval', False),
            )
            metrics = results.get('metrics', {})
            result_root = os.path.abspath(os.path.normpath(merged['result_root'] or os.path.join(project_root, 'result')))
            try:
                filepath = save_eval_result_json(
                    result_root,
                    merged['U_method'],
                    net_list[0],
                    dataset_list[0],
                    merged.get('proportion_of_group_unlearn'),
                    attack,
                    {
                        'precision': metrics.get('membership_precision', 0.0),
                        'recall': metrics.get('membership_recall', 0.0),
                        'f1': metrics.get('membership_f1', 0.0),
                        'I_Pre': metrics.get('I_Pre', 0.0),
                        'I_Rec': metrics.get('I_Rec', 0.0),
                        'I_F1': metrics.get('I_F1', 0.0),
                    },
                )
                print(f"\nResults saved to: {filepath}")
            except Exception as e:
                print(f"\nFailed to save results: {e}", file=sys.stderr)
                raise


if __name__ == '__main__':
    from parameter_parser import parameter_parser
    args = parameter_parser()
    if args.get('attack_method') not in ('TW_MIA', 'UW_MIA'):
        args['attack_method'] = 'TW_MIA'
    run_baseline_attack(args)
