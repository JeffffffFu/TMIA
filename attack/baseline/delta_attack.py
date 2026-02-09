import os
import sys
import random
import numpy as np
from typing import Dict, Any, List, Set, Tuple
from collections import defaultdict

from attack.TMIA.get_training_data import select_test_samples_by_pattern

attack_tmia_path = os.path.join(os.path.dirname(__file__), '../TMIA')
if attack_tmia_path not in sys.path:
    sys.path.insert(0, attack_tmia_path)

from .data_processor import (
    compute_breakpoint_confidence_delta_percentile,
    filter_breakpoint_deltas_by_threshold,
    build_path_and_find_timestamps,
    timestamp_to_index,
    extract_breakpoints_from_status_sequence
)
from .data_loader import compute_deltas_for_all_timestamps
from .evaluation import compute_delta_attack_metrics
try:
    from .utils import generate_all_combinations, parse_list_arg, save_eval_result_json, breakpoints_to_membership_timestamps
except ImportError:
    from utils import generate_all_combinations, parse_list_arg, save_eval_result_json, breakpoints_to_membership_timestamps


def run_delta_attack(shadow_args: Dict[str, Any],
                     target_args: Dict[str, Any],
                     num_shadow_samples: int = 500,
                     num_target_samples: int = 1000,
                     num_insert: int = 1,
                     num_remove: int = 1,
                     shadow_data_root: str = None,
                     target_data_root: str = None,
                     random_seed: int = None,
                     num_trials: int = 1,
                     use_ppl: bool = False,
                     enable_k_tolerance_eval: bool = False,
                     verbose: bool = True) -> Dict[str, Any]:
    if num_trials > 1:
        all_trial_results = []
        for trial_idx in range(num_trials):
            trial_seed = (random_seed if random_seed is not None else 42) + trial_idx
            np.random.seed(trial_seed)
            random.seed(trial_seed)
            
            result = run_delta_attack(
                shadow_args=shadow_args,
                target_args=target_args,
                num_shadow_samples=num_shadow_samples,
                num_target_samples=num_target_samples,
                num_insert=num_insert,
                num_remove=num_remove,
                shadow_data_root=shadow_data_root,
                target_data_root=target_data_root,
                random_seed=trial_seed,
                num_trials=1,
                use_ppl=use_ppl,
                enable_k_tolerance_eval=enable_k_tolerance_eval,
                verbose=False
            )
            
            all_trial_results.append(result['metrics'])
        
        metrics_keys = [
            'membership_precision', 'membership_recall', 'membership_f1',
            'insert_precision', 'insert_recall', 'insert_f1',
            'remove_precision', 'remove_recall', 'remove_f1',
            'I_Pre', 'I_Rec', 'I_F1'
        ]
        avg_metrics = {}
        for key in metrics_keys:
            values = [r.get(key, 0.0) for r in all_trial_results if key in r]
            if values:
                avg_metrics[key] = float(np.mean(values))
        stat_keys = [
            'total_intersection', 'total_predicted', 'total_true',
            'insert_correct', 'insert_total', 'insert_pred_total',
            'remove_correct', 'remove_total', 'remove_pred_total'
        ]
        for key in stat_keys:
            values = [r.get(key, 0) for r in all_trial_results if key in r]
            if values:
                avg_metrics[key] = int(np.mean(values))
        
        print("\n" + "=" * 70)
        print("Timestamp-level evaluation (T-Pre, T-Rec, T-F1)")
        print("=" * 70)
        print(f"  T-Pre: {avg_metrics.get('membership_precision', 0):.4f}")
        print(f"  T-Rec: {avg_metrics.get('membership_recall', 0):.4f}")
        print(f"  T-F1: {avg_metrics.get('membership_f1', 0):.4f}")
        print("\n" + "=" * 70)
        print("Interval-level evaluation (I-Pre, I-Rec, I-F1)")
        print("=" * 70)
        print(f"  I-Pre: {avg_metrics.get('I_Pre', 0):.4f}")
        print(f"  I-Rec: {avg_metrics.get('I_Rec', 0):.4f}")
        print(f"  I-F1: {avg_metrics.get('I_F1', 0):.4f}")
        print("=" * 70)
        
        return {
            'metrics': avg_metrics,
            'thresholds': result.get('thresholds', {}),
            'num_shadow_samples': num_shadow_samples,
            'num_target_samples': num_target_samples,
            'num_filtered_samples': result.get('num_filtered_samples', 0),
            'all_trial_results': all_trial_results
        }
    
    shadow_args = shadow_args.copy()
    if 'shadow_or_target' not in shadow_args:
        shadow_args['shadow_or_target'] = 'shadow'
    
    percentile_insert_delta, percentile_remove_delta = compute_breakpoint_confidence_delta_percentile(
        args=shadow_args,
        num_insert=num_insert,
        num_remove=num_remove,
        percentile=50,
        num_samples=num_shadow_samples,
        data_root=shadow_data_root,
        random_seed=random_seed,
        use_ppl=use_ppl
    )

    target_args = target_args.copy()
    if 'shadow_or_target' not in target_args:
        target_args['shadow_or_target'] = 'target'
    
    save_path, timestamp_dirs = build_path_and_find_timestamps(target_args, data_root=target_data_root)
    
    dataset_name = target_args.get('dataset_name', '')
    if dataset_name.lower() in ['cifar100', 'cinic10', 'cifar10', 'svhn']:
        num_insert_only = 50
        num_remove_only = 50
        num_insert_remove = 100
    else:
        num_insert_only = 400
        num_remove_only = 400
        num_insert_remove = 400
    
    balance_membership = target_args.get('balance_test_membership', True)
    ratio = 0.33
    
    test_sample_indices, test_breakpoint_indices_dict, test_samples_breakpoint_info = select_test_samples_by_pattern(
        save_path=save_path,
        timestamp_dirs=timestamp_dirs,
        num_insert_only=num_insert_only,
        num_remove_only=num_remove_only,
        num_insert_remove=num_insert_remove,
        random_seed=random_seed,
        balance_membership=balance_membership,
        target_member_ratio=ratio
    )
    selected_test_sample_indices = set(test_sample_indices)
    
    num_timestamps = len(timestamp_dirs)
    
    target_delta_data = compute_deltas_for_all_timestamps(
        args=target_args,
        sample_indices=test_sample_indices,
        num_timestamps=num_timestamps,
        data_root=target_data_root,
        use_ppl=use_ppl
    )
    filtered_delta_data = filter_breakpoint_deltas_by_threshold(
        delta_data=target_delta_data,
        insert_threshold=percentile_insert_delta,
        remove_threshold=percentile_remove_delta
    )
    num_timestamps = len(timestamp_dirs)
    
    predicted_breakpoints_by_sample = defaultdict(list)
    
    if 'insert_sample_indices' in filtered_delta_data:
        for sample_idx, timestamp in zip(
            filtered_delta_data['insert_sample_indices'],
            filtered_delta_data['insert_timestamps']
        ):
            timestamp_idx = timestamp_to_index(timestamp, timestamp_dirs)
            if timestamp_idx is not None:
                predicted_breakpoints_by_sample[sample_idx].append((timestamp_idx, 1))
    
    if 'remove_sample_indices' in filtered_delta_data:
        for sample_idx, timestamp in zip(
            filtered_delta_data['remove_sample_indices'],
            filtered_delta_data['remove_timestamps']
        ):
            timestamp_idx = timestamp_to_index(timestamp, timestamp_dirs)
            if timestamp_idx is not None:
                predicted_breakpoints_by_sample[sample_idx].append((timestamp_idx, 2))
    
    status_file = os.path.join(save_path, 'sample_status_converted.npy')
    if not os.path.exists(status_file):
        raise FileNotFoundError(f"Ground truth status file not found: {status_file}")
    
    ground_truth_status = np.load(status_file)
    all_sample_indices = selected_test_sample_indices.copy()
    
    all_sample_indices = {idx for idx in all_sample_indices if idx < len(ground_truth_status)}
    
    loaded_sample_indices = set()
    if 'insert_sample_indices' in target_delta_data:
        loaded_sample_indices.update(target_delta_data['insert_sample_indices'])
    if 'remove_sample_indices' in target_delta_data:
        loaded_sample_indices.update(target_delta_data['remove_sample_indices'])
    
    if len(all_sample_indices) == 0:
        if verbose:
            print("Warning: No predicted breakpoints found; cannot compute evaluation metrics")
        return {
            'metrics': {
                'membership_precision': 0.0,
                'membership_recall': 0.0,
                'membership_f1': 0.0,
                'insert_precision': 0.0,
                'insert_recall': 0.0,
                'insert_f1': 0.0,
                'remove_precision': 0.0,
                'remove_recall': 0.0,
                'remove_f1': 0.0,
                'total_intersection': 0,
                'total_predicted': 0,
                'total_true': 0,
                'insert_correct': 0,
                'insert_total': 0,
                'insert_pred_total': 0,
                'remove_correct': 0,
                'remove_total': 0,
                'remove_pred_total': 0
            },
            'thresholds': {
                'insert_threshold': percentile_insert_delta,
                'remove_threshold': percentile_remove_delta
            },
            'num_shadow_samples': num_shadow_samples,
            'num_target_samples': num_target_samples,
            'num_filtered_samples': 0
        }
    
    predicted_memberships_by_sample = {}
    true_memberships_by_sample = {}
    
    for sample_idx in all_sample_indices:
        if sample_idx >= len(ground_truth_status):
            continue
        true_status_seq = ground_truth_status[sample_idx]
        
        true_breakpoints = extract_breakpoints_from_status_sequence(true_status_seq)
        
        true_membership = breakpoints_to_membership_timestamps(true_breakpoints, num_timestamps)
        true_memberships_by_sample[sample_idx] = true_membership
        
        pred_breakpoints = predicted_breakpoints_by_sample.get(sample_idx, [])
        pred_membership = breakpoints_to_membership_timestamps(pred_breakpoints, num_timestamps)
        predicted_memberships_by_sample[sample_idx] = pred_membership
    
    metrics = compute_delta_attack_metrics(
        predicted_memberships_by_sample=predicted_memberships_by_sample,
        true_memberships_by_sample=true_memberships_by_sample,
        predicted_breakpoints_by_sample=predicted_breakpoints_by_sample,
        true_status_sequences={idx: ground_truth_status[idx] for idx in all_sample_indices if idx < len(ground_truth_status)},
        num_timestamps=num_timestamps,
        verbose=verbose
    )
    
    from .data_processor import convert_delta_predictions_to_tolerance_format
    import sys
    attack_tmia_path = os.path.join(os.path.dirname(__file__), '../TMIA')
    if attack_tmia_path not in sys.path:
        sys.path.insert(0, attack_tmia_path)
    from inference import compute_hit_rate_with_tolerance, compute_metrics_with_tolerance, compute_interval_level_metrics
    
    predictions_by_sample = convert_delta_predictions_to_tolerance_format(
        predicted_breakpoints_by_sample=predicted_breakpoints_by_sample,
        true_status_sequences={idx: ground_truth_status[idx] for idx in all_sample_indices if idx < len(ground_truth_status)},
        all_sample_indices=list(all_sample_indices),
        num_timestamps=num_timestamps
    )
    
    tolerance_metrics = {}
    for k in range(0, 6):
        metrics_k = compute_hit_rate_with_tolerance(predictions_by_sample, tolerance_theta=k)
        metrics_pr = compute_metrics_with_tolerance(predictions_by_sample, tolerance_k=k)
        metrics_k['insert_precision'] = metrics_pr['insert']['precision']
        metrics_k['insert_recall'] = metrics_pr['insert']['recall']
        metrics_k['insert_f1'] = metrics_pr['insert']['f1']
        metrics_k['remove_precision'] = metrics_pr['remove']['precision']
        metrics_k['remove_recall'] = metrics_pr['remove']['recall']
        metrics_k['remove_f1'] = metrics_pr['remove']['f1']
        tolerance_metrics[f'k_{k}'] = metrics_k

    if enable_k_tolerance_eval and verbose:
        print("\n" + "=" * 70)
        print("Tolerance window evaluation (K=0 to 5)")
        print("=" * 70)
        for k in range(0, 6):
            m = tolerance_metrics[f'k_{k}']
            print(f"--- Tolerance window K={k} ---")
            print(f"  insert_hit_rate: {m.get('insert_hit_rate', 0):.4f}, remove_hit_rate: {m.get('remove_hit_rate', 0):.4f}")
            print(f"  insert P/R/F1: {m.get('insert_precision', 0):.4f} / {m.get('insert_recall', 0):.4f} / {m.get('insert_f1', 0):.4f}")
            print(f"  remove P/R/F1: {m.get('remove_precision', 0):.4f} / {m.get('remove_recall', 0):.4f} / {m.get('remove_f1', 0):.4f}")
    
    interval_metrics = compute_interval_level_metrics(predictions_by_sample, num_timestamps, verbose=False)
    metrics['I_Pre'] = interval_metrics['I_Pre']
    metrics['I_Rec'] = interval_metrics['I_Rec']
    metrics['I_F1'] = interval_metrics['I_F1']
    
    metrics['tolerance_metrics'] = tolerance_metrics
    
    if verbose:
        print("\n" + "=" * 70)
        print("Timestamp-level evaluation (T-Pre, T-Rec, T-F1)")
        print("=" * 70)
        print(f"  T-Pre: {metrics['membership_precision']:.4f}")
        print(f"  T-Rec: {metrics['membership_recall']:.4f}")
        print(f"  T-F1: {metrics['membership_f1']:.4f}")
        print("\n" + "=" * 70)
        print("Interval-level evaluation (I-Pre, I-Rec, I-F1)")
        print("=" * 70)
        print(f"  I-Pre: {interval_metrics['I_Pre']:.4f}")
        print(f"  I-Rec: {interval_metrics['I_Rec']:.4f}")
        print(f"  I-F1: {interval_metrics['I_F1']:.4f}")
        print("=" * 70)
    
    return {
        'metrics': metrics,
        'thresholds': {
            'insert_threshold': percentile_insert_delta,
            'remove_threshold': percentile_remove_delta
        },
        'num_shadow_samples': num_shadow_samples,
        'num_target_samples': num_target_samples,
        'num_filtered_samples': len(predicted_breakpoints_by_sample)
    }


def run_batch_delta_attack(args: Dict[str, Any],
                           data_root: str = "../../save",
                           num_shadow_samples: int = 500,
                           num_target_samples: int = 1000,
                           num_insert: int = 1,
                           num_remove: int = 1,
                           result_root: str = None,
                           print_config: bool = True,
                           num_trials: int = 1,
                           use_ppl: bool = False,
                           enable_k_tolerance_eval: bool = False) -> Dict[str, Any]:
    if result_root is None:
        result_root = args.get('result_root', './result')
    import argparse
    
    class Args:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)
    
    temp_args = Args(args)
    combinations = generate_all_combinations(temp_args)
    total_combinations = len(combinations)
    
    if print_config:
        print(f"\nResults will be saved to: {result_root}/<U_method>/<model>/<dataset>/<proportion>/UW_MIA/result.json")
        print("=" * 80)
    
    all_results = []
    
    for idx, combination in enumerate(combinations, 1):
        train_net = combination['train_net_name']
        train_dataset = combination['train_dataset_name']
        test_net = combination['test_net_name']
        test_dataset = combination['test_dataset_name']
        train_proportion = combination['train_proportion']
        test_proportion = combination['test_proportion']
        base_args = combination['base_args']
        shadow_args = {
            'U_method': base_args['U_method'],
            'net_name': train_net,
            'dataset_name': train_dataset,
            'proportion_of_group_unlearn': train_proportion,
            'trial': base_args['trial'],
            'shadow_or_target': 'shadow'
        }
        
        target_args = {
            'U_method': base_args['U_method'],
            'net_name': test_net,
            'dataset_name': test_dataset,
            'proportion_of_group_unlearn': test_proportion,
            'trial': base_args['trial'],
            'shadow_or_target': 'target'
        }
        
        try:
            results = run_delta_attack(
                shadow_args=shadow_args,
                target_args=target_args,
                num_shadow_samples=num_shadow_samples,
                num_target_samples=num_target_samples,
                num_insert=num_insert,
                num_remove=num_remove,
                shadow_data_root=data_root,
                target_data_root=data_root,
                random_seed=base_args.get('seed', 42),
                num_trials=num_trials,
                use_ppl=use_ppl,
                enable_k_tolerance_eval=enable_k_tolerance_eval
            )
            
            metrics = results.get('metrics', {})
            
            membership_precision = metrics.get('membership_precision', -1.0)
            membership_recall = metrics.get('membership_recall', -1.0)
            membership_f1 = metrics.get('membership_f1', -1.0)
            insert_precision = metrics.get('insert_precision', -1.0)
            insert_recall = metrics.get('insert_recall', -1.0)
            insert_f1 = metrics.get('insert_f1', -1.0)
            remove_precision = metrics.get('remove_precision', -1.0)
            remove_recall = metrics.get('remove_recall', -1.0)
            remove_f1 = metrics.get('remove_f1', -1.0)
            
            save_eval_result_json(
                result_root,
                base_args['U_method'],
                test_net,
                test_dataset,
                test_proportion,
                'UW_MIA',
                {
                    'precision': membership_precision if membership_precision >= 0 else 0.0,
                    'recall': membership_recall if membership_recall >= 0 else 0.0,
                    'f1': membership_f1 if membership_f1 >= 0 else 0.0,
                    'I_Pre': metrics.get('I_Pre', 0.0),
                    'I_Rec': metrics.get('I_Rec', 0.0),
                    'I_F1': metrics.get('I_F1', 0.0),
                },
            )
            
            print(f"\n✓ Combination completed:")
            print(f"  Overlap - T-Pre/T-Rec/T-F1: {membership_precision:.4f}/{membership_recall:.4f}/{membership_f1:.4f}")
            
            all_results.append({
                'combination': combination,
                'membership_precision': membership_precision,
                'membership_recall': membership_recall,
                'membership_f1': membership_f1,
                'insert_precision': insert_precision,
                'insert_recall': insert_recall,
                'insert_f1': insert_f1,
                'remove_precision': remove_precision,
                'remove_recall': remove_recall,
                'remove_f1': remove_f1
            })
            
        except Exception as e:
            print(f"\n✗ Combination failed: {e}")
            save_eval_result_json(
                result_root,
                base_args['U_method'],
                test_net,
                test_dataset,
                test_proportion,
                'UW_MIA',
                {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'I_Pre': 0.0, 'I_Rec': 0.0, 'I_F1': 0.0},
            )
            all_results.append({
                'combination': combination,
                'membership_precision': -1.0,
                'membership_recall': -1.0,
                'membership_f1': -1.0,
                'insert_precision': -1.0,
                'insert_recall': -1.0,
                'insert_f1': -1.0,
                'remove_precision': -1.0,
                'remove_recall': -1.0,
                'remove_f1': -1.0
            })
    
    valid_results = [r for r in all_results if r['membership_precision'] >= 0]
    failed_results = [r for r in all_results if r['membership_precision'] < 0]
    
    print("\n" + "=" * 80)
    print("Batch UW-MIA attack completed")
    print("=" * 80)
    print(f"Success: {len(valid_results)}/{total_combinations}, Failed: {len(failed_results)}/{total_combinations}")
    if valid_results:
        avg_f1 = np.mean([r['membership_f1'] for r in valid_results])
        print(f"Average T-F1: {avg_f1:.4f}")
    print(f"\nResults saved to: {result_root}/<U_method>/<model>/<dataset>/<proportion>/UW_MIA/result.json")
    print("=" * 80)
    
    return {
        'all_results': all_results,
        'valid_results': valid_results,
        'failed_results': failed_results,
    }


def main():
    import sys
    import os as _os
    _root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '../..'))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from parameter_parser import parameter_parser
    from attack.baseline.baseline import run_baseline_attack
    args = parameter_parser()
    args['attack_method'] = 'UW_MIA'
    run_baseline_attack(args)


if __name__ == '__main__':
    main()

