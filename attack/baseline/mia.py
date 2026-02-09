import os
import sys
import numpy as np
from typing import Dict, Any, List, Tuple

from attack.TMIA.get_training_data import select_test_samples_by_pattern

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../TMIA'))

from .data_loader import DataLoader, load_outputs_and_labels_for_mia_threshold, load_training_data
from .data_processor import DataProcessor, build_path_and_find_timestamps
try:
    from .utils import generate_all_combinations, parse_list_arg, save_eval_result_json
except ImportError:
    from utils import generate_all_combinations, parse_list_arg, save_eval_result_json


class BlackBoxBenchmarks:
    def __init__(self, shadow_train_performance, shadow_test_performance, 
                 target_train_performance, target_test_performance, num_classes):
        self.num_classes = num_classes
        
        self.s_tr_outputs, self.s_tr_labels = shadow_train_performance
        self.s_te_outputs, self.s_te_labels = shadow_test_performance
        self.t_tr_outputs, self.t_tr_labels = target_train_performance
        self.t_te_outputs, self.t_te_labels = target_test_performance
        
        self.s_tr_corr = (np.argmax(self.s_tr_outputs, axis=1)==self.s_tr_labels).astype(int)
        self.s_te_corr = (np.argmax(self.s_te_outputs, axis=1)==self.s_te_labels).astype(int)
        self.t_tr_corr = (np.argmax(self.t_tr_outputs, axis=1)==self.t_tr_labels).astype(int)
        self.t_te_corr = (np.argmax(self.t_te_outputs, axis=1)==self.t_te_labels).astype(int)
        
        self.s_tr_conf = np.array([self.s_tr_outputs[i, self.s_tr_labels[i]] for i in range(len(self.s_tr_labels))])
        self.s_te_conf = np.array([self.s_te_outputs[i, self.s_te_labels[i]] for i in range(len(self.s_te_labels))])
        self.t_tr_conf = np.array([self.t_tr_outputs[i, self.t_tr_labels[i]] for i in range(len(self.t_tr_labels))])
        self.t_te_conf = np.array([self.t_te_outputs[i, self.t_te_labels[i]] for i in range(len(self.t_te_labels))])
        
        self.s_tr_entr = self._entr_comp(self.s_tr_outputs)
        self.s_te_entr = self._entr_comp(self.s_te_outputs)
        self.t_tr_entr = self._entr_comp(self.t_tr_outputs)
        self.t_te_entr = self._entr_comp(self.t_te_outputs)
        
        self.s_tr_m_entr = self._m_entr_comp(self.s_tr_outputs, self.s_tr_labels)
        self.s_te_m_entr = self._m_entr_comp(self.s_te_outputs, self.s_te_labels)
        self.t_tr_m_entr = self._m_entr_comp(self.t_tr_outputs, self.t_tr_labels)
        self.t_te_m_entr = self._m_entr_comp(self.t_te_outputs, self.t_te_labels)
    
    def _log_value(self, probs, small_value=1e-30):
        return -np.log(np.maximum(probs, small_value))
    
    def _entr_comp(self, probs):
        return np.sum(np.multiply(probs, self._log_value(probs)), axis=1)
    
    def _m_entr_comp(self, probs, true_labels):
        log_probs = self._log_value(probs)
        reverse_probs = 1 - probs
        log_reverse_probs = self._log_value(reverse_probs)
        modified_probs = np.copy(probs)
        modified_probs[range(true_labels.size), true_labels] = reverse_probs[range(true_labels.size), true_labels]
        modified_log_probs = np.copy(log_reverse_probs)
        modified_log_probs[range(true_labels.size), true_labels] = log_probs[range(true_labels.size), true_labels]
        return np.sum(np.multiply(modified_probs, modified_log_probs), axis=1)
    
    def _thre_setting(self, tr_values, te_values):
        value_list = np.concatenate((tr_values, te_values))
        thre, max_acc = 0, 0
        for value in value_list:
            tr_ratio = np.sum(tr_values >= value) / (len(tr_values) + 0.0)
            te_ratio = np.sum(te_values < value) / (len(te_values) + 0.0)
            acc = 0.5 * (tr_ratio + te_ratio)
            if acc > max_acc:
                thre, max_acc = value, acc
        return thre
    
    def _mem_inf_via_corr(self):
        t_tr_acc = np.sum(self.t_tr_corr) / (len(self.t_tr_corr) + 0.0)
        t_te_acc = np.sum(self.t_te_corr) / (len(self.t_te_corr) + 0.0)
        mem_inf_acc = 0.5 * (t_tr_acc + 1 - t_te_acc)
        return
    
    def _mem_inf_thre(self, v_name, s_tr_values, s_te_values, t_tr_values, t_te_values):
        t_tr_mem, t_te_non_mem = 0, 0
        for num in range(self.num_classes):
            thre = self._thre_setting(s_tr_values[self.s_tr_labels == num], s_te_values[self.s_te_labels == num])
            t_tr_mem += np.sum(t_tr_values[self.t_tr_labels == num] >= thre)
            t_te_non_mem += np.sum(t_te_values[self.t_te_labels == num] < thre)
        mem_inf_acc = 0.5 * (t_tr_mem / (len(self.t_tr_labels) + 0.0) + t_te_non_mem / (len(self.t_te_labels) + 0.0))
        return
    
    def _mem_inf_benchmarks(self, all_methods=True, benchmark_methods=[]):
        if (all_methods) or ('correctness' in benchmark_methods):
            self._mem_inf_via_corr()
        if (all_methods) or ('confidence' in benchmark_methods):
            self._mem_inf_thre('confidence', self.s_tr_conf, self.s_te_conf, self.t_tr_conf, self.t_te_conf)
        if (all_methods) or ('entropy' in benchmark_methods):
            self._mem_inf_thre('entropy', -self.s_tr_entr, -self.s_te_entr, -self.t_tr_entr, -self.t_te_entr)
        if (all_methods) or ('modified entropy' in benchmark_methods):
            self._mem_inf_thre('modified entropy', -self.s_tr_m_entr, -self.s_te_m_entr, -self.t_tr_m_entr, -self.t_te_m_entr)
        return


def load_target_data_for_mia(
    target_args: Dict[str, Any],
    target_data_root: str,
    num_per_pattern: int = 333,
    base_args: Dict[str, Any] = None,
    use_ppl: bool = False,
    normalize_ppl: bool = False,
    normalization_params: Dict[str, Any] = None,
    processor: DataProcessor = None,
) -> Dict[str, Any]:
    if 'shadow_or_target' not in target_args:
        target_args = target_args.copy()
        target_args['shadow_or_target'] = 'target'
    if base_args is None:
        base_args = target_args
    if processor is None:
        processor = DataProcessor()

    save_path, timestamp_dirs = build_path_and_find_timestamps(target_args, data_root=target_data_root)
    balance_membership = target_args.get('balance_test_membership', base_args.get('balance_test_membership', True))
    ratio =0.33
    dataset_name = target_args.get('dataset_name', base_args.get('dataset_name', ''))
    if target_args.get('attack_method', base_args.get('attack_method', '')) == 'mia_threshold':
        num_insert_only = 100
        num_remove_only = 100
        num_insert_remove = 10
    elif dataset_name.lower() in ['cifar100', 'cinic10', 'cifar10', 'svhn', 'simpleqa', 'squad']:
        num_insert_only = 50
        num_remove_only = 50
        num_insert_remove = 100
    else:
        num_insert_only = 400
        num_remove_only = 400
        num_insert_remove = 400

    test_sample_indices, test_breakpoint_indices_dict, test_samples_breakpoint_info = select_test_samples_by_pattern(
        save_path=save_path,
        timestamp_dirs=timestamp_dirs,
        num_insert_only=num_insert_only,
        num_remove_only=num_remove_only,
        num_insert_remove=num_insert_remove,
        random_seed=target_args.get('random_seed', base_args.get('random_seed', None)),
        balance_membership=balance_membership,
        target_member_ratio=ratio
    )

    print(f"\nTest sample selection done: {len(test_sample_indices)} samples in total")

    feature_type = 'PPL' if use_ppl else 'CT'
    print(f"\nLoading all target data and {feature_type} features...")

    norm_params = None
    if use_ppl and normalize_ppl and normalization_params is not None:
        norm_params = normalization_params
        print(f"  Using normalization parameters from training")
    elif use_ppl and normalize_ppl:
        print(f"  Warning: No normalization params provided; skipping normalization")

    target_data = processor.load_and_extract_ct_features(
        args=target_args,
        indices=test_sample_indices,
        data_root=target_data_root,
        use_ppl=use_ppl,
        normalize_ppl=normalize_ppl,
        normalization_params=norm_params
    )
    target_data['samples_breakpoint_info'] = test_samples_breakpoint_info
    return target_data


def evaluate_mia_attack(
    target_data: Dict[str, Any],
    predictions: Dict[str, Any],
    target_args: Dict[str, Any],
    target_data_root: str,
    data_root: str = None,
) -> Dict[str, Any]:
    if data_root is None:
        data_root = target_data_root
    if 'shadow_or_target' not in target_args:
        target_args = target_args.copy()
        target_args['shadow_or_target'] = 'target'

    base_dir = os.path.join(
        os.getcwd(),
        target_data_root,
        target_args['U_method'],
        target_args['net_name'],
        target_args['dataset_name'],
        str(target_args['proportion_of_group_unlearn']),
        target_args['shadow_or_target'],
        str(target_args['trial'])
    )
    status_file = os.path.join(base_dir, 'sample_status_converted.npy')
    converted_status_array = np.load(status_file)

    print(f"\nLoading ground truth status: {status_file}")
    print(f"  Number of timesteps: {converted_status_array.shape[1]}")

    all_true_status_list = []
    all_pred_status_list = []
    indices = predictions['indices']
    pred_status = predictions['predictions']

    for i, sample_idx in enumerate(indices):
        true_seq = converted_status_array[sample_idx]
        true_retain = np.isin(true_seq, [1, 2]).astype(int)
        pred_retain = pred_status[i]
        all_true_status_list.append(true_retain)
        all_pred_status_list.append(pred_retain)

    from .evaluation import evaluate_changepoint_accuracy
    changepoint_metrics = evaluate_changepoint_accuracy(
        all_true_status_list,
        all_pred_status_list
    )

    print(f"\n" + "=" * 70)
    print("Timestamp-level evaluation (T-Pre, T-Rec, T-F1)")
    print("=" * 70)
    print(f"  T-Pre: {changepoint_metrics['precision']:.4f}")
    print(f"  T-Rec: {changepoint_metrics['recall']:.4f}")
    print(f"  T-F1: {changepoint_metrics['f1']:.4f}")
    from .data_processor import convert_mia_predictions_to_tolerance_format
    attack_tmia_path = os.path.join(os.path.dirname(__file__), '../TMIA')
    if attack_tmia_path not in sys.path:
        sys.path.insert(0, attack_tmia_path)
    from inference import compute_hit_rate_with_tolerance, compute_metrics_with_tolerance, compute_interval_level_metrics

    all_true_status_sequences = []
    for i, sample_idx in enumerate(indices):
        true_seq = converted_status_array[sample_idx]
        all_true_status_sequences.append(true_seq)

    predictions_by_sample = convert_mia_predictions_to_tolerance_format(
        true_status_sequences=all_true_status_sequences,
        pred_status_sequences=all_pred_status_list,
        sample_indices=indices
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

    if target_args.get('enable_k_tolerance_eval', False):
        print(f"\n" + "=" * 70)
        print("Tolerance window evaluation (K=0 to 5)")
        print("=" * 70)
        for k in range(0, 6):
            m = tolerance_metrics[f'k_{k}']
            print(f"--- Tolerance window K={k} ---")
            print(f"  insert_hit_rate: {m.get('insert_hit_rate', 0):.4f}, remove_hit_rate: {m.get('remove_hit_rate', 0):.4f}")
            print(f"  insert P/R/F1: {m.get('insert_precision', 0):.4f} / {m.get('insert_recall', 0):.4f} / {m.get('insert_f1', 0):.4f}")
            print(f"  remove P/R/F1: {m.get('remove_precision', 0):.4f} / {m.get('remove_recall', 0):.4f} / {m.get('remove_f1', 0):.4f}")

    num_timestamps = converted_status_array.shape[1]
    print(f"\n" + "=" * 70)
    print("Interval-level evaluation (I-Pre, I-Rec, I-F1)")
    print("=" * 70)
    interval_metrics = compute_interval_level_metrics(predictions_by_sample, num_timestamps, verbose=False)
    changepoint_metrics['I_Pre'] = interval_metrics['I_Pre']
    changepoint_metrics['I_Rec'] = interval_metrics['I_Rec']
    changepoint_metrics['I_F1'] = interval_metrics['I_F1']
    print(f"  I-Pre: {interval_metrics['I_Pre']:.4f}")
    print(f"  I-Rec: {interval_metrics['I_Rec']:.4f}")
    print(f"  I-F1: {interval_metrics['I_F1']:.4f}")
    changepoint_metrics['tolerance_metrics'] = tolerance_metrics
    return changepoint_metrics



def run_mia_threshold_attack(
    args: Dict[str, Any],
    data_root: str = "../../save",
    checkpoint_dir: str = "./checkpoints",
    device: str = None,
    num_target_per_pattern: int = 333,
    num_trials: int = 1,
    seed: int = 42,
    debug: bool = True,
    print_config: bool = True,
    shadow_args: Dict[str, Any] = None,
    target_args: Dict[str, Any] = None,
    target_data_root: str = None,
) -> Dict[str, Any]:
    if shadow_args is None:
        shadow_args = args.copy()
    if target_args is None:
        target_args = args.copy()
    if target_data_root is None:
        target_data_root = data_root

    if args.get('use_ppl', False):
        raise ValueError("MIA threshold attack requires multi-class model output; use use_ppl=False (CT mode)")

    shadow_args_copy = shadow_args.copy()
    if 'shadow_or_target' not in shadow_args_copy:
        shadow_args_copy['shadow_or_target'] = 'shadow'
    if args.get('dataset_name', '') in ['simpleqa', 'squad']:
        num_unseen, num_retain, num_per_pattern = 10, 10, 0
    else:
        num_unseen, num_retain, num_per_pattern = 100, 100, 0

    _, _, shadow_indices = load_training_data(
        args=shadow_args_copy,
        num_unseen=num_unseen,
        num_retain=num_retain,
        num_per_pattern=num_per_pattern,
        data_root=data_root,
        use_ppl=False,
    )

    target_data = load_target_data_for_mia(
        target_args=target_args,
        target_data_root=target_data_root,
        num_per_pattern=num_target_per_pattern,
        base_args=args,
        use_ppl=False,
        normalize_ppl=False,
        normalization_params=None,
        processor=DataProcessor(),
    )

    target_args_copy = target_args.copy()
    target_args_copy['shadow_or_target'] = 'target'
    s_tr, s_te, t_tr, t_te, num_classes = load_outputs_and_labels_for_mia_threshold(
        shadow_args_copy,
        target_args_copy,
        shadow_indices,
        target_data['indices'],
        data_root=data_root,
        target_data_root=target_data_root,
    )

    MIA_bench = BlackBoxBenchmarks(s_tr, s_te, t_tr, t_te, num_classes=num_classes)
    MIA_bench._mem_inf_benchmarks()

    thresholds = {}
    for c in range(num_classes):
        mask_tr = (MIA_bench.s_tr_labels == c)
        mask_te = (MIA_bench.s_te_labels == c)
        if np.sum(mask_tr) == 0 or np.sum(mask_te) == 0:
            thresholds[c] = 0.5
        else:
            thresholds[c] = MIA_bench._thre_setting(
                MIA_bench.s_tr_conf[mask_tr],
                MIA_bench.s_te_conf[mask_te],
            )

    outputs = target_data['outputs']
    labels = target_data['labels']
    if outputs.ndim != 3 or labels.ndim != 2:
        raise ValueError("Target data must have outputs (N,T,C) and labels (N,T); do not use use_ppl mode")
    N, T, C = outputs.shape
    if np.any(outputs > 1.01) or np.any(outputs < -0.01):
        mx = np.max(outputs, axis=-1, keepdims=True)
        outputs = np.exp(outputs - mx) / (np.sum(np.exp(outputs - mx), axis=-1, keepdims=True) + 1e-30)

    pred = np.zeros((N, T), dtype=np.int64)
    for n in range(N):
        for t in range(T):
            c = int(labels[n, t])
            if c < 0:
                pred[n, t] = 0
            else:
                conf = float(outputs[n, t, c])
                thre_c = thresholds.get(c, 0.5)
                pred[n, t] = 1 if conf >= thre_c else 0

    predictions = {
        'predictions': pred,
        'probabilities': None,
        'indices': target_data['indices'],
        'ct_features': target_data['ct_features'],
        'outputs': target_data['outputs'],
        'labels': target_data['labels'],
        'samples_breakpoint_info': target_data.get('samples_breakpoint_info', {}),
    }

    evaluation_results = evaluate_mia_attack(
        target_data,
        predictions,
        target_args=target_args,
        target_data_root=target_data_root,
        data_root=data_root,
    )

    return {'predictions': predictions, 'evaluation': evaluation_results}


def run_batch_mia_threshold_attack(
    args: Dict[str, Any],
    data_root: str = "../../save",
    checkpoint_dir: str = "./checkpoints",
    device: str = None,
    num_target_per_pattern: int = 333,
    seed: int = 42,
    debug: bool = False,
    print_config: bool = True,
    result_root: str = None,
) -> Dict[str, Any]:
    if result_root is None:
        result_root = args.get('result_root', './result')
    class Args:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)

    temp_args_dict = args.copy()
    if 'use_ppl' not in temp_args_dict:
        temp_args_dict['use_ppl'] = False
    if 'normalize_ppl' not in temp_args_dict:
        temp_args_dict['normalize_ppl'] = False
    if 'debug' not in temp_args_dict:
        temp_args_dict['debug'] = debug
    temp_args = Args(temp_args_dict)
    combinations = generate_all_combinations(temp_args)
    total_combinations = len(combinations)

    if print_config:
        print(f"\nResults will be saved to: {result_root}/<U_method>/<model>/<dataset>/<proportion>/<attack_method>/result.json")
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
            'use_ppl': args.get('use_ppl', False),
            'normalize_ppl': args.get('normalize_ppl', False),
            'attack_method': args.get('attack_method', 'mia_threshold'),
        }
        target_args = {
            'U_method': base_args['U_method'],
            'net_name': test_net,
            'dataset_name': test_dataset,
            'proportion_of_group_unlearn': test_proportion,
            'trial': base_args['trial'],
            'use_ppl': args.get('use_ppl', False),
            'normalize_ppl': args.get('normalize_ppl', False),
            'attack_method': args.get('attack_method', 'mia_threshold'),
            'enable_k_tolerance_eval': args.get('enable_k_tolerance_eval', False),
        }
        test_same_as_train = base_args.get('test_same_as_train', False)
        target_data_root = data_root if test_same_as_train else base_args.get('target_data_root', data_root)

        try:
            results = run_mia_threshold_attack(
                args=shadow_args,
                data_root=data_root,
                checkpoint_dir=checkpoint_dir,
                device=device,
                num_target_per_pattern=num_target_per_pattern,
                num_trials=1,
                seed=seed,
                debug=debug,
                print_config=False,
                shadow_args=shadow_args,
                target_args=target_args,
                target_data_root=target_data_root,
            )
            evaluation = results.get('evaluation', {})
        except Exception as e:
            print(f"\n✗ Combination failed: {e}")
            evaluation = {}

        precision = evaluation.get('precision', -1.0)
        recall = evaluation.get('recall', -1.0)
        f1 = evaluation.get('f1', -1.0)
        save_eval_result_json(
            result_root,
            base_args['U_method'],
            test_net,
            test_dataset,
            test_proportion,
            'TW_MIA',
            {
                'precision': precision if precision >= 0 else 0.0,
                'recall': recall if recall >= 0 else 0.0,
                'f1': f1 if f1 >= 0 else 0.0,
                'I_Pre': evaluation.get('I_Pre', 0.0),
                'I_Rec': evaluation.get('I_Rec', 0.0),
                'I_F1': evaluation.get('I_F1', 0.0),
            },
        )
        all_results.append({'combination': combination, 'evaluation': evaluation})

        print(f"\n✓ Combination completed:")
        print(f"  Overlap - T-Pre/T-Rec/T-F1: {precision:.4f}/{recall:.4f}/{f1:.4f}")

    valid_results = [r for r in all_results if r['evaluation'].get('precision', -1) >= 0]
    failed_results = [r for r in all_results if r['evaluation'].get('precision', -1) < 0]
    print("\n" + "=" * 80)
    print("Batch MIA threshold attack completed")
    print("=" * 80)
    print(f"Success: {len(valid_results)}/{total_combinations}, Failed: {len(failed_results)}/{total_combinations}")
    if valid_results:
        avg_f1 = np.mean([r['evaluation'].get('f1', 0) for r in valid_results])
        print(f"Average T-F1: {avg_f1:.4f}")
    print(f"\nResults saved to: {result_root}/<U_method>/<model>/<dataset>/<proportion>/TW_MIA/result.json")
    print("=" * 80)

    return {'all_results': all_results}


def main():
    import sys
    import os as _os
    _root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '../..'))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from parameter_parser import parameter_parser
    from attack.baseline.baseline import run_baseline_attack
    args = parameter_parser()
    args['attack_method'] = 'TW_MIA'
    run_baseline_attack(args)


if __name__ == '__main__':
    main()

