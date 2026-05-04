import os
import math
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple


class DataLoader:
    def __init__(self):
        pass

    def load_data_by_indices(self,
                            args: Dict[str, Any],
                            indices: Optional[List[int]] = None,
                            data_root: str = "save",
                            use_ppl: bool = False) -> Dict[str, Any]:
        required_keys = ['U_method', 'net_name', 'dataset_name', 
                        'proportion_of_group_unlearn', 'shadow_or_target', 'trial']
        for key in required_keys:
            if key not in args:
                raise ValueError(f"args missing required parameter: {key}")
        
        base_dir = os.path.join(
            os.getcwd(),
            data_root,
            args['U_method'],
            args['net_name'],
            args['dataset_name'],
            str(args['proportion_of_group_unlearn']),
            args['shadow_or_target'],
            str(args['trial'])
        )
        
        if not os.path.exists(base_dir):
            raise FileNotFoundError(f"Directory does not exist: {base_dir}")
        
        available_timesteps = []
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path) and item.startswith('timestamp_'):
                try:
                    timestep_num = int(item.replace('timestamp_', ''))
                    available_timesteps.append(timestep_num)
                except ValueError:
                    continue
        
        available_timesteps.sort()
        
        if not available_timesteps:
            raise ValueError(f"No valid timestep data found in directory: {base_dir}")
        
        first_timestep_dir = os.path.join(base_dir, f'timestamp_{available_timesteps[0]}')
        sample_indices_path = os.path.join(first_timestep_dir, 'sample_indices.npy')
        
        if not os.path.exists(sample_indices_path):
            raise FileNotFoundError(f"Sample indices file not found: {sample_indices_path}")
        
        all_sample_indices = np.load(sample_indices_path)
        
        if indices is None:
            loaded_indices = all_sample_indices.tolist()
        else:
            loaded_indices = [idx for idx in indices if idx in all_sample_indices]
        
        if not loaded_indices:
            raise ValueError("No matching sample indices found")
        
        all_sequences_outputs = []
        all_sequences_labels = []
        
        for sample_idx in loaded_indices:
            sample_outputs = []
            sample_labels = []
            
            for timestep in available_timesteps:
                timestep_dir = os.path.join(base_dir, f'timestamp_{timestep}')
                
                try:
                    outputs_path = os.path.join(timestep_dir, 'outputs_current.npy')
                    sample_indices_path = os.path.join(timestep_dir, 'sample_indices.npy')
                    
                    if use_ppl:
                        required_files = [outputs_path, sample_indices_path]
                    else:
                        labels_path = os.path.join(timestep_dir, 'labels.npy')
                        required_files = [outputs_path, labels_path, sample_indices_path]
                    
                    if not all(os.path.exists(p) for p in required_files):
                        if len(sample_outputs) > 0:
                            sample_outputs.append(np.zeros_like(sample_outputs[0]))
                            if not use_ppl:
                                sample_labels.append(-1)
                        continue
                    
                    outputs = np.load(outputs_path)
                    sample_indices_t = np.load(sample_indices_path)
                    
                    if not use_ppl:
                        labels = np.load(labels_path)
                    
                    pos = np.where(sample_indices_t == sample_idx)[0]
                    
                    if len(pos) > 0:
                        if use_ppl:
                            ppl_value = outputs[pos[0], 0] if outputs.ndim > 1 else outputs[pos[0]]
                            sample_outputs.append(np.array([ppl_value]))
                        else:
                            sample_outputs.append(outputs[pos[0]])
                            sample_labels.append(labels[pos[0]])
                    else:
                        if len(sample_outputs) > 0:
                            if use_ppl:
                                sample_outputs.append(np.array([0.0]))
                            else:
                                sample_outputs.append(np.zeros_like(sample_outputs[0]))
                                sample_labels.append(-1)
                
                except Exception as e:
                    if len(sample_outputs) > 0:
                        if use_ppl:
                            sample_outputs.append(np.array([0.0]))
                        else:
                            sample_outputs.append(np.zeros_like(sample_outputs[0]))
                            sample_labels.append(-1)
                    continue
            
            if len(sample_outputs) == len(available_timesteps):
                all_sequences_outputs.append(np.array(sample_outputs))
                if not use_ppl:
                    all_sequences_labels.append(np.array(sample_labels))
            else:
                pass
        if not all_sequences_outputs:
            raise ValueError("Failed to load any complete time series")
        
        outputs_array = np.stack(all_sequences_outputs, axis=0)
        if use_ppl:
            labels_array = None
        else:
            labels_array = np.stack(all_sequences_labels, axis=0)
        return {
            'outputs': outputs_array,
            'labels': labels_array,
            'indices': loaded_indices,
            'timesteps': available_timesteps
        }
    
    def get_indices_by_pattern(self,
                               args: Dict[str, Any],
                               num_insert: int,
                               num_remove: int,
                            pattern_type: str = 'any',
                              data_root: str = "save") -> List[int]:
        base_dir = os.path.join(
            os.getcwd(),
            data_root,
            args['U_method'],
            args['net_name'],
            args['dataset_name'],
            str(args['proportion_of_group_unlearn']),
            args['shadow_or_target'],
            str(args['trial'])
        )
        
        if not os.path.exists(base_dir):
            raise FileNotFoundError(f"Directory does not exist: {base_dir}")
        
        status_file = os.path.join(base_dir, 'sample_status_converted.npy')
        if not os.path.exists(status_file):
            raise FileNotFoundError(f"Status file does not exist: {status_file}")
        
        converted_status_array = np.load(status_file)
        total_samples, num_timestamps = converted_status_array.shape
        num_unseen_per_sample = np.sum(converted_status_array == 0, axis=1)
        num_retain_per_sample = np.sum(converted_status_array == 1, axis=1)
        num_insert_per_sample = np.sum(converted_status_array == 2, axis=1)
        num_remove_per_sample = np.sum(converted_status_array == 3, axis=1)
        num_breakpoints_per_sample = num_insert_per_sample + num_remove_per_sample
        
        if pattern_type == 'unseen':
            if num_insert != 0 or num_remove != 0:
                pass
            mask = (num_breakpoints_per_sample == 0) & (num_unseen_per_sample == num_timestamps)
            pattern_desc = "All Unseen (no breakpoint)"
            
        elif pattern_type == 'retain':
            if num_insert != 0 or num_remove != 0:
                pass
            mask = (num_breakpoints_per_sample == 0) & (num_retain_per_sample == num_timestamps)
            pattern_desc = "All Retain (no breakpoint)"
            
        elif pattern_type == 'any':
            mask = (num_insert_per_sample == num_insert) & (num_remove_per_sample == num_remove)
            pattern_desc = f"{num_insert} insert, {num_remove} remove breakpoints"
            
        else:
            raise ValueError(f"Unsupported pattern_type: {pattern_type}. Supported: 'any', 'unseen', 'retain'")
        
        matching_indices = np.where(mask)[0].tolist()
        return matching_indices


def load_confidence_at_timestamp(sample_idx: int, timestamp_dir: str, use_ppl: bool = False) -> Optional[float]:
    outputs_file = os.path.join(timestamp_dir, "outputs_current.npy")
    indices_file = os.path.join(timestamp_dir, "sample_indices.npy")
    labels_file = os.path.join(timestamp_dir, "labels.npy")
    
    if not os.path.exists(outputs_file):
        return None
    
    outputs = np.load(outputs_file)
    
    if os.path.exists(indices_file):
        indices = np.load(indices_file)
    else:
        indices = np.arange(len(outputs))
    
    sample_pos = np.where(indices == sample_idx)[0]
    if len(sample_pos) == 0:
        return None
    
    sample_pos = sample_pos[0]
    
    sample_output = outputs[sample_pos]
    
    if sample_output.ndim == 0 or (sample_output.ndim > 0 and sample_output.shape[0] == 1):
        if use_ppl:
            ppl_value = float(sample_output) if sample_output.ndim == 0 else float(sample_output[0])
            if ppl_value <= 0:
                return None
            return math.log(ppl_value)
        else:
            return float(sample_output) if sample_output.ndim == 0 else float(sample_output[0])
    
    if not os.path.exists(labels_file):
        return float(np.max(sample_output))
    
    labels_data = np.load(labels_file)
    true_label = int(labels_data[sample_pos])
    
    if true_label < 0 or true_label >= len(sample_output):
        return float(np.max(sample_output))
    
    return float(sample_output[true_label])


def load_breakpoint_confidence_deltas(args: Dict[str, Any],
                                      num_insert: int,
                                      num_remove: int,
                                      breakpoint_type: str = 'both',
                                      num_samples: Optional[int] = None,
                                      data_root: Optional[str] = None,
                                      random_seed: Optional[int] = None,
                                      use_ppl: bool = False) -> Dict[str, Any]:
    import random
    from .data_processor import build_path_and_find_timestamps
    
    if breakpoint_type not in ['insert', 'remove', 'both']:
        raise ValueError(f"breakpoint_type must be 'insert', 'remove' or 'both', got: {breakpoint_type}")
    
    save_path, timestamp_dirs = build_path_and_find_timestamps(args, data_root=data_root)
    
    breakpoint_file = os.path.join(save_path, "breakpoint_indices.npy")
    if not os.path.exists(breakpoint_file):
        raise FileNotFoundError(f"Breakpoint file not found: {breakpoint_file}")
    
    breakpoint_indices_dict = np.load(breakpoint_file, allow_pickle=True).item()
    matching_samples = []
    for idx, breakpoint_info in breakpoint_indices_dict.items():
        insert_count = len(breakpoint_info.get('insert', []))
        remove_count = len(breakpoint_info.get('remove', []))
        
        if insert_count == num_insert and remove_count == num_remove:
            matching_samples.append(idx)
    
    if len(matching_samples) == 0:
        raise ValueError(f"No samples found with {num_insert} insert, {num_remove} remove")
    if num_samples is not None and num_samples < len(matching_samples):
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        selected_samples = random.sample(matching_samples, num_samples)
    else:
        selected_samples = matching_samples
    timestamp_to_idx = {}
    for idx, timestamp_dir in enumerate(timestamp_dirs):
        dir_name = os.path.basename(timestamp_dir)
        if dir_name.startswith("timestamp_"):
            timestamp_num = int(dir_name.split("_")[1])
        elif dir_name.isdigit():
            timestamp_num = int(dir_name)
        else:
            continue
        timestamp_to_idx[timestamp_num] = idx
    
    insert_deltas = []
    insert_sample_indices = []
    insert_timestamps = []
    remove_deltas = []
    remove_sample_indices = []
    remove_timestamps = []
    
    for sample_idx in selected_samples:
        breakpoint_info = breakpoint_indices_dict.get(sample_idx, {})
        insert_timestamps_list = breakpoint_info.get('insert', [])
        remove_timestamps_list = breakpoint_info.get('remove', [])
        
        if breakpoint_type in ['insert', 'both']:
            for insert_ts in insert_timestamps_list:
                if insert_ts not in timestamp_to_idx:
                    continue
                insert_idx = timestamp_to_idx[insert_ts]
                if insert_idx == 0:
                    continue
                prev_idx = insert_idx - 1
                prev_ts_dir = timestamp_dirs[prev_idx]
                insert_ts_dir = timestamp_dirs[insert_idx]
                
                prev_conf = load_confidence_at_timestamp(sample_idx, prev_ts_dir, use_ppl=use_ppl)
                insert_conf = load_confidence_at_timestamp(sample_idx, insert_ts_dir, use_ppl=use_ppl)
                
                if prev_conf is not None and insert_conf is not None:
                    delta = insert_conf - prev_conf
                    insert_deltas.append(delta)
                    insert_sample_indices.append(sample_idx)
                    insert_timestamps.append(insert_ts)
        
        if breakpoint_type in ['remove', 'both']:
            for remove_ts in remove_timestamps_list:
                if remove_ts not in timestamp_to_idx:
                    continue
                remove_idx = timestamp_to_idx[remove_ts]
                if remove_idx == 0:
                    continue
                
                prev_idx = remove_idx - 1
                prev_ts_dir = timestamp_dirs[prev_idx]
                remove_ts_dir = timestamp_dirs[remove_idx]
                
                prev_conf = load_confidence_at_timestamp(sample_idx, prev_ts_dir, use_ppl=use_ppl)
                remove_conf = load_confidence_at_timestamp(sample_idx, remove_ts_dir, use_ppl=use_ppl)
                
                if prev_conf is not None and remove_conf is not None:
                    delta = remove_conf - prev_conf
                    remove_deltas.append(delta)
                    remove_sample_indices.append(sample_idx)
                    remove_timestamps.append(remove_ts)
    
    if breakpoint_type == 'both':
        if len(insert_deltas) == 0 and len(remove_deltas) == 0:
            raise ValueError("No valid breakpoint confidence deltas found")
    elif breakpoint_type == 'insert':
        if len(insert_deltas) == 0:
            raise ValueError("No valid insert breakpoint confidence deltas found")
    elif breakpoint_type == 'remove':
        if len(remove_deltas) == 0:
            raise ValueError("No valid remove breakpoint confidence deltas found")
    
    if breakpoint_type == 'insert':
        return {
            'insert_deltas': insert_deltas,
            'insert_sample_indices': insert_sample_indices,
            'insert_timestamps': insert_timestamps
        }
    elif breakpoint_type == 'remove':
        return {
            'remove_deltas': remove_deltas,
            'remove_sample_indices': remove_sample_indices,
            'remove_timestamps': remove_timestamps
        }
    else:
        return {
            'insert_deltas': insert_deltas,
            'insert_sample_indices': insert_sample_indices,
            'insert_timestamps': insert_timestamps,
            'remove_deltas': remove_deltas,
            'remove_sample_indices': remove_sample_indices,
            'remove_timestamps': remove_timestamps
        }


def compute_deltas_for_all_timestamps(args: Dict[str, Any],
                                       sample_indices: List[int],
                                       num_timestamps: int,
                                       data_root: Optional[str] = None,
                                       use_ppl: bool = False) -> Dict[str, Any]:
    from .data_processor import build_path_and_find_timestamps
    import math
    
    save_path, timestamp_dirs = build_path_and_find_timestamps(args, data_root=data_root)
    
    if len(timestamp_dirs) != num_timestamps:
        num_timestamps = len(timestamp_dirs)
    insert_deltas = []
    insert_sample_indices = []
    insert_timestamps = []
    remove_deltas = []
    remove_sample_indices = []
    remove_timestamps = []
    for sample_idx in sample_indices:
        for ts_idx in range(1, num_timestamps):
            prev_ts_dir = timestamp_dirs[ts_idx - 1]
            current_ts_dir = timestamp_dirs[ts_idx]
            
            prev_conf = load_confidence_at_timestamp(sample_idx, prev_ts_dir, use_ppl=use_ppl)
            current_conf = load_confidence_at_timestamp(sample_idx, current_ts_dir, use_ppl=use_ppl)
            
            if prev_conf is not None and current_conf is not None:
                delta = current_conf - prev_conf
                
                dir_name = os.path.basename(current_ts_dir)
                if dir_name.startswith("timestamp_"):
                    timestamp_value = int(dir_name.split("_")[1])
                elif dir_name.isdigit():
                    timestamp_value = int(dir_name)
                else:
                    continue
                
                if delta > 0:
                    insert_deltas.append(delta)
                    insert_sample_indices.append(sample_idx)
                    insert_timestamps.append(timestamp_value)
                elif delta < 0:
                    remove_deltas.append(delta)
                    remove_sample_indices.append(sample_idx)
                    remove_timestamps.append(timestamp_value)
    return {
        'insert_deltas': insert_deltas,
        'insert_sample_indices': insert_sample_indices,
        'insert_timestamps': insert_timestamps,
        'remove_deltas': remove_deltas,
        'remove_sample_indices': remove_sample_indices,
        'remove_timestamps': remove_timestamps
    }


def _softmax_by_row(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    mx = np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(logits - mx)
    return exp / (np.sum(exp, axis=axis, keepdims=True) + 1e-30)


def load_outputs_and_labels_for_mia_threshold(
    shadow_args: Dict[str, Any],
    target_args: Dict[str, Any],
    shadow_indices: List[int],
    target_indices: List[int],
    data_root: str = "save",
    target_data_root: Optional[str] = None,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray],
          Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], int]:
    if target_data_root is None:
        target_data_root = data_root

    loader = DataLoader()

    def _load_and_split(args: Dict[str, Any], indices: List[int], root: str):
        base_dir = os.path.join(
            os.getcwd(),
            root,
            args['U_method'],
            args['net_name'],
            args['dataset_name'],
            str(args['proportion_of_group_unlearn']),
            args['shadow_or_target'],
            str(args['trial']),
        )
        status_file = os.path.join(base_dir, 'sample_status_converted.npy')
        if not os.path.exists(status_file):
            raise FileNotFoundError(f"Status file does not exist: {status_file}")
        status = np.load(status_file)

        data = loader.load_data_by_indices(
            args=args,
            indices=indices,
            data_root=root,
            use_ppl=False,
        )
        outputs = data['outputs']
        labels = data['labels']
        loaded_indices = data['indices']

        status_sub = status[np.array(loaded_indices)]
        member = np.isin(status_sub, [1, 2])

        N, T, C = outputs.shape
        outputs_flat = outputs.reshape(-1, C)
        labels_flat = labels.reshape(-1).astype(np.int32)
        member_flat = member.reshape(-1)

        valid = labels_flat >= 0
        outputs_flat = outputs_flat[valid]
        labels_flat = labels_flat[valid]
        member_flat = member_flat[valid]

        if np.any(outputs_flat > 1.01) or np.any(outputs_flat < -0.01):
            outputs_flat = _softmax_by_row(outputs_flat, axis=1)

        train_perf = (outputs_flat[member_flat], labels_flat[member_flat])
        test_perf = (outputs_flat[~member_flat], labels_flat[~member_flat])
        return train_perf, test_perf, C

    s_tr, s_te, num_classes = _load_and_split(shadow_args, shadow_indices, data_root)
    t_tr, t_te, _ = _load_and_split(target_args, target_indices, target_data_root)

    return s_tr, s_te, t_tr, t_te, num_classes


def load_ppl_member_nonmember_scores(
    args: Dict[str, Any],
    indices: List[int],
    data_root: str = "save",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flattened per-timestep scores for PPL-based TW-MIA: -log(PPL).
    Larger values are more member-like (lower perplexity). Aligns with
    delta path using log(PPL) as confidence direction.
    """
    loader = DataLoader()
    base_dir = os.path.join(
        os.getcwd(),
        data_root,
        args['U_method'],
        args['net_name'],
        args['dataset_name'],
        str(args['proportion_of_group_unlearn']),
        args['shadow_or_target'],
        str(args['trial']),
    )
    status_file = os.path.join(base_dir, 'sample_status_converted.npy')
    if not os.path.exists(status_file):
        raise FileNotFoundError(f"Status file does not exist: {status_file}")
    status = np.load(status_file)

    data = loader.load_data_by_indices(
        args=args,
        indices=indices,
        data_root=data_root,
        use_ppl=True,
    )
    outputs = data['outputs']
    loaded_indices = data['indices']
    status_sub = status[np.array(loaded_indices)]
    member = np.isin(status_sub, [1, 2])

    if outputs.ndim != 3 or outputs.shape[-1] != 1:
        raise ValueError(
            f"PPL TW-MIA expects outputs shape (N, T, 1), got {outputs.shape}"
        )

    ppl = outputs.reshape(-1)
    member_flat = member.reshape(-1)
    valid = np.isfinite(ppl) & (ppl > 0)
    ppl = ppl[valid]
    member_flat = member_flat[valid]
    scores = -np.log(np.maximum(ppl, 1e-30))
    return scores[member_flat], scores[~member_flat]


def load_training_data(args: Dict[str, Any],
                       num_unseen: int = 100,
                       num_retain: int = 100,
                       num_per_pattern: int = 50,
                       data_root: str = "../../save",
                       use_ppl: bool = False) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    from .data_processor import DataProcessor
    from .data_processor import build_path_and_find_timestamps
    from attack.TMIA.get_training_data import select_test_samples_by_pattern

    loader = DataLoader()
    processor = DataProcessor()

    base_dir = os.path.join(
        os.getcwd(),
        data_root,
        args['U_method'],
        args['net_name'],
        args['dataset_name'],
        str(args['proportion_of_group_unlearn']),
        args['shadow_or_target'],
        str(args['trial'])
    )
    status_file = os.path.join(base_dir, 'sample_status_converted.npy')
    ground_truth_status = np.load(status_file)

    _am = args.get('attack_method', '') or ''
    if _am in ('mia_threshold', 'TW_MIA'):
        save_path, timestamp_dirs = build_path_and_find_timestamps(args, data_root=data_root)
        ratio = args.get('training_target_member_ratio', 0.5)
        num_insert_only = args.get('training_num_insert_only', 0)
        num_remove_only = args.get('training_num_remove_only', 0)
        num_insert_remove = args.get('training_num_insert_remove', 20)
        if args.get('dataset_name', '').lower() == "svhn" and args.get('net_name', '').lower() == "simple_cnn":
            num_insert_remove = 50
        elif args.get('net_name', '').lower() == "mobilenet":
            ratio = 0.6
        elif args.get('net_name', '').lower() == "gpt2":
            num_insert_only = 100
            num_remove_only = 100
            num_insert_remove = 100
        selected_indices, _, _ = select_test_samples_by_pattern(
            save_path=save_path,
            timestamp_dirs=timestamp_dirs,
            num_insert_only=num_insert_only,
            num_remove_only=num_remove_only,
            num_insert_remove=num_insert_remove,
            random_seed=args.get('random_seed', None),
            balance_membership=args.get('balance_test_membership', True),
            target_member_ratio=ratio
        )
        if len(selected_indices) == 0:
            raise ValueError("No training samples selected for mia_threshold pattern-based sampling")

        selected_data = processor.load_and_extract_ct_features(
            args=args,
            indices=selected_indices,
            data_root=data_root,
            use_ppl=use_ppl
        )
        selected_ct = selected_data['ct_features']
        selected_labels = []
        for idx in selected_indices:
            sample_status = ground_truth_status[idx]
            sample_labels = np.isin(sample_status, [1, 2]).astype(np.int64)
            selected_labels.append(sample_labels)
        selected_labels = np.array(selected_labels)
        return selected_ct, selected_labels, selected_indices

    all_ct_features_list = []
    all_labels_list = []
    all_indices_list = []

    if num_unseen > 0:
        unseen_indices = loader.get_indices_by_pattern(
            args=args,
            num_insert=0,
            num_remove=0,
            pattern_type='unseen',
            data_root=data_root
        )
        actual_unseen = min(num_unseen, len(unseen_indices))
        unseen_indices = unseen_indices[:actual_unseen]
        unseen_data = processor.load_and_extract_ct_features(
            args=args,
            indices=unseen_indices,
            data_root=data_root,
            use_ppl=use_ppl
        )
        unseen_ct = unseen_data['ct_features']
        unseen_labels = np.zeros((len(unseen_indices), unseen_ct.shape[1]), dtype=np.int64)
        all_ct_features_list.append(unseen_ct)
        all_labels_list.append(unseen_labels)
        all_indices_list.extend(unseen_indices)

    if num_retain > 0:
        retain_indices = loader.get_indices_by_pattern(
            args=args,
            num_insert=0,
            num_remove=0,
            pattern_type='retain',
            data_root=data_root
        )
        actual_retain = min(num_retain, len(retain_indices))
        retain_indices = retain_indices[:actual_retain]
        retain_data = processor.load_and_extract_ct_features(
            args=args,
            indices=retain_indices,
            data_root=data_root,
            use_ppl=use_ppl
        )
        retain_ct = retain_data['ct_features']
        retain_labels = np.ones((len(retain_indices), retain_ct.shape[1]), dtype=np.int64)
        all_ct_features_list.append(retain_ct)
        all_labels_list.append(retain_labels)
        all_indices_list.extend(retain_indices)

    if num_per_pattern > 0:
        patterns = [
            {'name': '01', 'insert': 0, 'remove': 1},
            {'name': '10', 'insert': 1, 'remove': 0},
            {'name': '11', 'insert': 1, 'remove': 1}
        ]
        for pattern in patterns:
            pattern_indices = loader.get_indices_by_pattern(
                args=args,
                num_insert=pattern['insert'],
                num_remove=pattern['remove'],
                pattern_type='any',
                data_root=data_root
            )
            actual_num = min(num_per_pattern, len(pattern_indices))
            pattern_indices = pattern_indices[:actual_num]
            pattern_data = processor.load_and_extract_ct_features(
                args=args,
                indices=pattern_indices,
                data_root=data_root,
                use_ppl=use_ppl
            )
            pattern_ct = pattern_data['ct_features']
            pattern_labels = []
            for idx in pattern_indices:
                sample_status = ground_truth_status[idx]
                sample_labels = np.isin(sample_status, [1, 2]).astype(np.int64)
                pattern_labels.append(sample_labels)
            pattern_labels = np.array(pattern_labels)
            all_ct_features_list.append(pattern_ct)
            all_labels_list.append(pattern_labels)
            all_indices_list.extend(pattern_indices)

    if not all_ct_features_list:
        raise ValueError("All of num_unseen, num_retain, num_per_pattern are 0; cannot load any training data")

    all_ct_features = np.vstack(all_ct_features_list)
    all_labels = np.vstack(all_labels_list)
    return all_ct_features, all_labels, all_indices_list