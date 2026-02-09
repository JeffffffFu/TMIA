import os
import glob
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from .data_loader import DataLoader, load_confidence_at_timestamp, load_breakpoint_confidence_deltas


class DataProcessor:
    def __init__(self):
        self.data_loader = DataLoader()

    def extract_ct_features(self, posteriors: np.ndarray, labels: np.ndarray) -> np.ndarray:
        if posteriors.ndim == 2:
            features = []
            for posterior, label in zip(posteriors, labels):
                features.append([posterior[int(label)]])
            features = np.array(features)
        
        elif posteriors.ndim == 3:
            num_samples, num_timesteps, num_classes = posteriors.shape
            features = []
            for i in range(num_samples):
                sample_features = []
                for t in range(num_timesteps):
                    label = int(labels[i, t])
                    sample_features.append(posteriors[i, t, label])
                features.append(sample_features)
            features = np.array(features)
        else:
            raise ValueError(f"Unsupported dimension: {posteriors.ndim}")
        
        return features
    
    def extract_ct_features_from_sequences(self, 
                                          outputs: np.ndarray,
                                          labels: np.ndarray) -> np.ndarray:
        num_samples, num_timesteps, num_classes = outputs.shape
        
        sample_indices = np.arange(num_samples)[:, np.newaxis]
        time_indices = np.arange(num_timesteps)[np.newaxis, :]
        ct_sequences = outputs[sample_indices, time_indices, labels.astype(int)]
        
        return ct_sequences
    
    def load_and_extract_ct_features(self,
                                    args: Dict[str, Any],
                                    indices: Optional[List[int]] = None,
                                    data_root: str = "save",
                                    use_ppl: bool = False,
                                    normalize_ppl: bool = False,
                                    normalization_params: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        data = self.data_loader.load_data_by_indices(
            args=args, 
            indices=indices, 
            data_root=data_root,
            use_ppl=use_ppl
        )
        
        if use_ppl:
            ppl_features = data['outputs'].squeeze(axis=2)
            if normalize_ppl:
                if normalization_params is None:
                    normalized_ppl, norm_params = self.normalize_ppl_features(
                        ppl_features, 
                        method='zscore'
                    )
                else:
                    method = normalization_params.get('method', 'zscore')
                    normalized_ppl, _ = self.normalize_ppl_features(
                        ppl_features,
                        method=method,
                        mean=normalization_params.get('mean'),
                        std=normalization_params.get('std'),
                        min_val=normalization_params.get('min'),
                        max_val=normalization_params.get('max')
                    )
                ppl_features = normalized_ppl
                result = {
                    'ct_features': ppl_features,
                    'labels': data['labels'],
                    'outputs': data['outputs'],
                    'indices': data['indices'],
                    'timesteps': data['timesteps'],
                    'normalization_params': norm_params if normalization_params is None else normalization_params
                }
            else:
                result = {
                    'ct_features': ppl_features,
                    'labels': data['labels'],
                    'outputs': data['outputs'],
                    'indices': data['indices'],
                    'timesteps': data['timesteps']
                }
            
            return result
        else:
            ct_features = self.extract_ct_features_from_sequences(
                outputs=data['outputs'],
                labels=data['labels']
            )
            return {
                'ct_features': ct_features,
                'labels': data['labels'],
                'outputs': data['outputs'],
                'indices': data['indices'],
                'timesteps': data['timesteps']
            }
    
    def convert_status_to_changepoints(self, 
                                      status_sequence: np.ndarray) -> List[Dict[str, Any]]:
        changepoints = []
        
        for i in range(1, len(status_sequence)):
            prev_status = int(status_sequence[i-1])
            curr_status = int(status_sequence[i])
            
            if prev_status == 0 and curr_status == 1:
                changepoints.append({
                    'position': i,
                    'type': 'insert',
                    'transition': '0->1'
                })
            elif prev_status == 1 and curr_status == 0:
                changepoints.append({
                    'position': i,
                    'type': 'remove',
                    'transition': '1->0'
                })
        
        return changepoints
    
    def convert_batch_status_to_changepoints(self,
                                            status_sequences: np.ndarray) -> List[List[Dict[str, Any]]]:
        all_changepoints = []
        
        for status_seq in status_sequences:
            changepoints = self.convert_status_to_changepoints(status_seq)
            all_changepoints.append(changepoints)
        
        return all_changepoints
    
    def normalize_ppl_features(self,
                               ppl_features: np.ndarray,
                               method: str = 'zscore',
                               mean: Optional[float] = None,
                               std: Optional[float] = None,
                               min_val: Optional[float] = None,
                               max_val: Optional[float] = None) -> Tuple[np.ndarray, Dict[str, float]]:
        if method == 'zscore':
            if mean is None or std is None:
                mean = float(np.mean(ppl_features))
                std = float(np.std(ppl_features))
            
            if std == 0:
                normalized = ppl_features.copy()
            else:
                normalized = (ppl_features - mean) / std
            
            params = {
                'method': 'zscore',
                'mean': mean,
                'std': std
            }
            
        elif method == 'minmax':
            if min_val is None or max_val is None:
                min_val = float(np.min(ppl_features))
                max_val = float(np.max(ppl_features))
            
            if max_val == min_val:
                normalized = ppl_features.copy()
            else:
                normalized = (ppl_features - min_val) / (max_val - min_val)
            
            params = {
                'method': 'minmax',
                'min': min_val,
                'max': max_val
            }
            
        else:
            raise ValueError(f"Unsupported normalization method: {method}. Supported: 'zscore', 'minmax'")
        
        return normalized, params


def build_path_and_find_timestamps(args, data_root=None):
    U_method = args.get('U_method')
    net_name = args.get('net_name')
    dataset_name = args.get('dataset_name')
    proportion = args.get('proportion_of_group_unlearn')
    
    if data_root is None:
        data_root = args.get('data_root')
    if data_root is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        data_root = os.path.join(project_root, "save")
    
    if 'shadow_or_target' in args:
        model_type = args.get('shadow_or_target', 'target')
    elif args.get('is_shadow', False):
        model_type = 'shadow'
    else:
        model_type = args.get('model_type', 'target')
    
    trial = args.get('trial', 0)
    
    if U_method is None or net_name is None or dataset_name is None or proportion is None:
        raise ValueError("args must contain 'U_method', 'net_name', 'dataset_name', 'proportion_of_group_unlearn'")
    
    save_path = os.path.join(data_root, U_method, net_name, dataset_name, 
                             str(proportion), model_type, str(trial))
    
    if not os.path.exists(save_path):
        raise ValueError(f"Save path does not exist: {save_path}")
    timestamp_dirs = []
    
    for item in glob.glob(os.path.join(save_path, "timestamp_*")):
        if os.path.isdir(item):
            timestamp_dirs.append(item)
    
    if len(timestamp_dirs) == 0:
        for item in os.listdir(save_path):
            item_path = os.path.join(save_path, item)
            if os.path.isdir(item_path) and item.isdigit():
                timestamp_dirs.append(item_path)
    
    def extract_timestamp_num(path):
        dir_name = os.path.basename(path)
        if dir_name.startswith("timestamp_"):
            return int(dir_name.split("_")[1])
        elif dir_name.isdigit():
            return int(dir_name)
        else:
            return 0
    
    timestamp_dirs = sorted(timestamp_dirs, key=extract_timestamp_num)
    
    if len(timestamp_dirs) == 0:
        raise ValueError(f"No timestamp directory found: {save_path}")
    return save_path, timestamp_dirs


def filter_breakpoint_deltas_by_threshold(delta_data: Dict[str, Any],
                                          insert_threshold: float,
                                          remove_threshold: float) -> Dict[str, Any]:
    result = {}
    
    if 'insert_deltas' in delta_data:
        insert_deltas = delta_data['insert_deltas']
        insert_sample_indices = delta_data['insert_sample_indices']
        insert_timestamps = delta_data['insert_timestamps']
        
        filtered_insert_deltas = []
        filtered_insert_sample_indices = []
        filtered_insert_timestamps = []
        
        for delta, sample_idx, timestamp in zip(insert_deltas, insert_sample_indices, insert_timestamps):
            if delta > insert_threshold:
                filtered_insert_deltas.append(delta)
                filtered_insert_sample_indices.append(sample_idx)
                filtered_insert_timestamps.append(timestamp)
        
        result['insert_deltas'] = filtered_insert_deltas
        result['insert_sample_indices'] = filtered_insert_sample_indices
        result['insert_timestamps'] = filtered_insert_timestamps
    if 'remove_deltas' in delta_data:
        remove_deltas = delta_data['remove_deltas']
        remove_sample_indices = delta_data['remove_sample_indices']
        remove_timestamps = delta_data['remove_timestamps']
        
        filtered_remove_deltas = []
        filtered_remove_sample_indices = []
        filtered_remove_timestamps = []
        
        for delta, sample_idx, timestamp in zip(remove_deltas, remove_sample_indices, remove_timestamps):
            if delta < remove_threshold:
                filtered_remove_deltas.append(delta)
                filtered_remove_sample_indices.append(sample_idx)
                filtered_remove_timestamps.append(timestamp)
        
        result['remove_deltas'] = filtered_remove_deltas
        result['remove_sample_indices'] = filtered_remove_sample_indices
        result['remove_timestamps'] = filtered_remove_timestamps
    return result


def compute_delta_percentile(deltas: List[float], percentile: float) -> float:
    if len(deltas) == 0:
        return float('nan')
    return float(np.percentile(deltas, percentile))


def compute_breakpoint_confidence_delta_percentile(args: Dict[str, Any],
                                                   num_insert: int,
                                                   num_remove: int,
                                                   percentile: float = 50.0,
                                                   num_samples: Optional[int] = None,
                                                   data_root: Optional[str] = None,
                                                   random_seed: Optional[int] = None,
                                                   use_ppl: bool = False) -> Tuple[float, float]:
    result = load_breakpoint_confidence_deltas(
        args=args,
        num_insert=num_insert,
        num_remove=num_remove,
        breakpoint_type='both',
        num_samples=num_samples,
        data_root=data_root,
        random_seed=random_seed,
        use_ppl=use_ppl
    )
    
    insert_deltas = result['insert_deltas']
    remove_deltas = result['remove_deltas']
    
    percentile_insert_delta = compute_delta_percentile(insert_deltas, percentile)
    percentile_remove_delta = compute_delta_percentile(remove_deltas, percentile)
    return percentile_insert_delta, percentile_remove_delta


def timestamp_to_index(timestamp: int, timestamp_dirs: List[str]) -> Optional[int]:
    for idx, timestamp_dir in enumerate(timestamp_dirs):
        dir_name = os.path.basename(timestamp_dir)
        try:
            if dir_name.startswith("timestamp_"):
                timestamp_num = int(dir_name.split("_")[1])
            elif dir_name.isdigit():
                timestamp_num = int(dir_name)
            else:
                continue
            
            if timestamp_num == timestamp:
                return idx
        except (ValueError, IndexError):
            continue
    
    return None


def extract_breakpoints_from_status_sequence(status_seq: np.ndarray) -> List[Tuple[int, int]]:
    breakpoints = []
    
    for idx in range(len(status_seq)):
        status = status_seq[idx]
        if status == 2:
            breakpoints.append((idx, 1))
        elif status == 3:
            breakpoints.append((idx, 2))
    
    return breakpoints


def convert_mia_predictions_to_tolerance_format(
    true_status_sequences: List[np.ndarray],
    pred_status_sequences: List[np.ndarray],
    sample_indices: List[int]
) -> Dict[int, List[Dict[str, Any]]]:
    processor = DataProcessor()
    predictions_by_sample = {}
    
    for i, (true_seq, pred_seq) in enumerate(zip(true_status_sequences, pred_status_sequences)):
        sample_idx = sample_indices[i] if i < len(sample_indices) else i
        true_seq = np.array(true_seq)
        pred_seq = np.array(pred_seq)
        
        min_len = min(len(true_seq), len(pred_seq))
        true_seq = true_seq[:min_len]
        pred_seq = pred_seq[:min_len]
        
        true_insert_positions = set()
        true_remove_positions = set()
        for ts_idx in range(min_len):
            if true_seq[ts_idx] == 2:
                true_insert_positions.add(ts_idx)
            elif true_seq[ts_idx] == 3:
                true_remove_positions.add(ts_idx)
        
        pred_member_seq = (pred_seq == 1).astype(int)
        pred_changepoints = processor.convert_status_to_changepoints(pred_member_seq)
        pred_insert_positions = {cp['position'] for cp in pred_changepoints if cp['type'] == 'insert'}
        pred_remove_positions = {cp['position'] for cp in pred_changepoints if cp['type'] == 'remove'}
        
        sample_predictions = []
        for ts_idx in range(min_len):
            true_label = 0
            if ts_idx in true_insert_positions:
                true_label = 1
            elif ts_idx in true_remove_positions:
                true_label = 2
            
            pred_label = 0
            if ts_idx in pred_insert_positions:
                pred_label = 1
            elif ts_idx in pred_remove_positions:
                pred_label = 2
            
            sample_predictions.append({
                'sample_index': sample_idx,
                'timestamp_index': ts_idx,
                'predicted_label': pred_label,
                'true_label': true_label
            })
        
        predictions_by_sample[sample_idx] = sample_predictions
    
    return predictions_by_sample


def convert_delta_predictions_to_tolerance_format(
    predicted_breakpoints_by_sample: Dict[int, List[Tuple[int, int]]],
    true_status_sequences: Dict[int, np.ndarray],
    all_sample_indices: List[int],
    num_timestamps: int
) -> Dict[int, List[Dict[str, Any]]]:
    predictions_by_sample = {}
    
    for sample_idx in all_sample_indices:
        if sample_idx not in true_status_sequences:
            continue
        
        true_status_seq = true_status_sequences[sample_idx]
        
        pred_breakpoints = predicted_breakpoints_by_sample.get(sample_idx, [])
        pred_insert_set = {ts for ts, bp_type in pred_breakpoints if bp_type == 1}
        pred_remove_set = {ts for ts, bp_type in pred_breakpoints if bp_type == 2}
        
        true_insert_set = set()
        true_remove_set = set()
        for ts_idx in range(len(true_status_seq)):
            if true_status_seq[ts_idx] == 2:
                true_insert_set.add(ts_idx)
            elif true_status_seq[ts_idx] == 3:
                true_remove_set.add(ts_idx)
        
        sample_predictions = []
        num_ts = min(len(true_status_seq), num_timestamps)
        
        for ts_idx in range(num_ts):
            true_label = 0
            if ts_idx in true_insert_set:
                true_label = 1
            elif ts_idx in true_remove_set:
                true_label = 2
            
            pred_label = 0
            if ts_idx in pred_insert_set:
                pred_label = 1
            elif ts_idx in pred_remove_set:
                pred_label = 2
            
            sample_predictions.append({
                'sample_index': sample_idx,
                'timestamp_index': ts_idx,
                'predicted_label': pred_label,
                'true_label': true_label
            })
        
        predictions_by_sample[sample_idx] = sample_predictions
    
    return predictions_by_sample

