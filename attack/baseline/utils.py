import os
import csv
import json
from typing import Dict, Any, List


def save_eval_result_json(
    result_root: str,
    U_method: str,
    model_name: str,
    dataset_name: str,
    proportion: Any,
    attack_method: str,
    metrics: Dict[str, float],
) -> str:
    def pct(v):
        return round(float(v) * 100, 2)
    out = {
        "T-Pre": pct(metrics.get("T-Pre", metrics.get("precision", 0.0))),
        "T-Rec": pct(metrics.get("T-Rec", metrics.get("recall", 0.0))),
        "T-F1": pct(metrics.get("T-F1", metrics.get("f1", 0.0))),
        "I-Pre": pct(metrics.get("I-Pre", metrics.get("I_Pre", 0.0))),
        "I-Rec": pct(metrics.get("I-Rec", metrics.get("I_Rec", 0.0))),
        "I-F1": pct(metrics.get("I-F1", metrics.get("I_F1", 0.0))),
    }
    result_root = os.path.abspath(os.path.normpath(str(result_root)))
    parts = [result_root, str(U_method), str(model_name), str(dataset_name), str(proportion), str(attack_method)]
    dirpath = os.path.join(*parts)
    os.makedirs(dirpath, exist_ok=True)
    filepath = os.path.join(dirpath, "result.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    return filepath


def read_three_column_matrix_csv(filepath: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    matrix = {}
    if os.path.exists(filepath):
        with open(filepath, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
            if len(rows) > 0:
                header_row = rows[0]
                
                col_keys = []
                i = 1
                while i < len(header_row):
                    header = header_row[i]
                    if header.endswith('_precision'):
                        col_key = header.replace('_precision', '')
                        col_keys.append(col_key)
                        i += 3
                    else:
                        i += 1
                
                for row in rows[1:]:
                    if len(row) > 0:
                        row_key = row[0]
                        matrix[row_key] = {}
                        col_idx = 0
                        for col_key in col_keys:
                            matrix[row_key][col_key] = {}
                            base_idx = 1 + col_idx * 3
                            for metric_idx, metric in enumerate(['precision', 'recall', 'f1']):
                                idx = base_idx + metric_idx
                                if idx < len(row):
                                    value_str = row[idx]
                                    try:
                                        matrix[row_key][col_key][metric] = float(value_str) if value_str else 0.0
                                    except ValueError:
                                        matrix[row_key][col_key][metric] = 0.0
                                else:
                                    matrix[row_key][col_key][metric] = 0.0
                            col_idx += 1
    return matrix


def write_three_column_matrix_csv(filepath: str, matrix: Dict[str, Dict[str, Dict[str, float]]]):
    all_row_keys = sorted(set(matrix.keys()))
    all_col_keys = set()
    for row_dict in matrix.values():
        all_col_keys.update(row_dict.keys())
    all_col_keys = sorted(all_col_keys)
    
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['']
        for col_key in all_col_keys:
            header.extend([f"{col_key}_precision", f"{col_key}_recall", f"{col_key}_f1"])
        writer.writerow(header)
        
        for row_key in all_row_keys:
            row_data = [row_key]
            for col_key in all_col_keys:
                if col_key in matrix[row_key]:
                    row_data.extend([
                        f"{matrix[row_key][col_key].get('precision', 0.0):.4f}",
                        f"{matrix[row_key][col_key].get('recall', 0.0):.4f}",
                        f"{matrix[row_key][col_key].get('f1', 0.0):.4f}"
                    ])
                else:
                    row_data.extend(['', '', ''])
            writer.writerow(row_data)


def update_precision_recall_f1_merged_csv(filepath: str, 
                                         train_method: str, train_net: str, train_dataset: str,
                                         test_method: str, test_net: str, test_dataset: str,
                                         precision_value: float, recall_value: float, f1_value: float):
    row_key = f"{train_method}_{train_net}_{train_dataset}"
    col_key = f"{test_method}_{test_net}_{test_dataset}"
    
    matrix = read_three_column_matrix_csv(filepath)
    
    if precision_value < 0 or recall_value < 0 or f1_value < 0:
        precision_percent = precision_value
        recall_percent = recall_value
        f1_percent = f1_value
    else:
        precision_percent = precision_value * 100
        recall_percent = recall_value * 100
        f1_percent = f1_value * 100
    
    if row_key not in matrix:
        matrix[row_key] = {}
    if col_key not in matrix[row_key]:
        matrix[row_key][col_key] = {}
    matrix[row_key][col_key]['precision'] = precision_percent
    matrix[row_key][col_key]['recall'] = recall_percent
    matrix[row_key][col_key]['f1'] = f1_percent
    
    write_three_column_matrix_csv(filepath, matrix)


def update_insert_prf1_merged_csv(filepath: str,
                                  train_method: str, train_net: str, train_dataset: str,
                                  test_method: str, test_net: str, test_dataset: str,
                                  precision_value: float, recall_value: float, f1_value: float):
    row_key = f"{train_method}_{train_net}_{train_dataset}"
    col_key = f"{test_method}_{test_net}_{test_dataset}"
    
    matrix = read_three_column_matrix_csv(filepath)
    
    if precision_value < 0 or recall_value < 0 or f1_value < 0:
        precision_percent = precision_value
        recall_percent = recall_value
        f1_percent = f1_value
    else:
        precision_percent = precision_value * 100
        recall_percent = recall_value * 100
        f1_percent = f1_value * 100
    
    if row_key not in matrix:
        matrix[row_key] = {}
    if col_key not in matrix[row_key]:
        matrix[row_key][col_key] = {}
    matrix[row_key][col_key]['precision'] = precision_percent
    matrix[row_key][col_key]['recall'] = recall_percent
    matrix[row_key][col_key]['f1'] = f1_percent
    
    write_three_column_matrix_csv(filepath, matrix)


def update_remove_prf1_merged_csv(filepath: str,
                                   train_method: str, train_net: str, train_dataset: str,
                                   test_method: str, test_net: str, test_dataset: str,
                                   precision_value: float, recall_value: float, f1_value: float):
    row_key = f"{train_method}_{train_net}_{train_dataset}"
    col_key = f"{test_method}_{test_net}_{test_dataset}"
    
    matrix = read_three_column_matrix_csv(filepath)
    
    if precision_value < 0 or recall_value < 0 or f1_value < 0:
        precision_percent = precision_value
        recall_percent = recall_value
        f1_percent = f1_value
    else:
        precision_percent = precision_value * 100
        recall_percent = recall_value * 100
        f1_percent = f1_value * 100
    
    if row_key not in matrix:
        matrix[row_key] = {}
    if col_key not in matrix[row_key]:
        matrix[row_key][col_key] = {}
    matrix[row_key][col_key]['precision'] = precision_percent
    matrix[row_key][col_key]['recall'] = recall_percent
    matrix[row_key][col_key]['f1'] = f1_percent
    
    write_three_column_matrix_csv(filepath, matrix)


def read_single_value_matrix_csv(filepath: str) -> Dict[str, Dict[str, float]]:
    matrix = {}
    if os.path.exists(filepath):
        with open(filepath, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
            if len(rows) > 0:
                col_headers = rows[0][1:]
                for row in rows[1:]:
                    if len(row) > 0:
                        row_key = row[0]
                        matrix[row_key] = {}
                        for i, col_key in enumerate(col_headers):
                            if i + 1 < len(row):
                                value_str = row[i + 1]
                                try:
                                    matrix[row_key][col_key] = float(value_str) if value_str else 0.0
                                except ValueError:
                                    matrix[row_key][col_key] = 0.0
    return matrix


def write_single_value_matrix_csv(filepath: str, matrix: Dict[str, Dict[str, float]]):
    all_row_keys = sorted(set(matrix.keys()))
    all_col_keys = set()
    for row_dict in matrix.values():
        all_col_keys.update(row_dict.keys())
    all_col_keys = sorted(all_col_keys)
    
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = [''] + all_col_keys
        writer.writerow(header)
        
        for row_key in all_row_keys:
            row_data = [row_key]
            for col_key in all_col_keys:
                value = matrix[row_key].get(col_key, '')
                row_data.append(f"{value:.2f}" if isinstance(value, (int, float)) else value)
            writer.writerow(row_data)


def update_hit_rate_csv(filepath: str,
                        train_method: str, train_net: str, train_dataset: str,
                        test_method: str, test_net: str, test_dataset: str,
                        breakpoint_type: str,
                        hit_rate_value: float):
    row_key = f"{train_method}_{train_net}_{train_dataset}"
    col_key = f"{test_method}_{test_net}_{test_dataset}"
    
    matrix = read_single_value_matrix_csv(filepath)
    
    if hit_rate_value < 0:
        hit_rate_percent = hit_rate_value
    else:
        hit_rate_percent = hit_rate_value * 100
    
    if row_key not in matrix:
        matrix[row_key] = {}
    matrix[row_key][col_key] = hit_rate_percent
    
    write_single_value_matrix_csv(filepath, matrix)


def update_membership_metric_csv(filepath: str,
                                  train_method: str, train_net: str, train_dataset: str,
                                  test_method: str, test_net: str, test_dataset: str,
                                  metric_value: float):
    row_key = f"{train_method}_{train_net}_{train_dataset}"
    col_key = f"{test_method}_{test_net}_{test_dataset}"
    
    matrix = read_single_value_matrix_csv(filepath)
    
    if metric_value < 0:
        metric_percent = metric_value
    else:
        metric_percent = metric_value * 100
    
    if row_key not in matrix:
        matrix[row_key] = {}
    matrix[row_key][col_key] = metric_percent
    
    write_single_value_matrix_csv(filepath, matrix)


def read_tolerance_single_prf1_csv(filepath: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    data = {}
    if os.path.exists(filepath):
        with open(filepath, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
            if len(rows) > 1:
                for row in rows[1:]:
                    if len(row) >= 5:
                        row_key, col_key = row[0], row[1]
                        if row_key not in data:
                            data[row_key] = {}
                        try:
                            data[row_key][col_key] = {
                                'precision': float(row[2]) if row[2] else 0.0,
                                'recall': float(row[3]) if row[3] else 0.0,
                                'f1': float(row[4]) if row[4] else 0.0,
                            }
                        except (ValueError, IndexError):
                            pass
    return data


def write_tolerance_single_prf1_csv(filepath: str, data: Dict[str, Dict[str, Dict[str, float]]]):
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    all_row_keys = sorted(data.keys())
    all_col_keys = set()
    for row_dict in data.values():
        all_col_keys.update(row_dict.keys())
    all_col_keys = sorted(all_col_keys)
    header = ['train_config', 'test_config', 'precision', 'recall', 'f1']
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row_key in all_row_keys:
            for col_key in all_col_keys:
                vals = data[row_key].get(col_key, {})
                def fmt(v):
                    if v == '' or v is None:
                        return ''
                    return f"{float(v):.2f}" if isinstance(v, (int, float)) else v
                writer.writerow([
                    row_key, col_key,
                    fmt(vals.get('precision')),
                    fmt(vals.get('recall')),
                    fmt(vals.get('f1')),
                ])


def update_tolerance_single_prf1_csv(filepath: str,
                                      train_method: str, train_net: str, train_dataset: str,
                                      test_method: str, test_net: str, test_dataset: str,
                                      precision: float, recall: float, f1: float):
    row_key = f"{train_method}_{train_net}_{train_dataset}"
    col_key = f"{test_method}_{test_net}_{test_dataset}"
    data = read_tolerance_single_prf1_csv(filepath)
    if row_key not in data:
        data[row_key] = {}
    def pct(v):
        return v * 100 if v >= 0 else v
    data[row_key][col_key] = {
        'precision': pct(precision),
        'recall': pct(recall),
        'f1': pct(f1),
    }
    write_tolerance_single_prf1_csv(filepath, data)


def breakpoints_to_membership_timestamps(breakpoints: list, 
                                          num_timestamps: int) -> set:
    membership = set()
    
    if len(breakpoints) == 0:
        return membership

    sorted_breakpoints = sorted(breakpoints, key=lambda x: x[0])
    breakpoint_dict = {ts: bp_type for ts, bp_type in sorted_breakpoints}
    first_ts, first_type = sorted_breakpoints[0]
    current_status = 1 if first_type == 2 else 0
    
    if current_status == 1:
        for ts in range(first_ts):
            membership.add(ts)
    
    for ts in range(num_timestamps):
        if ts in breakpoint_dict:
            bp_type = breakpoint_dict[ts]
            if bp_type == 1:
                current_status = 1
                membership.add(ts)
            elif bp_type == 2:
                current_status = 0
        else:
            if current_status == 1:
                membership.add(ts)
    
    return membership


def parse_list_arg(arg_value: str) -> List[str]:
    if arg_value is None or arg_value == '':
        return []
    return [x.strip() for x in str(arg_value).split(',') if x.strip()]


def generate_all_combinations(args: Any) -> List[dict]:
    from itertools import product

    def getattr_default(obj, key, default=None):
        return getattr(obj, key, default) if hasattr(obj, key) else (obj.get(key, default) if isinstance(obj, dict) else default)

    train_net_names = parse_list_arg(getattr_default(args, 'train_net_names', ''))
    train_dataset_names = parse_list_arg(getattr_default(args, 'train_dataset_names', ''))
    test_net_names = parse_list_arg(getattr_default(args, 'test_net_names', ''))
    test_dataset_names = parse_list_arg(getattr_default(args, 'test_dataset_names', ''))
    if not test_net_names:
        test_net_names = train_net_names
    if not test_dataset_names:
        test_dataset_names = train_dataset_names

    use_same = getattr(args, 'test_same_as_train', True)
    proportion_of_group_unlearn = getattr_default(args, 'proportion_of_group_unlearn', 0.01)

    combinations = []
    if use_same or (train_net_names == test_net_names and train_dataset_names == test_dataset_names):
        for net, dataset in product(train_net_names, train_dataset_names):
            if dataset.lower() == 'mnli':
                prop = 0.005
            elif dataset.lower() == 'cinic10':
                prop = 0.001
            else:
                prop = proportion_of_group_unlearn
            combination = {
                'train_net_name': net,
                'train_dataset_name': dataset,
                'test_net_name': net,
                'test_dataset_name': dataset,
                'train_proportion': prop,
                'test_proportion': prop,
                'base_args': {
                    'U_method': getattr_default(args, 'U_method', 'continuous_update_finetune'),
                    'proportion_of_group_unlearn': proportion_of_group_unlearn,
                    'trial': getattr_default(args, 'trial', 0),
                    'data_root': getattr_default(args, 'data_root'),
                    'num_target_per_pattern': getattr_default(args, 'num_target_per_pattern', 333),
                    'num_epochs': getattr_default(args, 'num_epochs', 100),
                    'learning_rate': getattr_default(args, 'learning_rate', 0.0001),
                    'hidden_dim': getattr_default(args, 'hidden_dim', 32),
                    'checkpoint_dir': getattr_default(args, 'checkpoint_dir', './mia_checkpoints'),
                    'seed': getattr_default(args, 'seed', 42),
                    'num_trials': getattr_default(args, 'num_trials', 1),
                    'device': getattr_default(args, 'device', 'cuda:0'),
                    'debug': getattr_default(args, 'debug', False),
                    'test_same_as_train': True,
                }
            }
            combinations.append(combination)
    else:
        for train_net, train_dataset, test_net, test_dataset in product(
            train_net_names, train_dataset_names, test_net_names, test_dataset_names
        ):
            if train_dataset.lower() == 'mnli':
                train_proportion = 0.005
            elif train_dataset.lower() == 'cinic10':
                train_proportion = 0.001
            else:
                train_proportion = proportion_of_group_unlearn
            if test_dataset.lower() == 'mnli':
                test_proportion = 0.005
            elif test_dataset.lower() == 'cinic10':
                test_proportion = 0.001
            else:
                test_proportion = proportion_of_group_unlearn
            combination = {
                'train_net_name': train_net,
                'train_dataset_name': train_dataset,
                'test_net_name': test_net,
                'test_dataset_name': test_dataset,
                'train_proportion': train_proportion,
                'test_proportion': test_proportion,
                'base_args': {
                    'U_method': getattr_default(args, 'U_method', 'continuous_update_finetune'),
                    'proportion_of_group_unlearn': proportion_of_group_unlearn,
                    'trial': getattr_default(args, 'trial', 0),
                    'data_root': getattr_default(args, 'data_root'),
                    'num_target_per_pattern': getattr_default(args, 'num_target_per_pattern', 333),
                    'num_epochs': getattr_default(args, 'num_epochs', 100),
                    'learning_rate': getattr_default(args, 'learning_rate', 0.0001),
                    'hidden_dim': getattr_default(args, 'hidden_dim', 32),
                    'checkpoint_dir': getattr_default(args, 'checkpoint_dir', './mia_checkpoints'),
                    'seed': getattr_default(args, 'seed', 42),
                    'num_trials': getattr_default(args, 'num_trials', 1),
                    'device': getattr_default(args, 'device', 'cuda:0'),
                    'debug': getattr_default(args, 'debug', False),
                    'test_same_as_train': False,
                }
            }
            combinations.append(combination)

    return combinations
