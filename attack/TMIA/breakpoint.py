#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from attack.TMIA.get_training_data import build_path_and_find_timestamps, \
    process_breakpoint_and_select_sample, load_data_and_compute_confidence,\
    create_window_data, select_test_samples_by_pattern, select_samples_by_insert_remove_count, filter_outlier_windows, \
    compute_window_normalization_params, normalize_window_data
from attack.TMIA.inference import evaluate_model
from attack.TMIA.lstm import BiLSTMChangePoint
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import os
from collections import Counter
from attack.TMIA.inference import breakpoint_to_membership_timestamps

def process_breakpoint_data(
    args,
    sample_index=None,
    random_seed=None,
    is_llm=False,
    use_true_label=False,
    num_samples=1000
):
    save_path, timestamp_dirs = build_path_and_find_timestamps(args)
    
    window_size = args.get('window_size', 5)
    sample_indices, breakpoint_indices_dict, samples_breakpoint_info = process_breakpoint_and_select_sample(
        save_path=save_path,
        timestamp_dirs=timestamp_dirs,
        sample_index=sample_index,
        random_seed=random_seed,
        num_samples=num_samples,
        window_size=window_size
    )
    
    ppl_normalization_params = None
    samples_timestamps_data = load_data_and_compute_confidence(
        timestamp_dirs=timestamp_dirs,
        sample_indices=sample_indices,
        samples_breakpoint_info=samples_breakpoint_info,
        is_llm=is_llm,
        use_true_label=use_true_label,
        ppl_normalization_params=ppl_normalization_params
    )
    if is_llm and ppl_normalization_params is not None:
        return samples_timestamps_data, ppl_normalization_params
    
    return samples_timestamps_data



class WindowDataset(Dataset):
    def __init__(self, window_data, seq_len=10, window_size=5, use_stats=False):
        self.window_data = window_data
        self.seq_len = seq_len
        self.window_size = window_size
        self.use_stats = use_stats
    
    def __len__(self):
        return len(self.window_data)
    
    def _compute_stats(self, window_sequence):
        seq_len = len(window_sequence)
        center_idx = self.window_size
        
        if center_idx > 0:
            front_window = window_sequence[:center_idx]
            front_mean = torch.mean(front_window)
            front_var = torch.var(front_window, unbiased=False)
        else:
            front_mean = torch.tensor(0.0)
            front_var = torch.tensor(0.0)
        if center_idx + 1 < seq_len:
            back_window = window_sequence[center_idx + 1:]
            back_mean = torch.mean(back_window)
            back_var = torch.var(back_window, unbiased=False)
        else:
            back_mean = torch.tensor(0.0)
            back_var = torch.tensor(0.0)
        
        return torch.stack([front_mean, front_var, back_mean, back_var])
    
    def __getitem__(self, idx):
        item = self.window_data[idx]
        window_sequence = item['window_sequence']
        
        if len(window_sequence) > self.seq_len:
            window_sequence = window_sequence[:self.seq_len]
        elif len(window_sequence) < self.seq_len:
            padding = [window_sequence[-1]] * (self.seq_len - len(window_sequence))
            window_sequence = window_sequence + padding
        
        window_sequence_tensor = torch.tensor(window_sequence, dtype=torch.float32).unsqueeze(1)
        label = torch.tensor(item['label'], dtype=torch.long)
        
        if self.use_stats:
            stats = self._compute_stats(torch.tensor(window_sequence, dtype=torch.float32))
            return window_sequence_tensor, label, stats
        else:
            return window_sequence_tensor, label


def train_lstm_model(model, train_loader, val_loader, device, args, class_weights=None):
    optimizer = optim.Adam(model.parameters(), lr=args.get('lr', 0.0001), weight_decay=args.get('weight_decay', 1e-5))
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    num_epochs = 500

    if  args['net_name']=='llama8b' and args['dataset_name']=='squad':
        num_epochs = 200

    best_val_acc = 0.0
    train_losses = []
    val_accs = []
    
    model = model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for batch in train_loader:
            if len(batch) == 3:
                batch_data, batch_labels, batch_stats = batch
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                batch_stats = batch_stats.to(device)
            else:
                batch_data, batch_labels = batch
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                batch_stats = None
            optimizer.zero_grad()
            if batch_stats is not None:
                outputs = model(batch_data, stats=batch_stats)
            else:
                outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100.0 * train_correct / train_total
        train_losses.append(avg_train_loss)
        
        train_class_correct = [0, 0, 0]
        train_class_total = [0, 0, 0]
        model.eval()
        with torch.no_grad():
            for batch in train_loader:
                if len(batch) == 3:
                    batch_data, batch_labels, batch_stats = batch
                    batch_data = batch_data.to(device)
                    batch_labels = batch_labels.to(device)
                    batch_stats = batch_stats.to(device)
                    outputs = model(batch_data, stats=batch_stats)
                else:
                    batch_data, batch_labels = batch
                    batch_data = batch_data.to(device)
                    batch_labels = batch_labels.to(device)
                    outputs = model(batch_data)
                _, predicted = torch.max(outputs.data, 1)
                for i in range(batch_labels.size(0)):
                    label = batch_labels[i].item()
                    train_class_total[label] += 1
                    if predicted[i] == batch_labels[i]:
                        train_class_correct[label] += 1
        model.train()
        if train_acc > best_val_acc:
            best_val_acc = train_acc


    return {
        'train_losses': train_losses,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc
    }

def attack_breakpoint(args):
    train_args = args.copy()
    train_args['is_shadow'] = True
    train_args['model_type'] = 'shadow'
    is_llm=False
    if args['U_method']=='continuous_update_finetune_LLM':
        is_llm=True
    result = process_breakpoint_data(
        args=train_args,
        sample_index=None,
        random_seed=args.get('random_seed', 42),
        is_llm=is_llm,
        use_true_label=args.get('use_true_label', True),
        num_samples=args.get('num_samples', 1000)
    )
    
    if isinstance(result, tuple) and len(result) == 2:
        samples_timestamps_data, ppl_normalization_params = result
    else:
        samples_timestamps_data = result
        ppl_normalization_params = None
    window_size = args.get('window_size')
    window_data = create_window_data(samples_timestamps_data, window_size=window_size, skip_boundary=True)
    if is_llm:
        window_data, outlier_window_indices = filter_outlier_windows(window_data, outlier_filter=True, z_threshold=3.0)
    class_data = {0: [], 1: [], 2: []}
    for item in window_data:
        class_data[item['label']].append(item)
    class_ratios = args.get('class_sampling_ratios', [0.6 ,1.0, 1.0])
    if args['net_name']=='llama8b' and args['dataset_name']=='squad':
        class_ratios = args.get('class_sampling_ratios', [1.0, 1.0, 1.0])
    elif args['net_name']=='simple_cnn' and args['dataset_name']=='cinic10' :
        class_ratios = args.get('class_sampling_ratios', [0.3, 1.0, 1.0])
    elif args['net_name']=='mobilenet' and args['dataset_name']=='cinic10':
        class_ratios = args.get('class_sampling_ratios', [1.5, 1.0, 1.0])
    available_counts = [len(class_data[i]) for i in range(3) if len(class_data[i]) > 0]

    print("TMIA Start ----------------")

    min_available = min(available_counts)
    target_counts = {}
    for class_label in range(3):
        if len(class_data[class_label]) > 0:
            target_count = int(min_available * class_ratios[class_label])
            target_count = min(target_count, len(class_data[class_label]))
            target_counts[class_label] = target_count
        else:
            target_counts[class_label] = 0
    sampled_window_data = []
    random_seed = args.get('random_seed', 42)
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    for class_label in range(3):
        if target_counts[class_label] > 0:
            if len(class_data[class_label]) > target_counts[class_label]:
                sampled = random.sample(class_data[class_label], target_counts[class_label])
            else:
                sampled = class_data[class_label]
            sampled_window_data.extend(sampled)
    use_normalization = args.get('use_window_normalization', False)
    window_normalization_params = None
    if use_normalization:
        mean, std = compute_window_normalization_params(sampled_window_data)
        if mean is not None and std is not None:
            window_normalization_params = (mean, std)
            args['window_normalization_params'] = window_normalization_params
            sampled_window_data = normalize_window_data(sampled_window_data, mean, std)
        else:
            use_normalization = False
            args['window_normalization_params'] = None
    else:
        args['window_normalization_params'] = None
    seq_len = 2 * window_size + 1
    use_stats = True
    train_dataset = WindowDataset(sampled_window_data, seq_len=seq_len, window_size=window_size, use_stats=use_stats)
    
    batch_size = args.get('batch_size', 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    model = BiLSTMChangePoint(
        hidden_size=args.get('hidden_size', 32),
        fc_hidden_size=args.get('fc_hidden_size', 32),
        use_stats=use_stats
    )
    device = torch.device('cuda' if torch.cuda.is_available() and args.get('use_cuda', True) else 'cpu')
    use_class_weights = args.get('use_class_weights', False)
    class_weights = None
    if use_class_weights:
        sampled_label_counts = Counter([item['label'] for item in sampled_window_data])
        total_samples = len(sampled_window_data)
        class_weights_list = []
        for class_label in range(3):
            if sampled_label_counts[class_label] > 0:
                weight = total_samples / (3.0 * sampled_label_counts[class_label])
            else:
                weight = 0.0
            class_weights_list.append(weight)
        class_weights = torch.tensor(class_weights_list, dtype=torch.float32)
    elif 'class_weights' in args and args['class_weights'] is not None:
        class_weights = torch.tensor(args['class_weights'], dtype=torch.float32)
    results = train_lstm_model(model, train_loader, None, device, args, class_weights=class_weights)
    num_attacks = args.get('num_attacks', 1)
    test_args = args.copy()
    test_args['is_shadow'] = False
    test_args['model_type'] = 'target'
    
    save_path, timestamp_dirs = build_path_and_find_timestamps(test_args)
    
    all_original_metrics = []
    all_processed_metrics = []
    all_test_data_stats = []
    all_original_class_metrics = []
    all_processed_class_metrics = []
    all_tolerance_metrics_original = []
    all_tolerance_metrics_processed = []
    all_interval_level_metrics_processed = []
    
    base_random_seed = args.get('random_seed', 42)
    balance_membership = args.get('balance_test_membership', True)
    target_member_ratio = args.get('target_test_member_ratio', 0.66)
    if args['U_method'] == 'continuous_update_finetune_multiple_update':
        cases = [
            {'num_insert': 1, 'num_remove': 1, 'num_samples': 200, 'case_name': '1insert_1remove'},
            {'num_insert': 2, 'num_remove': 2, 'num_samples': 200, 'case_name': '2insert_2remove'},
            {'num_insert': 3, 'num_remove': 3, 'num_samples': 200, 'case_name': '3insert_3remove'},
            {'num_insert': 4, 'num_remove': 4, 'num_samples': 200, 'case_name': '4insert_4remove'},
            {'num_insert': 5, 'num_remove': 5, 'num_samples': 200, 'case_name': '5insert_5remove'},
        ]
        
        for case_idx, case_config in enumerate(cases):
            case_all_original_metrics = []
            case_all_processed_metrics = []
            case_all_test_data_stats = []

            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname((os.path.dirname(script_dir)))
            result_base_path = os.path.join(project_root, "result")
            case_results_dir = os.path.join(result_base_path, args.get('U_method'), args.get('net_name'),
                args.get('dataset_name'), str(args.get('proportion_of_group_unlearn')),
                args.get('attack_method'), str(args.get('window_size', 5)), case_config['case_name'])
            os.makedirs(case_results_dir, exist_ok=True)
            args['output_dir'] = case_results_dir
            
            for attack_idx in range(num_attacks):
                attack_random_seed = base_random_seed + attack_idx if base_random_seed is not None else None
                verbose_selection = (attack_idx == 0)
                balance_membership =True
                target_member_ratio =0.66
                test_sample_indices, test_breakpoint_indices_dict, test_samples_breakpoint_info = \
                    select_samples_by_insert_remove_count(
                        save_path=save_path,
                        timestamp_dirs=timestamp_dirs,
                        num_insert=case_config['num_insert'],
                        num_remove=case_config['num_remove'],
                        num_samples=case_config['num_samples'],
                        random_seed=attack_random_seed,
                        verbose=verbose_selection,
                        balance_membership=balance_membership,
                        target_member_ratio=target_member_ratio
                    )
                
                if len(test_sample_indices) == 0:
                    continue
                test_samples_timestamps_data = load_data_and_compute_confidence(
                    timestamp_dirs=timestamp_dirs,
                    sample_indices=test_sample_indices,
                    samples_breakpoint_info=test_samples_breakpoint_info,
                    is_llm=is_llm,
                    use_true_label=args.get('use_true_label', True),
                    ppl_normalization_params=ppl_normalization_params if is_llm else None
                )
                actual_timestamps = []
                for timestamp_dir in timestamp_dirs:
                    timestamp_num = int(os.path.basename(timestamp_dir).split("_")[1])
                    actual_timestamps.append(timestamp_num)
                
                test_samples_data = []
                for i, sample_idx in enumerate(test_sample_indices):
                    confidences, labels = test_samples_timestamps_data[i]
                    sample_info = test_samples_breakpoint_info.get(sample_idx, {})
                    
                    actual_insert_ts = sample_info.get('insert', [])
                    actual_remove_ts = sample_info.get('remove', [])
                    
                    insert_timestamps_indices = []
                    remove_timestamps_indices = []
                    ts_to_idx = {ts: idx for idx, ts in enumerate(actual_timestamps)}
                    
                    for ts in actual_insert_ts:
                        if ts in ts_to_idx:
                            insert_timestamps_indices.append(ts_to_idx[ts])
                    
                    for ts in actual_remove_ts:
                        if ts in ts_to_idx:
                            remove_timestamps_indices.append(ts_to_idx[ts])
                    
                    timestamps = list(range(len(confidences)))
                    max_idx = len(confidences) - 1
                    insert_timestamps_indices = [idx for idx in insert_timestamps_indices if 0 <= idx <= max_idx]
                    remove_timestamps_indices = [idx for idx in remove_timestamps_indices if 0 <= idx <= max_idx]
                    
                    test_samples_data.append({
                        'sample_index': sample_idx,
                        'insert_timestamps': insert_timestamps_indices,
                        'remove_timestamps': remove_timestamps_indices,
                        'timestamps': timestamps,
                        'confidences': confidences,
                        'labels': labels
                    })
                
                if len(test_samples_data) > 0:
                    num_ts_per_sample = len(test_samples_data[0]['timestamps'])
                    total_ts = len(test_samples_data) * num_ts_per_sample
                    
                    total_mem_count = 0
                    total_nonmem_count = 0
                    for sample_data in test_samples_data:
                        insert_ts_indices = sample_data.get('insert_timestamps', [])
                        remove_ts_indices = sample_data.get('remove_timestamps', [])
                        breakpoints = [(ts, 1) for ts in insert_ts_indices] + [(ts, 2) for ts in remove_ts_indices]
                        membership_ts = breakpoint_to_membership_timestamps(breakpoints, num_ts_per_sample)
                        total_mem_count += len(membership_ts)
                        total_nonmem_count += (num_ts_per_sample - len(membership_ts))
                    
                    mem_ratio = total_mem_count / total_ts if total_ts > 0 else 0.0
                    nonmem_ratio = total_nonmem_count / total_ts if total_ts > 0 else 0.0
                    
                    test_data_stats = {
                        'num_samples': len(test_samples_data),
                        'num_timestamps_per_sample': num_ts_per_sample,
                        'total_timestamps': total_ts,
                        'member_count': total_mem_count,
                        'nonmember_count': total_nonmem_count,
                        'member_ratio': mem_ratio,
                        'nonmember_ratio': nonmem_ratio
                    }
                    case_all_test_data_stats.append(test_data_stats)
                test_window_data = create_window_data(test_samples_timestamps_data, window_size=window_size, skip_boundary=False)
                if args.get('use_window_normalization', False) and args.get('window_normalization_params') is not None:
                    norm_mean, norm_std = args['window_normalization_params']
                    test_window_data = normalize_window_data(test_window_data, norm_mean, norm_std)
                test_metrics = evaluate_model(model, test_samples_data, test_window_data, device, args, verbose=False)
                if not isinstance(test_metrics, dict) or 'original_metrics' not in test_metrics:
                    continue
                
                case_all_original_metrics.append(test_metrics['original_metrics'])
                case_all_processed_metrics.append(test_metrics['processed_metrics'])
            if len(case_all_original_metrics) > 0:
                metric_keys = [
                    'membership_precision', 'membership_recall', 'membership_f1',
                    'insert_precision', 'insert_recall', 'insert_f1',
                    'remove_precision', 'remove_recall', 'remove_f1'
                ]
                
                case_avg_original_metrics = {}
                for key in metric_keys:
                    values = [m[key] for m in case_all_original_metrics if key in m]
                    case_avg_original_metrics[key] = sum(values) / len(values) if values else 0.0
                
                case_avg_processed_metrics = {}
                for key in metric_keys:
                    values = [m[key] for m in case_all_processed_metrics if key in m]
                    case_avg_processed_metrics[key] = sum(values) / len(values) if values else 0.0
        return {
            'model': model,
            'results': results,
            'window_data': sampled_window_data,
            'original_window_data': window_data,
            'test_metrics': None,
            'num_attacks': num_attacks,
            'model_file': None,
            'results_file': None
        }
    

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname((os.path.dirname(script_dir)))
    result_base_path = os.path.join(project_root, "result")


    U_method = args.get('U_method')
    net_name = args.get('net_name')
    dataset_name = args.get('dataset_name')
    proportion = args.get('proportion_of_group_unlearn')
    attack_method = args.get('attack_method')
    window_size = args.get('window_size', 5)
    results_dir = os.path.join(result_base_path, U_method, net_name, dataset_name,
                               str(proportion), attack_method, str(window_size))
    os.makedirs(results_dir, exist_ok=True)
    
    args['output_dir'] = results_dir
    
    for attack_idx in range(num_attacks):
        attack_random_seed = base_random_seed + attack_idx if base_random_seed is not None else None
        num_insert_only=400
        num_remove_only=400
        num_insert_remove=400
        if args['dataset_name'] =='cifar10' or args['dataset_name'] =='cifar100' or args['dataset_name'] =='cinic10':
            num_insert_only=50
            num_remove_only=50
            num_insert_remove =100
        verbose_selection = (attack_idx == 0)
        test_sample_indices, test_breakpoint_indices_dict, test_samples_breakpoint_info = \
            select_test_samples_by_pattern(
                save_path=save_path,
                timestamp_dirs=timestamp_dirs,
                num_insert_only=num_insert_only,
                num_remove_only=num_remove_only,
                num_insert_remove=num_insert_remove,
                random_seed=attack_random_seed,
                balance_membership=balance_membership,
                target_member_ratio=target_member_ratio,
                verbose=verbose_selection
            )
        test_samples_timestamps_data = load_data_and_compute_confidence(
            timestamp_dirs=timestamp_dirs,
            sample_indices=test_sample_indices,
            samples_breakpoint_info=test_samples_breakpoint_info,
            is_llm=is_llm,
            use_true_label=args.get('use_true_label', True),
            ppl_normalization_params=ppl_normalization_params if is_llm else None
        )
        actual_timestamps = []
        for timestamp_dir in timestamp_dirs:
            timestamp_num = int(os.path.basename(timestamp_dir).split("_")[1])
            actual_timestamps.append(timestamp_num)
        
        test_samples_data = []
        for i, sample_idx in enumerate(test_sample_indices):
            confidences, labels = test_samples_timestamps_data[i]
            sample_info = test_samples_breakpoint_info.get(sample_idx, {})
            
            actual_insert_ts = sample_info.get('insert', [])
            actual_remove_ts = sample_info.get('remove', [])
            
            insert_timestamps_indices = []
            remove_timestamps_indices = []
            ts_to_idx = {ts: idx for idx, ts in enumerate(actual_timestamps)}
            
            for ts in actual_insert_ts:
                if ts in ts_to_idx:
                    insert_timestamps_indices.append(ts_to_idx[ts])
            
            for ts in actual_remove_ts:
                if ts in ts_to_idx:
                    remove_timestamps_indices.append(ts_to_idx[ts])
            
            timestamps = list(range(len(confidences)))
            max_idx = len(confidences) - 1
            insert_timestamps_indices = [idx for idx in insert_timestamps_indices if 0 <= idx <= max_idx]
            remove_timestamps_indices = [idx for idx in remove_timestamps_indices if 0 <= idx <= max_idx]
            
            test_samples_data.append({
                'sample_index': sample_idx,
                'insert_timestamps': insert_timestamps_indices,
                'remove_timestamps': remove_timestamps_indices,
                'timestamps': timestamps,
                'confidences': confidences,
                'labels': labels
            })
        
        if len(test_samples_data) > 0:
            num_ts_per_sample = len(test_samples_data[0]['timestamps'])
            total_ts = len(test_samples_data) * num_ts_per_sample
            
            total_mem_count = 0
            total_nonmem_count = 0
            for sample_data in test_samples_data:
                insert_ts_indices = sample_data.get('insert_timestamps', [])
                remove_ts_indices = sample_data.get('remove_timestamps', [])
                breakpoints = [(ts, 1) for ts in insert_ts_indices] + [(ts, 2) for ts in remove_ts_indices]
                membership_ts = breakpoint_to_membership_timestamps(breakpoints, num_ts_per_sample)
                total_mem_count += len(membership_ts)
                total_nonmem_count += (num_ts_per_sample - len(membership_ts))
            
            mem_ratio = total_mem_count / total_ts if total_ts > 0 else 0.0
            nonmem_ratio = total_nonmem_count / total_ts if total_ts > 0 else 0.0
            
            test_data_stats = {
                'num_samples': len(test_samples_data),
                'num_timestamps_per_sample': num_ts_per_sample,
                'total_timestamps': total_ts,
                'member_count': total_mem_count,
                'nonmember_count': total_nonmem_count,
                'member_ratio': mem_ratio,
                'nonmember_ratio': nonmem_ratio
            }
            all_test_data_stats.append(test_data_stats)
            
        test_window_data = create_window_data(test_samples_timestamps_data, window_size=window_size, skip_boundary=False)
        
        if args.get('use_window_normalization', False) and args.get('window_normalization_params') is not None:
            norm_mean, norm_std = args['window_normalization_params']
            test_window_data = normalize_window_data(test_window_data, norm_mean, norm_std)
        
        test_metrics = evaluate_model(model, test_samples_data, test_window_data, device, args, verbose=False)
        if not isinstance(test_metrics, dict) or 'original_metrics' not in test_metrics:
            raise ValueError(f"evaluate_model返回值格式错误，缺少 'original_metrics' 键")
        all_original_metrics.append(test_metrics['original_metrics'])
        all_processed_metrics.append(test_metrics['processed_metrics'])
        
        if 'original_class_metrics' in test_metrics:
            all_original_class_metrics.append(test_metrics['original_class_metrics'])
        if 'processed_class_metrics' in test_metrics:
            all_processed_class_metrics.append(test_metrics['processed_class_metrics'])
        if 'tolerance_metrics_original' in test_metrics:
            all_tolerance_metrics_original.append(test_metrics['tolerance_metrics_original'])
        if 'tolerance_metrics_processed' in test_metrics:
            all_tolerance_metrics_processed.append(test_metrics['tolerance_metrics_processed'])
        if 'interval_level_metrics_processed' in test_metrics:
            all_interval_level_metrics_processed.append(test_metrics['interval_level_metrics_processed'])
    
    metric_keys = [
        'membership_precision', 'membership_recall', 'membership_f1',
        'insert_precision', 'insert_recall', 'insert_f1',
        'remove_precision', 'remove_recall', 'remove_f1'
    ]
    
    avg_original_metrics = {}
    for key in metric_keys:
        values = [m[key] for m in all_original_metrics if key in m]
        avg_original_metrics[key] = sum(values) / len(values) if values else 0.0
    avg_processed_metrics = {}
    for key in metric_keys:
        values = [m[key] for m in all_processed_metrics if key in m]
        avg_processed_metrics[key] = sum(values) / len(values) if values else 0.0
    
    original_metrics = avg_original_metrics
    processed_metrics = avg_processed_metrics
    avg_original_class_metrics = {}
    if len(all_original_class_metrics) > 0:
        if 'breakpoint_combined' in all_original_class_metrics[0]:
            combined_list = [m.get('breakpoint_combined', {}) for m in all_original_class_metrics if 'breakpoint_combined' in m]
            if len(combined_list) > 0:
                avg_combined = {
                    'precision': sum(c.get('precision', 0) for c in combined_list) / len(combined_list),
                    'recall': sum(c.get('recall', 0) for c in combined_list) / len(combined_list),
                    'f1': sum(c.get('f1', 0) for c in combined_list) / len(combined_list),
                    'tp': int(sum(c.get('tp', 0) for c in combined_list) / len(combined_list)),
                    'fp': int(sum(c.get('fp', 0) for c in combined_list) / len(combined_list)),
                    'fn': int(sum(c.get('fn', 0) for c in combined_list) / len(combined_list))
                }
                avg_original_class_metrics['breakpoint_combined'] = avg_combined
    avg_processed_class_metrics = {}
    if len(all_processed_class_metrics) > 0:
        if 'breakpoint_combined' in all_processed_class_metrics[0]:
            combined_list = [m.get('breakpoint_combined', {}) for m in all_processed_class_metrics if 'breakpoint_combined' in m]
            if len(combined_list) > 0:
                avg_combined = {
                    'precision': sum(c.get('precision', 0) for c in combined_list) / len(combined_list),
                    'recall': sum(c.get('recall', 0) for c in combined_list) / len(combined_list),
                    'f1': sum(c.get('f1', 0) for c in combined_list) / len(combined_list),
                    'tp': int(sum(c.get('tp', 0) for c in combined_list) / len(combined_list)),
                    'fp': int(sum(c.get('fp', 0) for c in combined_list) / len(combined_list)),
                    'fn': int(sum(c.get('fn', 0) for c in combined_list) / len(combined_list))
                }
                avg_processed_class_metrics['breakpoint_combined'] = avg_combined
    avg_tolerance_metrics_original = {}
    if len(all_tolerance_metrics_original) > 0:
        for k in range(1, 6):
            k_key = f'k_{k}'
            k_metrics_list = [m.get(k_key, {}) for m in all_tolerance_metrics_original if k_key in m]
            if len(k_metrics_list) > 0:
                avg_k_metrics = {}
                insert_list = [km.get('insert', {}) for km in k_metrics_list if 'insert' in km]
                if len(insert_list) > 0:
                    avg_k_metrics['insert'] = {
                        'precision': sum(i.get('precision', 0) for i in insert_list) / len(insert_list),
                        'recall': sum(i.get('recall', 0) for i in insert_list) / len(insert_list),
                        'f1': sum(i.get('f1', 0) for i in insert_list) / len(insert_list)
                    }
                remove_list = [km.get('remove', {}) for km in k_metrics_list if 'remove' in km]
                if len(remove_list) > 0:
                    avg_k_metrics['remove'] = {
                        'precision': sum(r.get('precision', 0) for r in remove_list) / len(remove_list),
                        'recall': sum(r.get('recall', 0) for r in remove_list) / len(remove_list),
                        'f1': sum(r.get('f1', 0) for r in remove_list) / len(remove_list)
                    }
                combined_list = [km.get('breakpoint_combined', {}) for km in k_metrics_list if 'breakpoint_combined' in km]
                if len(combined_list) > 0:
                    avg_k_metrics['breakpoint_combined'] = {
                        'precision': sum(c.get('precision', 0) for c in combined_list) / len(combined_list),
                        'recall': sum(c.get('recall', 0) for c in combined_list) / len(combined_list),
                        'f1': sum(c.get('f1', 0) for c in combined_list) / len(combined_list)
                    }
                avg_tolerance_metrics_original[k_key] = avg_k_metrics
    avg_tolerance_metrics_processed = {}
    if len(all_tolerance_metrics_processed) > 0:
        for k in range(1, 6):
            k_key = f'k_{k}'
            k_metrics_list = [m.get(k_key, {}) for m in all_tolerance_metrics_processed if k_key in m]
            if len(k_metrics_list) > 0:
                avg_k_metrics = {}
                insert_list = [km.get('insert', {}) for km in k_metrics_list if 'insert' in km]
                if len(insert_list) > 0:
                    avg_k_metrics['insert'] = {
                        'precision': sum(i.get('precision', 0) for i in insert_list) / len(insert_list),
                        'recall': sum(i.get('recall', 0) for i in insert_list) / len(insert_list),
                        'f1': sum(i.get('f1', 0) for i in insert_list) / len(insert_list)
                    }
                remove_list = [km.get('remove', {}) for km in k_metrics_list if 'remove' in km]
                if len(remove_list) > 0:
                    avg_k_metrics['remove'] = {
                        'precision': sum(r.get('precision', 0) for r in remove_list) / len(remove_list),
                        'recall': sum(r.get('recall', 0) for r in remove_list) / len(remove_list),
                        'f1': sum(r.get('f1', 0) for r in remove_list) / len(remove_list)
                    }
                combined_list = [km.get('breakpoint_combined', {}) for km in k_metrics_list if 'breakpoint_combined' in km]
                if len(combined_list) > 0:
                    avg_k_metrics['breakpoint_combined'] = {
                        'precision': sum(c.get('precision', 0) for c in combined_list) / len(combined_list),
                        'recall': sum(c.get('recall', 0) for c in combined_list) / len(combined_list),
                        'f1': sum(c.get('f1', 0) for c in combined_list) / len(combined_list)
                    }
                
                avg_tolerance_metrics_processed[k_key] = avg_k_metrics

    # Timestamp-level and Interval-level evaluation output
    t_pre = processed_metrics.get('membership_precision', 0.0)
    t_rec = processed_metrics.get('membership_recall', 0.0)
    t_f1 = processed_metrics.get('membership_f1', 0.0)
    if len(all_interval_level_metrics_processed) > 0:
        i_pre = sum(m.get('I_Pre', 0.0) for m in all_interval_level_metrics_processed) / len(all_interval_level_metrics_processed)
        i_rec = sum(m.get('I_Rec', 0.0) for m in all_interval_level_metrics_processed) / len(all_interval_level_metrics_processed)
        i_f1 = sum(m.get('I_F1', 0.0) for m in all_interval_level_metrics_processed) / len(all_interval_level_metrics_processed)
    else:
        i_pre = i_rec = i_f1 = 0.0

    print("\nTimestamp-level evaluation (T-Pre, T-Rec, T-F1)")
    print("======================================================================")
    print(f"  T-Pre: {t_pre:.4f}")
    print(f"  T-Rec: {t_rec:.4f}")
    print(f"  T-F1: {t_f1:.4f}")
    print("\n======================================================================")
    print("Interval-level evaluation (I-Pre, I-Rec, I-F1)")
    print("======================================================================")
    print(f"  I-Pre: {i_pre:.4f}")
    print(f"  I-Rec: {i_rec:.4f}")
    print(f"  I-F1: {i_f1:.4f}")
    print("======================================================================\n")

    print("TMIA Finish ----------------")
