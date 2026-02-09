#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transfer Attack Evaluation Script
加载已保存的攻击模型，对不同的目标数据集进行评估
"""
import os
import torch
from attack.TMIA.get_training_data import (
    build_path_and_find_timestamps,
    select_test_samples_by_pattern,
    load_data_and_compute_confidence,
    create_window_data
)
from attack.TMIA.lstm import BiLSTMChangePoint
from attack.TMIA.inference import evaluate_model


def load_attack_model(model_path):

    checkpoint = torch.load(model_path, map_location='cpu')
    
    model_config = checkpoint['model_config']
    model = BiLSTMChangePoint(
        hidden_size=model_config['hidden_size'],
        fc_hidden_size=model_config['fc_hidden_size'],
        use_stats=model_config['use_stats']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint


def transfer_attack(source_config, target_config, model_path, num_attacks=3, num_samples_per_attack=1200):

    result_base_path = source_config.get('result_base_path') or target_config.get('result_base_path')
    if result_base_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
        result_base_path = os.path.join(project_root, "result")
    attack_method = target_config.get('attack_method', source_config.get('attack_method', 'Transfer_dataset'))
    output_dir = os.path.join(result_base_path, attack_method)
    os.makedirs(output_dir, exist_ok=True)

    if model_path is None:
        
        path_components = [result_base_path]
        
        if 'U_method' in source_config and source_config['U_method']:
            path_components.append(source_config['U_method'])
        
        path_components.extend([
            source_config['net_name'],
            source_config['dataset_name'],
            str(source_config['proportion']),
            source_config['attack_method'],
            str(source_config['window_size']),
            'model.pth'
        ])
        
        model_path = os.path.join(*path_components)
    
    model, checkpoint = load_attack_model(model_path)
    model_config = checkpoint['model_config']
    window_size = model_config['window_size']
    
    target_args = target_config.copy()
    target_args['save_base_path'] = target_config.get('save_base_path')
    target_save_path, target_timestamp_dirs = build_path_and_find_timestamps(target_args)
    
    is_llm = (target_config.get('U_method') == 'continuous_update_finetune_LLM')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    all_original_metrics = []
    all_processed_metrics = []
    all_interval_level_processed = []
    base_random_seed = target_config.get('random_seed', 42)
    balance_membership = target_config.get('balance_test_membership', True)
    target_member_ratio = target_config.get('target_test_member_ratio', 0.66)
    
    for attack_idx in range(num_attacks):
        attack_random_seed = base_random_seed + attack_idx if base_random_seed is not None else None
        verbose_selection = (attack_idx == 0)
        
        test_sample_indices, test_breakpoint_indices_dict, test_samples_breakpoint_info = \
            select_test_samples_by_pattern(
                save_path=target_save_path,
                timestamp_dirs=target_timestamp_dirs,
                num_insert_only=num_samples_per_attack // 3,
                num_remove_only=num_samples_per_attack // 3,
                num_insert_remove=num_samples_per_attack // 3,
                random_seed=attack_random_seed,
                balance_membership=balance_membership,
                target_member_ratio=target_member_ratio,
                verbose=verbose_selection
            )
        
        test_samples_timestamps_data = load_data_and_compute_confidence(
            timestamp_dirs=target_timestamp_dirs,
            sample_indices=test_sample_indices,
            samples_breakpoint_info=test_samples_breakpoint_info,
            is_llm=is_llm,
            use_true_label=target_config.get('use_true_label', True),
            ppl_normalization_params=None  # 不使用标准化
        )
        
        # 构建测试样本数据
        actual_timestamps = []
        for timestamp_dir in target_timestamp_dirs:
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
        
        test_window_data = create_window_data(
            test_samples_timestamps_data, 
            window_size=window_size, 
            skip_boundary=False
        )
        
        eval_args = target_config.copy()
        eval_args['window_size'] = window_size
        eval_args['output_dir'] = output_dir
        test_metrics = evaluate_model(
            model, test_samples_data, test_window_data, device, 
            eval_args, verbose=False
        )

        if isinstance(test_metrics, dict) and 'original_metrics' in test_metrics:
            all_original_metrics.append(test_metrics['original_metrics'])
            all_processed_metrics.append(test_metrics['processed_metrics'])
            if 'interval_level_metrics_processed' in test_metrics:
                all_interval_level_processed.append(test_metrics['interval_level_metrics_processed'])
    
    # 5. 计算平均结果
    if len(all_original_metrics) == 0:
        return None
    
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

    # Interval-level 指标平均（对应 processed）
    avg_interval_level_processed = None
    if all_interval_level_processed:
        il_keys = ['I_Pre', 'I_Rec', 'I_F1']
        avg_interval_level_processed = {}
        for key in il_keys:
            values = [m[key] for m in all_interval_level_processed if key in m]
            avg_interval_level_processed[key] = sum(values) / len(values) if values else 0.0
    
    # 6. 打印结果

    return {
        'original_metrics': avg_original_metrics,
        'processed_metrics': avg_processed_metrics,
        'interval_level_metrics_processed': avg_interval_level_processed,
        'source_config': source_config,
        'target_config': target_config
    }

def transfer(args):
    attack_method = args.get('attack_method')
    window_size = args.get('window_size')
    num_attacks = 1
    all_results = {}
    
    if attack_method == 'Transfer_unlearning':
        # Unlearning Method Transfer Evaluation

        shadow_methods = ['continuous_update_finetune', 'continuous_update_finetune_GA', 'continuous_update_finetune_NPO']
        target_methods = ['continuous_update_finetune', 'continuous_update_finetune_GA', 'continuous_update_finetune_NPO']
        method_names = ['Finetune', 'GA', 'NPO']
        method_name_map = {
            'continuous_update_finetune': 'Finetune',
            'continuous_update_finetune_GA': 'GA',
            'continuous_update_finetune_NPO': 'NPO'
        }
        dataset_name = 'sst5'
        net_name = 'pythia70m'

        #for cifar10
        shadow_methods = ['continuous_update_finetune', 'continuous_update_finetune_GA', 'continuous_update_finetune_scrub']
        target_methods = ['continuous_update_finetune', 'continuous_update_finetune_GA', 'continuous_update_finetune_scrub']
        method_names = ['Finetune', 'GA', 'scrub']  # 用于显示
        method_name_map = {
            'continuous_update_finetune': 'Finetune',
            'continuous_update_finetune_GA': 'GA',
            'continuous_update_finetune_scrub': 'scrub'
        }
        dataset_name = 'cifar10'
        net_name = 'simple_cnn'

        proportion = 0.01
        
        print("="*80)
        print("Unlearning Method Transfer Evaluation (SST5, Pythia-70M)")
        print("="*80)
        print(f"Shadow Methods: {method_names}")
        print(f"Target Methods: {method_names}")
        print(f"Dataset: {dataset_name}, Model: {net_name}")
        print(f"Total combinations: {len(shadow_methods) * len(target_methods)}")
        print("="*80 + "\n")
        
        for shadow_method in shadow_methods:
            for target_method in target_methods:
                shadow_name = method_name_map[shadow_method]
                target_name = method_name_map[target_method]
                print(f"\n{'='*80}")
                print(f"Evaluating: Shadow={shadow_name} -> Target={target_name}")
                print(f"{'='*80}\n")
                
                source_config = {
                    'net_name': net_name,
                    'dataset_name': dataset_name,
                    'proportion': proportion,
                    'attack_method': 'TMIA',
                    'window_size': window_size,
                    'U_method': shadow_method  # 添加 U_method 用于构建模型路径
                }
                
                target_config = {
                    'U_method': target_method,
                    'net_name': net_name,
                    'dataset_name': dataset_name,
                    'proportion_of_group_unlearn': proportion,
                    'trial': 0,
                    'is_shadow': False,
                    'use_true_label': True,
                    'random_seed': 42
                }


        header = "Shadow\\Target"
        print(f"\n{header:<15}", end='')
        for target_method in target_methods:
            print(f"{method_name_map[target_method]:<15}", end='')
        print()
        print("-" * 80)
        
        for shadow_method in shadow_methods:
            print(f"{method_name_map[shadow_method]:<15}", end='')
            for target_method in target_methods:
                key = (shadow_method, target_method)
                if key in all_results:
                    metrics = all_results[key]['processed_metrics']
                    precision = metrics.get('membership_precision', 0.0)
                    recall = metrics.get('membership_recall', 0.0)
                    print(f"P:{precision:.2f} R:{recall:.2f}  ", end='')
                else:
                    print(f"{'N/A':<15}", end='')
            print()
        
        print("="*80)
        
        # 保存结果
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
        result_dir = os.path.join(project_root, "result", attack_method)
        os.makedirs(result_dir, exist_ok=True)
        
        results_file = os.path.join(result_dir, "results.txt")
        with open(results_file, 'w', encoding='utf-8') as f:
            header = "Shadow\\Target"
            f.write(f"{header:<15}")
            for target_method in target_methods:
                f.write(f"{method_name_map[target_method]:<25}")
            f.write("\n")
            f.write("-" * 120 + "\n")
            
            for shadow_method in shadow_methods:
                f.write(f"{method_name_map[shadow_method]:<15}")
                for target_method in target_methods:
                    key = (shadow_method, target_method)
                    if key in all_results:
                        metrics = all_results[key]['processed_metrics']
                        precision = metrics.get('membership_precision', 0.0)
                        recall = metrics.get('membership_recall', 0.0)
                        f1 = metrics.get('membership_f1', 0.0)
                        f.write(f"{precision*100:.2f} / {recall*100:.2f} / {f1*100:.2f}  ")
                    else:
                        f.write(f"{'N/A':<25}")
                f.write("\n")
        
        results_interval_file = os.path.join(result_dir, "results_interval.txt")
        with open(results_interval_file, 'w', encoding='utf-8') as f:
            f.write(f"{header:<15}")
            for target_method in target_methods:
                f.write(f"{method_name_map[target_method]:<25}")
            f.write("\n")
            f.write("-" * 120 + "\n")
            for shadow_method in shadow_methods:
                f.write(f"{method_name_map[shadow_method]:<15}")
                for target_method in target_methods:
                    key = (shadow_method, target_method)
                    if key in all_results and all_results[key].get('interval_level_metrics_processed'):
                        m = all_results[key]['interval_level_metrics_processed']
                        i_pre = m.get('I_Pre', 0.0)
                        i_rec = m.get('I_Rec', 0.0)
                        i_f1 = m.get('I_F1', 0.0)
                        f.write(f"{i_pre*100:.2f} / {i_rec*100:.2f} / {i_f1*100:.2f}  ")
                    else:
                        f.write(f"{'N/A':<25}")
                f.write("\n")
        

    elif attack_method == 'Transfer_model':
        # Model Transfer Evaluation

        shadow_models = ['pythia70m', 'T5', 'gpt2']
        target_models = ['pythia70m', 'T5', 'gpt2']
        dataset_name = 'sst5'

        shadow_models = ['resnet18', 'simple_cnn', 'mobilenet']
        target_models = ['resnet18', 'simple_cnn', 'mobilenet']
        dataset_name = 'cifar10'

        proportion = 0.01
        
        print("="*80)
        print("Model Transfer Evaluation (SST5, Finetune)")
        print("="*80)
        print(f"Shadow Models: {shadow_models}")
        print(f"Target Models: {target_models}")
        print(f"Dataset: {dataset_name}")
        print(f"Total combinations: {len(shadow_models) * len(target_models)}")
        print("="*80 + "\n")
        
        for shadow_model in shadow_models:
            for target_model in target_models:
                print(f"\n{'='*80}")
                print(f"Evaluating: Shadow={shadow_model.upper()} -> Target={target_model.upper()}")
                print(f"{'='*80}\n")
                
                source_config = {
                    'net_name': shadow_model,
                    'dataset_name': dataset_name,
                    'proportion': proportion,
                    'attack_method': 'TMIA',
                    'window_size': window_size,
                    'U_method': 'continuous_update_finetune'
                }
                
                target_config = {
                    'U_method': 'continuous_update_finetune',
                    'net_name': target_model,
                    'dataset_name': dataset_name,
                    'proportion_of_group_unlearn': proportion,
                    'trial': 0,
                    'is_shadow': False,
                    'use_true_label': True,
                    'random_seed': 42
                }

        
        for shadow in shadow_models:
            print(f"{shadow.upper():<15}", end='')
            for target in target_models:
                key = (shadow, target)
                if key in all_results:
                    metrics = all_results[key]['processed_metrics']
                    precision = metrics.get('membership_precision', 0.0)
                    recall = metrics.get('membership_recall', 0.0)
                    print(f"P:{precision:.2f} R:{recall:.2f}  ", end='')
                else:
                    print(f"{'N/A':<15}", end='')
            print()
        
        print("="*80)
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
        result_dir = os.path.join(project_root, "result", attack_method)
        os.makedirs(result_dir, exist_ok=True)
        
        results_file = os.path.join(result_dir, "results.txt")
        with open(results_file, 'w', encoding='utf-8') as f:
            header = "Shadow\\Target"
            f.write(f"{header:<15}")
            for target in target_models:
                f.write(f"{target.upper():<25}")
            f.write("\n")
            f.write("-" * 120 + "\n")
            
            for shadow in shadow_models:
                f.write(f"{shadow.upper():<15}")
                for target in target_models:
                    key = (shadow, target)
                    if key in all_results:
                        metrics = all_results[key]['processed_metrics']
                        precision = metrics.get('membership_precision', 0.0)
                        recall = metrics.get('membership_recall', 0.0)
                        f1 = metrics.get('membership_f1', 0.0)
                        f.write(f"{precision*100:.2f} / {recall*100:.2f} / {f1*100:.2f}  ")
                    else:
                        f.write(f"{'N/A':<25}")
                f.write("\n")
        
        results_interval_file = os.path.join(result_dir, "results_interval.txt")
        with open(results_interval_file, 'w', encoding='utf-8') as f:
            f.write(f"{header:<15}")
            for target in target_models:
                f.write(f"{target.upper():<25}")
            f.write("\n")
            f.write("-" * 120 + "\n")
            for shadow in shadow_models:
                f.write(f"{shadow.upper():<15}")
                for target in target_models:
                    key = (shadow, target)
                    if key in all_results and all_results[key].get('interval_level_metrics_processed'):
                        m = all_results[key]['interval_level_metrics_processed']
                        i_pre = m.get('I_Pre', 0.0)
                        i_rec = m.get('I_Rec', 0.0)
                        i_f1 = m.get('I_F1', 0.0)
                        f.write(f"{i_pre*100:.2f} / {i_rec*100:.2f} / {i_f1*100:.2f}  ")
                    else:
                        f.write(f"{'N/A':<25}")
                f.write("\n")
        
        print(f"\n结果已保存到: {results_file}")
        print(f"Interval-level 结果已保存到: {results_interval_file}")
    
    else:
        # Dataset Transfer Evaluation
        shadow_datasets = ['sst5', 'news20', 'mnli']
        target_datasets = ['sst5', 'news20', 'mnli']
        net_name = 'pythia70m'
        #
        shadow_datasets = ['cifar10', 'svhn', 'cinic10']
        target_datasets = ['cifar10', 'svhn', 'cinic10']
        net_name = 'simple_cnn'
        
        def get_proportion(dataset_name):
            if dataset_name == 'mnli':
                return 0.005
            elif dataset_name =='cinic10':
                return 0.001
            else:
                return  0.01


        print("="*80)
        print("Dataset Transfer Evaluation (Pythia-70M)")
        print("="*80)
        print(f"Shadow Datasets: {shadow_datasets}")
        print(f"Target Datasets: {target_datasets}")
        print(f"Total combinations: {len(shadow_datasets) * len(target_datasets)}")
        print("="*80 + "\n")
        
        for shadow_dataset in shadow_datasets:
            for target_dataset in target_datasets:
                print(f"\n{'='*80}")
                print(f"Evaluating: Shadow={shadow_dataset.upper()} -> Target={target_dataset.upper()}")
                print(f"{'='*80}\n")
                
                source_proportion = get_proportion(shadow_dataset)
                target_proportion = get_proportion(target_dataset)
                
                source_config = {
                    'net_name': net_name,
                    'dataset_name': shadow_dataset,
                    'proportion': source_proportion,
                    'attack_method': 'TMIA',
                    'window_size': window_size,
                    'U_method': 'continuous_update_finetune'  # 添加 U_method 用于构建模型路径
                }
                target_config = {
                    'U_method': 'continuous_update_finetune',
                    'net_name': net_name,
                    'dataset_name': target_dataset,
                    'proportion_of_group_unlearn': target_proportion,
                    'trial': 0,
                    'is_shadow': False,
                    'use_true_label': True,
                    'random_seed': 42
                }
                try:
                    result = transfer_attack(
                        source_config, target_config, 
                        model_path=None, 
                        num_attacks=num_attacks
                    )
                    if result:
                        all_results[(shadow_dataset, target_dataset)] = result
                except Exception as e:
                    print(f"错误: {e}")
                    import traceback
                    traceback.print_exc()
        
        print("\n" + "="*80)
        print("Dataset Transfer 汇总结果")
        print("="*80)
        header = "Shadow\\Target"
        print(f"\n{header:<15}", end='')
        for target in target_datasets:
            print(f"{target.upper():<15}", end='')
        print()
        print("-" * 80)
        
        for shadow in shadow_datasets:
            print(f"{shadow.upper():<15}", end='')
            for target in target_datasets:
                key = (shadow, target)
                if key in all_results:
                    metrics = all_results[key]['processed_metrics']
                    precision = metrics.get('membership_precision', 0.0)
                    recall = metrics.get('membership_recall', 0.0)
                    print(f"P:{precision:.2f} R:{recall:.2f}  ", end='')
                else:
                    print(f"{'N/A':<15}", end='')
            print()
        
        print("="*80)
        
        # 保存结果到文件
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
        result_dir = os.path.join(project_root, "result", attack_method)
        os.makedirs(result_dir, exist_ok=True)
        
        results_file = os.path.join(result_dir, "results.txt")
        with open(results_file, 'w', encoding='utf-8') as f:
            header = "Shadow\\Target"
            f.write(f"{header:<15}")
            for target in target_datasets:
                f.write(f"{target.upper():<25}")
            f.write("\n")
            f.write("-" * 120 + "\n")
            
            for shadow in shadow_datasets:
                f.write(f"{shadow.upper():<15}")
                for target in target_datasets:
                    key = (shadow, target)
                    if key in all_results:
                        metrics = all_results[key]['processed_metrics']
                        precision = metrics.get('membership_precision', 0.0)
                        recall = metrics.get('membership_recall', 0.0)
                        f1 = metrics.get('membership_f1', 0.0)
                        f.write(f"{precision*100:.2f} / {recall*100:.2f} / {f1*100:.2f}  ")
                    else:
                        f.write(f"{'N/A':<25}")
                f.write("\n")
        results_interval_file = os.path.join(result_dir, "results_interval.txt")
        with open(results_interval_file, 'w', encoding='utf-8') as f:
            f.write(f"{header:<15}")
            for target in target_datasets:
                f.write(f"{target.upper():<25}")
            f.write("\n")
            f.write("-" * 120 + "\n")
            for shadow in shadow_datasets:
                f.write(f"{shadow.upper():<15}")
                for target in target_datasets:
                    key = (shadow, target)
                    if key in all_results and all_results[key].get('interval_level_metrics_processed'):
                        m = all_results[key]['interval_level_metrics_processed']
                        i_pre = m.get('I_Pre', 0.0)
                        i_rec = m.get('I_Rec', 0.0)
                        i_f1 = m.get('I_F1', 0.0)
                        f.write(f"{i_pre*100:.2f} / {i_rec*100:.2f} / {i_f1*100:.2f}  ")
                    else:
                        f.write(f"{'N/A':<25}")
                f.write("\n")
        print(f"\n结果已保存到: {results_file}")
        print(f"Interval-level 结果已保存到: {results_interval_file}")

