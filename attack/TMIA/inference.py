#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from collections import defaultdict
import json
import os

def breakpoint_to_membership_timestamps(breakpoints, num_timestamps):
    membership = set()
    
    sorted_breakpoints = sorted(breakpoints, key=lambda x: x[0])
    
    if len(sorted_breakpoints) == 0:
        return membership
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


def breakpoint_to_membership_intervals(breakpoints, num_timestamps):
    intervals = []
    sorted_bps = sorted(breakpoints, key=lambda x: x[0])
    if not sorted_bps:
        return intervals

    first_ts, first_type = sorted_bps[0]
    in_segment = False
    start = None

    if first_type == 2:
        seg = set(range(0, first_ts))
        if seg:
            intervals.append(seg)
        in_segment = False
    else:
        start = first_ts
        in_segment = True

    for ts, bp_type in sorted_bps[1:]:
        if bp_type == 2:
            if in_segment and start is not None:
                seg = set(range(start, ts))
                if seg:
                    intervals.append(seg)
                in_segment = False
        else:
            if not in_segment:
                start = ts
                in_segment = True

    if in_segment and start is not None:
        seg = set(range(start, num_timestamps))
        if seg:
            intervals.append(seg)
    return intervals


def compute_class_metrics(predictions):
    class_stats = {0: {'tp': 0, 'fp': 0, 'fn': 0},
                   1: {'tp': 0, 'fp': 0, 'fn': 0},
                   2: {'tp': 0, 'fp': 0, 'fn': 0}}
    
    total_correct = 0
    total_count = len(predictions)
    
    for pred in predictions:
        pred_label = pred['predicted_label']
        true_label = pred['true_label']
        
        if pred_label == true_label:
            total_correct += 1
            class_stats[true_label]['tp'] += 1
        else:
            class_stats[pred_label]['fp'] += 1
            class_stats[true_label]['fn'] += 1
    
    total_accuracy = total_correct / total_count if total_count > 0 else 0.0
    class_metrics = {}
    for class_label in [0, 1, 2]:
        stats = class_stats[class_label]
        tp = stats['tp']
        fp = stats['fp']
        fn = stats['fn']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        class_metrics[f'class_{class_label}'] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    breakpoint_tp = class_stats[1]['tp'] + class_stats[2]['tp']
    breakpoint_fp = class_stats[1]['fp'] + class_stats[2]['fp']
    breakpoint_fn = class_stats[1]['fn'] + class_stats[2]['fn']
    
    breakpoint_precision = breakpoint_tp / (breakpoint_tp + breakpoint_fp) if (breakpoint_tp + breakpoint_fp) > 0 else 0.0
    breakpoint_recall = breakpoint_tp / (breakpoint_tp + breakpoint_fn) if (breakpoint_tp + breakpoint_fn) > 0 else 0.0
    breakpoint_f1 = 2 * (breakpoint_precision * breakpoint_recall) / (breakpoint_precision + breakpoint_recall) if (breakpoint_precision + breakpoint_recall) > 0 else 0.0
    
    class_metrics['breakpoint_combined'] = {
        'precision': breakpoint_precision,
        'recall': breakpoint_recall,
        'f1': breakpoint_f1,
        'tp': breakpoint_tp,
        'fp': breakpoint_fp,
        'fn': breakpoint_fn
    }
    return {
        'total_accuracy': total_accuracy,
        **class_metrics
    }
def compute_hit_rate_with_tolerance(predictions_by_sample, tolerance_theta=1):
    total_insert_hits = 0
    total_insert_total = 0
    total_remove_hits = 0
    total_remove_total = 0
    
    for sample_idx, sample_predictions in predictions_by_sample.items():
        sorted_predictions = sorted(sample_predictions, key=lambda x: x['timestamp_index'])
        
        true_insert = [p['timestamp_index'] for p in sorted_predictions if p['true_label'] == 1]
        true_remove = [p['timestamp_index'] for p in sorted_predictions if p['true_label'] == 2]
        
        pred_insert = [p['timestamp_index'] for p in sorted_predictions if p['predicted_label'] == 1]
        pred_remove = [p['timestamp_index'] for p in sorted_predictions if p['predicted_label'] == 2]
        
        for true_ts in true_insert:
            total_insert_total += 1
            for pred_ts in pred_insert:
                if abs(pred_ts - true_ts) <= tolerance_theta:
                    total_insert_hits += 1
                    break
        
        for true_ts in true_remove:
            total_remove_total += 1
            for pred_ts in pred_remove:
                if abs(pred_ts - true_ts) <= tolerance_theta:
                    total_remove_hits += 1
                    break
    
    insert_hit_rate = total_insert_hits / total_insert_total if total_insert_total > 0 else 0.0
    remove_hit_rate = total_remove_hits / total_remove_total if total_remove_total > 0 else 0.0
    return {
        'insert_hit_rate': insert_hit_rate,
        'remove_hit_rate': remove_hit_rate,
        'insert_hits': total_insert_hits,
        'insert_total': total_insert_total,
        'remove_hits': total_remove_hits,
        'remove_total': total_remove_total
    }


def compute_metrics_with_tolerance(predictions_by_sample, tolerance_k=1):
    true_insert_by_sample = {}
    true_remove_by_sample = {}
    pred_insert_by_sample = {}
    pred_remove_by_sample = {}
    
    for sample_idx, sample_predictions in predictions_by_sample.items():
        sorted_predictions = sorted(sample_predictions, key=lambda x: x['timestamp_index'])
        
        true_insert = [p['timestamp_index'] for p in sorted_predictions if p['true_label'] == 1]
        true_remove = [p['timestamp_index'] for p in sorted_predictions if p['true_label'] == 2]
        
        pred_insert = [p['timestamp_index'] for p in sorted_predictions if p['predicted_label'] == 1]
        pred_remove = [p['timestamp_index'] for p in sorted_predictions if p['predicted_label'] == 2]
        
        true_insert_by_sample[sample_idx] = true_insert
        true_remove_by_sample[sample_idx] = true_remove
        pred_insert_by_sample[sample_idx] = pred_insert
        pred_remove_by_sample[sample_idx] = pred_remove
    
    def match_with_tolerance(pred_points, true_points, k):
        matched_pairs = []
        unmatched_pred = list(pred_points)
        unmatched_true = list(true_points)
        
        for pred_ts in list(unmatched_pred):
            best_match = None
            min_dist = float('inf')
            
            for true_ts in unmatched_true:
                dist = abs(pred_ts - true_ts)
                if dist <= k and dist < min_dist:
                    min_dist = dist
                    best_match = true_ts
            
            if best_match is not None:
                matched_pairs.append((pred_ts, best_match))
                unmatched_pred.remove(pred_ts)
                unmatched_true.remove(best_match)
        
        return matched_pairs, unmatched_pred, unmatched_true
    
    all_matched_insert = []
    all_unmatched_pred_insert = []
    all_unmatched_true_insert = []
    
    for sample_idx in predictions_by_sample.keys():
        matched, unmatched_pred, unmatched_true = match_with_tolerance(
            pred_insert_by_sample.get(sample_idx, []),
            true_insert_by_sample.get(sample_idx, []),
            tolerance_k
        )
        all_matched_insert.extend(matched)
        all_unmatched_pred_insert.extend(unmatched_pred)
        all_unmatched_true_insert.extend(unmatched_true)
    
    insert_tp = len(all_matched_insert)
    insert_fp = len(all_unmatched_pred_insert)
    insert_fn = len(all_unmatched_true_insert)
    
    insert_precision = insert_tp / (insert_tp + insert_fp) if (insert_tp + insert_fp) > 0 else 0.0
    insert_recall = insert_tp / (insert_tp + insert_fn) if (insert_tp + insert_fn) > 0 else 0.0
    insert_f1 = 2 * (insert_precision * insert_recall) / (insert_precision + insert_recall) if (insert_precision + insert_recall) > 0 else 0.0
    
    all_matched_remove = []
    all_unmatched_pred_remove = []
    all_unmatched_true_remove = []
    
    for sample_idx in predictions_by_sample.keys():
        matched, unmatched_pred, unmatched_true = match_with_tolerance(
            pred_remove_by_sample.get(sample_idx, []),
            true_remove_by_sample.get(sample_idx, []),
            tolerance_k
        )
        all_matched_remove.extend(matched)
        all_unmatched_pred_remove.extend(unmatched_pred)
        all_unmatched_true_remove.extend(unmatched_true)
    
    remove_tp = len(all_matched_remove)
    remove_fp = len(all_unmatched_pred_remove)
    remove_fn = len(all_unmatched_true_remove)
    
    remove_precision = remove_tp / (remove_tp + remove_fp) if (remove_tp + remove_fp) > 0 else 0.0
    remove_recall = remove_tp / (remove_tp + remove_fn) if (remove_tp + remove_fn) > 0 else 0.0
    remove_f1 = 2 * (remove_precision * remove_recall) / (remove_precision + remove_recall) if (remove_precision + remove_recall) > 0 else 0.0
    
    combined_tp = insert_tp + remove_tp
    combined_fp = insert_fp + remove_fp
    combined_fn = insert_fn + remove_fn
    
    combined_precision = combined_tp / (combined_tp + combined_fp) if (combined_tp + combined_fp) > 0 else 0.0
    combined_recall = combined_tp / (combined_tp + combined_fn) if (combined_tp + combined_fn) > 0 else 0.0
    combined_f1 = 2 * (combined_precision * combined_recall) / (combined_precision + combined_recall) if (combined_precision + combined_recall) > 0 else 0.0
    
    return {
        'insert': {
            'precision': insert_precision,
            'recall': insert_recall,
            'f1': insert_f1,
            'tp': insert_tp,
            'fp': insert_fp,
            'fn': insert_fn
        },
        'remove': {
            'precision': remove_precision,
            'recall': remove_recall,
            'f1': remove_f1,
            'tp': remove_tp,
            'fp': remove_fp,
            'fn': remove_fn
        },
        'breakpoint_combined': {
            'precision': combined_precision,
            'recall': combined_recall,
            'f1': combined_f1,
            'tp': combined_tp,
            'fp': combined_fp,
            'fn': combined_fn
        }
    }


def compute_membership_interval_metrics(predictions_by_sample, num_timestamps, verbose=True):
    all_predicted_memberships = []
    all_true_memberships = []
    all_predictions = []
    for sample_preds in predictions_by_sample.values():
        all_predictions.extend(sample_preds)

    class_correct = {0: 0, 1: 0, 2: 0}
    class_predicted = {0: 0, 1: 0, 2: 0}
    class_true = {0: 0, 1: 0, 2: 0}
    
    for p in all_predictions:
        pred_label = p['predicted_label']
        true_label = p['true_label']
        class_predicted[pred_label] += 1
        class_true[true_label] += 1
        
        if pred_label == true_label:
            class_correct[pred_label] += 1
    
    class_accuracies = {}
    for class_label in [0, 1, 2]:
        acc = class_correct[class_label] / class_predicted[class_label] if class_predicted[class_label] > 0 else 0.0
        class_accuracies[f'class_{class_label}_accuracy'] = acc
        class_accuracies[f'class_{class_label}_correct'] = class_correct[class_label]
        class_accuracies[f'class_{class_label}_predicted'] = class_predicted[class_label]
        class_accuracies[f'class_{class_label}_true'] = class_true[class_label]
    
    insert_precision = class_accuracies.get('class_1_accuracy', 0.0)
    insert_recall = class_correct[1] / class_true[1] if class_true[1] > 0 else 0.0
    insert_f1 = 2 * (insert_precision * insert_recall) / (insert_precision + insert_recall) if (insert_precision + insert_recall) > 0 else 0.0
    
    remove_precision = class_accuracies.get('class_2_accuracy', 0.0)
    remove_recall = class_correct[2] / class_true[2] if class_true[2] > 0 else 0.0
    remove_f1 = 2 * (remove_precision * remove_recall) / (remove_precision + remove_recall) if (remove_precision + remove_recall) > 0 else 0.0
    
    for sample_idx, sample_predictions in predictions_by_sample.items():
        sorted_predictions = sorted(sample_predictions, key=lambda x: x['timestamp_index'])
        
        pred_insert_points = [p['timestamp_index'] for p in sorted_predictions if p['predicted_label'] == 1]
        pred_remove_points = [p['timestamp_index'] for p in sorted_predictions if p['predicted_label'] == 2]
        
        true_insert_points = [p['timestamp_index'] for p in sorted_predictions if p['true_label'] == 1]
        true_remove_points = [p['timestamp_index'] for p in sorted_predictions if p['true_label'] == 2]
        
        pred_breakpoints = [(ts, 1) for ts in pred_insert_points] + [(ts, 2) for ts in pred_remove_points]
        pred_membership = breakpoint_to_membership_timestamps(pred_breakpoints, num_timestamps)
        all_predicted_memberships.append(pred_membership)
        
        true_breakpoints = [(ts, 1) for ts in true_insert_points] + [(ts, 2) for ts in true_remove_points]
        true_membership = breakpoint_to_membership_timestamps(true_breakpoints, num_timestamps)
        all_true_memberships.append(true_membership)
    
    total_intersection = 0
    total_predicted = 0
    total_true = 0
    total_false_positive = 0
    total_false_negative = 0
    for pred_set, true_set in zip(all_predicted_memberships, all_true_memberships):
        intersection = pred_set & true_set
        false_positive = pred_set - true_set
        false_negative = true_set - pred_set
        
        total_intersection += len(intersection)
        total_predicted += len(pred_set)
        total_true += len(true_set)
        total_false_positive += len(false_positive)
        total_false_negative += len(false_negative)
    
    precision = total_intersection / total_predicted if total_predicted > 0 else 0.0
    recall = total_intersection / total_true if total_true > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    result_dict = {
        'membership_precision': precision,
        'membership_recall': recall,
        'membership_f1': f1,
        'insert_precision': insert_precision,
        'insert_recall': insert_recall,
        'insert_f1': insert_f1,
        'remove_precision': remove_precision,
        'remove_recall': remove_recall,
        'remove_f1': remove_f1,
    }
    return result_dict


def compute_interval_level_metrics(predictions_by_sample, num_timestamps, verbose=True):
    sum_i_pre = 0.0
    sum_i_rec = 0.0
    n_samples = 0
    total_pred = 0
    total_true = 0

    for sample_idx in sorted(predictions_by_sample.keys()):
        sample_predictions = predictions_by_sample[sample_idx]
        sorted_predictions = sorted(sample_predictions, key=lambda x: x['timestamp_index'])
        pred_insert_points = [p['timestamp_index'] for p in sorted_predictions if p['predicted_label'] == 1]
        pred_remove_points = [p['timestamp_index'] for p in sorted_predictions if p['predicted_label'] == 2]
        true_insert_points = [p['timestamp_index'] for p in sorted_predictions if p['true_label'] == 1]
        true_remove_points = [p['timestamp_index'] for p in sorted_predictions if p['true_label'] == 2]

        pred_bps = [(ts, 1) for ts in pred_insert_points] + [(ts, 2) for ts in pred_remove_points]
        true_bps = [(ts, 1) for ts in true_insert_points] + [(ts, 2) for ts in true_remove_points]
        pred_intervals = breakpoint_to_membership_intervals(pred_bps, num_timestamps)
        true_intervals = breakpoint_to_membership_intervals(true_bps, num_timestamps)

        total_pred += len(pred_intervals)
        total_true += len(true_intervals)

        def overlap(ii, jj):
            return len(ii & jj) > 0

        G_pred = []
        for I_i in pred_intervals:
            matches = [j for j, Ij in enumerate(true_intervals) if overlap(I_i, Ij)]
            G_pred.append(matches)

        G_true = []
        for Ij in true_intervals:
            matches = [i for i, I_i in enumerate(pred_intervals) if overlap(I_i, Ij)]
            G_true.append(matches)

        i_pre_list = []
        for i, I_i in enumerate(pred_intervals):
            g = G_pred[i]
            if not g:
                i_pre_list.append(0.0)
                continue
            vals = [len(I_i & true_intervals[j]) / len(I_i) for j in g]
            i_pre_list.append(sum(vals) / len(vals))

        i_rec_list = []
        for j, Ij in enumerate(true_intervals):
            g = G_true[j]
            if not g:
                i_rec_list.append(0.0)
                continue
            vals = [len(pred_intervals[i] & Ij) / len(Ij) for i in g]
            i_rec_list.append(sum(vals) / len(vals))

        n_pred = len(pred_intervals)
        n_true = len(true_intervals)
        I_Pre_x = sum(i_pre_list) / n_pred if n_pred else 0.0
        I_Rec_x = sum(i_rec_list) / n_true if n_true else 0.0
        denom = I_Pre_x + I_Rec_x
        I_F1_x = (2.0 * I_Pre_x * I_Rec_x / denom) if denom > 0 else 0.0

        sum_i_pre += I_Pre_x
        sum_i_rec += I_Rec_x
        n_samples += 1

    I_Pre_macro = sum_i_pre / n_samples if n_samples else 0.0
    I_Rec_macro = sum_i_rec / n_samples if n_samples else 0.0
    denom_macro = I_Pre_macro + I_Rec_macro
    I_F1_macro = (2.0 * I_Pre_macro * I_Rec_macro / denom_macro) if denom_macro > 0 else 0.0

    return {
        'I_Pre': I_Pre_macro,
        'I_Rec': I_Rec_macro,
        'I_F1': I_F1_macro,
    }


def predict_breakpoints_from_model(model, window_data, device, window_size=5, use_stats=True):
    import torch.nn.functional as F
    
    def compute_stats(window_seq, win_size):
        seq_len = len(window_seq)
        center_idx = win_size
        
        if center_idx > 0:
            front_window = window_seq[:center_idx]
            front_mean = torch.mean(front_window)
            front_var = torch.var(front_window, unbiased=False)
        else:
            front_mean = torch.tensor(0.0)
            front_var = torch.tensor(0.0)
        
        if center_idx + 1 < seq_len:
            back_window = window_seq[center_idx + 1:]
            back_mean = torch.mean(back_window)
            back_var = torch.var(back_window, unbiased=False)
        else:
            back_mean = torch.tensor(0.0)
            back_var = torch.tensor(0.0)
        
        return torch.stack([front_mean, front_var, back_mean, back_var])
    
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for item in window_data:
            window_sequence = item['window_sequence']
            seq_len = 2 * window_size + 1
            if len(window_sequence) > seq_len:
                window_sequence = window_sequence[:seq_len]
            elif len(window_sequence) < seq_len:
                padding = [window_sequence[-1]] * (seq_len - len(window_sequence))
                window_sequence = window_sequence + padding
            
            x = torch.tensor(window_sequence, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
            
            model_use_stats = getattr(model, 'use_stats', use_stats)
            if model_use_stats:
                window_seq_tensor = torch.tensor(window_sequence, dtype=torch.float32)
                stats = compute_stats(window_seq_tensor, window_size).unsqueeze(0).to(device)
                output = model(x, stats=stats)
            else:
                output = model(x)
            
            probs = F.softmax(output, dim=1)
            predicted_label = torch.argmax(output, dim=1).item()
            confidence_scores = probs[0].cpu().numpy().tolist()
            
            predictions.append({
                'sample_index': item['sample_index'],
                'timestamp_index': item['timestamp_index'],
                'predicted_label': predicted_label,
                'true_label': item['label'],
                'confidence_scores': confidence_scores,
                'window_sequence': window_sequence
            })
    
    return predictions


def collapse_consecutive_breakpoints(predictions_by_sample):
    processed_predictions = {}
    
    for sample_idx, sample_predictions in predictions_by_sample.items():
        sorted_predictions = sorted(sample_predictions, key=lambda x: x['timestamp_index'])
        
        entrance_indices = [i for i, p in enumerate(sorted_predictions) if p['predicted_label'] == 1]
        exit_indices = [i for i, p in enumerate(sorted_predictions) if p['predicted_label'] == 2]
        
        to_remove_entrance = set()
        if len(entrance_indices) > 0:
            runs = []
            current_run = [entrance_indices[0]]
            for i in range(1, len(entrance_indices)):
                if entrance_indices[i] == entrance_indices[i-1] + 1:
                    current_run.append(entrance_indices[i])
                else:
                    if len(current_run) > 1:
                        runs.append(current_run)
                    current_run = [entrance_indices[i]]
            if len(current_run) > 1:
                runs.append(current_run)
            
            for run in runs:
                best_idx = max(run, key=lambda i: sorted_predictions[i]['confidence_scores'][1])
                to_remove_entrance.update(run)
                to_remove_entrance.discard(best_idx)
        
        to_remove_exit = set()
        if len(exit_indices) > 0:
            runs = []
            current_run = [exit_indices[0]]
            for i in range(1, len(exit_indices)):
                if exit_indices[i] == exit_indices[i-1] + 1:
                    current_run.append(exit_indices[i])
                else:
                    if len(current_run) > 1:
                        runs.append(current_run)
                    current_run = [exit_indices[i]]
            if len(current_run) > 1:
                runs.append(current_run)
            
            for run in runs:
                best_idx = max(run, key=lambda i: sorted_predictions[i]['confidence_scores'][2])
                to_remove_exit.update(run)
                to_remove_exit.discard(best_idx)
        
        processed = []
        for i, pred in enumerate(sorted_predictions):
            if i in to_remove_entrance:
                new_pred = pred.copy()
                new_pred['predicted_label'] = 0
                processed.append(new_pred)
            elif i in to_remove_exit:
                new_pred = pred.copy()
                new_pred['predicted_label'] = 0
                processed.append(new_pred)
            else:
                processed.append(pred)
        
        processed_predictions[sample_idx] = processed
    
    return processed_predictions
def evaluate_model(model, test_samples_data, test_window_data, device, args, verbose=True):
    window_size = args.get('window_size', 5)
    verbose = args.get('verbose', verbose)
    
    if len(test_samples_data) > 0:
        num_timestamps = len(test_samples_data[0].get('timestamps', []))
    else:
        num_timestamps = 0
    
    use_stats = True
    predictions = predict_breakpoints_from_model(model, test_window_data, device, window_size, use_stats=use_stats)
    
    predictions_by_sample = defaultdict(list)
    for pred in predictions:
        sample_idx = pred['sample_index']
        predictions_by_sample[sample_idx].append(pred)
    
    processed_predictions_by_sample = collapse_consecutive_breakpoints(predictions_by_sample)
    
    processed_predictions = []
    for sample_idx, sample_preds in processed_predictions_by_sample.items():
        processed_predictions.extend(sample_preds)
    
    original_metrics = compute_membership_interval_metrics(predictions_by_sample, num_timestamps, verbose=verbose)
    processed_metrics = compute_membership_interval_metrics(processed_predictions_by_sample, num_timestamps, verbose=verbose)
    interval_level_original = compute_interval_level_metrics(predictions_by_sample, num_timestamps, verbose=verbose)
    interval_level_processed = compute_interval_level_metrics(processed_predictions_by_sample, num_timestamps, verbose=verbose)
    
    result = {
        'original_metrics': original_metrics,
        'processed_metrics': processed_metrics,
        'interval_level_metrics_original': interval_level_original,
        'interval_level_metrics_processed': interval_level_processed,
    }
    
    output_dir = args.get('output_dir', './results')
    os.makedirs(output_dir, exist_ok=True)
    
    json_file = os.path.join(output_dir, 'result.json')
    
    json_result = {
        'T-Pre': round(processed_metrics.get('membership_precision', 0.0) * 100, 2),
        'T-Rec': round(processed_metrics.get('membership_recall', 0.0) * 100, 2),
        'T-F1': round(processed_metrics.get('membership_f1', 0.0) * 100, 2),
        'I-Pre': round(interval_level_processed.get('I_Pre', 0.0) * 100, 2),
        'I-Rec': round(interval_level_processed.get('I_Rec', 0.0) * 100, 2),
        'I-F1': round(interval_level_processed.get('I_F1', 0.0) * 100, 2),
    }
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_result, f, indent=2, ensure_ascii=False)
    
    return result

