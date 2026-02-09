import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any, List


def evaluate_accuracy(model: nn.Module,
                     data_loader: DataLoader,
                     device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> float:
    model.eval()
    model.to(device)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total if total > 0 else 0.0
    
    return accuracy


def evaluate_model(model: nn.Module,
                   train_loader: DataLoader,
                   val_loader: DataLoader = None,
                   device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> Dict[str, float]:
    results = {}
    
    train_acc = evaluate_accuracy(model, train_loader, device)
    results['train_accuracy'] = train_acc
    if val_loader is not None:
        val_acc = evaluate_accuracy(model, val_loader, device)
        results['val_accuracy'] = val_acc
    return results


def load_and_evaluate(model_path: str,
                     model: nn.Module,
                     train_loader: DataLoader,
                     val_loader: DataLoader = None,
                     device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> Dict[str, float]:
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    results = evaluate_model(model, train_loader, val_loader, device)
    
    return results


def compute_retain_status_overlap_metrics(true_retain_status_list: List[np.ndarray],
                                          pred_retain_status_list: List[np.ndarray]) -> Dict[str, float]:
    if len(true_retain_status_list) != len(pred_retain_status_list):
        raise ValueError(
            f"Number of true status sequences ({len(true_retain_status_list)}) does not match "
            f"number of predicted status sequences ({len(pred_retain_status_list)})"
        )
    
    num_sequences = len(true_retain_status_list)
    
    if num_sequences == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'num_samples': 0
        }
    
    precisions = []
    recalls = []
    
    for true_status, pred_status in zip(true_retain_status_list, pred_retain_status_list):
        true_status = np.array(true_status)
        pred_status = np.array(pred_status)
        
        min_len = min(len(true_status), len(pred_status))
        if min_len == 0:
            continue
            
        true_status = true_status[:min_len]
        pred_status = pred_status[:min_len]
        
        matches = (true_status == pred_status)
        
        true_in_retain = (true_status == 1)
        pred_in_retain = (pred_status == 1)
        
        num_true_in_retain = np.sum(true_in_retain)
        num_pred_in_retain = np.sum(pred_in_retain)
        
        matched_in_retain = matches & true_in_retain & pred_in_retain
        num_matched_in_retain = np.sum(matched_in_retain)
        
        if num_pred_in_retain > 0:
            precision_seq = num_matched_in_retain / num_pred_in_retain
        else:
            precision_seq = 1.0 if num_true_in_retain == 0 else 0.0
        
        if num_true_in_retain > 0:
            recall_seq = num_matched_in_retain / num_true_in_retain
        else:
            recall_seq = 1.0 if num_pred_in_retain == 0 else 0.0
        
        precisions.append(precision_seq)
        recalls.append(recall_seq)
    
    precision = np.mean(precisions) if len(precisions) > 0 else 0.0
    recall = np.mean(recalls) if len(recalls) > 0 else 0.0
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'num_samples': len(precisions)
    }


def compute_f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def evaluate_changepoint_accuracy(true_status_sequences: List[np.ndarray],
                                  pred_status_sequences: List[np.ndarray]) -> Dict[str, Any]:
    from .data_processor import DataProcessor
    
    processor = DataProcessor()
    
    overlap_metrics = compute_retain_status_overlap_metrics(true_status_sequences, pred_status_sequences)
    precision = overlap_metrics['precision']
    recall = overlap_metrics['recall']
    f1 = compute_f1_score(precision, recall)
    
    insert_correct = 0
    insert_total = 0
    insert_pred_total = 0
    
    remove_correct = 0
    remove_total = 0
    remove_pred_total = 0
    
    for true_seq, pred_seq in zip(true_status_sequences, pred_status_sequences):
        true_seq = np.array(true_seq)
        pred_seq = np.array(pred_seq)
        
        min_len = min(len(true_seq), len(pred_seq))
        true_seq = true_seq[:min_len]
        pred_seq = pred_seq[:min_len]
        
        true_changepoints = processor.convert_status_to_changepoints(true_seq)
        pred_changepoints = processor.convert_status_to_changepoints(pred_seq)
        
        true_insert_positions = {cp['position'] for cp in true_changepoints if cp['type'] == 'insert'}
        true_remove_positions = {cp['position'] for cp in true_changepoints if cp['type'] == 'remove'}
        
        pred_insert_positions = {cp['position'] for cp in pred_changepoints if cp['type'] == 'insert'}
        pred_remove_positions = {cp['position'] for cp in pred_changepoints if cp['type'] == 'remove'}
        
        insert_total += len(true_insert_positions)
        insert_pred_total += len(pred_insert_positions)
        insert_correct += len(true_insert_positions & pred_insert_positions)
        
        remove_total += len(true_remove_positions)
        remove_pred_total += len(pred_remove_positions)
        remove_correct += len(true_remove_positions & pred_remove_positions)
    
    insert_precision = insert_correct / insert_pred_total if insert_pred_total > 0 else 0.0
    insert_recall = insert_correct / insert_total if insert_total > 0 else 0.0
    insert_f1 = compute_f1_score(insert_precision, insert_recall)
    
    remove_precision = remove_correct / remove_pred_total if remove_pred_total > 0 else 0.0
    remove_recall = remove_correct / remove_total if remove_total > 0 else 0.0
    remove_f1 = compute_f1_score(remove_precision, remove_recall)
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        
        'insert_precision': float(insert_precision),
        'insert_recall': float(insert_recall),
        'insert_f1': float(insert_f1),
        
        'remove_precision': float(remove_precision),
        'remove_recall': float(remove_recall),
        'remove_f1': float(remove_f1),
        
        'insert_correct': int(insert_correct),
        'insert_total': int(insert_total),
        'insert_pred_total': int(insert_pred_total),
        'remove_correct': int(remove_correct),
        'remove_total': int(remove_total),
        'remove_pred_total': int(remove_pred_total)
    }


def calculate_member_nonmember_distribution(status_sequences: List[np.ndarray]) -> Dict[str, Any]:
    total_timesteps = 0
    member_count = 0
    nonmember_count = 0
    
    for status_seq in status_sequences:
        status_seq = np.array(status_seq)
        total_timesteps += len(status_seq)
        
        member_mask = np.isin(status_seq, [1, 2])
        nonmember_mask = np.isin(status_seq, [0, 3])
        
        member_count += np.sum(member_mask)
        nonmember_count += np.sum(nonmember_mask)
    
    num_samples = len(status_sequences)
    member_ratio = member_count / total_timesteps if total_timesteps > 0 else 0.0
    nonmember_ratio = nonmember_count / total_timesteps if total_timesteps > 0 else 0.0
    
    return {
        'num_samples': num_samples,
        'total_timesteps': total_timesteps,
        'member_count': int(member_count),
        'nonmember_count': int(nonmember_count),
        'member_ratio': float(member_ratio),
        'nonmember_ratio': float(nonmember_ratio)
    }


def compute_delta_attack_metrics(predicted_memberships_by_sample: Dict[int, set],
                                  true_memberships_by_sample: Dict[int, set],
                                  predicted_breakpoints_by_sample: Dict[int, List[tuple]],
                                  true_status_sequences: Dict[int, np.ndarray],
                                  num_timestamps: int,
                                  verbose: bool = True) -> Dict[str, Any]:
    total_intersection = 0
    total_predicted = 0
    total_true = 0
    total_false_positive = 0
    total_false_negative = 0
    
    for sample_idx in predicted_memberships_by_sample.keys():
        if sample_idx not in true_memberships_by_sample:
            continue
        
        pred_set = predicted_memberships_by_sample[sample_idx]
        true_set = true_memberships_by_sample[sample_idx]
        
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
    
    insert_correct = 0
    insert_total = 0
    insert_pred_total = 0
    
    remove_correct = 0
    remove_total = 0
    remove_pred_total = 0
    
    for sample_idx in predicted_breakpoints_by_sample.keys():
        if sample_idx not in true_status_sequences:
            continue
        
        true_status_seq = true_status_sequences[sample_idx]
        
        true_insert_positions = set()
        true_remove_positions = set()
        
        for idx in range(len(true_status_seq)):
            if true_status_seq[idx] == 2:
                true_insert_positions.add(idx)
            elif true_status_seq[idx] == 3:
                true_remove_positions.add(idx)
        
        pred_breakpoints = predicted_breakpoints_by_sample.get(sample_idx, [])
        pred_insert_positions = {ts for ts, bp_type in pred_breakpoints if bp_type == 1}
        pred_remove_positions = {ts for ts, bp_type in pred_breakpoints if bp_type == 2}
        
        insert_total += len(true_insert_positions)
        insert_pred_total += len(pred_insert_positions)
        insert_correct += len(true_insert_positions & pred_insert_positions)
        
        remove_total += len(true_remove_positions)
        remove_pred_total += len(pred_remove_positions)
        remove_correct += len(true_remove_positions & pred_remove_positions)
    
    insert_precision = insert_correct / insert_pred_total if insert_pred_total > 0 else 0.0
    insert_recall = insert_correct / insert_total if insert_total > 0 else 0.0
    insert_f1 = 2 * (insert_precision * insert_recall) / (insert_precision + insert_recall) if (insert_precision + insert_recall) > 0 else 0.0
    
    remove_precision = remove_correct / remove_pred_total if remove_pred_total > 0 else 0.0
    remove_recall = remove_correct / remove_total if remove_total > 0 else 0.0
    remove_f1 = 2 * (remove_precision * remove_recall) / (remove_precision + remove_recall) if (remove_precision + remove_recall) > 0 else 0.0
    
    return {
        'membership_precision': float(precision),
        'membership_recall': float(recall),
        'membership_f1': float(f1),
        'insert_precision': float(insert_precision),
        'insert_recall': float(insert_recall),
        'insert_f1': float(insert_f1),
        'remove_precision': float(remove_precision),
        'remove_recall': float(remove_recall),
        'remove_f1': float(remove_f1),
        'total_intersection': int(total_intersection),
        'total_predicted': int(total_predicted),
        'total_true': int(total_true),
        'insert_correct': int(insert_correct),
        'insert_total': int(insert_total),
        'insert_pred_total': int(insert_pred_total),
        'remove_correct': int(remove_correct),
        'remove_total': int(remove_total),
        'remove_pred_total': int(remove_pred_total)
    }
