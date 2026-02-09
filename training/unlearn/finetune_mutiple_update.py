from data.load_data import get_data
from data.prepare_data import split_dataset, split_dataset4
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os
from torch.utils.data import Dataset, Subset, DataLoader, ConcatDataset
from model.DNN import DNN


def get_training_config(args):

    dataset_name = args.get('dataset_name', '').lower()
    
    config = {
        'num_alternations': args.get('num_alternations', 4),
        'remove_epoch': 3,
        'retain_epoch': 1,
        'remove_lr': args.get('forget_lr', args.get('lr', 0.001)),
        'retain_lr': args.get('lr', 0.001),
        'remove_weight_decay': 5e-4,
        'retain_weight_decay': 5e-4,
        'insert_to_remove_cooldown': args.get('insert_to_remove_cooldown', 30),
        'remove_to_insert_cooldown': args.get('remove_to_insert_cooldown', 30),
    }

    if dataset_name == 'sst5':
        config['num_alternations'] = 4
    elif dataset_name == 'mnli':
        config['num_alternations'] = 6
        config['remove_epoch'] = 4
        config['retain_epoch'] = 1
    elif dataset_name == 'cifar100' or dataset_name == 'cifar10':
        config['remove_epoch'] = 1
        config['retain_epoch'] = 1
        config['num_alternations'] = 30
        config['insert_to_remove_cooldown'] = 15
        config['remove_to_insert_cooldown'] = 15
    elif dataset_name == 'svhn':
        config['remove_epoch'] = 1
        config['retain_epoch'] = 1
        config['num_alternations'] = 20
        config['insert_to_remove_cooldown'] = 15
        config['remove_to_insert_cooldown'] = 15
    elif dataset_name == 'cinic10':
        config['remove_epoch'] = 1
        config['retain_epoch'] = 1
        config['num_alternations'] = 30
        config['remove_lr'] = 0.5 * config['remove_lr']
        config['insert_to_remove_cooldown'] = 15
        config['remove_to_insert_cooldown'] = 15
    return config


class WrongLabelDataset(Dataset):

    def __init__(self, dataset, num_classes):
        self.dataset = dataset
        self.num_classes = num_classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if isinstance(sample, dict):
            data = {k: v for k, v in sample.items() if k != 'labels'}
            true_label = sample['labels']
            if isinstance(true_label, torch.Tensor):
                true_label = true_label.item()
            wrong_label = (true_label + 1) % self.num_classes
            return {**data, 'labels': torch.tensor(wrong_label, dtype=torch.long)}
        else:
            data, true_label = sample
            if isinstance(true_label, torch.Tensor):
                true_label = true_label.item()
            wrong_label = (true_label + 1) % self.num_classes
            return data, torch.tensor(wrong_label, dtype=torch.long)


class MixedLabelDataset(Dataset):

    def __init__(self, correct_label_dataset, wrong_label_dataset):
        self.correct_dataset = correct_label_dataset
        self.wrong_dataset = wrong_label_dataset
        self.correct_len = len(correct_label_dataset) if correct_label_dataset is not None else 0
        self.wrong_len = len(wrong_label_dataset) if wrong_label_dataset is not None else 0

    def __len__(self):
        return self.correct_len + self.wrong_len

    def __getitem__(self, idx):
        if idx < self.correct_len:
            if self.correct_dataset is not None:
                return self.correct_dataset[idx]
            else:
                raise IndexError(f"Index {idx} out of range for correct_dataset")
        else:
            if self.wrong_dataset is not None:
                return self.wrong_dataset[idx - self.correct_len]
            else:
                raise IndexError(f"Index {idx} out of range for wrong_dataset")


def finetune_with_wrong_labels(model, forget_loader, optimizer, criterion, args, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        for batch in forget_loader:
            if isinstance(batch, dict):
                data = {k: v.to(args['device']) for k, v in batch.items() if k != 'labels'}
                target = batch['labels'].to(args['device'])
            else:
                data, target = batch
                data, target = data.to(args['device']), target.to(args['device'])

            optimizer.zero_grad()
            output = model.forward_propagation(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()


def finetune_with_correct_labels(model, insert_loader, optimizer, criterion, args, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        for batch in insert_loader:
            if isinstance(batch, dict):
                data = {k: v.to(args['device']) for k, v in batch.items() if k != 'labels'}
                target = batch['labels'].to(args['device'])
            else:
                data, target = batch
                data, target = data.to(args['device']), target.to(args['device'])

            optimizer.zero_grad()
            output = model.forward_propagation(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()


def finetune_on_remove_set(model, remove_loader, args, num_epochs=3, lr=None, weight_decay=None):
    if lr is None:
        lr = args.get('lr', 0.001)
    if weight_decay is None:
        weight_decay = 5e-4
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(num_epochs):
        for batch in remove_loader:
            if isinstance(batch, dict):
                data = {k: v.to(args['device']) for k, v in batch.items() if k != 'labels'}
                target = batch['labels'].to(args['device'])
            else:
                data, target = batch
                data, target = data.to(args['device']), target.to(args['device'])

            optimizer.zero_grad()
            output = model.forward_propagation(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()


def finetune_on_retain_set(model, retain_loader, args, num_epochs=2, lr=None, weight_decay=None):
    if lr is None:
        lr = args.get('lr', 0.001)
    if weight_decay is None:
        weight_decay = 5e-4
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(num_epochs):
        for batch in retain_loader:
            if isinstance(batch, dict):
                data = {k: v.to(args['device']) for k, v in batch.items() if k != 'labels'}
                target = batch['labels'].to(args['device'])
            else:
                data, target = batch
                data, target = data.to(args['device']), target.to(args['device'])

            optimizer.zero_grad()
            output = model.forward_propagation(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()


def evaluate_four_sets(model, all_samples, sample_status, remove_indices, insert_indices, args):
    train_acc = 0.0
    test_acc = 0.0
    insert_acc = 0.0
    remove_acc = 0.0

    train_set_indices = np.where(sample_status == 1)[0].tolist()
    if len(train_set_indices) > 0:
        train_set_dataset = Subset(all_samples, train_set_indices)
        train_set_loader = DataLoader(
            train_set_dataset,
            batch_size=args['batch_size'],
            shuffle=False
        )
        train_acc = model.test_model_acc(train_set_loader)

    test_set_indices = np.where(sample_status == 0)[0].tolist()
    if len(test_set_indices) > 0:
        test_set_dataset = Subset(all_samples, test_set_indices)
        test_set_loader = DataLoader(
            test_set_dataset,
            batch_size=args['batch_size'],
            shuffle=False
        )
        test_acc = model.test_model_acc(test_set_loader)

    if len(insert_indices) > 0:
        insert_dataset = Subset(all_samples, insert_indices)
        insert_loader_eval = DataLoader(
            insert_dataset,
            batch_size=args['batch_size'],
            shuffle=False
        )
        insert_acc = model.test_model_acc(insert_loader_eval)

    if len(remove_indices) > 0:
        remove_dataset_eval = Subset(all_samples, remove_indices)
        remove_loader_eval = DataLoader(
            remove_dataset_eval,
            batch_size=args['batch_size'],
            shuffle=False
        )
        remove_acc = model.test_model_acc(remove_loader_eval)

    return train_acc, test_acc, insert_acc, remove_acc


def train_single_model_mutiple_update(args, train_dataset, test_dataset, num_classes, trial, total_unlearn_steps,is_shadow=False):
    model_prefix = "[SHADOW]" if is_shadow else ""
    model_type = "shadow" if is_shadow else "target"

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args['batch_size'], shuffle=False)


    original_model = DNN(args)
    original_model.train_model(train_loader, test_loader)

    training_config = get_training_config(args)

    update_proportion = args['proportion_of_group_unlearn']

    adaptive_cooldown = args.get('adaptive_cooldown', True)
    if adaptive_cooldown and total_unlearn_steps < 300:
        base_cooldown = training_config['insert_to_remove_cooldown']
        cooldown_factor = max(0.5, total_unlearn_steps / 300.0)
        adjusted_cooldown = max(8, int(base_cooldown * cooldown_factor))
        training_config['insert_to_remove_cooldown'] = adjusted_cooldown
        training_config['remove_to_insert_cooldown'] = adjusted_cooldown
        print(f"  {model_prefix} -> Adaptive cooldown: {adjusted_cooldown} (base: {base_cooldown}, steps: {total_unlearn_steps})")

    print(f'\n{model_prefix} ========== {model_type.upper()} MODEL - The {trial}-th trial ==========')

    all_samples = ConcatDataset([train_dataset, test_dataset])
    total_samples = len(all_samples)

    sample_status = np.zeros(total_samples, dtype=int)
    sample_status[:len(train_dataset)] = 1

    print(f"  {model_prefix} -> Total samples: {total_samples}")
    print(f"  {model_prefix} -> Initial training set size: {len(train_dataset)} (status=1)")
    print(f"  {model_prefix} -> Initial test set size: {len(test_dataset)} (status=0)")

    current_train_indices = list(range(len(train_dataset)))
    current_train_dataset = Subset(all_samples, current_train_indices)

    current_model = DNN(args)
    current_model.load_state_dict(original_model.state_dict())

    save_path = os.getcwd() + f"/save/{args['U_method']}/{args['net_name']}/{args['dataset_name']}/{args['proportion_of_group_unlearn']}/{model_type}/{trial}/"
    os.makedirs(save_path, exist_ok=True)

    sample_status_history = []
    for i in range(total_samples):
        sample_status_history.append([sample_status[i]])

    fixed_unseen_indices = set()
    print(f"  {model_prefix} -> No fixed unseen samples (all samples can participate in insert/remove operations)")
    current_train_dataset = Subset(all_samples, current_train_indices)

    last_insert_timestamp = {i: None for i in range(total_samples)}
    last_remove_timestamp = {i: None for i in range(total_samples)}
    insert_to_remove_cooldown = training_config['insert_to_remove_cooldown']
    remove_to_insert_cooldown = training_config['remove_to_insert_cooldown']

    sample_insert_count = {i: 0 for i in range(total_samples)}
    sample_remove_count = {i: 0 for i in range(total_samples)}

    max_operation_multiplier = args.get('max_operation_multiplier', 3)
    use_priority_selection = args.get('use_priority_selection', True)

    target_operation_distribution = args.get('target_operation_distribution', True)
    target_max_operations = args.get('target_max_operations', 5)

    operation_quota_enabled = args.get('operation_quota_enabled', True)

    sample_target_combination = {}
    if target_operation_distribution:
        all_combinations = []
        for insert_cnt in range(1, target_max_operations + 1):
            for remove_cnt in range(1, target_max_operations + 1):
                all_combinations.append((insert_cnt, remove_cnt))

        samples_per_combination = max(1, total_samples // len(all_combinations))
        combination_counts = {combo: 0 for combo in all_combinations}

        for i in range(total_samples):
            available_combos = [c for c in all_combinations if combination_counts[c] < samples_per_combination]
            if len(available_combos) == 0:
                available_combos = all_combinations
            target_combo = min(available_combos, key=lambda c: combination_counts[c])
            sample_target_combination[i] = target_combo
            combination_counts[target_combo] += 1
        
        print(f"  {model_prefix} -> Target combination assignment: {samples_per_combination} samples per combination")
        print(f"  {model_prefix} -> Combination distribution: {dict(sorted(combination_counts.items()))}")
    else:
        sample_target_combination = None
    
    if operation_quota_enabled and target_operation_distribution:
        quota_per_combination = max(1, total_samples // 25)
        operation_quota = {}
        for insert_cnt in range(1, target_max_operations + 1):
            for remove_cnt in range(1, target_max_operations + 1):
                operation_quota[(insert_cnt, remove_cnt)] = 0
        print(f"  {model_prefix} -> Operation quota enabled: {quota_per_combination} samples per combination")
    else:
        operation_quota = None
        quota_per_combination = None

    timestamp_logs = []

    for k in range(total_unlearn_steps):
        print(f'\n  {model_prefix} -> Timestamp {k}')

        remove_indices = []
        insert_indices = []

        if len(current_train_indices) > 0:
            eligible_for_remove = [
                idx for idx in current_train_indices
                if (last_insert_timestamp[idx] is None or (k - last_insert_timestamp[idx] >= insert_to_remove_cooldown))
                and (not target_operation_distribution or (
                    sample_remove_count[idx] < target_max_operations
                    and (sample_target_combination is None or
                         sample_remove_count[idx] < sample_target_combination[idx][1])
                ))
            ]

            if len(eligible_for_remove) > 0:
                base_num_to_remove = max(1, int(len(current_train_indices) * update_proportion))
                max_num_to_remove = max(base_num_to_remove, int(len(current_train_indices) * update_proportion * max_operation_multiplier))

                num_to_remove = min(len(eligible_for_remove), max_num_to_remove)

                if use_priority_selection and len(eligible_for_remove) > num_to_remove:
                    if target_operation_distribution and sample_target_combination is not None:
                        def get_operation_priority_with_target(idx):
                            insert_cnt = sample_insert_count[idx]
                            remove_cnt = sample_remove_count[idx]
                            target_insert, target_remove = sample_target_combination[idx]

                            if insert_cnt >= target_insert and remove_cnt >= target_remove:
                                return (9999, 9999, 9999)

                            distance_to_target = abs(insert_cnt - target_insert) + abs(remove_cnt - target_remove)

                            return (distance_to_target, insert_cnt + remove_cnt, remove_cnt)
                        
                        eligible_for_remove_sorted = sorted(eligible_for_remove, key=get_operation_priority_with_target)
                        remove_indices = eligible_for_remove_sorted[:num_to_remove]
                    elif target_operation_distribution and operation_quota is not None:
                        def get_operation_priority_with_quota(idx):
                            insert_cnt = sample_insert_count[idx]
                            remove_cnt = sample_remove_count[idx]
                            total_ops = insert_cnt + remove_cnt

                            if insert_cnt >= target_max_operations or remove_cnt >= target_max_operations:
                                return (1000, total_ops, abs(insert_cnt - remove_cnt), remove_cnt)

                            next_remove_cnt = remove_cnt + 1
                            quota_key = (insert_cnt, next_remove_cnt)
                            if quota_key in operation_quota and operation_quota[quota_key] >= quota_per_combination:
                                return (500, total_ops, abs(insert_cnt - remove_cnt), remove_cnt)

                            return (total_ops, abs(insert_cnt - remove_cnt), remove_cnt)
                        
                        eligible_for_remove_sorted = sorted(eligible_for_remove, key=get_operation_priority_with_quota)
                        remove_indices = eligible_for_remove_sorted[:num_to_remove]
                    elif target_operation_distribution:
                        def get_operation_priority(idx):
                            insert_cnt = sample_insert_count[idx]
                            remove_cnt = sample_remove_count[idx]
                            total_ops = insert_cnt + remove_cnt
                            if insert_cnt >= target_max_operations or remove_cnt >= target_max_operations:
                                return (1000, total_ops, abs(insert_cnt - remove_cnt), remove_cnt)
                            return (total_ops, abs(insert_cnt - remove_cnt), remove_cnt)

                        eligible_for_remove_sorted = sorted(eligible_for_remove, key=get_operation_priority)
                        remove_indices = eligible_for_remove_sorted[:num_to_remove]
                    else:
                        eligible_for_remove_sorted = sorted(eligible_for_remove, key=lambda idx: sample_remove_count[idx])
                        remove_indices = eligible_for_remove_sorted[:num_to_remove]
                else:
                    remove_indices = random.sample(eligible_for_remove, num_to_remove)

                for idx in remove_indices:
                    sample_remove_count[idx] += 1
                    if operation_quota is not None:
                        insert_cnt = sample_insert_count[idx]
                        remove_cnt = sample_remove_count[idx]
                        quota_key = (insert_cnt, remove_cnt)
                        if quota_key in operation_quota:
                            operation_quota[quota_key] += 1
                
                print(f"    {model_prefix} -> Remove: {len(remove_indices)} samples from training set (eligible: {len(eligible_for_remove)}, base: {base_num_to_remove}, max: {max_num_to_remove})")
            else:
                print(
                    f"    {model_prefix} -> Skip remove: all {len(current_train_indices)} samples in cooldown after insert")

        available_test_indices = [
            idx for idx in range(total_samples)
            if sample_status[idx] == 0
               and idx not in fixed_unseen_indices
               and idx not in remove_indices
               and (last_remove_timestamp[idx] is None or (k - last_remove_timestamp[idx] >= remove_to_insert_cooldown))
               and (not target_operation_distribution or (
                   sample_insert_count[idx] < target_max_operations
                   and (sample_target_combination is None or
                        sample_insert_count[idx] < sample_target_combination[idx][0])
               ))
        ]

        if len(available_test_indices) > 0:
            current_train_size_after_remove = len(current_train_indices) - len(remove_indices)
            base_num_to_insert = max(1, int(current_train_size_after_remove * update_proportion))
            max_num_to_insert = max(base_num_to_insert, int(current_train_size_after_remove * update_proportion * max_operation_multiplier))

            num_to_insert = min(len(available_test_indices), max_num_to_insert)

            if use_priority_selection and len(available_test_indices) > num_to_insert:
                if target_operation_distribution and sample_target_combination is not None:
                    def get_operation_priority_with_target(idx):
                        insert_cnt = sample_insert_count[idx]
                        remove_cnt = sample_remove_count[idx]
                        target_insert, target_remove = sample_target_combination[idx]

                        if insert_cnt >= target_insert and remove_cnt >= target_remove:
                            return (9999, 9999, 9999)

                        distance_to_target = abs(insert_cnt - target_insert) + abs(remove_cnt - target_remove)

                        return (distance_to_target, insert_cnt + remove_cnt, insert_cnt)

                    available_test_indices_sorted = sorted(available_test_indices, key=get_operation_priority_with_target)
                    insert_indices = available_test_indices_sorted[:num_to_insert]
                elif target_operation_distribution and operation_quota is not None:
                    def get_operation_priority_with_quota(idx):
                        insert_cnt = sample_insert_count[idx]
                        remove_cnt = sample_remove_count[idx]
                        total_ops = insert_cnt + remove_cnt

                        if insert_cnt >= target_max_operations or remove_cnt >= target_max_operations:
                            return (1000, total_ops, abs(insert_cnt - remove_cnt), insert_cnt)

                        next_insert_cnt = insert_cnt + 1
                        quota_key = (next_insert_cnt, remove_cnt)
                        if quota_key in operation_quota and operation_quota[quota_key] >= quota_per_combination:
                            return (500, total_ops, abs(insert_cnt - remove_cnt), insert_cnt)

                        return (total_ops, abs(insert_cnt - remove_cnt), insert_cnt)

                    available_test_indices_sorted = sorted(available_test_indices, key=get_operation_priority_with_quota)
                    insert_indices = available_test_indices_sorted[:num_to_insert]
                elif target_operation_distribution:
                    def get_operation_priority(idx):
                        insert_cnt = sample_insert_count[idx]
                        remove_cnt = sample_remove_count[idx]
                        total_ops = insert_cnt + remove_cnt
                        if insert_cnt >= target_max_operations or remove_cnt >= target_max_operations:
                            return (1000, total_ops, abs(insert_cnt - remove_cnt), insert_cnt)
                        return (total_ops, abs(insert_cnt - remove_cnt), insert_cnt)

                    available_test_indices_sorted = sorted(available_test_indices, key=get_operation_priority)
                    insert_indices = available_test_indices_sorted[:num_to_insert]
                else:
                    available_test_indices_sorted = sorted(available_test_indices, key=lambda idx: sample_insert_count[idx])
                    insert_indices = available_test_indices_sorted[:num_to_insert]
            else:
                insert_indices = random.sample(available_test_indices, num_to_insert)

            for idx in insert_indices:
                sample_insert_count[idx] += 1
                if operation_quota is not None:
                    insert_cnt = sample_insert_count[idx]
                    remove_cnt = sample_remove_count[idx]
                    quota_key = (insert_cnt, remove_cnt)
                    if quota_key in operation_quota:
                        operation_quota[quota_key] += 1
            
            print(f"    {model_prefix} -> Insert: {len(insert_indices)} samples from test set to training set (eligible: {len(available_test_indices)}, base: {base_num_to_insert}, max: {max_num_to_insert})")
        else:
            print(
                f"    {model_prefix} -> Skip insert: no available samples (all in cooldown after remove)")

        for idx in remove_indices:
            sample_status[idx] = 0
            last_remove_timestamp[idx] = k
        current_train_indices = [idx for idx in current_train_indices if idx not in remove_indices]

        for idx in insert_indices:
            sample_status[idx] = 1
            last_insert_timestamp[idx] = k
        current_train_indices.extend(insert_indices)

        has_any_operation = len(remove_indices) > 0 or len(insert_indices) > 0

        if not has_any_operation:
            print(f"    {model_prefix} -> Skip model update: all samples in cooldown (no remove/insert operations)")
        else:
            num_alternations = training_config['num_alternations']
            remove_epoch = training_config['remove_epoch']
            retain_epoch = training_config['retain_epoch']
            remove_lr = training_config['remove_lr']
            retain_lr = training_config['retain_lr']
            remove_weight_decay = training_config['remove_weight_decay']
            retain_weight_decay = training_config['retain_weight_decay']

            remove_loader = None
            if len(remove_indices) > 0:
                remove_dataset = Subset(all_samples, remove_indices)
                wrong_label_dataset = WrongLabelDataset(remove_dataset, num_classes)
                remove_loader = DataLoader(
                    wrong_label_dataset,
                    batch_size=args['batch_size'],
                    shuffle=True
                )

            retain_indices = np.where(sample_status == 1)[0].tolist()
            retain_loader = None
            if len(retain_indices) > 0:
                current_train_dataset = Subset(all_samples, retain_indices)
                retain_loader = DataLoader(
                    current_train_dataset,
                    batch_size=args['batch_size'],
                    shuffle=True
                )

            for alt_iter in range(num_alternations):
                print(f"    {model_prefix} -> Alternation {alt_iter + 1}/{num_alternations}:")

                if remove_loader is not None:
                    finetune_on_remove_set(current_model, remove_loader, args, 
                                          num_epochs=remove_epoch, lr=remove_lr, weight_decay=remove_weight_decay)
                    print(
                        f"      {model_prefix} -> Remove finetune (wrong labels): {len(remove_indices)} samples, {remove_epoch} epochs, lr={remove_lr}")

                    train_acc, test_acc, insert_acc, remove_acc = evaluate_four_sets(
                        current_model, all_samples, sample_status, remove_indices, insert_indices, args
                    )
                    insert_acc_str = f"{insert_acc:.4f} (n={len(insert_indices)})" if len(insert_indices) > 0 else "N/A (n=0)"
                    remove_acc_str = f"{remove_acc:.4f} (n={len(remove_indices)})" if len(remove_indices) > 0 else "N/A (n=0)"
                    print(
                        f"      {model_prefix} -> After remove finetune - Train: {train_acc:.4f}, Test: {test_acc:.4f}, Insert: {insert_acc_str}, Remove: {remove_acc_str}")

                if retain_loader is not None:
                    finetune_on_retain_set(current_model, retain_loader, args, 
                                          num_epochs=retain_epoch, lr=retain_lr, weight_decay=retain_weight_decay)
                    print(
                        f"      {model_prefix} -> Retain finetune (correct labels): {len(retain_indices)} samples, {retain_epoch} epochs, lr={retain_lr}")

                    train_acc, test_acc, insert_acc, remove_acc = evaluate_four_sets(
                        current_model, all_samples, sample_status, remove_indices, insert_indices, args
                    )
                    insert_acc_str = f"{insert_acc:.4f} (n={len(insert_indices)})" if len(insert_indices) > 0 else "N/A (n=0)"
                    remove_acc_str = f"{remove_acc:.4f} (n={len(remove_indices)})" if len(remove_indices) > 0 else "N/A (n=0)"
                    print(
                        f"      {model_prefix} -> After retain finetune - Train: {train_acc:.4f}, Test: {test_acc:.4f}, Insert: {insert_acc_str}, Remove: {remove_acc_str}")

        for idx in fixed_unseen_indices:
            sample_status[idx] = 0

        current_train_indices = np.where(sample_status == 1)[0].tolist()

        print(f"    {model_prefix} -> Current training set size: {len(current_train_indices)}")
        print(f"    {model_prefix} -> Samples in training set (status=1): {np.sum(sample_status == 1)}")
        print(f"    {model_prefix} -> Samples not in training set (status=0): {np.sum(sample_status == 0)}")

        train_acc, test_acc, insert_acc, remove_acc_after_finetune = evaluate_four_sets(
            current_model, all_samples, sample_status, remove_indices, insert_indices, args
        )

        insert_acc_str = f"{insert_acc:.4f} (n={len(insert_indices)})" if len(insert_indices) > 0 else "N/A (n=0)"
        remove_acc_str = f"{remove_acc_after_finetune:.4f} (n={len(remove_indices)})" if len(remove_indices) > 0 else "N/A (n=0)"
        print(
            f"    {model_prefix} -> Final accuracy - Train: {train_acc:.4f}, Test: {test_acc:.4f}, Insert: {insert_acc_str}, Remove: {remove_acc_str}")

        for i in range(total_samples):
            sample_status_history[i].append(sample_status[i])

        all_samples_loader = DataLoader(
            all_samples,
            batch_size=args['batch_size'],
            shuffle=False
        )

        outputs_current = []
        labels_list = []
        indices_list = []

        current_model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(all_samples_loader):
                if isinstance(batch, dict):
                    data = {k: v.to(args['device']) for k, v in batch.items() if k != 'labels'}
                    labels = batch['labels']
                else:
                    data, labels = batch
                    data = data.to(args['device'])
                    labels = labels

                logits_current = current_model.forward_propagation(data)

                probs_current = F.softmax(logits_current, dim=1)

                outputs_current.append(probs_current.cpu().numpy())
                labels_list.append(labels.numpy() if isinstance(labels, torch.Tensor) else labels)

                batch_start_idx = batch_idx * args['batch_size']
                batch_size_actual = len(labels)
                batch_indices = list(range(batch_start_idx, batch_start_idx + batch_size_actual))
                indices_list.extend(batch_indices)

                assert max(
                    batch_indices) < total_samples, f"Index {max(batch_indices)} out of range (total_samples={total_samples})"

        outputs_current = np.concatenate(outputs_current, axis=0)
        labels_list = np.concatenate(labels_list)
        indices_array = np.array(indices_list)

        timestamp_save_path = f"{save_path}/timestamp_{k}/"
        os.makedirs(timestamp_save_path, exist_ok=True)

        np.save(f"{timestamp_save_path}/outputs_current.npy", outputs_current)
        np.save(f"{timestamp_save_path}/labels.npy", labels_list)
        np.save(f"{timestamp_save_path}/sample_indices.npy", indices_array)

        torch.save(current_model.state_dict(), f"{timestamp_save_path}/model_state_dict.pth")

        log_entry = f"Timestamp {k}:\n"
        log_entry += f"  -> Final accuracy - Train: {train_acc:.4f}, Test: {test_acc:.4f}, Insert: {insert_acc:.4f}, Remove: {remove_acc_after_finetune:.4f}\n"
        if has_any_operation:
            log_entry += f"  -> Alternations: {training_config['num_alternations']} (remove: {training_config['remove_epoch']} epochs, retain: {training_config['retain_epoch']} epochs)\n"
        else:
            log_entry += f"  -> Alternations: skipped (no operations)\n"
        timestamp_logs.append(log_entry)

    sample_status_array = np.array(sample_status_history)
    np.save(f"{save_path}/sample_status_history.npy", sample_status_array)
    print(f"\n  {model_prefix} -> Saved sample status history to {save_path}/sample_status_history.npy")
    print(f"      {model_prefix} Shape: {sample_status_array.shape} (samples x timestamps)")

    num_timestamps_with_initial = sample_status_array.shape[1]
    num_timestamps = num_timestamps_with_initial - 1

    converted_status_array = np.zeros((total_samples, num_timestamps), dtype=int)

    for i in range(total_samples):
        for k in range(1, num_timestamps_with_initial):
            prev_status = sample_status_array[i, k - 1]
            curr_status = sample_status_array[i, k]

            if prev_status == 0 and curr_status == 0:
                converted_status_array[i, k - 1] = 0
            elif prev_status == 1 and curr_status == 1:
                converted_status_array[i, k - 1] = 1
            elif prev_status == 0 and curr_status == 1:
                converted_status_array[i, k - 1] = 2
            elif prev_status == 1 and curr_status == 0:
                converted_status_array[i, k - 1] = 3

    np.save(f"{save_path}/sample_status_converted.npy", converted_status_array)

    converted_stats = {
        0: np.sum(converted_status_array == 0),
        1: np.sum(converted_status_array == 1),
        2: np.sum(converted_status_array == 2),
        3: np.sum(converted_status_array == 3)
    }

    num_unseen_per_sample = np.sum(converted_status_array == 0, axis=1)
    num_retain_per_sample = np.sum(converted_status_array == 1, axis=1)
    num_insert_per_sample = np.sum(converted_status_array == 2, axis=1)
    num_remove_per_sample = np.sum(converted_status_array == 3, axis=1)
    num_breakpoints_per_sample = num_insert_per_sample + num_remove_per_sample

    all_unseen_samples = np.where((num_breakpoints_per_sample == 0) & (num_unseen_per_sample == num_timestamps))[0]
    all_retain_samples = np.where((num_breakpoints_per_sample == 0) & (num_retain_per_sample == num_timestamps))[0]

    pattern_statistics = {
        'all_unseen': len(all_unseen_samples),
        'all_retain': len(all_retain_samples),
    }

    for num_insert in range(num_timestamps + 1):
        for num_remove in range(num_timestamps + 1):
            if num_insert + num_remove == 0:
                continue
            if num_insert + num_remove > num_timestamps:
                continue
            mask = (num_insert_per_sample == num_insert) & (num_remove_per_sample == num_remove)
            count = np.sum(mask)
            if count > 0:
                pattern_key = f"{num_insert}_insert_{num_remove}_remove"
                pattern_statistics[pattern_key] = count

    breakpoint_indices_dict = {}
    for i in range(total_samples):
        insert_positions = np.where(converted_status_array[i, :] == 2)[0].tolist()
        remove_positions = np.where(converted_status_array[i, :] == 3)[0].tolist()

        insert_timestamps = insert_positions
        remove_timestamps = remove_positions

        if len(insert_timestamps) > 0 or len(remove_timestamps) > 0:
            breakpoint_indices_dict[i] = {
                'insert': insert_timestamps,
                'remove': remove_timestamps
            }

    breakpoint_indices_file = f"{save_path}/breakpoint_indices.npy"
    np.save(breakpoint_indices_file, breakpoint_indices_dict, allow_pickle=True)
    print(f"  {model_prefix} -> Saved breakpoint indices to {breakpoint_indices_file}")

    all_unseen_count = pattern_statistics['all_unseen']
    all_retain_count = pattern_statistics['all_retain']
    all_unseen_pct = (all_unseen_count / total_samples) * 100 if total_samples > 0 else 0.0
    all_retain_pct = (all_retain_count / total_samples) * 100 if total_samples > 0 else 0.0
    print(f"      {model_prefix} {'All Unseen (no breakpoint)':<40} {all_unseen_count:<15} {all_unseen_pct:>13.2f}%")
    print(f"      {model_prefix} {'All Retain (no breakpoint)':<40} {all_retain_count:<15} {all_retain_pct:>13.2f}%")

    for pattern_key in sorted(pattern_statistics.keys()):
        if pattern_key in ['all_unseen', 'all_retain']:
            continue
        count = pattern_statistics[pattern_key]
        percentage = (count / total_samples) * 100 if total_samples > 0 else 0.0
        parts = pattern_key.split('_')
        num_insert = int(parts[0])
        num_remove = int(parts[2])
        pattern_desc = f"{num_insert} insert, {num_remove} remove breakpoints"
        print(f"      {model_prefix} {pattern_desc:<40} {count:<15} {percentage:>13.2f}%")

    main_stats_file_path = f"{save_path}/status_statistics.txt"
    with open(main_stats_file_path, 'w', encoding='utf-8') as f:
        f.write(f"{model_type.upper()} Model - Trial {trial} - Converted Status Pattern Statistics\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total samples: {total_samples}\n")
        f.write(f"Total timestamps: {num_timestamps}\n\n")
        f.write(f"Status encoding:\n")
        f.write(f"  0: Unseen (0->0, not breakpoint)\n")
        f.write(f"  1: Retain (1->1, not breakpoint)\n")
        f.write(f"  2: Insert (0->1, breakpoint)\n")
        f.write(f"  3: Remove (1->0, breakpoint)\n\n")
        f.write(f"Pattern Distribution:\n")
        f.write(f"{'Pattern':<50} {'Count':<15} {'Percentage':<15}\n")
        f.write("-" * 80 + "\n")

        f.write(f"{'All Unseen (no breakpoint)':<50} {all_unseen_count:<15} {all_unseen_pct:>13.2f}%\n")
        f.write(f"{'All Retain (no breakpoint)':<50} {all_retain_count:<15} {all_retain_pct:>13.2f}%\n")

        for pattern_key in sorted(pattern_statistics.keys()):
            if pattern_key in ['all_unseen', 'all_retain']:
                continue
            count = pattern_statistics[pattern_key]
            percentage = (count / total_samples) * 100 if total_samples > 0 else 0.0
            parts = pattern_key.split('_')
            num_insert = int(parts[0])
            num_remove = int(parts[2])
            pattern_desc = f"{num_insert} insert, {num_remove} remove breakpoints"
            f.write(f"{pattern_desc:<50} {count:<15} {percentage:>13.2f}%\n")

        f.write(f"\n\nDetailed Status Count Statistics:\n")
        f.write(f"{'Status Type':<30} {'Total Count':<20} {'Average per Sample':<20}\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Unseen (0->0)':<30} {converted_stats[0]:<20} {converted_stats[0] / total_samples:.2f}\n")
        f.write(f"{'Retain (1->1)':<30} {converted_stats[1]:<20} {converted_stats[1] / total_samples:.2f}\n")
        f.write(f"{'Insert (0->1)':<30} {converted_stats[2]:<20} {converted_stats[2] / total_samples:.2f}\n")
        f.write(f"{'Remove (1->0)':<30} {converted_stats[3]:<20} {converted_stats[3] / total_samples:.2f}\n")

        f.write(f"\n\nMember Interval Length Distribution:\n")
        f.write(f"{'Member Interval Length':<30} {'Sample Count':<20} {'Percentage':<15}\n")
        f.write("-" * 65 + "\n")

        sample_member_lengths = []

        for i in range(total_samples):
            status_sequence = sample_status_array[i, :]
            member_length = np.sum(status_sequence == 1)
            sample_member_lengths.append(member_length)

        if len(sample_member_lengths) > 0:
            max_length = max(sample_member_lengths)
            min_length = min(sample_member_lengths)

            interval_ranges = []
            step = 40
            current_start = 0
            while current_start <= max_length:
                current_end = current_start + step - 1
                if current_start == 0:
                    interval_ranges.append((1, current_end))
                else:
                    interval_ranges.append((current_start, current_end))
                current_start += step

            interval_counts = {}
            interval_samples = {}

            for sample_idx, length in enumerate(sample_member_lengths):
                found = False
                for start, end in interval_ranges:
                    if start <= length <= end:
                        range_key = f"{start}-{end}"
                        interval_counts[range_key] = interval_counts.get(range_key, 0) + 1
                        if range_key not in interval_samples:
                            interval_samples[range_key] = []
                        interval_samples[range_key].append(sample_idx)
                        found = True
                        break
                if not found:
                    if len(interval_ranges) > 0:
                        last_start, last_end = interval_ranges[-1]
                        if length > last_end:
                            range_key = f">{last_end}"
                            interval_counts[range_key] = interval_counts.get(range_key, 0) + 1
                            if range_key not in interval_samples:
                                interval_samples[range_key] = []
                            interval_samples[range_key].append(sample_idx)
                        elif length == 0:
                            range_key = "0"
                            interval_counts[range_key] = interval_counts.get(range_key, 0) + 1
                            if range_key not in interval_samples:
                                interval_samples[range_key] = []
                            interval_samples[range_key].append(sample_idx)

            if "0" in interval_counts:
                count = interval_counts["0"]
                percentage = (count / total_samples) * 100 if total_samples > 0 else 0.0
                f.write(f"{'0':<30} {count:<20} {percentage:>13.2f}%\n")

            for start, end in interval_ranges:
                range_key = f"{start}-{end}"
                count = interval_counts.get(range_key, 0)
                if count > 0:
                    percentage = (count / total_samples) * 100 if total_samples > 0 else 0.0
                    f.write(f"{range_key:<30} {count:<20} {percentage:>13.2f}%\n")

            if len(interval_ranges) > 0:
                last_start, last_end = interval_ranges[-1]
                if f">{last_end}" in interval_counts:
                    count = interval_counts[f">{last_end}"]
                    percentage = (count / total_samples) * 100 if total_samples > 0 else 0.0
                    f.write(f">{last_end:<29} {count:<20} {percentage:>13.2f}%\n")

            f.write(f"\nSummary:\n")
            f.write(f"  Total samples: {total_samples}\n")
            f.write(f"  Average member length: {np.mean(sample_member_lengths):.2f}\n")
            f.write(f"  Min member length: {min_length}\n")
            f.write(f"  Max member length: {max_length}\n")
            f.write(f"  Median member length: {np.median(sample_member_lengths):.2f}\n")

            member_interval_samples_file = f"{save_path}/member_interval_samples.txt"
            with open(member_interval_samples_file, 'w', encoding='utf-8') as f_samples:
                f_samples.write(f"{model_type.upper()} Model - Trial {trial} - Member Interval Sample Indices\n")
                f_samples.write("=" * 70 + "\n\n")
                f_samples.write(f"Total samples: {total_samples}\n")
                f_samples.write(f"Total timestamps: {num_timestamps}\n\n")

                if "0" in interval_samples:
                    f_samples.write(f"Interval: 0 (Member length = 0)\n")
                    f_samples.write(f"Sample count: {len(interval_samples['0'])}\n")
                    f_samples.write(f"Sample indices: {interval_samples['0']}\n\n")

                for start, end in interval_ranges:
                    range_key = f"{start}-{end}"
                    if range_key in interval_samples:
                        f_samples.write(f"Interval: {range_key} (Member length: {start} to {end})\n")
                        f_samples.write(f"Sample count: {len(interval_samples[range_key])}\n")
                        f_samples.write(f"Sample indices: {interval_samples[range_key]}\n\n")

                if len(interval_ranges) > 0:
                    last_start, last_end = interval_ranges[-1]
                    if f">{last_end}" in interval_samples:
                        f_samples.write(f"Interval: >{last_end} (Member length > {last_end})\n")
                        f_samples.write(f"Sample count: {len(interval_samples[f'>{last_end}'])}\n")
                        f_samples.write(f"Sample indices: {interval_samples[f'>{last_end}']}\n\n")

            member_interval_samples_npy = f"{save_path}/member_interval_samples.npy"
            np.save(member_interval_samples_npy, interval_samples, allow_pickle=True)
            print(f"  {model_prefix} -> Saved member interval sample indices to {member_interval_samples_file}")
            print(f"  {model_prefix} -> Saved member interval sample indices (npy) to {member_interval_samples_npy}")
        else:
            f.write("No member intervals found.\n")
    
    print(f"  {model_prefix} -> Saved main status statistics to {main_stats_file_path}")

    log_file_path = f"{save_path}/timestamp_logs.txt"
    with open(log_file_path, 'w', encoding='utf-8') as f:
        f.write(f"{model_type.upper()} Model - Trial {trial} - Timestamp Logs\n")
        f.write("=" * 50 + "\n\n")
        for log_entry in timestamp_logs:
            f.write(log_entry + "\n")
    print(f"  {model_prefix} -> Saved timestamp logs to {log_file_path}")

    if target_operation_distribution:
        print(f"\n  {model_prefix} -> Final operation count distribution:")
        operation_distribution = {}
        for i in range(total_samples):
            insert_cnt = sample_insert_count[i]
            remove_cnt = sample_remove_count[i]
            key = (insert_cnt, remove_cnt)
            operation_distribution[key] = operation_distribution.get(key, 0) + 1

        print(f"    {model_prefix} {'Insert':<8} {'Remove':<8} {'Count':<10} {'Percentage':<10}")
        print(f"    {model_prefix} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")
        for insert_cnt in range(target_max_operations + 1):
            for remove_cnt in range(target_max_operations + 1):
                key = (insert_cnt, remove_cnt)
                count = operation_distribution.get(key, 0)
                if count > 0:
                    percentage = (count / total_samples) * 100
                    print(f"    {model_prefix} {insert_cnt:<8} {remove_cnt:<8} {count:<10} {percentage:>9.2f}%")

        target_combinations = []
        for insert_cnt in range(1, target_max_operations + 1):
            for remove_cnt in range(1, target_max_operations + 1):
                key = (insert_cnt, remove_cnt)
                if operation_distribution.get(key, 0) > 0:
                    target_combinations.append(key)
        
        print(f"    {model_prefix} -> Target combinations (1-{target_max_operations} operations): {len(target_combinations)}/{target_max_operations * target_max_operations}")
        if len(target_combinations) < target_max_operations * target_max_operations:
            missing = []
            for insert_cnt in range(1, target_max_operations + 1):
                for remove_cnt in range(1, target_max_operations + 1):
                    if (insert_cnt, remove_cnt) not in target_combinations:
                        missing.append((insert_cnt, remove_cnt))
            print(f"    {model_prefix} -> Missing combinations: {missing[:10]}..." if len(missing) > 10 else f"    {model_prefix} -> Missing combinations: {missing}")


def continuous_update_finetune_multiple_update(args):
    print("dataset and net_name:", args['dataset_name'], args['net_name'])
    total_unlearn_steps = args.get('total_unlearn_steps', 200)
    update_proportion = args['proportion_of_group_unlearn']



    print(f"  -> Total unlearn steps: {total_unlearn_steps}")
    print(f"  -> Update proportion per step: {update_proportion * 100}%")

    train_data, test_data = get_data(args['dataset_name'], args['net_name'])
    target_m, shadow_m = split_dataset4(train_data, args['random'])
    target_um, shadow_um = split_dataset4(test_data, args['random'])

    temp_model = DNN(args)
    num_classes = temp_model.num_classes
    del temp_model


    for t in range(args['trials']):
        print(f'\n========== The {t}-th trial ==========')

        train_single_model_mutiple_update(
            args=args,
            train_dataset=target_m,
            test_dataset=target_um,
            num_classes=num_classes,
            trial=t,
            total_unlearn_steps=total_unlearn_steps,
            is_shadow=False
        )
