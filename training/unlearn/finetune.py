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


class WrongLabelDataset(Dataset):

    def __init__(self, dataset, num_classes):
        self.dataset = dataset
        self.num_classes = num_classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if isinstance(sample, dict):
            # 文本数据
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


def finetune_on_remove_set(model, remove_loader, args, num_epochs=3, learning_rate=None, weight_decay=None):
    if learning_rate is None:
        learning_rate = args.get('forget_lr', args.get('lr', 0.001))
    if weight_decay is None:
        weight_decay = args.get('remove_weight_decay', 5e-4)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    model.train()
    model.model.train()
    for epoch in range(num_epochs):
        for batch in remove_loader:
            if isinstance(batch, dict):
                data = {k: v.to(args['device']) for k, v in batch.items() if k != 'labels'}
                target = batch['labels'].to(args['device'])
            else:
                data, target = batch
                data, target = data.to(args['device']), target.to(args['device'])

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()


def finetune_on_retain_set(model, retain_loader, args, num_epochs=2, learning_rate=None, weight_decay=None):
    if learning_rate is None:
        learning_rate = args.get('retain_lr', args.get('lr', 0.001))
    if weight_decay is None:
        weight_decay = args.get('retain_weight_decay', 5e-4)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    model.train()
    model.model.train()
    for epoch in range(num_epochs):
        for batch in retain_loader:
            if isinstance(batch, dict):
                data = {k: v.to(args['device']) for k, v in batch.items() if k != 'labels'}
                target = batch['labels'].to(args['device'])
            else:
                data, target = batch
                data, target = data.to(args['device']), target.to(args['device'])

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()


def get_training_config(args):

    dataset_name = args.get('dataset_name', '').lower()
    
    # 默认配置
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
    elif dataset_name == 'cifar100' or dataset_name=='cifar10':
        config['remove_epoch'] = 1
        config['retain_epoch'] = 1
        config['num_alternations'] =30
        config['insert_to_remove_cooldown'] = 15
        config['remove_to_insert_cooldown'] = 15
    elif dataset_name == 'svhn' :
        config['remove_epoch'] = 1
        config['retain_epoch'] = 1
        config['num_alternations'] =20
        config['insert_to_remove_cooldown'] = 15
        config['remove_to_insert_cooldown'] = 15
    elif dataset_name == 'cinic10':
        config['remove_epoch'] = 1
        config['retain_epoch'] = 1
        config['num_alternations'] = 30
        config['remove_lr'] = 0.5* config['remove_lr']
        config['insert_to_remove_cooldown'] = 15
        config['remove_to_insert_cooldown'] = 15
    return config


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


def train_single_model(args, train_dataset, test_dataset, num_classes, trial, total_unlearn_steps,is_shadow=False):

    model_prefix = "[SHADOW]" if is_shadow else ""
    model_type = "shadow" if is_shadow else "target"

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args['batch_size'], shuffle=False)

    original_model = DNN(args)
    original_model.print_dropout_info()

    original_model.train_model(train_loader, test_loader)


    update_proportion = args['proportion_of_group_unlearn']
    
    training_config = get_training_config(args)
    num_alternations = training_config['num_alternations']
    remove_epoch = training_config['remove_epoch']
    retain_epoch = training_config['retain_epoch']
    remove_lr = training_config['remove_lr']
    retain_lr = training_config['retain_lr']
    remove_weight_decay = training_config['remove_weight_decay']
    retain_weight_decay = training_config['retain_weight_decay']
    
    print(f'\n{model_prefix} ========== {model_type.upper()} MODEL - The {trial}-th trial ==========')
    print(f"  {model_prefix} -> Training config:")
    print(f"    {model_prefix} ->   Num alternations: {num_alternations}")
    print(f"    {model_prefix} ->   Remove epoch: {remove_epoch}, LR: {remove_lr}, Weight decay: {remove_weight_decay}")
    print(f"    {model_prefix} ->   Retain epoch: {retain_epoch}, LR: {retain_lr}, Weight decay: {retain_weight_decay}")


    all_samples = ConcatDataset([train_dataset, test_dataset])
    total_samples = len(all_samples)


    sample_status = np.zeros(total_samples, dtype=int)
    sample_status[:len(train_dataset)] = 1

    print(f"  {model_prefix} -> Total samples: {total_samples}")
    print(f"  {model_prefix} -> Initial training set size: {len(train_dataset)} (status=1)")
    print(f"  {model_prefix} -> Initial test set size: {len(test_dataset)} (status=0)")

    current_train_indices = list(range(len(train_dataset)))
    current_train_dataset = Subset(all_samples, current_train_indices)

    current_model = original_model

    save_path = os.getcwd() + f"/save/{args['U_method']}/{args['net_name']}/{args['dataset_name']}/{args['proportion_of_group_unlearn']}/{model_type}/{trial}/"
    os.makedirs(save_path, exist_ok=True)

    sample_status_history = []
    for i in range(total_samples):
        sample_status_history.append([sample_status[i]])

    min_fixed_unseen = 200
    num_fixed_unseen = max(min_fixed_unseen, int(total_samples * 0.01))
    all_indices = list(range(total_samples))
    initial_unseen_indices = np.where(sample_status == 0)[0].tolist()

    if len(initial_unseen_indices) >= num_fixed_unseen:
        fixed_unseen_indices = set(random.sample(initial_unseen_indices, num_fixed_unseen))
    else:
        fixed_unseen_indices = set(random.sample(initial_unseen_indices, len(initial_unseen_indices)))
        remaining_needed = num_fixed_unseen - len(fixed_unseen_indices)
        other_indices = [idx for idx in all_indices if idx not in fixed_unseen_indices]
        additional_fixed = random.sample(other_indices, min(remaining_needed, len(other_indices)))
        fixed_unseen_indices.update(additional_fixed)
        for idx in additional_fixed:
            sample_status[idx] = 0
            if idx in current_train_indices:
                current_train_indices.remove(idx)
    print(
        f"  {model_prefix} -> Fixed {len(fixed_unseen_indices)} samples ({len(fixed_unseen_indices) / total_samples * 100:.2f}%) to remain unseen permanently (minimum: {min_fixed_unseen})")

    for i in range(total_samples):
        sample_status_history[i][0] = sample_status[i]

    current_train_indices = [idx for idx in current_train_indices if idx not in fixed_unseen_indices]
    current_train_dataset = Subset(all_samples, current_train_indices)
    
    initial_train_size = len(current_train_indices)
    print(f"  {model_prefix} -> Initial train size (after fixed_unseen): {initial_train_size} (used for insert/remove calculation)")


    last_insert_timestamp = {i: None for i in range(total_samples)}
    last_remove_timestamp = {i: None for i in range(total_samples)}
    insert_to_remove_cooldown = training_config.get('insert_to_remove_cooldown', args.get('insert_to_remove_cooldown', 30))
    remove_to_insert_cooldown = training_config.get('remove_to_insert_cooldown', args.get('remove_to_insert_cooldown', 30))
    print(f"  {model_prefix} -> Cooldown periods: insert_to_remove={insert_to_remove_cooldown}, remove_to_insert={remove_to_insert_cooldown}")
    timestamp_logs = []

    for k in range(total_unlearn_steps):
        print(f'\n  {model_prefix} -> Timestamp {k}')

        remove_indices = []
        insert_indices = []

        num_to_remove = max(1, int(initial_train_size * update_proportion))
        if num_to_remove > 0 and len(current_train_indices) > 0:

            eligible_for_remove = [
                idx for idx in current_train_indices
                if last_insert_timestamp[idx] is None or (k - last_insert_timestamp[idx] >= insert_to_remove_cooldown)
            ]

            if len(eligible_for_remove) >= num_to_remove:
                remove_indices = random.sample(eligible_for_remove, min(num_to_remove, len(eligible_for_remove)))
                print(f"    {model_prefix} -> Remove: {len(remove_indices)} samples from training set")
            else:
                print(
                    f"    {model_prefix} -> Skip remove: only {len(eligible_for_remove)} eligible samples (need {num_to_remove}, {len(current_train_indices) - len(eligible_for_remove)} in cooldown after insert)")


        available_test_indices = [
            idx for idx in range(total_samples)
            if sample_status[idx] == 0
               and idx not in fixed_unseen_indices
               and idx not in remove_indices
               and (last_remove_timestamp[idx] is None or (k - last_remove_timestamp[idx] >= remove_to_insert_cooldown))
        ]
        num_to_insert = max(1, int(initial_train_size * update_proportion))

        if num_to_insert > 0 and len(available_test_indices) >= num_to_insert:
            insert_indices = random.sample(
                available_test_indices,
                min(num_to_insert, len(available_test_indices))
            )
            print(f"    {model_prefix} -> Insert: {len(insert_indices)} samples from test set to training set")
        else:
            if num_to_insert > 0 and len(available_test_indices) < num_to_insert:
                print(
                    f"    {model_prefix} -> Skip insert: only {len(available_test_indices)} available samples (need {num_to_insert}, {len(fixed_unseen_indices)} fixed as unseen)")


        for idx in remove_indices:
            sample_status[idx] = 0
            last_remove_timestamp[idx] = k
        current_train_indices = [idx for idx in current_train_indices if idx not in remove_indices]

        for idx in insert_indices:
            sample_status[idx] = 1
            last_insert_timestamp[idx] = k
        current_train_indices.extend(insert_indices)


        has_any_operation = len(remove_indices) > 0 or len(insert_indices) > 0
        skip_last_retain_finetune = False

        if not has_any_operation:
            print(f"    {model_prefix} -> Skip model update: all samples in cooldown (no remove/insert operations)")
        else:

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
                                          num_epochs=remove_epoch, 
                                          learning_rate=remove_lr,
                                          weight_decay=remove_weight_decay)
                    print(
                        f"      {model_prefix} -> Remove finetune (wrong labels): {len(remove_indices)} samples, {remove_epoch} epochs")

                    train_acc, test_acc, insert_acc, remove_acc = evaluate_four_sets(
                        current_model, all_samples, sample_status, remove_indices, insert_indices, args
                    )
                    print(
                        f"      {model_prefix} -> After remove finetune - Train: {train_acc:.4f}, Test: {test_acc:.4f}, Insert: {insert_acc:.4f}, Remove: {remove_acc:.4f}")


                is_last_alternation = (alt_iter == num_alternations - 1)
                should_skip_retain = is_last_alternation and skip_last_retain_finetune
                
                if retain_loader is not None and not should_skip_retain:
                    finetune_on_retain_set(current_model, retain_loader, args, 
                                         num_epochs=retain_epoch,
                                         learning_rate=retain_lr,
                                         weight_decay=retain_weight_decay)
                    print(
                        f"      {model_prefix} -> Retain finetune (correct labels): {len(retain_indices)} samples, {retain_epoch} epochs")

                    train_acc, test_acc, insert_acc, remove_acc = evaluate_four_sets(
                        current_model, all_samples, sample_status, remove_indices, insert_indices, args
                    )
                    print(
                        f"      {model_prefix} -> After retain finetune - Train: {train_acc:.4f}, Test: {test_acc:.4f}, Insert: {insert_acc:.4f}, Remove: {remove_acc:.4f}")
                elif should_skip_retain:
                    print(f"      {model_prefix} -> Skipping retain finetune (last alternation, skip_last_retain_finetune=True)")

        for idx in fixed_unseen_indices:
            sample_status[idx] = 0

        current_train_indices = np.where(sample_status == 1)[0].tolist()

        print(f"    {model_prefix} -> Current training set size: {len(current_train_indices)}")
        print(f"    {model_prefix} -> Samples in training set (status=1): {np.sum(sample_status == 1)}")
        print(f"    {model_prefix} -> Samples not in training set (status=0): {np.sum(sample_status == 0)}")

        train_acc, test_acc, insert_acc, remove_acc_after_finetune = evaluate_four_sets(
            current_model, all_samples, sample_status, remove_indices, insert_indices, args
        )

        print(
            f"    {model_prefix} -> Final accuracy - Train: {train_acc:.4f}, Test: {test_acc:.4f}, Insert: {insert_acc:.4f}, Remove: {remove_acc_after_finetune:.4f}")


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

        log_entry = f"Timestamp {k}:\n"
        log_entry += f"  -> Final accuracy - Train: {train_acc:.4f}, Test: {test_acc:.4f}, Insert: {insert_acc:.4f}, Remove: {remove_acc_after_finetune:.4f}\n"
        log_entry += f"  -> Alternations: {num_alternations} (remove: {remove_epoch} epochs, retain: {retain_epoch} epochs)\n"
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

def continuous_update_finetune(args):
    print("dataset and net_name:", args['dataset_name'], args['net_name'])

    # 参数控制
    total_unlearn_steps =  80

    if args['dataset_name'] =='cifar10' or args['dataset_name'] =='cifar100' or args['dataset_name'] =='cinic10' or args['dataset_name']=='svhn':
        total_unlearn_steps = 50

    update_proportion = args['proportion_of_group_unlearn']
    print(f"  -> Total unlearn steps: {total_unlearn_steps}")
    print(f"  -> Update proportion per step: {update_proportion * 100}%")

    train_data, test_data = get_data(args['dataset_name'], args['net_name'], U_method=args.get('U_method'))
    target_m, shadow_m = split_dataset4(train_data, args['random'])
    target_um, shadow_um = split_dataset4(test_data, args['random'])

    temp_model = DNN(args)
    num_classes = temp_model.num_classes
    del temp_model


    for t in range(args['trials']):
        print(f'\n========== The {t}-th trial ==========')


        train_single_model(
            args=args,
            train_dataset=target_m,
            test_dataset=target_um,
            num_classes=num_classes,
            trial=t,
            total_unlearn_steps=total_unlearn_steps,
            is_shadow=False
        )

        # train_single_model(
        #     args=args,
        #     train_dataset=shadow_m,
        #     test_dataset=shadow_um,
        #     num_classes=num_classes,
        #     trial=t,
        #     total_unlearn_steps=total_unlearn_steps,
        #     is_shadow=True
        # )