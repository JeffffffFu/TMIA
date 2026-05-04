from data.load_data import get_data
from data.prepare_data import split_dataset4
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os
from torch.utils.data import Dataset, Subset, DataLoader, ConcatDataset
from model.DNN import DNN


def NPO_unlearn(unlearned_model, forget_loader, optimizer, original_model, args):
    """NPO方法：在forget set上进行NPO unlearn"""
    beta = 0.1
    for forget_batch in forget_loader:
        if isinstance(forget_batch, dict):
            inputs_f = {k: v.to(args['device']) for k, v in forget_batch.items() if k != 'labels'}
            targets_f = forget_batch['labels'].to(args['device'])
        else:
            inputs_f, targets_f = forget_batch
            inputs_f, targets_f = inputs_f.to(args['device']), targets_f.to(args['device'])

        optimizer.zero_grad()

        with torch.no_grad():
            outputs_target = original_model.forward_propagation(inputs_f)

        outputs_theta = unlearned_model.forward_propagation(inputs_f)

        probs_theta = torch.softmax(outputs_theta, dim=1)
        probs_target = torch.softmax(outputs_target, dim=1)

        target_probs_theta = torch.gather(probs_theta, 1, targets_f.unsqueeze(1)).squeeze(1)
        target_probs_target = torch.gather(probs_target, 1, targets_f.unsqueeze(1)).squeeze(1)

        epsilon = 1e-8
        log_ratio = torch.log(target_probs_theta + epsilon) - torch.log(target_probs_target + epsilon)

        npo_loss = -(2.0 / beta) * torch.log(torch.sigmoid(-beta * log_ratio) + epsilon)
        npo_loss = npo_loss.mean()

        npo_loss.backward()
        optimizer.step()


def refine_remained(unlearned_model, retain_loader, optimizer, criterion, args):
    """NPO方法：在retain set上进行正常训练"""
    for batch in retain_loader:
        if isinstance(batch, dict):
            data = {k: v.to(args['device']) for k, v in batch.items() if k != 'labels'}
            target = batch['labels'].to(args['device'])
        else:
            data, target = batch
            data, target = data.to(args['device']), target.to(args['device'])

        optimizer.zero_grad()
        output = unlearned_model.forward_propagation(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def NPO_train_on_remove_set(model, original_model, remove_loader, args, num_epochs=3):
    """在Remove Set上使用NPO方法进行反学习"""
    # 根据数据集设置学习率
    lr_npo = args['lr']
    if args['dataset_name'] == 'sst5':
        lr_npo = args['lr'] *3.0
    elif args['dataset_name'] == 'news20':
        lr_npo = args['lr'] * 3.0
    elif args['dataset_name'] == 'rte':
        lr_npo = args['lr']
    elif args['dataset_name'] == 'mrpc':
        lr_npo = args['lr']
    
    optimizer_NPO = optim.Adam(model.parameters(), lr=lr_npo, weight_decay=5e-4)
    
    model.train()
    for epoch in range(num_epochs):
        NPO_unlearn(model, remove_loader, optimizer_NPO, original_model, args)


def NPO_train_on_retain_set(model, retain_loader, args, num_epochs=1):
    """在Retain Set上使用NPO方法进行正常训练"""
    optimizer_remained = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(num_epochs):
        refine_remained(model, retain_loader, optimizer_remained, criterion, args)


def evaluate_four_sets(model, all_samples, sample_status, remove_indices, insert_indices, args):
    """评估四个set的准确率：train_acc, test_acc, insert_acc, remove_acc"""
    train_acc = 0.0
    test_acc = 0.0
    insert_acc = 0.0
    remove_acc = 0.0

    # 评估training set accuracy：所有状态为1的样本
    train_set_indices = np.where(sample_status == 1)[0].tolist()
    if len(train_set_indices) > 0:
        train_set_dataset = Subset(all_samples, train_set_indices)
        train_set_loader = DataLoader(
            train_set_dataset,
            batch_size=args['batch_size'],
            shuffle=False
        )
        train_acc = model.test_model_acc(train_set_loader)

    # 评估test set accuracy：所有状态为0的样本
    test_set_indices = np.where(sample_status == 0)[0].tolist()
    if len(test_set_indices) > 0:
        test_set_dataset = Subset(all_samples, test_set_indices)
        test_set_loader = DataLoader(
            test_set_dataset,
            batch_size=args['batch_size'],
            shuffle=False
        )
        test_acc = model.test_model_acc(test_set_loader)

    # 评估insert set的准确率
    if len(insert_indices) > 0:
        insert_dataset = Subset(all_samples, insert_indices)
        insert_loader_eval = DataLoader(
            insert_dataset,
            batch_size=args['batch_size'],
            shuffle=False
        )
        insert_acc = model.test_model_acc(insert_loader_eval)

    # 评估remove set的准确率
    if len(remove_indices) > 0:
        remove_dataset_eval = Subset(all_samples, remove_indices)
        remove_loader_eval = DataLoader(
            remove_dataset_eval,
            batch_size=args['batch_size'],
            shuffle=False
        )
        remove_acc = model.test_model_acc(remove_loader_eval)

    return train_acc, test_acc, insert_acc, remove_acc


def train_single_model_NPO(args, train_dataset, test_dataset, num_classes, trial, total_unlearn_steps, is_shadow=False):
    """
    训练单个模型（target或shadow）的连续更新NPO unlearn过程

    Args:
        args: 配置参数字典
        train_dataset: 训练数据集（target_m 或 shadow_m）
        test_dataset: 测试数据集（test_data 或 shadow_um）
        num_classes: 类别数
        trial: trial编号
        total_unlearn_steps: 总unlearn步数
        is_shadow: 是否为shadow model，默认为False（target model）

    Returns:
        None（所有结果保存到文件）
    """
    # 根据is_shadow设置前缀和保存路径
    model_prefix = "[SHADOW]" if is_shadow else ""
    model_type = "shadow" if is_shadow else "target"

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args['batch_size'], shuffle=False)

    original_model = DNN(args)
    original_model.train_model(train_loader, test_loader)

    # 参数控制
    update_proportion = args['proportion_of_group_unlearn']
    # 统一冷却期：无论是remove还是insert，20个timestamp内不能再进行任何操作
    cooldown_period = args.get('cooldown_period', 20)
    num_alternations = args.get('num_alternations', 8)

    random_seed = args.get('random_seed', args.get('random', 3407))

    print(f'\n{model_prefix} ========== {model_type.upper()} MODEL - The {trial}-th trial ==========')

    # 设置随机种子
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # 1. 初始化索引跟踪系统
    # 合并训练集和测试集的所有样本，为每个样本分配唯一索引
    all_samples = ConcatDataset([train_dataset, test_dataset])
    total_samples = len(all_samples)

    # 初始化状态：train_dataset中的样本为1，test_dataset中的样本为0
    sample_status = np.zeros(total_samples, dtype=int)
    sample_status[:len(train_dataset)] = 1  # 训练集中的样本状态为1

    print(f"  {model_prefix} -> Total samples: {total_samples}")
    print(f"  {model_prefix} -> Initial training set size: {len(train_dataset)} (status=1)")
    print(f"  {model_prefix} -> Initial test set size: {len(test_dataset)} (status=0)")

    # 当前训练集（初始为train_dataset）
    current_train_indices = list(range(len(train_dataset)))
    current_train_dataset = Subset(all_samples, current_train_indices)

    # 当前模型（从原始模型开始）
    current_model = DNN(args)
    current_model.load_state_dict(original_model.state_dict())

    # 保存路径
    save_path = os.getcwd() + f"/save/{args['U_method']}/{args['net_name']}/{args['dataset_name']}/{args['proportion_of_group_unlearn']}/{model_type}/{trial}/"
    os.makedirs(save_path, exist_ok=True)
    
    # 初始化存储结构：二维list存储每个样本在每个timestamp的状态
    sample_status_history = []
    for i in range(total_samples):
        sample_status_history.append([sample_status[i]])  # 初始状态

    # 初始化冷却期跟踪：统一记录每个样本最后一次更新的timestamp
    last_update_timestamp = {i: None for i in range(total_samples)}

    timestamp_logs = []

    # 连续更新学习
    for k in range(total_unlearn_steps):
        print(f'\n  {model_prefix} -> Timestamp {k}')

        # 初始化每个timestamp的变量
        remove_indices = []
        insert_indices = []

        # 2. Remove操作：从训练集中选择数据进行remove
        num_to_remove = max(1, int(len(current_train_indices) * update_proportion))
        if num_to_remove > 0 and len(current_train_indices) > 0:
            # 排除在冷却期内的样本
            eligible_for_remove = [
                idx for idx in current_train_indices
                if last_update_timestamp[idx] is None or (k - last_update_timestamp[idx] >= cooldown_period)
            ]

            if len(eligible_for_remove) >= num_to_remove:
                remove_indices = random.sample(eligible_for_remove, min(num_to_remove, len(eligible_for_remove)))
                print(f"    {model_prefix} -> Remove: {len(remove_indices)} samples from training set")
            else:
                print(
                    f"    {model_prefix} -> Skip remove: only {len(eligible_for_remove)} eligible samples (need {num_to_remove}, {len(current_train_indices) - len(eligible_for_remove)} in cooldown)")

        # 3. Insert操作：从所有状态为0的样本中选择数据insert到训练集
        available_test_indices = [
            idx for idx in range(total_samples)
            if sample_status[idx] == 0
               and idx not in remove_indices
               and (last_update_timestamp[idx] is None or (k - last_update_timestamp[idx] >= cooldown_period))
        ]
        num_to_insert = max(1, int(len(current_train_indices) * update_proportion))

        if num_to_insert > 0 and len(available_test_indices) >= num_to_insert:
            insert_indices = random.sample(
                available_test_indices,
                min(num_to_insert, len(available_test_indices))
            )
            print(f"    {model_prefix} -> Insert: {len(insert_indices)} samples from test set to training set")
        else:
            if num_to_insert > 0 and len(available_test_indices) < num_to_insert:
                print(
                    f"    {model_prefix} -> Skip insert: only {len(available_test_indices)} available samples (need {num_to_insert})")

        # 4. 更新状态（在交替训练之前先更新状态）
        # Remove的样本状态改为0
        for idx in remove_indices:
            sample_status[idx] = 0
            last_update_timestamp[idx] = k  # 记录更新操作timestamp
        current_train_indices = [idx for idx in current_train_indices if idx not in remove_indices]

        # Insert的样本状态改为1
        for idx in insert_indices:
            sample_status[idx] = 1
            last_update_timestamp[idx] = k  # 记录更新操作timestamp
        current_train_indices.extend(insert_indices)

        # 5. 检查是否有任何操作（remove或insert）
        # 如果所有样本都在冷却期，既没有remove也没有insert操作，则跳过模型更新
        has_any_operation = len(remove_indices) > 0 or len(insert_indices) > 0

        if not has_any_operation:
            print(f"    {model_prefix} -> Skip model update: all samples in cooldown (no remove/insert operations)")
        else:
            # 保存当前timestamp开始时的模型状态，作为NPO的参考模型
            reference_model = DNN(args)
            reference_model.load_state_dict(current_model.state_dict())
            
            # 交替进行Remove Set（NPO unlearn）训练和Retain Set（正常训练）Finetune
            if args['dataset_name'] == 'sst5':
                num_alternations = args.get('num_alternations', 4)
            elif args['dataset_name'] == 'mnli' and args['net_name'] == 'gpt2':
                num_alternations = args.get('num_alternations', 6)
            else:
                num_alternations = args.get('num_alternations', 8)

            # 准备remove set
            remove_loader = None
            if len(remove_indices) > 0:
                remove_dataset = Subset(all_samples, remove_indices)
                remove_loader = DataLoader(
                    remove_dataset,
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

            # 设置epoch数
            if args['dataset_name'] == 'news20':
                remove_epoch = 3
                retain_epoch = 1
            else:
                remove_epoch = 3
                retain_epoch = 1

            # 交替训练
            for alt_iter in range(num_alternations):
                print(f"    {model_prefix} -> Alternation {alt_iter + 1}/{num_alternations}:")

                # Step 1: 在Remove Set上使用NPO方法进行反学习（遗忘效果）
                if remove_loader is not None:
                    NPO_train_on_remove_set(current_model, reference_model, remove_loader, args, num_epochs=remove_epoch)
                    print(
                        f"      {model_prefix} -> Remove unlearn (NPO): {len(remove_indices)} samples, {remove_epoch} epochs")

                    train_acc, test_acc, insert_acc, remove_acc = evaluate_four_sets(
                        current_model, all_samples, sample_status, remove_indices, insert_indices, args
                    )
                    print(
                        f"      {model_prefix} -> After remove unlearn - Train: {train_acc:.4f}, Test: {test_acc:.4f}, Insert: {insert_acc:.4f}, Remove: {remove_acc:.4f}")

                # Step 2: 在Retain Set上使用NPO方法进行正常训练（恢复性能）
                if retain_loader is not None:
                    NPO_train_on_retain_set(current_model, retain_loader, args, num_epochs=retain_epoch)
                    print(
                        f"      {model_prefix} -> Retain train (NPO normal): {len(retain_indices)} samples, {retain_epoch} epochs")

                    train_acc, test_acc, insert_acc, remove_acc = evaluate_four_sets(
                        current_model, all_samples, sample_status, remove_indices, insert_indices, args
                    )
                    print(
                        f"      {model_prefix} -> After retain train - Train: {train_acc:.4f}, Test: {test_acc:.4f}, Insert: {insert_acc:.4f}, Remove: {remove_acc:.4f}")

        # 更新current_train_indices以反映最终状态
        current_train_indices = np.where(sample_status == 1)[0].tolist()

        print(f"    {model_prefix} -> Current training set size: {len(current_train_indices)}")
        print(f"    {model_prefix} -> Samples in training set (status=1): {np.sum(sample_status == 1)}")
        print(f"    {model_prefix} -> Samples not in training set (status=0): {np.sum(sample_status == 0)}")

        # 6. 在所有交替训练完成后，最终评估四个set的准确率
        train_acc, test_acc, insert_acc, remove_acc_after_finetune = evaluate_four_sets(
            current_model, all_samples, sample_status, remove_indices, insert_indices, args
        )

        # 打印最终准确率
        print(
            f"    {model_prefix} -> Final accuracy - Train: {train_acc:.4f}, Test: {test_acc:.4f}, Insert: {insert_acc:.4f}, Remove: {remove_acc_after_finetune:.4f}")

        # 更新每个样本的状态历史（记录当前timestamp的状态）
        for i in range(total_samples):
            sample_status_history[i].append(sample_status[i])

        # 保存每个样本的output（只保存当前模型的output）
        all_samples_loader = DataLoader(
            all_samples,
            batch_size=args['batch_size'],
            shuffle=False
        )

        # 保存每个样本的output
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

                # 只计算当前模型的输出（logits）
                logits_current = current_model.forward_propagation(data)

                # 转换为概率
                probs_current = F.softmax(logits_current, dim=1)

                outputs_current.append(probs_current.cpu().numpy())
                labels_list.append(labels.numpy() if isinstance(labels, torch.Tensor) else labels)

                batch_start_idx = batch_idx * args['batch_size']
                batch_size_actual = len(labels)
                batch_indices = list(range(batch_start_idx, batch_start_idx + batch_size_actual))
                indices_list.extend(batch_indices)

                # 验证：确保索引在有效范围内
                assert max(
                    batch_indices) < total_samples, f"Index {max(batch_indices)} out of range (total_samples={total_samples})"

        outputs_current = np.concatenate(outputs_current, axis=0)
        labels_list = np.concatenate(labels_list)
        indices_array = np.array(indices_list)

        # 保存到文件
        timestamp_save_path = f"{save_path}/timestamp_{k}/"
        os.makedirs(timestamp_save_path, exist_ok=True)

        np.save(f"{timestamp_save_path}/outputs_current.npy", outputs_current)
        np.save(f"{timestamp_save_path}/labels.npy", labels_list)
        np.save(f"{timestamp_save_path}/sample_indices.npy", indices_array)

        # 记录日志
        log_entry = f"Timestamp {k}:\n"
        log_entry += f"  -> Final accuracy - Train: {train_acc:.4f}, Test: {test_acc:.4f}, Insert: {insert_acc:.4f}, Remove: {remove_acc_after_finetune:.4f}\n"
        log_entry += f"  -> Alternations: {num_alternations} (remove: {remove_epoch} epochs, retain: {retain_epoch} epochs)\n"
        timestamp_logs.append(log_entry)

    # 6. 在trial结束后，统一保存样本状态历史
    sample_status_array = np.array(sample_status_history)
    np.save(f"{save_path}/sample_status_history.npy", sample_status_array)
    print(f"\n  {model_prefix} -> Saved sample status history to {save_path}/sample_status_history.npy")
    print(f"      {model_prefix} Shape: {sample_status_array.shape} (samples x timestamps)")

    # 6.5. 转换sample_status_history为包含breakpoint信息的状态编码
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

    # 保存转换后的状态数组
    np.save(f"{save_path}/sample_status_converted.npy", converted_status_array)

    # 统计转换后的状态分布
    converted_stats = {
        0: np.sum(converted_status_array == 0),
        1: np.sum(converted_status_array == 1),
        2: np.sum(converted_status_array == 2),
        3: np.sum(converted_status_array == 3)
    }

    # 7. 基于转换后的状态进行详细统计
    num_unseen_per_sample = np.sum(converted_status_array == 0, axis=1)
    num_retain_per_sample = np.sum(converted_status_array == 1, axis=1)
    num_insert_per_sample = np.sum(converted_status_array == 2, axis=1)
    num_remove_per_sample = np.sum(converted_status_array == 3, axis=1)
    num_breakpoints_per_sample = num_insert_per_sample + num_remove_per_sample

    # 统计详细模式
    all_unseen_samples = np.where((num_breakpoints_per_sample == 0) & (num_unseen_per_sample == num_timestamps))[0]
    all_retain_samples = np.where((num_breakpoints_per_sample == 0) & (num_retain_per_sample == num_timestamps))[0]

    # 统计各种模式的数量
    pattern_statistics = {
        'all_unseen': len(all_unseen_samples),
        'all_retain': len(all_retain_samples),
    }

    # 对于有breakpoint的样本，按insert和remove的数量分类
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

    # 保存breakpoint索引
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

    # 保存breakpoint索引到文件
    breakpoint_indices_file = f"{save_path}/breakpoint_indices.npy"
    np.save(breakpoint_indices_file, breakpoint_indices_dict, allow_pickle=True)
    print(f"  {model_prefix} -> Saved breakpoint indices to {breakpoint_indices_file}")

    # No breakpoint情况
    all_unseen_count = pattern_statistics['all_unseen']
    all_retain_count = pattern_statistics['all_retain']
    all_unseen_pct = (all_unseen_count / total_samples) * 100 if total_samples > 0 else 0.0
    all_retain_pct = (all_retain_count / total_samples) * 100 if total_samples > 0 else 0.0
    print(f"      {model_prefix} {'All Unseen (no breakpoint)':<40} {all_unseen_count:<15} {all_unseen_pct:>13.2f}%")
    print(f"      {model_prefix} {'All Retain (no breakpoint)':<40} {all_retain_count:<15} {all_retain_pct:>13.2f}%")

    # 有breakpoint情况
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

    # 保存主要统计结果到文件
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

        # No breakpoint情况
        f.write(f"{'All Unseen (no breakpoint)':<50} {all_unseen_count:<15} {all_unseen_pct:>13.2f}%\n")
        f.write(f"{'All Retain (no breakpoint)':<50} {all_retain_count:<15} {all_retain_pct:>13.2f}%\n")

        # 有breakpoint情况
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

        # 添加详细的状态统计
        f.write(f"\n\nDetailed Status Count Statistics:\n")
        f.write(f"{'Status Type':<30} {'Total Count':<20} {'Average per Sample':<20}\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Unseen (0->0)':<30} {converted_stats[0]:<20} {converted_stats[0] / total_samples:.2f}\n")
        f.write(f"{'Retain (1->1)':<30} {converted_stats[1]:<20} {converted_stats[1] / total_samples:.2f}\n")
        f.write(f"{'Insert (0->1)':<30} {converted_stats[2]:<20} {converted_stats[2] / total_samples:.2f}\n")
        f.write(f"{'Remove (1->0)':<30} {converted_stats[3]:<20} {converted_stats[3] / total_samples:.2f}\n")
    print(f"  {model_prefix} -> Saved main status statistics to {main_stats_file_path}")

    # 8. 保存所有timestamp的打印信息
    log_file_path = f"{save_path}/timestamp_logs.txt"
    with open(log_file_path, 'w', encoding='utf-8') as f:
        f.write(f"{model_type.upper()} Model - Trial {trial} - Timestamp Logs\n")
        f.write("=" * 50 + "\n\n")
        for log_entry in timestamp_logs:
            f.write(log_entry + "\n")
    print(f"  {model_prefix} -> Saved timestamp logs to {log_file_path}")


def continuous_update_finetune_NPO(args):
    print("dataset and net_name:", args['dataset_name'], args['net_name'])

    # 参数控制
    total_unlearn_steps = args.get('total_unlearn_steps', 50)
    update_proportion = args['proportion_of_group_unlearn']  # 0.1% 的数据更新比例

    print(f"  -> Total unlearn steps: {total_unlearn_steps}")
    print(f"  -> Update proportion per step: {update_proportion * 100}%")

    train_data, test_data = get_data(args['dataset_name'], args['net_name'])
    target_m, shadow_m = split_dataset4(train_data, args['random'])
    target_um, shadow_um = split_dataset4(test_data, args['random'])

    # 获取类别数
    temp_model = DNN(args)
    num_classes = temp_model.num_classes
    del temp_model

    for t in range(args['trials']):
        print(f'\n========== The {t}-th trial ==========')

        # 训练target model
        train_single_model_NPO(
            args=args,
            train_dataset=target_m,
            test_dataset=target_um,
            num_classes=num_classes,
            trial=t,
            total_unlearn_steps=total_unlearn_steps,
            is_shadow=False
        )
        # 训练shadow model
        train_single_model_NPO(
            args=args,
            train_dataset=shadow_m,
            test_dataset=shadow_um,
            num_classes=num_classes,
            trial=t,
            total_unlearn_steps=total_unlearn_steps,
            is_shadow=True
        )
