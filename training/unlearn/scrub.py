from data.load_data import get_data
from data.prepare_data import split_dataset, split_dataset4
import torch
import torch.nn as nn
import torch.optim as optim
from model.DNN import DNN
from utlis.utils import sample_target_samples, save_output
from torch.utils.data import ConcatDataset, Dataset, Subset, DataLoader
import copy
import time
import numpy as np
import random
import os
import torch.nn.functional as F


def finetune_on_retain_set(model, retain_loader, args, num_epochs=2, learning_rate=None, weight_decay=None):
    if learning_rate is None:
        learning_rate = args.get('retain_finetune_lr', args.get('scrub_lr', args.get('lr', 0.001)))
    if weight_decay is None:
        weight_decay = args.get('retain_finetune_weight_decay', 5e-4)
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


class DistillKL(nn.Module):

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T ** 2) / y_s.shape[0]
        return loss


def scrub_unlearn(current_model, forget_loader, retain_loader, test_loader, args,
                  unlearn_epochs=None, learning_rate=None, T=None, scrub_beta=None,
                  scrub_gamma=None, scrub_alpha=None, smoothing=None, m_steps=None):
    device = args['device']
    
    T = T if T is not None else args.get('scrub_T', 1)
    scrub_beta = scrub_beta if scrub_beta is not None else args.get('scrub_beta', 0.0)
    scrub_gamma = scrub_gamma if scrub_gamma is not None else args.get('scrub_gamma', 3.0)
    scrub_alpha = scrub_alpha if scrub_alpha is not None else args.get('scrub_alpha', 0.1)
    smoothing = smoothing if smoothing is not None else args.get('scrub_smoothing', 0.0)
    m_steps = m_steps if m_steps is not None else args.get('scrub_m_steps', 1)
    unlearn_epochs = unlearn_epochs if unlearn_epochs is not None else args.get('scrub_unlearn_epochs', 30)
    lr = learning_rate if learning_rate is not None else args.get('scrub_lr', 0.001)

    teacher = copy.deepcopy(current_model)
    student = copy.deepcopy(current_model)
    model_t = copy.deepcopy(teacher)
    model_s = copy.deepcopy(student)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(T)
    criterion_kd = DistillKL(T)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)
    criterion_list.append(criterion_div)
    criterion_list.append(criterion_kd)

    optimizer = optim.Adam(trainable_list.parameters(),
                           lr=lr, weight_decay=5e-4)

    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.to(device)
        criterion_list.to(device)
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True

    def avg_fn(averaged_model_parameter, model_parameter, num_averaged):
        return (1 - scrub_beta) * averaged_model_parameter + scrub_beta * model_parameter

    swa_model = torch.optim.swa_utils.AveragedModel(model_s, avg_fn=avg_fn)
    swa_model.to(device)

    for epoch in range(1, unlearn_epochs + 1):
        maximize_loss = 0
        if forget_loader is not None and epoch <= m_steps:
            maximize_loss = train_distill(forget_loader, module_list, swa_model,
                                          criterion_list, optimizer, scrub_gamma, scrub_alpha, scrub_beta, smoothing,
                                          "maximize", device,
                                          quiet=False)
        if retain_loader is not None:
            train_acc, train_loss = train_distill(retain_loader, module_list, swa_model, criterion_list,
                                                  optimizer, scrub_gamma, scrub_alpha, scrub_beta, smoothing,
                                                  "minimize", device, quiet=False)

    return model_s


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def param_dist(model, swa_model, p):
    dist = 0.
    for p1, p2 in zip(model.parameters(), swa_model.parameters()):
        dist += torch.norm(p1 - p2, p='fro')
    return p * dist


def train_distill(train_loader, module_list, swa_model, criterion_list, optimizer, scrub_gamma, scrub_alpha, scrub_beta,
                  smoothing, split, device, quiet=False):
    for module in module_list:
        module.train()
    module_list[-1].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    kd_losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    loss = 0.0

    for idx, data in enumerate(train_loader):
        if isinstance(data, dict):
            input = data.get('input_ids') or data.get('input')
            target = data.get('labels') or data.get('target')
            if input is None:
                input = next(v for k, v in data.items() if k != 'labels' and isinstance(v, torch.Tensor))
            if target is None:
                target = data.get('labels')
        else:
            input, target = data

        data_time.update(time.time() - end)

        if not isinstance(input, dict):
            if input.dtype != torch.float32 and input.dtype != torch.float64:
                input = input.float()

        if torch.cuda.is_available():
            if isinstance(input, dict):
                input = {k: v.to(device) for k, v in input.items()}
            else:
                input = input.to(device)
            target = target.to(device)

        model_s.train()
        if isinstance(input, dict):
            logit_s = model_s.model(input)
        else:
            logit_s = model_s.model(input)
        with torch.no_grad():
            model_t.eval()
            if isinstance(input, dict):
                logit_t = model_t.forward_propagation(input)
            else:
                logit_t = model_t.forward_propagation(input)

    loss_cls = criterion_cls(logit_s, target)
    loss_div = criterion_div(logit_s, logit_t)
    loss_kd = 0
    if split == "minimize":
        loss = scrub_gamma * loss_cls + scrub_alpha * loss_div + scrub_beta * loss_kd
    elif split == "maximize":
        loss = -loss_div

    loss = loss + param_dist(model_s, swa_model, smoothing)

    if split == "minimize" and not quiet:
        acc1, _ = accuracy(logit_s, target, topk=(1, 1))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
    elif split == "maximize" and not quiet:
        kd_losses.update(loss.item(), input.size(0))

    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    batch_time.update(time.time() - end)
    end = time.time()

    if split == "minimize":
        return top1.avg, losses.avg
    else:
        return kd_losses.avg


def adjust_learning_rate(epoch, learning_rate, lr_decay_epochs, lr_decay_rate, optimizer):
    steps = np.sum(epoch > np.asarray(lr_decay_epochs))
    new_lr = learning_rate
    if steps > 0:
        new_lr = learning_rate * (lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    return new_lr, optimizer


def get_training_config_scrub(args):
    dataset_name = args.get('dataset_name', 'default').lower()
    net_name = args.get('net_name', 'default').lower()

    config = {
        'unlearn_epochs': args.get('scrub_unlearn_epochs', 30),
        'learning_rate': args.get('scrub_lr', 0.001),
        'T': args.get('scrub_T', 1),
        'scrub_beta': args.get('scrub_beta', 0.0),
        'scrub_gamma': args.get('scrub_gamma', 0.99),
        'scrub_alpha': args.get('scrub_alpha', 0.1),
        'smoothing': args.get('scrub_smoothing', 0.0),
        'm_steps': args.get('scrub_m_steps', 3),
    }

    if dataset_name == 'sst5':
        config['unlearn_epochs'] = args.get('scrub_unlearn_epochs', 20)
    elif dataset_name == 'mnli':
        config['unlearn_epochs'] = args.get('scrub_unlearn_epochs', 30)
    elif dataset_name == 'cifar100':
        config['unlearn_epochs'] = args.get('scrub_unlearn_epochs', 20)
        config['learning_rate'] = args.get('scrub_lr', 0.0005)
    elif dataset_name == 'cifar10':
        config['scrub_gamma']= 0.4
        config['scrub_alpha']= 0.01


    return config


def train_single_model_scrub(args, train_dataset, test_dataset, num_classes, trial, total_unlearn_steps,
                             is_shadow=False):
    model_prefix = "[SHADOW]" if is_shadow else ""
    model_type = "shadow" if is_shadow else "target"

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args['batch_size'], shuffle=False)

    original_model = DNN(args)
    original_model.train_model(train_loader, test_loader)

    update_proportion = args['proportion_of_group_unlearn']
    cooldown_period = args.get('cooldown_period', 15)

    training_config = get_training_config_scrub(args)
    unlearn_epochs = training_config['unlearn_epochs']
    learning_rate = training_config['learning_rate']
    T = training_config['T']
    scrub_beta = training_config['scrub_beta']
    scrub_gamma = training_config['scrub_gamma']
    scrub_alpha = training_config['scrub_alpha']
    smoothing = training_config['smoothing']
    m_steps = training_config['m_steps']
    num_alternations = 3

    random_seed = args.get('random_seed', args.get('random', 3407))

    print(f'\n{model_prefix} ========== {model_type.upper()} MODEL - The {trial}-th trial ==========')
    print(f"  {model_prefix} -> Training config:")
    print(f"    {model_prefix} ->   Num alternations: {num_alternations}")
    print(f"    {model_prefix} ->   Unlearn epochs: {unlearn_epochs}, LR: {learning_rate}")
    print(f"    {model_prefix} ->   T: {T}, Gamma: {scrub_gamma}, Alpha: {scrub_alpha}, Beta: {scrub_beta}")
    print(f"    {model_prefix} ->   Smoothing: {smoothing}, M steps: {m_steps}")

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

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

    last_update_timestamp = {i: None for i in range(total_samples)}

    timestamp_logs = []

    for k in range(total_unlearn_steps):
        print(f'\n  {model_prefix} -> Timestamp {k}')

        remove_indices = []
        insert_indices = []

        num_to_remove = max(1, int(len(current_train_indices) * update_proportion))
        if num_to_remove > 0 and len(current_train_indices) > 0:
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

        for idx in remove_indices:
            sample_status[idx] = 0
            last_update_timestamp[idx] = k
        current_train_indices = [idx for idx in current_train_indices if idx not in remove_indices]

        for idx in insert_indices:
            sample_status[idx] = 1
            last_update_timestamp[idx] = k
        current_train_indices.extend(insert_indices)

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
            print(
                f"    {model_prefix} -> Final retain set size: {len(retain_indices)} samples (after remove {len(remove_indices)} and insert {len(insert_indices)})")

        has_any_operation = len(remove_indices) > 0 or len(insert_indices) > 0

        if not has_any_operation:
            print(f"    {model_prefix} -> Skip model update: all samples in cooldown (no remove/insert operations)")
        elif retain_loader is None:
            print(f"    {model_prefix} -> Skip model update: retain set is empty")
        else:
            print(f"    {model_prefix} -> Applying SCRUB unlearning...")

            for alt_iter in range(num_alternations):
                print(f"    {model_prefix} -> Alternation {alt_iter + 1}/{num_alternations}:")

                current_model = scrub_unlearn(
                    current_model, remove_loader, retain_loader, test_loader, args,
                    unlearn_epochs=unlearn_epochs,
                    learning_rate=learning_rate,
                    T=T,
                    scrub_beta=scrub_beta,
                    scrub_gamma=scrub_gamma,
                    scrub_alpha=scrub_alpha,
                    smoothing=smoothing,
                    m_steps=m_steps
                )
                print(
                    f"      {model_prefix} -> SCRUB train: {len(retain_indices)} retain samples, {len(remove_indices) if remove_indices else 0} remove samples, {unlearn_epochs} epochs")

                train_acc, test_acc, insert_acc, remove_acc = evaluate_four_sets(
                    current_model, all_samples, sample_status, remove_indices, insert_indices, args
                )
                print(
                    f"      {model_prefix} -> After SCRUB train - Train: {train_acc:.4f}, Test: {test_acc:.4f}, Insert: {insert_acc:.4f}, Remove: {remove_acc:.4f}")

                if retain_loader is not None:

                    retain_finetune_epochs = args.get('retain_finetune_epochs', 1)
                    retain_finetune_lr = args.get('retain_finetune_lr', learning_rate)
                    finetune_on_retain_set(
                        current_model, retain_loader, args,
                        num_epochs=retain_finetune_epochs,
                        learning_rate=retain_finetune_lr
                    )
                    print(
                        f"      {model_prefix} -> Retain finetune: {len(retain_indices)} retain samples, {retain_finetune_epochs} epochs")

                    train_acc, test_acc, insert_acc, remove_acc = evaluate_four_sets(
                        current_model, all_samples, sample_status, remove_indices, insert_indices, args
                    )
                    print(
                        f"      {model_prefix} -> After retain finetune - Train: {train_acc:.4f}, Test: {test_acc:.4f}, Insert: {insert_acc:.4f}, Remove: {remove_acc:.4f}")

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

                probs_current = torch.nn.functional.softmax(logits_current, dim=1)

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
        log_entry += f"  -> Alternations: {num_alternations} (epochs: {unlearn_epochs})\n"
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
    print(f"  {model_prefix} -> Saved main status statistics to {main_stats_file_path}")

    log_file_path = f"{save_path}/timestamp_logs.txt"
    with open(log_file_path, 'w', encoding='utf-8') as f:
        f.write(f"{model_type.upper()} Model - Trial {trial} - Timestamp Logs\n")
        f.write("=" * 50 + "\n\n")
        for log_entry in timestamp_logs:
            f.write(log_entry + "\n")
    print(f"  {model_prefix} -> Saved timestamp logs to {log_file_path}")
def continuous_update_finetune_scrub(args):
    print("dataset and net_name:", args['dataset_name'], args['net_name'])

    total_unlearn_steps = args.get('total_unlearn_steps', 50)
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

        train_single_model_scrub(
            args=args,
            train_dataset=target_m,
            test_dataset=target_um,
            num_classes=num_classes,
            trial=t,
            total_unlearn_steps=total_unlearn_steps,
            is_shadow=False
        )
        train_single_model_scrub(
            args=args,
            train_dataset=shadow_m,
            test_dataset=shadow_um,
            num_classes=num_classes,
            trial=t,
            total_unlearn_steps=total_unlearn_steps,
            is_shadow=True
        )