import os

import pandas as pd
from torch.utils.data import DataLoader, Subset
import random
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import copy
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import datasets, transforms
def sample_target_samples(train_data, proportion, dataset_name ,unlearn_label=False):
    num_unlearn_samples=0
    if unlearn_label:
        target_label = 5
        if isinstance(train_data[0], dict):
            unlearn_indices = [i for i, sample in enumerate(train_data) if sample['labels'] == target_label]
        else:
            # 图像数据
            unlearn_indices = [i for i, (_, label) in enumerate(train_data) if label == target_label]
        selected_samples = Subset(train_data, unlearn_indices)

        all_indices = list(range(len(train_data)))
        remaining_indices = list(set(all_indices) - set(unlearn_indices))
        unlearn_dataset = Subset(train_data, remaining_indices)
    else:
        all_indices = list(range(len(train_data)))
        if proportion <=0:
            raise ValueError("Proportion must be greater than zero")
        elif proportion >=1:
            num_unlearn_samples= int(proportion)
        else:
            num_unlearn_samples = int(len(train_data) * proportion)

        random_indices = random.sample(all_indices, num_unlearn_samples)
        selected_samples = Subset(train_data, random_indices)

        remaining_indices = list(set(all_indices) - set(random_indices))
        unlearn_dataset = Subset(train_data, remaining_indices)

    return selected_samples, unlearn_dataset
# for over-well, return 2 target sets
def sample_target_samples2(train_data_transformed,train_data, proportion, dataset_name ,unlearn_label=False):
    num_unlearn_samples=0
    if unlearn_label:
        target_label = 5
        # 处理不同类型的数据格式
        if isinstance(train_data_transformed[0], dict):
            # 文本数据（SST5等）
            unlearn_indices = [i for i, sample in enumerate(train_data_transformed) if sample['labels'] == target_label]
        else:
            # 图像数据
            unlearn_indices = [i for i, (_, label) in enumerate(train_data_transformed) if label == target_label]
        selected_samples_transformed = Subset(train_data_transformed, unlearn_indices)

        all_indices = list(range(len(train_data_transformed)))
        remaining_indices = list(set(all_indices) - set(unlearn_indices))
        unlearn_dataset_transformed = Subset(train_data_transformed, remaining_indices)

    else:
        all_indices = list(range(len(train_data_transformed)))
        if proportion <=0:
            raise ValueError("Proportion must be greater than zero")
        elif proportion >=1:
            num_unlearn_samples= int(proportion)
        else:
            num_unlearn_samples = int(len(train_data_transformed) * proportion)

        random_indices = random.sample(all_indices, num_unlearn_samples)
        selected_samples_transformed = Subset(train_data_transformed, random_indices)

        remaining_indices = list(set(all_indices) - set(random_indices))
        unlearn_dataset_transformed = Subset(train_data_transformed, remaining_indices)

        selected_samples=Subset(train_data, random_indices)
        unlearn_dataset= Subset(train_data, remaining_indices)

    return selected_samples_transformed, unlearn_dataset_transformed,selected_samples,unlearn_dataset






def l1_regularization(model):
    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)


def pruning_model(model, px):
    print("Apply Unstructured L1 Pruning Globally (all conv layers)")
    parameters_to_prune = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            parameters_to_prune.append((m, "weight"))

    parameters_to_prune = tuple(parameters_to_prune)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )
    return model

def check_sparsity(model):
    sum_list = 0
    zero_sum = 0

    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            sum_list = sum_list + float(m.weight.nelement())
            zero_sum = zero_sum + float(torch.sum(m.weight == 0))

    if zero_sum:
        remain_weight_ratie = 100 * (1 - zero_sum / sum_list)
        print("* remain weight ratio = ", 100 * (1 - zero_sum / sum_list), "%")
    else:
        print("no weight for calculating sparsity")
        remain_weight_ratie = None

    return remain_weight_ratie

def extract_mask(model_dict):
    new_dict = {}
    for key in model_dict.keys():
        if "mask" in key:
            new_dict[key] = copy.deepcopy(model_dict[key])

    return new_dict

def remove_prune(model):
    print("Remove hooks for multiplying masks (all conv layers)")
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.remove(m, "weight")

def prune_model_custom(model, mask_dict,args):
    print("Pruning with custom mask (all conv layers)")
    model=model.to(args['device'])
    device = args['device']

    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            mask_name = name + ".weight_mask"
            if mask_name in mask_dict.keys():
                # 确保mask在正确的设备上，并且dtype匹配
                mask = mask_dict[mask_name]
                if isinstance(mask, torch.Tensor):
                    # 获取模型权重的dtype和设备
                    weight_dtype = m.weight.dtype
                    # 将mask移到正确的设备并匹配dtype
                    mask = mask.to(device=device, dtype=weight_dtype)
                prune.CustomFromMask.apply(
                    m, "weight", mask=mask
                )
            else:
                print("Can not find [{}] in mask_dict".format(mask_name))
    return model

def get_gradient_norm(model,target_m,args):
    train_loader = torch.utils.data.DataLoader(
        target_m, batch_size=1, shuffle=False)
    model.eval()
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    criterion = nn.CrossEntropyLoss()
    for data, target in train_loader:
        data, target = data.to(args['device']), target.to(args['device'])
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
    param_norm = nn.utils.parameters_to_vector(model.parameters()).norm()

    return param_norm

def save_output(shadow_or_target,args,original_model,unlearned_model,forget_set,retain_set,test_set,shadow_um,t,k=-1):

    if k!=-1:
        save_path = os.getcwd() + f"/save2/{args['U_method']}/{args['net_name']}/{args['dataset_name']}/{args['proportion_of_group_unlearn']}/{shadow_or_target}/{t}/{k}"
    else:
        save_path = os.getcwd() + f"/save2/{args['U_method']}/{args['net_name']}/{args['dataset_name']}/{args['proportion_of_group_unlearn']}/{shadow_or_target}/{t}/"
    os.makedirs(save_path, exist_ok=True)
    #target_list=['target','well-over','well-well','over-over','over-well','high_conf','low_conf','low_entropy','high_entropy','low_asr','high_asr','random','dp_0.5','dp_1.0','dp_1.5','dp_2.0','dp_0.69','dp_0.96','in','out']
    target_list=['shadow']

    print("unlearned sample:------------------")
    for i, idx in enumerate(forget_set.indices):
        sample = forget_set.dataset[idx]
        
        if isinstance(sample, dict):
            data = sample
            label = sample['labels']
        else:
            data, label = sample
            
        save_path_target_sample = f'{save_path}/{i}'
        os.makedirs(save_path_target_sample, exist_ok=True)
        unlearned_sample_original_model_posterior = original_model.predict_proba(data)
        unlearned_sample_unlearned_model_posterior = unlearned_model.predict_proba(data)
        torch.save(unlearned_sample_original_model_posterior,
                   f"{save_path_target_sample}/unlearned_sample_original_model_posterior.pth")
        torch.save(unlearned_sample_unlearned_model_posterior,
                   f"{save_path_target_sample}/unlearned_sample_unlearned_model_posterior.pth")
        torch.save(label, f"{save_path_target_sample}/target_sample_label.pth")

    print("retain sample:------------------")
    retain_sample, _ = sample_target_samples(retain_set, int(len(forget_set)), args['dataset_name'])

    for i, idx in enumerate(retain_sample.indices):
        sample = retain_sample.dataset[idx]
        
        # 处理不同类型的数据格式
        if isinstance(sample, dict):
            # 文本数据（SST5等）
            data = sample
            label = sample['labels']
        else:
            # 图像数据
            data, label = sample
            
        save_path_retain_sample = f'{save_path}/{i}'
        if not os.path.exists(save_path_retain_sample):
            os.makedirs(save_path_retain_sample)
        retain_sample_original_model_posterior = original_model.predict_proba(data)
        retain_sample_unlearned_model_posterior = unlearned_model.predict_proba(data)
        torch.save(retain_sample_original_model_posterior,
                   f"{save_path_retain_sample}/retain_sample_original_model_posterior.pth")
        torch.save(retain_sample_unlearned_model_posterior,
                   f"{save_path_retain_sample}/retain_sample_unlearned_model_posterior.pth")
        torch.save(label, f"{save_path_retain_sample}/retain_sample_label.pth")

    print("unseen sample:------------------")
    if shadow_or_target in target_list:
        unseen_sample, _ = sample_target_samples(shadow_um, int(len(forget_set)), args['dataset_name'])
    else:
        unseen_sample, _ = sample_target_samples(test_set, int(len(forget_set)),  args['dataset_name'])

    for i, idx in enumerate(unseen_sample.indices):
        sample = unseen_sample.dataset[idx]
        
        # 处理不同类型的数据格式
        if isinstance(sample, dict):
            # 文本数据（SST5等）
            data = sample
            label = sample['labels']
        else:
            # 图像数据
            data, label = sample
            
        save_path_unseen_sample = f'{save_path}/{i}'
        if not os.path.exists(save_path_unseen_sample):
            os.makedirs(save_path_unseen_sample)
        unseen_sample_original_model_posterior = original_model.predict_proba(data)
        unseen_sample_unlearned_model_posterior = unlearned_model.predict_proba(data)
        torch.save(unseen_sample_original_model_posterior,
                   f"{save_path_unseen_sample}/unseen_sample_original_model_posterior.pth")
        torch.save(unseen_sample_unlearned_model_posterior,
                   f"{save_path_unseen_sample}/unseen_sample_unlearned_model_posterior.pth")
        torch.save(label, f"{save_path_unseen_sample}/unseen_sample_label.pth")

    # 为文本数据创建自定义collate函数
    def collate_fn(batch):
        """自定义collate函数，处理文本和图像数据"""
        if isinstance(batch[0], dict):
            # 文本数据（SST5等）
            input_ids = torch.stack([item['input_ids'] for item in batch])
            attention_mask = torch.stack([item['attention_mask'] for item in batch])
            labels = torch.stack([item['labels'] for item in batch])
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }
        else:
            # 图像数据，使用默认collate
            return torch.utils.data.dataloader.default_collate(batch)
    
    forget_loader = torch.utils.data.DataLoader(
        forget_set, batch_size=args['batch_size'], shuffle=True, collate_fn=collate_fn)
    retain_loader = torch.utils.data.DataLoader(
        retain_set, batch_size=args['batch_size'], shuffle=True, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args['batch_size'], shuffle=True, collate_fn=collate_fn)

    forget_set_acc_original = original_model.test_model_acc(forget_loader)
    retain_set_acc_original = original_model.test_model_acc(retain_loader)
    test_acc_original = original_model.test_model_acc(test_loader)
    forget_set_acc_unlearned = unlearned_model.test_model_acc(forget_loader)
    retain_set_acc_unlearned = unlearned_model.test_model_acc(retain_loader)
    test_acc_unlearned = unlearned_model.test_model_acc(test_loader)

    metrics_dict = {
        "forget_set_acc_original": round(forget_set_acc_original, 4),
        "retain_set_acc_original": round(retain_set_acc_original, 4),
        "test_acc_original": round(test_acc_original, 4),
        "forget_set_acc_unlearned": round(forget_set_acc_unlearned, 4),
        "retain_set_acc_unlearned": round(retain_set_acc_unlearned, 4),
        "test_acc_unlearned": round(test_acc_unlearned, 4),
    }
    print(metrics_dict)
    # 创建 DataFrame，转置存储 (指标名 | 对应值)
    data_to_save = pd.DataFrame(list(metrics_dict.items()), columns=["Metric", "Value"])

    # 按 **纵向格式** 存储 CSV
    data_to_save.to_csv(f"{save_path}/model_performance.csv", index=False)


import torch
from torch.utils.data import Subset
from tqdm import tqdm
import random
import numpy as np


def calculate_confidence_with_subsets(model, train_loader, K, device):

    model.to(device)
    model.eval()  # 设置为评估模式
    K = int(K)

    indices = list(range(len(train_loader.dataset)))
    confidences = []  # 存储 Top-2 置信度差值
    entropies = []  # 存储信息熵

    # 获取模型输出
    samples = [train_loader.dataset[i] for i in indices]
    
    # 处理不同类型的数据格式
    if isinstance(samples[0], dict):
        # 文本数据（SST5等）
        inputs = samples  # 直接传递字典列表
        outputs = model(inputs)
    else:
        # 图像数据
        inputs = torch.stack([sample[0] for sample in samples]).to(device)
        outputs = model(inputs)
    
    probabilities = torch.softmax(outputs, dim=1)

    # 计算 Top-2 confidence
    top2_probs, _ = torch.topk(probabilities, k=2, dim=1)
    confidence = top2_probs[:, 0] - top2_probs[:, 1]
    confidences.extend(confidence.cpu().detach().numpy())

    # 计算信息熵
    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)  # 避免 log(0)
    entropies.extend(entropy.cpu().detach().numpy())

    # 按照置信度从低到高排序
    sorted_confidence_indices = [idx for _, idx in sorted(zip(confidences, indices), key=lambda x: x[0])]
    sorted_confidence_values = sorted(confidences)

    # 按照熵从高到低排序（高熵表示高不确定性）
    sorted_entropy_indices = [idx for _, idx in sorted(zip(entropies, indices), key=lambda x: x[0], reverse=True)]
    sorted_entropy_values = sorted(entropies, reverse=True)

    # 获取 Top-2 confidence 相关的样本
    low_confidence_indices = sorted_confidence_indices[:K]  # 置信度最低（高不确定性）
    high_confidence_indices = sorted_confidence_indices[-K:]  # 置信度最高（低不确定性）

    # 获取 Entropy 相关的样本
    high_entropy_indices = sorted_entropy_indices[:K]  # 熵最高（高不确定性）
    low_entropy_indices = sorted_entropy_indices[-K:]  # 熵最低（低不确定性）

    # 随机选择 K 个样本
    random_indices = random.sample(indices, K)

    # 获取剩余样本的索引
    all_indices = set(indices)
    remaining_after_high_indices = list(all_indices - set(high_confidence_indices))
    remaining_after_low_indices = list(all_indices - set(low_confidence_indices))
    remaining_after_high_entropy_indices = list(all_indices - set(high_entropy_indices))
    remaining_after_low_entropy_indices = list(all_indices - set(low_entropy_indices))
    remaining_after_random_indices = list(all_indices - set(random_indices))

    # 创建子集
    high_confidence_subset = Subset(train_loader.dataset, high_confidence_indices)
    remaining_after_high = Subset(train_loader.dataset, remaining_after_high_indices)

    low_confidence_subset = Subset(train_loader.dataset, low_confidence_indices)
    remaining_after_low = Subset(train_loader.dataset, remaining_after_low_indices)

    high_entropy_subset = Subset(train_loader.dataset, high_entropy_indices)
    remaining_after_high_entropy = Subset(train_loader.dataset, remaining_after_high_entropy_indices)

    low_entropy_subset = Subset(train_loader.dataset, low_entropy_indices)
    remaining_after_low_entropy = Subset(train_loader.dataset, remaining_after_low_entropy_indices)

    random_subset = Subset(train_loader.dataset, random_indices)
    remaining_after_random = Subset(train_loader.dataset, remaining_after_random_indices)

    return (
        high_confidence_subset, remaining_after_high,
        low_confidence_subset, remaining_after_low,
        high_entropy_subset, remaining_after_high_entropy,
        low_entropy_subset, remaining_after_low_entropy,
        random_subset,remaining_after_random
    )


class TransformedDataset(Dataset):
    def __init__(self, subset):
        self.subset = subset  # 传入 Subset
        self.transform =  transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])  # 传入 transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]  # 获取原始数据
        image = self.transform(image)  # 重新应用 transform
        return image, label

def get_top_bottom_n_indices(lst, n):
    sorted_indices = np.argsort(lst)  # 获取从小到大的索引
    bottom_n_indices = sorted_indices[:n].tolist()  # 前 n 小的索引
    top_n_indices = sorted_indices[-n:].tolist()  # 前 n 大的索引
    return top_n_indices, bottom_n_indices


def add_gaussian_noise(image, mean=0, std=0.1):

    noise = torch.randn_like(image) * std + mean
    noisy_image = image + noise
    return noisy_image