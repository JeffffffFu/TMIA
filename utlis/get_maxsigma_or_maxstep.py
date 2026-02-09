#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据目标epsilon计算所需的noise_multiplier (sigma)
"""
import numpy as np
from scipy.optimize import minimize_scalar
from utlis.compute_dp_sgd import apply_dp_sgd_analysis

def get_noise_multiplier(eps, delta, sample_rate, steps, alphas):
    """
    根据目标epsilon计算所需的noise_multiplier (sigma)
    
    Args:
        eps: 目标epsilon（隐私预算）
        delta: 失败概率
        sample_rate: 采样率 q = batch_size / dataset_size
        steps: 训练步数
        alphas: RDP orders列表
    
    Returns:
        float: 所需的noise_multiplier (sigma)
    """
    def objective(sigma):
        """
        目标函数：计算当前sigma对应的epsilon，返回与目标epsilon的差值
        """
        try:
            computed_eps, _ = apply_dp_sgd_analysis(sample_rate, sigma, steps, alphas, delta)
            # 返回 (computed_eps - target_eps)^2，使得computed_eps尽可能接近target_eps
            return (computed_eps - eps) ** 2
        except:
            # 如果计算失败，返回一个很大的值
            return 1e10
    
    # 使用二分搜索或优化算法找到满足条件的sigma
    # sigma的范围通常在 [0.1, 10.0] 之间
    # 使用minimize_scalar进行优化
    result = minimize_scalar(objective, bounds=(0.1, 10.0), method='bounded')
    
    if result.success:
        optimal_sigma = result.x
        # 验证结果：计算实际epsilon
        actual_eps, _ = apply_dp_sgd_analysis(sample_rate, optimal_sigma, steps, alphas, delta)
        if abs(actual_eps - eps) < 0.1:  # 允许0.1的误差
            return optimal_sigma
        else:
            # 如果误差太大，尝试更精确的搜索
            # 使用更小的搜索范围
            refined_result = minimize_scalar(
                objective, 
                bounds=(max(0.1, optimal_sigma - 0.5), min(10.0, optimal_sigma + 0.5)), 
                method='bounded'
            )
            if refined_result.success:
                return refined_result.x
            return optimal_sigma
    else:
        # 如果优化失败，使用二分搜索
        return _binary_search_sigma(eps, delta, sample_rate, steps, alphas)


def _binary_search_sigma(eps, delta, sample_rate, steps, alphas, low=0.1, high=10.0, tolerance=0.01, max_iter=50):
    """
    使用二分搜索找到满足目标epsilon的sigma
    
    Args:
        eps: 目标epsilon
        delta: 失败概率
        sample_rate: 采样率
        steps: 训练步数
        alphas: RDP orders
        low: 搜索下界
        high: 搜索上界
        tolerance: 容忍误差
        max_iter: 最大迭代次数
    
    Returns:
        float: 找到的sigma值
    """
    for _ in range(max_iter):
        mid = (low + high) / 2.0
        try:
            computed_eps, _ = apply_dp_sgd_analysis(sample_rate, mid, steps, alphas, delta)
            if abs(computed_eps - eps) < tolerance:
                return mid
            elif computed_eps > eps:
                # 如果计算的epsilon大于目标，需要增加sigma（更多噪声）
                low = mid
            else:
                # 如果计算的epsilon小于目标，可以减少sigma（更少噪声）
                high = mid
        except:
            # 如果计算失败，增加sigma
            low = mid
    
    # 返回中间值
    return (low + high) / 2.0

