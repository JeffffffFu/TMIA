import os
import numpy as np
import glob
import random
from attack.TMIA.inference import breakpoint_to_membership_timestamps

def build_path_and_find_timestamps(args):
    U_method = args.get('U_method')
    net_name = args.get('net_name')
    dataset_name = args.get('dataset_name')
    proportion = args.get('proportion_of_group_unlearn')

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    save_base_path = os.path.join(project_root, "save")

    if args.get('is_shadow', False):
        model_type = 'shadow'
        if args['attack_method'] == 'Transfer_dataset' or args['attack_method'] == 'Transfer_model' or args[
            'attack_method'] == 'Transfer_unlearning':
                proportion=0.01
        elif args['net_name']=='pythia70m_dropout':
            net_name='pythia70m'
        elif  args['net_name']=='simple_cnn_dropout':
            net_name = 'simple_cnn'
        elif U_method=='continuous_update_finetune_dp' or U_method=='continuous_update_finetune_multiple_update':
            U_method='continuous_update_finetune'

        elif  dataset_name=='mnli':
            proportion = 0.005
        elif  dataset_name=='cinic10':
            proportion = 0.001
        elif dataset_name=='svhn':
            proportion=0.01
        else:
            proportion = 0.01
    else:
        model_type = args.get('model_type', 'target')

    trial = args.get('trial', 0)


    save_path = os.path.join(save_base_path, U_method, net_name, dataset_name,
                             str(proportion), model_type, str(trial))


    timestamp_dirs = []
    for item in glob.glob(os.path.join(save_path, "timestamp_*")):
        if os.path.isdir(item):
            timestamp_dirs.append(item)
    timestamp_dirs = sorted(timestamp_dirs,
                           key=lambda x: int(os.path.basename(x).split("_")[1]))

    if len(timestamp_dirs) == 0:
        raise ValueError(f"未找到timestamp目录: {save_path}")

    return save_path, timestamp_dirs


def process_breakpoint_and_select_sample(
    save_path,
    timestamp_dirs,
    sample_index=None,
    random_seed=None,
    num_samples=1,
    window_size=None
):
    breakpoint_file = os.path.join(save_path, "breakpoint_indices.npy")
    breakpoint_indices_dict = {}
    samples_with_one_insert_one_remove = []
    samples_with_one_insert_only = []
    samples_with_one_remove_only = []
    if os.path.exists(breakpoint_file):
        breakpoint_indices_dict = np.load(breakpoint_file, allow_pickle=True).item()

        num_timestamps = len(timestamp_dirs)

        if window_size is not None and window_size > 0:
            actual_timestamps = []
            for timestamp_dir in timestamp_dirs:
                timestamp_num = int(os.path.basename(timestamp_dir).split("_")[1])
                actual_timestamps.append(timestamp_num)
            ts_to_idx = {ts: idx for idx, ts in enumerate(actual_timestamps)}

            filtered_samples_1i1r = []
            filtered_samples_1i = []
            filtered_samples_1r = []

            for idx, breakpoint_info in breakpoint_indices_dict.items():
                insert_count = len(breakpoint_info.get('insert', []))
                remove_count = len(breakpoint_info.get('remove', []))

                insert_ts_list = breakpoint_info.get('insert', [])
                remove_ts_list = breakpoint_info.get('remove', [])
                insert_indices = [ts_to_idx[ts] for ts in insert_ts_list if ts in ts_to_idx]
                remove_indices = [ts_to_idx[ts] for ts in remove_ts_list if ts in ts_to_idx]
                all_bp_indices = insert_indices + remove_indices

                is_in_boundary = any(bp_idx < window_size or bp_idx >= (num_timestamps - window_size)
                                      for bp_idx in all_bp_indices)

                if not is_in_boundary:
                    if insert_count == 1 and remove_count == 1:
                        filtered_samples_1i1r.append(idx)
                    elif insert_count == 1 and remove_count == 0:
                        filtered_samples_1i.append(idx)
                    elif insert_count == 0 and remove_count == 1:
                        filtered_samples_1r.append(idx)

            samples_with_one_insert_one_remove = filtered_samples_1i1r
            samples_with_one_insert_only = filtered_samples_1i
            samples_with_one_remove_only = filtered_samples_1r
        else:
            for idx, breakpoint_info in breakpoint_indices_dict.items():
                insert_count = len(breakpoint_info.get('insert', []))
                remove_count = len(breakpoint_info.get('remove', []))
                if insert_count == 1 and remove_count == 1:
                    samples_with_one_insert_one_remove.append(idx)
                elif insert_count == 1 and remove_count == 0:
                    samples_with_one_insert_only.append(idx)
                elif insert_count == 0 and remove_count == 1:
                    samples_with_one_remove_only.append(idx)

    if num_samples == 1:
        if sample_index is None:
            if len(samples_with_one_insert_one_remove) > 0:
                if random_seed is not None:
                    random.seed(random_seed)
                    np.random.seed(random_seed)
                    selected_indices = [random.choice(samples_with_one_insert_one_remove)]
                else:
                    selected_indices = [samples_with_one_insert_one_remove[0]]
            else:
                available_supplement = samples_with_one_insert_only + samples_with_one_remove_only
                if len(available_supplement) > 0:
                    if random_seed is not None:
                        random.seed(random_seed)
                        np.random.seed(random_seed)
                        selected_indices = [random.choice(available_supplement)]
                    else:
                        selected_indices = [available_supplement[0]]
                else:
                    first_timestamp_dir = timestamp_dirs[0]
                    indices_file = os.path.join(first_timestamp_dir, "sample_indices.npy")
                    if os.path.exists(indices_file):
                        indices = np.load(indices_file)
                        selected_indices = [indices[0] if len(indices) > 0 else 0]
                    else:
                        selected_indices = [0]
        else:
            selected_indices = [sample_index]
    else:
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        selected_indices = []
        num_needed = num_samples

        num_from_1i1r = min(num_needed, len(samples_with_one_insert_one_remove))
        if num_from_1i1r > 0:
            if random_seed is not None:
                selected_from_1i1r = random.sample(samples_with_one_insert_one_remove, num_from_1i1r)
            else:
                selected_from_1i1r = samples_with_one_insert_one_remove[:num_from_1i1r]
            selected_indices.extend(selected_from_1i1r)
            num_needed -= num_from_1i1r

        if num_needed > 0:
            available_supplement = samples_with_one_insert_only + samples_with_one_remove_only

            if len(available_supplement) > 0:
                num_from_supplement = min(num_needed, len(available_supplement))
                if random_seed is not None:
                    selected_from_supplement = random.sample(available_supplement, num_from_supplement)
                else:
                    selected_from_supplement = available_supplement[:num_from_supplement]
                selected_indices.extend(selected_from_supplement)
                num_needed -= num_from_supplement

    samples_breakpoint_info = {}
    invalid_samples = []

    for idx in selected_indices:
        sample_breakpoint_info = breakpoint_indices_dict.get(idx, {})
        insert_timestamps = sample_breakpoint_info.get('insert', [])
        remove_timestamps = sample_breakpoint_info.get('remove', [])

        insert_count = len(insert_timestamps)
        remove_count = len(remove_timestamps)

        if insert_count != 1 or remove_count != 1:
            invalid_samples.append({
                'index': idx,
                'insert_count': insert_count,
                'remove_count': remove_count,
                'insert_timestamps': insert_timestamps,
                'remove_timestamps': remove_timestamps
            })
        samples_breakpoint_info[idx] = {
            'insert': insert_timestamps,
            'remove': remove_timestamps
        }

    return selected_indices, breakpoint_indices_dict, samples_breakpoint_info


def calculate_sample_membership_stats(sample_idx, breakpoint_info, num_timestamps, timestamp_dirs=None):
    insert_ts = breakpoint_info.get('insert', [])
    remove_ts = breakpoint_info.get('remove', [])

    if timestamp_dirs is not None:
        actual_timestamps = []
        for timestamp_dir in timestamp_dirs:
            timestamp_num = int(os.path.basename(timestamp_dir).split("_")[1])
            actual_timestamps.append(timestamp_num)

        ts_to_idx = {ts: idx for idx, ts in enumerate(actual_timestamps)}

        insert_ts_indices = [ts_to_idx[ts] for ts in insert_ts if ts in ts_to_idx]
        remove_ts_indices = [ts_to_idx[ts] for ts in remove_ts if ts in ts_to_idx]
    else:
        insert_ts_indices = insert_ts
        remove_ts_indices = remove_ts

    max_idx = num_timestamps - 1
    insert_ts_indices = [idx for idx in insert_ts_indices if 0 <= idx <= max_idx]
    remove_ts_indices = [idx for idx in remove_ts_indices if 0 <= idx <= max_idx]

    breakpoints = [(ts, 1) for ts in insert_ts_indices] + [(ts, 2) for ts in remove_ts_indices]

    membership_timestamps = breakpoint_to_membership_timestamps(breakpoints, num_timestamps)

    member_count = len(membership_timestamps)
    nonmember_count = num_timestamps - member_count
    member_ratio = member_count / num_timestamps if num_timestamps > 0 else 0.0
    return {
        'sample_idx': sample_idx,
        'member_count': member_count,
        'nonmember_count': nonmember_count,
        'total_count': num_timestamps,
        'member_ratio': member_ratio,
        'insert_timestamps': insert_ts_indices,
        'remove_timestamps': remove_ts_indices
    }

def select_test_samples_by_pattern(
    save_path,
    timestamp_dirs,
    num_insert_only=400,
    num_remove_only=400,
    num_insert_remove=400,
    random_seed=None,
    balance_membership=True,
    target_member_ratio=0.66,
    verbose=True
):
    if isinstance(target_member_ratio, (tuple, list)) and len(target_member_ratio) == 2:
        member_part, nonmember_part = target_member_ratio
        target_member_ratio = member_part / (member_part + nonmember_part)
    breakpoint_file = os.path.join(save_path, "breakpoint_indices.npy")
    breakpoint_indices_dict = {}

    num_timestamps = len(timestamp_dirs)

    samples_insert_only = []
    samples_remove_only = []
    samples_insert_remove = []
    samples_other = {}

    if os.path.exists(breakpoint_file):
        breakpoint_indices_dict = np.load(breakpoint_file, allow_pickle=True).item()

        for idx, breakpoint_info in breakpoint_indices_dict.items():
            insert_count = len(breakpoint_info.get('insert', []))
            remove_count = len(breakpoint_info.get('remove', []))

            if insert_count == 1 and remove_count == 0:
                samples_insert_only.append(idx)
            elif insert_count == 0 and remove_count == 1:
                samples_remove_only.append(idx)
            elif insert_count == 1 and remove_count == 1:
                samples_insert_remove.append(idx)
            else:
                pattern_key = (insert_count, remove_count)
                if pattern_key not in samples_other:
                    samples_other[pattern_key] = []
                samples_other[pattern_key].append(idx)

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    def select_balanced_samples(candidate_indices, target_num, category_name, ts_dirs, target_ratio, current_total_member=0, current_total_nonmember=0):
        if len(candidate_indices) == 0:
            return [], 0, 0

        actual_num = min(target_num, len(candidate_indices))

        if not balance_membership:
            sampled = random.sample(candidate_indices, actual_num) if len(candidate_indices) > actual_num else candidate_indices
            total_member = 0
            total_nonmember = 0
            for idx in sampled:
                stats = calculate_sample_membership_stats(idx, breakpoint_indices_dict[idx], num_timestamps, ts_dirs)
                total_member += stats['member_count']
                total_nonmember += stats['nonmember_count']
            return sampled, total_member, total_nonmember

        candidate_stats = []
        for idx in candidate_indices:
            stats = calculate_sample_membership_stats(idx, breakpoint_indices_dict[idx], num_timestamps, ts_dirs)
            candidate_stats.append((idx, stats))

        candidate_stats.sort(key=lambda x: x[1]['member_ratio'], reverse=True)

        selected = []
        selected_member = 0
        selected_nonmember = 0

        all_member = current_total_member
        all_nonmember = current_total_nonmember

        remaining = candidate_stats.copy()

        for _ in range(actual_num):
            if len(remaining) == 0:
                break

            best_idx = None
            best_stats = None
            best_score = float('inf')

            current_ratio = all_member / (all_member + all_nonmember) if (all_member + all_nonmember) > 0 else target_ratio

            if current_ratio < target_ratio:
                if len(remaining) > 0:
                    best_idx, best_stats = remaining[0]
                    best_score = 0
            else:
                for idx, stats in remaining:
                    new_member = all_member + stats['member_count']
                    new_nonmember = all_nonmember + stats['nonmember_count']
                    new_ratio = new_member / (new_member + new_nonmember) if (new_member + new_nonmember) > 0 else target_ratio

                    score = abs(new_ratio - target_ratio)

                    if score < best_score:
                        best_score = score
                        best_idx = idx
                        best_stats = stats

            if best_idx is not None:
                selected.append(best_idx)
                selected_member += best_stats['member_count']
                selected_nonmember += best_stats['nonmember_count']
                all_member += best_stats['member_count']
                all_nonmember += best_stats['nonmember_count']
                remaining = [(idx, stats) for idx, stats in remaining if idx != best_idx]

        return selected, selected_member, selected_nonmember

    selected_indices = []
    total_member_count = 0
    total_nonmember_count = 0

    def evaluate_category_ratio(candidate_indices, category_name):
        if len(candidate_indices) == 0:
            return None, 0.0

        sample_size = min(50, len(candidate_indices))
        sample_indices = random.sample(candidate_indices, sample_size) if len(candidate_indices) > sample_size else candidate_indices

        total_member = 0
        total_nonmember = 0
        for idx in sample_indices:
            stats = calculate_sample_membership_stats(idx, breakpoint_indices_dict[idx], num_timestamps, timestamp_dirs)
            total_member += stats['member_count']
            total_nonmember += stats['nonmember_count']

        if total_member + total_nonmember == 0:
            return None, 0.0

        avg_ratio = total_member / (total_member + total_nonmember)
        return sample_indices, avg_ratio

    threshold_ratio = target_member_ratio * 0.5

    if len(samples_insert_only) > 0:
        should_skip = False
        if balance_membership:
            _, category_avg_ratio = evaluate_category_ratio(samples_insert_only, "1 insert only")
            if category_avg_ratio < threshold_ratio:
                should_skip = True

        if not should_skip:
            sampled, member_count, nonmember_count = select_balanced_samples(
                samples_insert_only, num_insert_only, "1 insert only",
                timestamp_dirs, target_member_ratio, total_member_count, total_nonmember_count
            )
            selected_indices.extend(sampled)
            total_member_count += member_count
            total_nonmember_count += nonmember_count

    if len(samples_remove_only) > 0:
        should_skip = False
        if balance_membership:
            _, category_avg_ratio = evaluate_category_ratio(samples_remove_only, "1 remove only")
            if category_avg_ratio < threshold_ratio:
                should_skip = True

        if not should_skip:
            sampled, member_count, nonmember_count = select_balanced_samples(
                samples_remove_only, num_remove_only, "1 remove only",
                timestamp_dirs, target_member_ratio, total_member_count, total_nonmember_count
            )
            selected_indices.extend(sampled)
            total_member_count += member_count
            total_nonmember_count += nonmember_count

    if len(samples_insert_remove) > 0:
        should_skip = False
        if balance_membership:
            _, category_avg_ratio = evaluate_category_ratio(samples_insert_remove, "1 insert 1 remove")
            if category_avg_ratio < threshold_ratio:
                should_skip = True

        if not should_skip:
            sampled, member_count, nonmember_count = select_balanced_samples(
                samples_insert_remove, num_insert_remove, "1 insert 1 remove",
                timestamp_dirs, target_member_ratio, total_member_count, total_nonmember_count
            )
            selected_indices.extend(sampled)
            total_member_count += member_count
            total_nonmember_count += nonmember_count

    need_select_from_other = False
    if len(samples_other) > 0:
        if len(selected_indices) == 0:
            need_select_from_other = True
        else:
            if balance_membership:
                current_ratio = total_member_count / (total_member_count + total_nonmember_count) if (total_member_count + total_nonmember_count) > 0 else 0
                if current_ratio < target_member_ratio:
                    need_select_from_other = True

    if need_select_from_other:
        if len(selected_indices) == 0:
            target_num_samples = max(num_insert_only, num_remove_only, num_insert_remove, 100)
            needed_member = None
        else:
            if total_nonmember_count > 0:
                required_total = total_nonmember_count / (1 - target_member_ratio)
                required_member = required_total * target_member_ratio
                needed_member = max(0, int(required_member - total_member_count))
            else:
                needed_member = None
                target_num_samples = 100

        all_other_samples = []
        for pattern_key, sample_list in samples_other.items():
            for idx in sample_list:
                all_other_samples.append((idx, pattern_key))

        other_candidate_stats = []
        for idx, pattern_key in all_other_samples:
            if idx not in selected_indices:
                stats = calculate_sample_membership_stats(idx, breakpoint_indices_dict[idx], num_timestamps, timestamp_dirs)
                other_candidate_stats.append((idx, pattern_key, stats))

        other_candidate_stats.sort(key=lambda x: x[2]['member_ratio'], reverse=True)

        selected_from_other = []
        current_member_from_other = 0
        current_nonmember_from_other = 0

        for idx, pattern_key, stats in other_candidate_stats:
            if len(selected_indices) == 0:
                if len(selected_from_other) >= target_num_samples:
                    new_ratio = current_member_from_other / (current_member_from_other + current_nonmember_from_other) if (current_member_from_other + current_nonmember_from_other) > 0 else 0
                    if balance_membership and new_ratio >= target_member_ratio * 0.90:
                        break
                    if len(selected_from_other) >= target_num_samples * 1.5:
                        break
                selected_from_other.append((idx, pattern_key, stats))
                current_member_from_other += stats['member_count']
                current_nonmember_from_other += stats['nonmember_count']
            else:
                if needed_member is not None and current_member_from_other < needed_member * 2:
                    selected_from_other.append((idx, pattern_key, stats))
                    current_member_from_other += stats['member_count']
                    current_nonmember_from_other += stats['nonmember_count']

                    new_ratio = (total_member_count + current_member_from_other) / (total_member_count + total_nonmember_count + current_member_from_other + current_nonmember_from_other) if (total_member_count + total_nonmember_count + current_member_from_other + current_nonmember_from_other) > 0 else 0
                    if new_ratio >= target_member_ratio * 0.95:
                        break

        if len(selected_from_other) > 0:
                    other_indices = [idx for idx, _, _ in selected_from_other]
                    other_member_count = sum([stats['member_count'] for _, _, stats in selected_from_other])
                    other_nonmember_count = sum([stats['nonmember_count'] for _, _, stats in selected_from_other])

                    selected_indices.extend(other_indices)
                    total_member_count += other_member_count
                    total_nonmember_count += other_nonmember_count

    samples_breakpoint_info = {}
    for idx in selected_indices:
        sample_breakpoint_info = breakpoint_indices_dict.get(idx, {})
        samples_breakpoint_info[idx] = {
            'insert': sample_breakpoint_info.get('insert', []),
            'remove': sample_breakpoint_info.get('remove', [])
        }

    return selected_indices, breakpoint_indices_dict, samples_breakpoint_info


def select_samples_by_insert_remove_count(
    save_path,
    timestamp_dirs,
    num_insert,
    num_remove,
    num_samples,
    random_seed=None,
    verbose=True,
    balance_membership=True,
    target_member_ratio=0.66
):
    if isinstance(target_member_ratio, (tuple, list)) and len(target_member_ratio) == 2:
        member_part, nonmember_part = target_member_ratio
        target_member_ratio = member_part / (member_part + nonmember_part)
    elif not isinstance(target_member_ratio, (int, float)) or not (0 <= target_member_ratio <= 1):
        raise ValueError(f"target_member_ratio必须是0到1之间的float，或者是长度为2的元组/列表，但得到: {target_member_ratio}")

    breakpoint_file = os.path.join(save_path, "breakpoint_indices.npy")
    breakpoint_indices_dict = {}

    num_timestamps = len(timestamp_dirs)

    if os.path.exists(breakpoint_file):
        breakpoint_indices_dict = np.load(breakpoint_file, allow_pickle=True).item()
    else:
        return [], {}, {}

    matching_samples = []
    for idx, breakpoint_info in breakpoint_indices_dict.items():
        insert_count = len(breakpoint_info.get('insert', []))
        remove_count = len(breakpoint_info.get('remove', []))

        if insert_count == num_insert and remove_count == num_remove:
            matching_samples.append(idx)

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    actual_num = min(num_samples, len(matching_samples))

    if balance_membership and len(matching_samples) > 0:
        def select_balanced_samples(candidate_indices, target_num, ts_dirs, target_ratio):
            if len(candidate_indices) == 0:
                return []

            actual_target_num = min(target_num, len(candidate_indices))

            candidate_stats = []
            for idx in candidate_indices:
                stats = calculate_sample_membership_stats(idx, breakpoint_indices_dict[idx], num_timestamps, ts_dirs)
                candidate_stats.append((idx, stats))

            candidate_stats.sort(key=lambda x: x[1]['member_ratio'])

            selected = []
            total_member = 0
            total_nonmember = 0

            remaining = candidate_stats.copy()

            for _ in range(actual_target_num):
                if len(remaining) == 0:
                    break

                best_idx = None
                best_stats = None
                best_score = float('inf')

                current_ratio = total_member / (total_member + total_nonmember) if (total_member + total_nonmember) > 0 else target_ratio

                for idx, stats in remaining:
                    new_member = total_member + stats['member_count']
                    new_nonmember = total_nonmember + stats['nonmember_count']
                    new_ratio = new_member / (new_member + new_nonmember) if (new_member + new_nonmember) > 0 else target_ratio

                    score = abs(new_ratio - target_ratio)

                    if score < best_score:
                        best_score = score
                        best_idx = idx
                        best_stats = stats

                if best_idx is not None:
                    selected.append(best_idx)
                    total_member += best_stats['member_count']
                    total_nonmember += best_stats['nonmember_count']
                    remaining = [(idx, stats) for idx, stats in remaining if idx != best_idx]

            return selected

        selected_indices = select_balanced_samples(matching_samples, actual_num, timestamp_dirs, target_member_ratio)
    else:
        selected_indices = random.sample(matching_samples, actual_num) if len(matching_samples) > actual_num else matching_samples

    samples_breakpoint_info = {idx: breakpoint_indices_dict[idx] for idx in selected_indices}

    return selected_indices, breakpoint_indices_dict, samples_breakpoint_info


def compute_ppl_normalization_params(timestamp_dirs, sample_indices, samples_breakpoint_info, outlier_filter=True, z_threshold=3.0):
    sample_avg_ppl = {}

    for sample_index in sample_indices:
        sample_ppl_values = []

        sample_breakpoint_info = samples_breakpoint_info.get(sample_index, {})

        for timestamp_dir in timestamp_dirs:
            outputs_file = os.path.join(timestamp_dir, "outputs_current.npy")
            indices_file = os.path.join(timestamp_dir, "sample_indices.npy")

            if not os.path.exists(outputs_file):
                continue

            outputs = np.load(outputs_file)
            if os.path.exists(indices_file):
                indices = np.load(indices_file)
            else:
                indices = np.arange(len(outputs))

            sample_pos = np.where(indices == sample_index)[0]
            if len(sample_pos) == 0:
                continue

            sample_pos = sample_pos[0]

            ppl_value = outputs[sample_pos, 0] if outputs.ndim > 1 else outputs[sample_pos]
            ppl_value = np.clip(ppl_value, 1e-10, 1e10)
            sample_ppl_values.append(float(ppl_value))

        if len(sample_ppl_values) > 0:
            sample_avg_ppl[sample_index] = np.mean(sample_ppl_values)

    if len(sample_avg_ppl) == 0:
        raise ValueError("未找到任何PPL值用于计算标准化参数")

    all_ppl_values = []
    for sample_index in sample_indices:
        sample_breakpoint_info = samples_breakpoint_info.get(sample_index, {})

        for timestamp_dir in timestamp_dirs:
            outputs_file = os.path.join(timestamp_dir, "outputs_current.npy")
            indices_file = os.path.join(timestamp_dir, "sample_indices.npy")

            if not os.path.exists(outputs_file):
                continue

            outputs = np.load(outputs_file)
            if os.path.exists(indices_file):
                indices = np.load(indices_file)
            else:
                indices = np.arange(len(outputs))

            sample_pos = np.where(indices == sample_index)[0]
            if len(sample_pos) == 0:
                continue

            sample_pos = sample_pos[0]
            ppl_value = outputs[sample_pos, 0] if outputs.ndim > 1 else outputs[sample_pos]
            ppl_value = np.clip(ppl_value, 1e-10, 1e10)
            all_ppl_values.append(float(ppl_value))

    all_ppl_array = np.array(all_ppl_values)
    outlier_sample_indices = []

    if outlier_filter and len(sample_avg_ppl) > 0:
        avg_ppl_values = np.array(list(sample_avg_ppl.values()))
        initial_mean = np.mean(avg_ppl_values)
        initial_std = np.std(avg_ppl_values)

        if initial_std > 1e-10:
            for sample_index, avg_ppl in sample_avg_ppl.items():
                z_score = abs((avg_ppl - initial_mean) / initial_std)
                if z_score > z_threshold:
                    outlier_sample_indices.append(sample_index)

        if len(outlier_sample_indices) > 0:
            outlier_set = set(outlier_sample_indices)
            all_ppl_values = []
            for sample_index in sample_indices:
                if sample_index in outlier_set:
                    continue

                sample_breakpoint_info = samples_breakpoint_info.get(sample_index, {})
                for timestamp_dir in timestamp_dirs:
                    outputs_file = os.path.join(timestamp_dir, "outputs_current.npy")
                    indices_file = os.path.join(timestamp_dir, "sample_indices.npy")

                    if not os.path.exists(outputs_file):
                        continue

                    outputs = np.load(outputs_file)
                    if os.path.exists(indices_file):
                        indices = np.load(indices_file)
                    else:
                        indices = np.arange(len(outputs))

                    sample_pos = np.where(indices == sample_index)[0]
                    if len(sample_pos) == 0:
                        continue

                    sample_pos = sample_pos[0]
                    ppl_value = outputs[sample_pos, 0] if outputs.ndim > 1 else outputs[sample_pos]
                    ppl_value = np.clip(ppl_value, 1e-10, 1e10)
                    all_ppl_values.append(float(ppl_value))

            all_ppl_array = np.array(all_ppl_values)

    if len(all_ppl_array) == 0:
        raise ValueError("过滤异常值后没有剩余PPL值，请调整过滤参数")

    mean = float(np.mean(all_ppl_array))
    std = float(np.std(all_ppl_array))

    if std < 1e-10:
        std = 1.0

    return mean, std, outlier_sample_indices

def load_data_and_compute_confidence(
    timestamp_dirs,
    sample_indices,
    samples_breakpoint_info,
    is_llm=False,
    use_true_label=True,
    ppl_normalization_params=None,
    confidence_mode='probability'
):
    all_samples_data = []

    confidence_stats = {'min': [], 'max': [], 'mean': []}

    for sample_index in sample_indices:
        sample_breakpoint_info = samples_breakpoint_info.get(sample_index, {})
        insert_timestamps = set(sample_breakpoint_info.get('insert', []))
        remove_timestamps = set(sample_breakpoint_info.get('remove', []))

        confidences = []
        labels = []

        for timestamp_dir in timestamp_dirs:
            timestamp_num = int(os.path.basename(timestamp_dir).split("_")[1])

            outputs_file = os.path.join(timestamp_dir, "outputs_current.npy")
            indices_file = os.path.join(timestamp_dir, "sample_indices.npy")
            labels_file = os.path.join(timestamp_dir, "labels.npy")

            if not os.path.exists(outputs_file):
                continue

            outputs = np.load(outputs_file)
            if os.path.exists(indices_file):
                indices = np.load(indices_file)
            else:
                indices = np.arange(len(outputs))

            labels_data = None
            if use_true_label or confidence_mode == 'label_only':
                if os.path.exists(labels_file):
                    labels_data = np.load(labels_file)
                elif confidence_mode == 'label_only':
                    continue

            sample_pos = np.where(indices == sample_index)[0]
            if len(sample_pos) == 0:
                continue

            sample_pos = sample_pos[0]

            if is_llm:
                if confidence_mode == 'label_only':
                    raise ValueError("label_only模式不支持LLM模型，请使用probability模式")

                ppl_value = outputs[sample_pos, 0] if outputs.ndim > 1 else outputs[sample_pos]
                ppl_value = np.clip(ppl_value, 1e-10, 1e10)

                if ppl_normalization_params is not None:
                    ppl_mean, ppl_std = ppl_normalization_params
                    normalized_ppl = (ppl_value - ppl_mean) / ppl_std
                    normalized_ppl_clipped = np.clip(normalized_ppl, -50, 50)
                    if normalized_ppl_clipped > 0:
                        exp_neg = np.exp(-normalized_ppl_clipped)
                        confidence = exp_neg / (1.0 + exp_neg)
                    else:
                        confidence = 1.0 / (1.0 + np.exp(normalized_ppl_clipped))
                else:
                    confidence = 1.0 / (1.0 + ppl_value)
            else:
                sample_output = outputs[sample_pos]

                if confidence_mode == 'label_only':
                    if labels_data is None:
                        continue

                    predicted_label = int(np.argmax(sample_output))
                    true_label = int(labels_data[sample_pos])

                    confidence = 1.0 if predicted_label == true_label else 0.0
                else:
                    if use_true_label and labels_data is not None:
                        true_label = int(labels_data[sample_pos])
                        if true_label < 0 or true_label >= len(sample_output):
                            confidence = np.max(sample_output)
                        else:
                            confidence = float(sample_output[true_label])
                    else:
                        confidence = float(np.max(sample_output))

            if timestamp_num in insert_timestamps:
                label = 1
            elif timestamp_num in remove_timestamps:
                label = 2
            else:
                label = 0
            confidences.append(float(confidence))
            labels.append(label)

        if len(confidences) == 0:
            continue

        confidence_stats['min'].append(min(confidences))
        confidence_stats['max'].append(max(confidences))
        confidence_stats['mean'].append(np.mean(confidences))

        all_samples_data.append([confidences, labels])

    if len(all_samples_data) == 0:
        raise ValueError(f"未找到任何样本的数据")

    return all_samples_data

def create_window_data(samples_timestamps_data, window_size=5, skip_boundary=False):
    window_data_list = []
    skipped_class0_boundary = 0

    for sample_idx, (confidences, labels) in enumerate(samples_timestamps_data):
        seq_len = len(confidences)

        for timestamp_idx in range(seq_len):
            label = labels[timestamp_idx]

            if skip_boundary and label == 0:
                if timestamp_idx < window_size or timestamp_idx >= (seq_len - window_size):
                    skipped_class0_boundary += 1
                    continue

            actual_start = max(0, timestamp_idx - window_size)
            actual_end = min(seq_len, timestamp_idx + window_size + 1)

            actual_sequence = confidences[actual_start:actual_end]

            needed_left = max(0, window_size - timestamp_idx)
            needed_right = max(0, (timestamp_idx + window_size + 1) - seq_len)

            if needed_left > 0:
                padding_left = [confidences[0]] * needed_left
                actual_sequence = padding_left + actual_sequence

            if needed_right > 0:
                padding_right = [confidences[-1]] * needed_right
                actual_sequence = actual_sequence + padding_right

            window_sequence = actual_sequence
            assert len(window_sequence) == 2 * window_size + 1, \
                f"窗口大小不正确: 期望 {2 * window_size + 1}, 实际 {len(window_sequence)}, " \
                f"timestamp_idx={timestamp_idx}, seq_len={seq_len}, needed_left={needed_left}, needed_right={needed_right}"

            window_data_list.append({
                'window_sequence': window_sequence,
                'label': label,
                'sample_index': sample_idx,
                'timestamp_index': timestamp_idx
            })

    return window_data_list


def compute_window_normalization_params(window_data):
    all_values = []
    for item in window_data:
        window_sequence = item['window_sequence']
        all_values.extend(window_sequence)

    if len(all_values) == 0:
        return None, None

    mean = np.mean(all_values)
    std = np.std(all_values)

    if std == 0:
        return None, None

    return mean, std


def normalize_window_data(window_data, mean, std):
    if mean is None or std is None:
        return window_data

    normalized_window_data = []
    for item in window_data:
        normalized_item = item.copy()
        window_sequence = item['window_sequence']
        normalized_sequence = [(x - mean) / std for x in window_sequence]
        normalized_item['window_sequence'] = normalized_sequence
        normalized_window_data.append(normalized_item)

    return normalized_window_data


def filter_outlier_windows(window_data, outlier_filter=True, z_threshold=3.0):
    if not outlier_filter or len(window_data) == 0:
        return window_data, []

    window_features = []

    for window_item in window_data:
        window_sequence = window_item['window_sequence']
        window_mean = np.mean(window_sequence)
        window_features.append(window_mean)

    window_features = np.array(window_features)

    feature_mean = np.mean(window_features)
    feature_std = np.std(window_features)

    outlier_window_indices = []
    if feature_std > 1e-10:
        for idx, feature_value in enumerate(window_features):
            z_score = abs((feature_value - feature_mean) / feature_std)
            if z_score > z_threshold:
                outlier_window_indices.append(idx)

    filtered_window_data = [window_data[i] for i in range(len(window_data)) if i not in outlier_window_indices]

    return filtered_window_data, outlier_window_indices
