#!/usr/bin/env python3
"""
Query Token Length Distribution Analysis

分析数据集中的 query token 长度分布，用来确定最优的长度分类阈值
"""

import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_dataset
from transformers import AutoTokenizer
from exp.batch_scaling_experiment import extract_question, set_random_seed


def load_tokenizer(model_path: str = "intfloat/e5-large-v2"):
    """加载 e5 tokenizer"""
    print(f"Loading tokenizer from {model_path}...")
    return AutoTokenizer.from_pretrained(model_path, use_fast=True)


def compute_token_lengths(
    queries: List[str],
    tokenizer,
    batch_size: int = 512,
    show_progress: bool = True,
) -> np.ndarray:
    """计算所有 query 的 token 长度"""
    lengths = []

    for i, query in enumerate(queries):
        if show_progress and (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1} / {len(queries)} queries...")

        encoded = tokenizer(query, truncation=False, add_special_tokens=True)
        lengths.append(len(encoded["input_ids"]))

    return np.array(lengths)


def print_statistics(lengths: np.ndarray, dataset_name: str = "Dataset"):
    """打印 token 长度的详细统计"""
    print(f"\n{'='*70}")
    print(f"Token Length Statistics for {dataset_name}")
    print(f"{'='*70}\n")

    print(f"Total queries: {len(lengths)}")
    print(f"\nBasic Statistics:")
    print(f"  Min:       {np.min(lengths):6.0f} tokens")
    print(f"  Max:       {np.max(lengths):6.0f} tokens")
    print(f"  Mean:      {np.mean(lengths):6.2f} tokens")
    print(f"  Median:    {np.median(lengths):6.2f} tokens")
    print(f"  Std Dev:   {np.std(lengths):6.2f} tokens")
    print(f"  25th %ile: {np.percentile(lengths, 25):6.2f} tokens")
    print(f"  75th %ile: {np.percentile(lengths, 75):6.2f} tokens")
    print(f"  95th %ile: {np.percentile(lengths, 95):6.2f} tokens")
    print(f"  99th %ile: {np.percentile(lengths, 99):6.2f} tokens")

    print(f"\nPercentile Distribution:")
    percentiles = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(lengths, p)
        print(f"  {p:3d}%: {val:6.1f} tokens")


def suggest_5_level_thresholds(lengths: np.ndarray) -> Dict[str, Tuple[int, int]]:
    """
    基于分布建议 5 个梯度的分类阈值
    返回 {"very_short": (min, max), "short": (min, max), ...}
    """
    # 使用百分位数作为参考
    p20 = np.percentile(lengths, 20)
    p40 = np.percentile(lengths, 40)
    p60 = np.percentile(lengths, 60)
    p80 = np.percentile(lengths, 80)

    # 向上/向下舍入到最近的 5
    def round_threshold(x, direction='up'):
        if direction == 'up':
            return int(np.ceil(x / 5) * 5)
        else:
            return int(np.floor(x / 5) * 5)

    very_short_max = round_threshold(p20, 'up')
    short_max = round_threshold(p40, 'up')
    medium_max = round_threshold(p60, 'up')
    long_max = round_threshold(p80, 'up')
    very_long_max = int(np.max(lengths)) + 1

    return {
        "very_short": (0, very_short_max),
        "short": (very_short_max + 1, short_max),
        "medium": (short_max + 1, medium_max),
        "long": (medium_max + 1, long_max),
        "very_long": (long_max + 1, very_long_max),
    }


def suggest_5_level_thresholds_balanced(lengths: np.ndarray) -> Dict[str, Tuple[int, int]]:
    """
    基于分布建议 5 个梯度的分类阈值（更均衡的版本）
    尽量让各类的 query 数量相近
    """
    # 使用百分位数来均衡分布
    p20 = np.percentile(lengths, 20)
    p40 = np.percentile(lengths, 40)
    p60 = np.percentile(lengths, 60)
    p80 = np.percentile(lengths, 80)

    # 向上舍入到最近的整数
    very_short_max = int(np.ceil(p20))
    short_max = int(np.ceil(p40))
    medium_max = int(np.ceil(p60))
    long_max = int(np.ceil(p80))
    very_long_max = int(np.max(lengths)) + 1

    return {
        "very_short": (0, very_short_max),
        "short": (very_short_max + 1, short_max),
        "medium": (short_max + 1, medium_max),
        "long": (medium_max + 1, long_max),
        "very_long": (long_max + 1, very_long_max),
    }


def validate_and_print_thresholds(
    lengths: np.ndarray,
    thresholds: Dict[str, Tuple[int, int]]
) -> None:
    """验证并打印分类结果"""
    print(f"\n{'='*70}")
    print("Suggested 5-Level Classification:")
    print(f"{'='*70}\n")

    results = {}
    for category, (min_len, max_len) in thresholds.items():
        count = np.sum((lengths >= min_len) & (lengths <= max_len))
        pct = 100.0 * count / len(lengths)
        avg_len = np.mean(lengths[(lengths >= min_len) & (lengths <= max_len)]) if count > 0 else 0

        results[category] = {
            "range": f"[{min_len}, {max_len}]",
            "count": count,
            "percentage": pct,
            "avg": avg_len,
        }

        print(f"{category:12s}: tokens {min_len:4d}-{max_len:4d} | "
              f"count: {count:6d} ({pct:5.1f}%) | "
              f"avg: {avg_len:6.2f}")

    return results


def print_histogram_text(
    lengths: np.ndarray,
    thresholds: Dict[str, Tuple[int, int]],
    bins: int = 50
) -> None:
    """打印文本直方图"""
    print(f"\n{'='*70}")
    print("Token Length Distribution (Text Histogram):")
    print(f"{'='*70}\n")

    hist, edges = np.histogram(lengths, bins=bins)
    max_hist = np.max(hist)

    for i, count in enumerate(hist):
        bar_len = int(50 * count / max_hist)
        bar = "█" * bar_len
        edge_start = int(edges[i])
        edge_end = int(edges[i + 1])
        pct = 100.0 * count / len(lengths)
        print(f"{edge_start:4d}-{edge_end:4d} | {bar:50s} | {count:5d} ({pct:5.1f}%)")


def generate_distribution_plot(
    lengths: np.ndarray,
    thresholds: Dict[str, Tuple[int, int]],
    output_file: str = "token_distribution.png"
) -> None:
    """生成分布可视化图表"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 直方图
    ax = axes[0, 0]
    ax.hist(lengths, bins=100, edgecolor='black', alpha=0.7)
    for category, (min_len, max_len) in thresholds.items():
        ax.axvline(min_len, color='red', linestyle='--', alpha=0.5)
    ax.axvline(np.max(lengths), color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel("Token Count")
    ax.set_ylabel("Frequency")
    ax.set_title("Token Length Distribution (Histogram)")
    ax.grid(True, alpha=0.3)

    # 2. 累积分布函数
    ax = axes[0, 1]
    sorted_lengths = np.sort(lengths)
    cumulative = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths) * 100
    ax.plot(sorted_lengths, cumulative, linewidth=2)
    for category, (min_len, max_len) in thresholds.items():
        ax.axvline(min_len, color='red', linestyle='--', alpha=0.5, label=category)
    ax.set_xlabel("Token Count")
    ax.set_ylabel("Cumulative Percentage (%)")
    ax.set_title("CDF: Token Length Distribution")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # 3. 箱线图
    ax = axes[1, 0]
    box_data = []
    labels = []
    for category, (min_len, max_len) in thresholds.items():
        mask = (lengths >= min_len) & (lengths <= max_len)
        box_data.append(lengths[mask])
        labels.append(category)
    ax.boxplot(box_data, labels=labels)
    ax.set_ylabel("Token Count")
    ax.set_title("Token Length Distribution by Category (Box Plot)")
    ax.grid(True, alpha=0.3, axis='y')

    # 4. 各类别的统计柱状图
    ax = axes[1, 1]
    categories = list(thresholds.keys())
    means = []
    counts = []
    for category, (min_len, max_len) in thresholds.items():
        mask = (lengths >= min_len) & (lengths <= max_len)
        if np.sum(mask) > 0:
            means.append(np.mean(lengths[mask]))
            counts.append(np.sum(mask))
        else:
            means.append(0)
            counts.append(0)

    x = np.arange(len(categories))
    width = 0.35

    ax2 = ax.twinx()
    bars1 = ax.bar(x - width/2, means, width, label="Avg Tokens", alpha=0.8)
    bars2 = ax2.bar(x + width/2, counts, width, label="Count", alpha=0.8, color='orange')

    ax.set_xlabel("Category")
    ax.set_ylabel("Avg Token Count", color='blue')
    ax2.set_ylabel("Query Count", color='orange')
    ax.set_title("Category-wise Statistics")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='orange')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved distribution plot to {output_file}")
    plt.close()


def analyze_dataset(
    dataset_name: str,
    split: str = "validation",
    source: str = "huggingface",
    sample_size: int = None,
    random_seed: int = 2026,
):
    """完整的数据集分析流程"""
    print(f"\n{'='*70}")
    print(f"Analyzing {dataset_name} ({split} split)")
    print(f"{'='*70}\n")

    # 1. 加载数据集
    print(f"Loading {dataset_name} from {source}...")
    if source == "huggingface":
        dataset = load_dataset(dataset_name, split=split)
    else:
        raise NotImplementedError(f"Source {source} not supported")

    # 2. 提取 query
    print("Extracting questions...")
    queries = [extract_question(record) for record in dataset]
    queries = [q for q in queries if q]

    print(f"Extracted {len(queries)} valid queries")

    # 3. 采样（如果指定）
    if sample_size is not None and sample_size < len(queries):
        print(f"Sampling {sample_size} queries...")
        set_random_seed(random_seed)
        rng = np.random.RandomState(random_seed)
        indices = rng.choice(len(queries), sample_size, replace=False)
        queries = [queries[i] for i in sorted(indices.tolist())]

    # 4. 计算 token 长度
    print("Computing token lengths...")
    tokenizer = load_tokenizer()
    lengths = compute_token_lengths(queries, tokenizer)

    # 5. 打印统计信息
    print_statistics(lengths, f"{dataset_name}:{split}")

    # 6. 建议分类阈值
    thresholds_percentile = suggest_5_level_thresholds(lengths)
    thresholds_balanced = suggest_5_level_thresholds_balanced(lengths)

    print(f"\n{'='*70}")
    print("Method 1: Percentile-based (rounded to 5s)")
    print(f"{'='*70}")
    results1 = validate_and_print_thresholds(lengths, thresholds_percentile)

    print(f"\n{'='*70}")
    print("Method 2: Balanced percentile (more balanced distribution)")
    print(f"{'='*70}")
    results2 = validate_and_print_thresholds(lengths, thresholds_balanced)

    # 7. 文本直方图
    print_histogram_text(lengths, thresholds_balanced, bins=40)

    # 8. 生成可视化
    print("\nGenerating visualization...")
    safe_dataset = re.sub(r"[^a-zA-Z0-9._-]+", "_", dataset_name).strip("_")
    safe_split = re.sub(r"[^a-zA-Z0-9._-]+", "_", split).strip("_")
    output_file = f"token_distribution_{safe_dataset}_{safe_split}.png"
    generate_distribution_plot(lengths, thresholds_balanced, output_file=output_file)

    # 9. 导出配置建议
    print(f"\n{'='*70}")
    print("YAML Configuration Suggestion (5-level)")
    print(f"{'='*70}\n")

    print("query_length_analysis:")
    print("  categories:")
    for category, (min_len, max_len) in thresholds_balanced.items():
        if min_len == 0:
            print(f"    {category}:")
            print(f"      max_tokens: {max_len}")
        elif max_len > 100000:  # 最后一个
            print(f"    {category}:")
            print(f"      min_tokens: {min_len}")
        else:
            print(f"    {category}:")
            print(f"      min_tokens: {min_len}")
            print(f"      max_tokens: {max_len}")

    return lengths, thresholds_balanced, results2


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze token length distribution in a dataset")
    parser.add_argument("--dataset", type=str, default="natural_questions",
                       help="Dataset name (default: natural_questions)")
    parser.add_argument("--split", type=str, default="train",
                       help="Dataset split (default: train)")
    parser.add_argument("--sample_size", type=int, default=None,
                       help="Sample size (default: use all queries)")
    parser.add_argument("--seed", type=int, default=2026,
                       help="Random seed (default: 2026)")

    args = parser.parse_args()

    lengths, thresholds, results = analyze_dataset(
        dataset_name=args.dataset,
        split=args.split,
        sample_size=args.sample_size,
        random_seed=args.seed,
    )

    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()
