#!/usr/bin/env python3
"""
Query Length Experiment 结果分析脚本示例

这个脚本展示如何加载和分析实验结果
"""

import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path


def load_results(output_dir: str, nprobe: int, llm_model: str):
    """加载实验结果"""
    output_path = Path(output_dir)

    csv_file = output_path / f"results_qlength_{nprobe}_{llm_model}.csv"
    json_file = output_path / f"detailed_log_qlength_{nprobe}_{llm_model}.json"

    # 加载 CSV
    df = pd.read_csv(csv_file)

    # 加载 JSON
    with open(json_file) as f:
        config_and_results = json.load(f)

    return df, config_and_results


def print_summary(df, config_and_results):
    """打印摘要统计"""
    print("=" * 60)
    print("Query Length Experiment Results Summary")
    print("=" * 60)
    print()

    config = config_and_results["experiment_config"]
    print(f"Dataset: {config['dataset']}")
    print(f"LLM Model: {config['llm_model']}")
    print(f"nprobe: {config['nprobe']}")
    print(f"Batch Size: {config['batch_size']}")
    print()

    print("Results by Query Length Category:")
    print("-" * 60)
    print(df.to_string(index=False))
    print()

    # 计算增长率
    short_latency = df[df["category"] == "short"]["total_time_ms"].values[0]
    long_latency = df[df["category"] == "long"]["total_time_ms"].values[0]
    growth = (long_latency - short_latency) / short_latency * 100

    short_throughput = df[df["category"] == "short"]["throughput_qps"].values[0]
    long_throughput = df[df["category"] == "long"]["throughput_qps"].values[0]
    throughput_drop = (short_throughput - long_throughput) / short_throughput * 100

    print("Key Insights:")
    print(f"  - Latency increase (short → long): {growth:.1f}%")
    print(f"  - Throughput drop (short → long): {throughput_drop:.1f}%")
    print()


def compare_multiple_results(output_dirs: list, nprobe: int, llm_model: str):
    """对比多个实验结果（不同 nprobe）"""
    print("=" * 80)
    print(f"Comparing Query Length Results across Different nprobe Values")
    print(f"LLM Model: {llm_model}")
    print("=" * 80)
    print()

    all_results = {}

    for output_dir in output_dirs:
        df, config = load_results(output_dir, nprobe, llm_model)
        nprobe_val = config["experiment_config"]["nprobe"]
        all_results[nprobe_val] = df

    # 创建对比表
    for category in ["short", "medium", "long"]:
        print(f"\n{category.upper()} Queries:")
        print("-" * 80)

        comparison_data = []
        for nprobe_val, df in sorted(all_results.items()):
            row = df[df["category"] == category].iloc[0]
            comparison_data.append({
                "nprobe": nprobe_val,
                "embedding_ms": row["avg_embedding_time_ms"],
                "retrieval_ms": row["avg_retrieval_time_ms"],
                "generation_ms": row["avg_generation_time_ms"],
                "total_ms": row["total_time_ms"],
                "throughput_qps": row["throughput_qps"],
            })

        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))


def generate_comparison_plot(csv_file: str, output_path: str = "comparison.png"):
    """生成对比可视化"""
    df = pd.read_csv(csv_file)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 总延迟对比
    ax = axes[0, 0]
    bars = ax.bar(df["category"], df["total_time_ms"], color=["green", "orange", "red"])
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Total Latency by Query Length Category")
    ax.grid(True, alpha=0.3, axis="y")
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom')

    # 2. 吞吐量对比
    ax = axes[0, 1]
    bars = ax.bar(df["category"], df["throughput_qps"], color=["green", "orange", "red"])
    ax.set_ylabel("Throughput (QPS)")
    ax.set_title("Throughput by Query Length Category")
    ax.grid(True, alpha=0.3, axis="y")
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')

    # 3. 各阶段时间对比
    ax = axes[1, 0]
    x = range(len(df))
    width = 0.25
    ax.bar([i - width for i in x], df["avg_embedding_time_ms"], width, label="Embedding", alpha=0.8)
    ax.bar([i for i in x], df["avg_retrieval_time_ms"], width, label="Retrieval", alpha=0.8)
    ax.bar([i + width for i in x], df["avg_generation_time_ms"], width, label="Generation", alpha=0.8)
    ax.set_ylabel("Time (ms)")
    ax.set_title("Stage-wise Latency Breakdown")
    ax.set_xticks(x)
    ax.set_xticklabels(df["category"])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # 4. 平均 token 长度
    ax = axes[1, 1]
    bars = ax.bar(df["category"], df["avg_token_len"], color=["blue", "purple", "maroon"])
    ax.set_ylabel("Average Token Count")
    ax.set_title("Average Query Length by Category")
    ax.grid(True, alpha=0.3, axis="y")
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {output_path}")
    plt.close()


def export_markdown_report(df, config_and_results, output_file: str = "report.md"):
    """生成 Markdown 格式的报告"""
    config = config_and_results["experiment_config"]

    with open(output_file, "w") as f:
        f.write("# Query Length Impact Analysis Report\n\n")

        f.write("## Experiment Configuration\n\n")
        f.write(f"- **Dataset**: {config['dataset']} ({config['dataset_split']})\n")
        f.write(f"- **LLM Model**: {config['llm_model']}\n")
        f.write(f"- **nprobe**: {config['nprobe']}\n")
        f.write(f"- **Batch Size**: {config['batch_size']}\n")
        f.write(f"- **Sample Size per Category**: {config['sample_size_per_category']}\n\n")

        f.write("## Category Definitions\n\n")
        cats = config['categories_config']
        f.write(f"- **Short**: token count ≤ {cats['short'].get('max_tokens', 'N/A')}\n")
        f.write(f"- **Medium**: {cats['medium'].get('min_tokens', 'N/A')} ≤ token count ≤ {cats['medium'].get('max_tokens', 'N/A')}\n")
        f.write(f"- **Long**: token count > {cats['long'].get('min_tokens', 'N/A')}\n\n")

        f.write("## Results\n\n")

        f.write("### Summary Table\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n")

        f.write("### Key Findings\n\n")
        short_row = df[df["category"] == "short"].iloc[0]
        long_row = df[df["category"] == "long"].iloc[0]

        latency_increase = (long_row["total_time_ms"] - short_row["total_time_ms"]) / short_row["total_time_ms"] * 100
        throughput_decrease = (short_row["throughput_qps"] - long_row["throughput_qps"]) / short_row["throughput_qps"] * 100
        embedding_increase = (long_row["avg_embedding_time_ms"] - short_row["avg_embedding_time_ms"]) / short_row["avg_embedding_time_ms"] * 100

        f.write(f"1. **Total Latency**: {latency_increase:.1f}% increase from short to long queries\n")
        f.write(f"   - Short: {short_row['total_time_ms']:.1f} ms\n")
        f.write(f"   - Long: {long_row['total_time_ms']:.1f} ms\n\n")

        f.write(f"2. **Throughput**: {throughput_decrease:.1f}% decrease from short to long queries\n")
        f.write(f"   - Short: {short_row['throughput_qps']:.4f} QPS\n")
        f.write(f"   - Long: {long_row['throughput_qps']:.4f} QPS\n\n")

        f.write(f"3. **Embedding Stage**: {embedding_increase:.1f}% increase\n")
        f.write(f"   - Short: {short_row['avg_embedding_time_ms']:.1f} ms\n")
        f.write(f"   - Long: {long_row['avg_embedding_time_ms']:.1f} ms\n")
        f.write(f"   - *Note: 随 token 数增加而增加（预期表现）*\n\n")

        f.write("4. **Retrieval Stage**: 基本保持稳定\n")
        retrieval_change = (long_row["avg_retrieval_time_ms"] - short_row["avg_retrieval_time_ms"]) / short_row["avg_retrieval_time_ms"] * 100
        f.write(f"   - Change: {retrieval_change:.1f}%\n\n")

        f.write("5. **Generation Stage**: 基本保持稳定\n")
        generation_change = (long_row["avg_generation_time_ms"] - short_row["avg_generation_time_ms"]) / short_row["avg_generation_time_ms"] * 100
        f.write(f"   - Change: {generation_change:.1f}%\n")
        f.write(f"   - *Note: 受 context 长度影响不大*\n\n")

    print(f"Saved markdown report to {output_file}")


def main():
    """示例使用"""
    # 示例 1: 加载和打印单个实验结果
    print("Example 1: Loading and Printing Single Experiment Results")
    print()

    output_dir = "../output/query_length_experiment"
    nprobe = 128
    llm_model = "llama3-8B"

    try:
        df, config_and_results = load_results(output_dir, nprobe, llm_model)
        print_summary(df, config_and_results)
    except FileNotFoundError as e:
        print(f"Result files not found: {e}")
        print("Please run the experiment first:")
        print(f"  python query_length_experiment.py --config query_length_config.yaml --nprobe {nprobe} --llm_model {llm_model}")
        return

    # 示例 2: 生成对比图表
    print("\nExample 2: Generating Comparison Plot")
    csv_file = Path(output_dir) / f"results_qlength_{nprobe}_{llm_model}.csv"
    generate_comparison_plot(str(csv_file), "query_length_comparison.png")

    # 示例 3: 导出 Markdown 报告
    print("\nExample 3: Exporting Markdown Report")
    export_markdown_report(df, config_and_results, "query_length_report.md")

    print("\nAll analyses completed! Check the output files.")


if __name__ == "__main__":
    main()
