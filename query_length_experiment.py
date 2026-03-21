#!/usr/bin/env python3
"""
Query Length Analysis Experiment

研究不同长度的 query 对 RAG 系统各个阶段（embedding/retrieval/generation）的影响。
- 固定 batch_size
- 按 e5 tokenizer token 数分为 short/medium/long 三类
- 各类采样 200 条 query
- 对比各阶段平均时间和吞吐量
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

# 复用 batch_scaling_experiment 中的工具和类
from batch_scaling_experiment import (
    QueryEmbeddingStage,
    RetrievalStage,
    GenerationStage,
    BatchRecord,
    TimingResult,
    extract_question,
    build_prompt,
    synchronize_if_needed,
    set_random_seed,
    load_yaml_config,
    resolve_path,
    validate_local_file,
    filter_batch_sizes,
    split_queries_for_warmup,
)


@dataclass
class CategoryTimingResult:
    """某个 query 长度类别的统计结果"""
    category_name: str  # "short", "medium", "long"
    num_queries: int
    avg_token_len: float  # 该类别 query 的平均 token 数
    embedding_total_sec: float
    retrieval_total_sec: float
    generation_total_sec: float

    embedding_avg_ms: float = field(init=False)
    retrieval_avg_ms: float = field(init=False)
    generation_avg_ms: float = field(init=False)
    total_sec: float = field(init=False)
    total_ms: float = field(init=False)
    throughput_qps: float = field(init=False)

    def __post_init__(self) -> None:
        if self.num_queries <= 0:
            raise ValueError("num_queries must be positive.")

        self.embedding_avg_ms = (self.embedding_total_sec * 1000.0) / self.num_queries
        self.retrieval_avg_ms = (self.retrieval_total_sec * 1000.0) / self.num_queries
        self.generation_avg_ms = (self.generation_total_sec * 1000.0) / self.num_queries
        self.total_sec = (
            self.embedding_total_sec + self.retrieval_total_sec + self.generation_total_sec
        )
        self.total_ms = self.total_sec * 1000.0
        self.throughput_qps = self.num_queries / self.total_sec if self.total_sec > 0 else math.inf


def compute_token_lengths(
    queries: List[str],
    tokenizer: AutoTokenizer,
) -> List[int]:
    """
    计算每条 query 的 token 数（使用 e5 tokenizer）

    Args:
        queries: query 列表
        tokenizer: e5 tokenizer 对象

    Returns:
        token 长度列表，对应每条 query
    """
    token_lengths = []
    for query in queries:
        # 不 padding，直接统计 token 数
        encoded = tokenizer(query, truncation=False, add_special_tokens=True)
        token_lengths.append(len(encoded["input_ids"]))
    return token_lengths


def print_length_distribution(queries: List[str], lengths: List[int], logger: logging.Logger) -> None:
    """打印 token 长度分布统计"""
    logger.info("=== Query Token Length Distribution ===")
    logger.info(f"Total queries: {len(queries)}")
    logger.info(f"Min tokens: {min(lengths)}, Max tokens: {max(lengths)}, Mean tokens: {np.mean(lengths):.1f}")

    # 分段统计
    ranges = [(0, 5), (6, 10), (11, 15), (16, 20), (21, 30), (31, 100), (101, 1000)]
    for start, end in ranges:
        count = sum(1 for l in lengths if start <= l <= end)
        pct = 100.0 * count / len(lengths) if len(lengths) > 0 else 0
        logger.info(f"  [{start:3d}, {end:3d}]: {count:4d} queries ({pct:5.1f}%)")


def bin_queries_by_length(
    queries: List[str],
    lengths: List[int],
    config: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, Tuple[List[str], List[int]]]:
    """
    按 token 长度分类 query

    Args:
        queries: query 列表
        lengths: 对应的 token 长度列表
        config: 配置 dict，包含 query_length_analysis.categories
        logger: logger

    Returns:
        {"short": (queries, lengths), "medium": ..., "long": ...}
    """
    categories_config = config["query_length_analysis"]["categories"]

    result = {}
    for category in ["short", "medium", "long"]:
        cat_config = categories_config[category]
        min_tokens = cat_config.get("min_tokens", 0)
        max_tokens = cat_config.get("max_tokens", float("inf"))

        # 筛选落在该范围的 query
        selected_indices = [
            i for i, l in enumerate(lengths)
            if min_tokens <= l <= max_tokens
        ]
        cat_queries = [queries[i] for i in selected_indices]
        cat_lengths = [lengths[i] for i in selected_indices]

        result[category] = (cat_queries, cat_lengths)
        logger.info(
            f"Category '{category}': {len(cat_queries)} queries, "
            f"token range [{min_tokens}, {max_tokens}], "
            f"mean tokens: {np.mean(cat_lengths) if cat_lengths else 0:.1f}"
        )

    return result


def sample_category(
    queries: List[str],
    lengths: List[int],
    sample_size: int,
    random_seed: int,
) -> Tuple[List[str], List[int]]:
    """
    从某个类别中随机采样

    Args:
        queries: 该类别的所有 query
        lengths: 对应的 token 长度
        sample_size: 采样数
        random_seed: 随机种子

    Returns:
        (采样后的 queries, 对应的 lengths)
    """
    if len(queries) <= sample_size:
        return queries, lengths

    rng = np.random.RandomState(random_seed)
    indices = rng.choice(len(queries), sample_size, replace=False)
    sampled_queries = [queries[i] for i in sorted(indices.tolist())]
    sampled_lengths = [lengths[i] for i in sorted(indices.tolist())]

    return sampled_queries, sampled_lengths


class QueryLengthExperiment:
    """Query 长度分析实验"""

    def __init__(
        self,
        config_path: Path,
        nprobe: int,
        llm_model: str,
        output_dir: Optional[Path],
        index_path_override: Optional[Path],
        test_mode: bool,
    ) -> None:
        config_path = config_path.resolve()
        self.config = load_yaml_config(config_path)

        # 解析和验证路径
        configured_index_path = self.config["retrieval"]["index_path"]
        if index_path_override is not None:
            configured_index_path = str(index_path_override.resolve())
        resolved_index_path = resolve_path(configured_index_path, config_path.parent)
        validate_local_file(resolved_index_path, "FAISS index", config_path)
        self.config["retrieval"]["index_path"] = str(resolved_index_path)

        self.nprobe = nprobe
        self.llm_model = llm_model
        self.test_mode = test_mode
        self.config["debug"]["test_mode"] = test_mode

        gpu_id = self.config["system"].get("gpu_id")
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        self.output_dir = output_dir
        if self.output_dir is None:
            configured = Path(self.config["output"]["output_dir"])
            if configured.is_absolute():
                self.output_dir = configured
            else:
                self.output_dir = (config_path.parent / configured).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = self._build_logger()
        self.logger.info("Experiment config path: %s", config_path)
        self.logger.info("nprobe=%s llm_model=%s test_mode=%s", nprobe, llm_model, test_mode)

        set_random_seed(self.config["dataset"]["random_seed"])
        torch.set_num_threads(self.config["system"]["num_workers"])

        # 初始化各阶段
        self.embedding_stage = QueryEmbeddingStage(self.config, self.logger)
        self.retrieval_stage = RetrievalStage(self.config, self.logger)
        self.retrieval_stage.set_nprobe(nprobe)
        self.generation_stage = GenerationStage(self.config, llm_model, self.logger)

        # 加载并预处理 query
        all_queries = self._load_queries()
        self.warmup_queries, self.eval_queries = split_queries_for_warmup(
            all_queries,
            self.config["timing"],
        )

        self.logger.info(
            "Warmup queries: %s | Evaluation queries: %s",
            len(self.warmup_queries),
            len(self.eval_queries),
        )

        # 按长度分类 query（仅用 eval_queries）
        self.length_categories = self._bin_and_sample_queries()
        self.results: List[CategoryTimingResult] = []

    def _build_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"query_length.{self.nprobe}.{self.llm_model}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.handlers.clear()

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        file_handler = logging.FileHandler(
            self.output_dir / f"experiment_log_qlength_{self.nprobe}_{self.llm_model}.log",
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        return logger

    def _load_queries(self) -> List[str]:
        dataset_config = self.config["dataset"]
        dataset_name = dataset_config["name"]
        split = dataset_config["split"]
        self.logger.info("Loading dataset: %s split=%s", dataset_name, split)

        if dataset_config["source"] != "huggingface":
            raise NotImplementedError("Only HuggingFace datasets are supported.")

        dataset = load_dataset(dataset_name, split=split)
        queries = [extract_question(record) for record in dataset]
        queries = [query for query in queries if query]

        if not queries:
            raise ValueError(f"No questions could be extracted from dataset {dataset_name}.")

        sample_queries = dataset_config.get("sample_queries")
        if self.test_mode:
            sample_queries = self.config["debug"]["test_sample_size"]

        if sample_queries is not None and sample_queries < len(queries):
            rng = np.random.RandomState(dataset_config["random_seed"])
            indices = rng.choice(len(queries), sample_queries, replace=False)
            queries = [queries[index] for index in sorted(indices.tolist())]

        self.logger.info("Loaded %s queries", len(queries))
        return queries

    def _bin_and_sample_queries(self) -> Dict[str, Tuple[List[str], List[int]]]:
        """按长度分类并采样 query"""
        self.logger.info("Computing token lengths for all queries...")
        lengths = compute_token_lengths(self.eval_queries, self.embedding_stage.tokenizer)

        if self.config["query_length_analysis"].get("print_distribution", False):
            print_length_distribution(self.eval_queries, lengths, self.logger)

        # 分类
        self.logger.info("Binning queries by length...")
        binned = bin_queries_by_length(self.eval_queries, lengths, self.config, self.logger)

        # 采样
        sample_size = self.config["query_length_analysis"]["sample_size"]
        random_seed = self.config["dataset"]["random_seed"]

        sampled = {}
        for category in ["short", "medium", "long"]:
            cat_queries, cat_lengths = binned[category]
            sampled_queries, sampled_lengths = sample_category(
                cat_queries,
                cat_lengths,
                sample_size,
                random_seed + ord(category[0]),  # 不同类别不同随机种子
            )
            sampled[category] = (sampled_queries, sampled_lengths)
            self.logger.info(
                f"Sampled {len(sampled_queries)} queries for category '{category}'"
            )

        return sampled

    def run_category_experiment(self, category_name: str, queries: List[str], lengths: List[int]) -> CategoryTimingResult:
        """
        对某一类 query 执行实验

        Args:
            category_name: "short", "medium", "long"
            queries: 该类别的所有 query
            lengths: 对应的 token 长度列表

        Returns:
            CategoryTimingResult
        """
        batch_size = self.config["query_length_analysis"]["batch_size"]

        self.logger.info(
            "Running category '%s' with %s queries, batch_size=%s",
            category_name,
            len(queries),
            batch_size,
        )

        total_embedding_sec = 0.0
        total_retrieval_sec = 0.0
        total_generation_sec = 0.0
        effective_queries = 0

        # Warmup（可选）
        if self.warmup_queries:
            self.logger.info("Running warmup for category '%s'...", category_name)
            for start_idx in range(0, len(self.warmup_queries), batch_size):
                warmup_batch = self.warmup_queries[start_idx : start_idx + batch_size]
                embeddings, _ = self.embedding_stage(warmup_batch)
                retrieved_docs, _ = self.retrieval_stage(embeddings)
                self.generation_stage(warmup_batch, retrieved_docs)

        # 评估
        for batch_index, start_idx in enumerate(range(0, len(queries), batch_size), start=1):
            batch_queries = queries[start_idx : start_idx + batch_size]

            embeddings, embedding_sec = self.embedding_stage(batch_queries)
            retrieved_docs, retrieval_sec = self.retrieval_stage(embeddings)
            _, generation_sec = self.generation_stage(batch_queries, retrieved_docs)

            total_embedding_sec += embedding_sec
            total_retrieval_sec += retrieval_sec
            total_generation_sec += generation_sec
            effective_queries += len(batch_queries)

            if batch_index % self.config["debug"].get("print_interval", 10) == 0:
                self.logger.info(f"  batch {batch_index}: {len(batch_queries)} queries processed")

        avg_token_len = np.mean(lengths) if lengths else 0.0

        result = CategoryTimingResult(
            category_name=category_name,
            num_queries=effective_queries,
            avg_token_len=avg_token_len,
            embedding_total_sec=total_embedding_sec,
            retrieval_total_sec=total_retrieval_sec,
            generation_total_sec=total_generation_sec,
        )

        self.logger.info(
            "Category '%s': embedding=%.3fms, retrieval=%.3fms, generation=%.3fms, "
            "throughput=%.3fqps, avg_tokens=%.1f",
            category_name,
            result.embedding_avg_ms,
            result.retrieval_avg_ms,
            result.generation_avg_ms,
            result.throughput_qps,
            avg_token_len,
        )

        return result

    def run(self) -> None:
        """对所有 query 长度类别执行实验"""
        # 自动获取所有类别（支持 3 级、5 级或更多）
        categories = sorted(self.length_categories.keys())

        for category in categories:
            queries, lengths = self.length_categories[category]
            result = self.run_category_experiment(category, queries, lengths)
            self.results.append(result)

    def save_results(self) -> None:
        """保存结果到 CSV 和 JSON"""
        rows = [
            {
                "llm_model": self.llm_model,
                "nprobe": self.nprobe,
                "category": result.category_name,
                "num_queries": result.num_queries,
                "avg_token_len": round(result.avg_token_len, 2),
                "avg_embedding_time_ms": round(result.embedding_avg_ms, 4),
                "avg_retrieval_time_ms": round(result.retrieval_avg_ms, 4),
                "avg_generation_time_ms": round(result.generation_avg_ms, 4),
                "total_time_ms": round(result.total_ms, 4),
                "throughput_qps": round(result.throughput_qps, 6),
            }
            for result in self.results
        ]

        csv_path = self.output_dir / self.config["output"]["csv_filename"].format(
            nprobe=self.nprobe,
            llm_model=self.llm_model,
        )
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        self.logger.info("Saved CSV to %s", csv_path)

        json_path = self.output_dir / self.config["output"]["json_filename"].format(
            nprobe=self.nprobe,
            llm_model=self.llm_model,
        )
        payload = {
            "experiment_config": {
                "dataset": self.config["dataset"]["name"],
                "dataset_split": self.config["dataset"]["split"],
                "nprobe": self.nprobe,
                "llm_model": self.llm_model,
                "batch_size": self.config["query_length_analysis"]["batch_size"],
                "sample_size_per_category": self.config["query_length_analysis"]["sample_size"],
                "categories_config": self.config["query_length_analysis"]["categories"],
            },
            "results": [asdict(result) for result in self.results],
        }
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        self.logger.info("Saved JSON to %s", json_path)

    def generate_plots(self) -> None:
        """生成对比柱状图"""
        if not self.config["output"].get("generate_plots", False):
            return

        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        categories = [result.category_name for result in self.results]
        embedding_times = [result.embedding_avg_ms for result in self.results]
        retrieval_times = [result.retrieval_avg_ms for result in self.results]
        generation_times = [result.generation_avg_ms for result in self.results]

        x = np.arange(len(categories))
        width = 0.25

        plt.figure(figsize=(10, 6))
        plt.bar(x - width, embedding_times, width, label="Embedding")
        plt.bar(x, retrieval_times, width, label="Retrieval")
        plt.bar(x + width, generation_times, width, label="Generation")

        plt.xlabel("Query Length Category")
        plt.ylabel("Average Time per Query (ms)")
        plt.title(f"Latency by Query Length | nprobe={self.nprobe} | llm={self.llm_model}")
        plt.xticks(x, categories)
        plt.legend()
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()

        plot_format = self.config["output"].get("plot_format", "png")
        dpi = self.config["output"].get("plot_dpi", 300)
        stem = f"{self.nprobe}_{self.llm_model}"

        plt.savefig(plots_dir / f"latency_by_query_length_{stem}.{plot_format}", dpi=dpi)
        plt.close()

        # 吞吐量对比图
        throughputs = [result.throughput_qps for result in self.results]

        plt.figure(figsize=(10, 6))
        plt.bar(categories, throughputs, color=["green", "orange", "red"])
        plt.xlabel("Query Length Category")
        plt.ylabel("Throughput (queries/sec)")
        plt.title(f"Throughput by Query Length | nprobe={self.nprobe} | llm={self.llm_model}")
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()

        plt.savefig(plots_dir / f"throughput_by_query_length_{stem}.{plot_format}", dpi=dpi)
        plt.close()

        self.logger.info("Saved plots to %s", plots_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query length impact analysis for one-shot RAG."
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to the YAML config file.")
    parser.add_argument("--nprobe", type=int, required=True, help="FAISS nprobe value.")
    parser.add_argument("--llm_model", type=str, required=True, help="Generator model name from config.")
    parser.add_argument(
        "--index_path",
        type=Path,
        default=None,
        help="Override retrieval.index_path for the FAISS index file.",
    )
    parser.add_argument("--output_dir", type=Path, default=None, help="Directory for CSV, JSON and plots.")
    parser.add_argument("--test_mode", action="store_true", help="Run in test mode with smaller dataset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment = QueryLengthExperiment(
        config_path=args.config,
        nprobe=args.nprobe,
        llm_model=args.llm_model,
        output_dir=args.output_dir,
        index_path_override=args.index_path,
        test_mode=args.test_mode,
    )
    experiment.run()
    experiment.save_results()
    experiment.generate_plots()
    experiment.logger.info("Experiment completed successfully!")


if __name__ == "__main__":
    main()
