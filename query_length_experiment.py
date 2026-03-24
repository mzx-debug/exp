#!/usr/bin/env python3
"""
Query length impact analysis for one-shot RAG.

Supports two modes:
1) natural_binning: bin natural-questions queries by token length.
2) controlled_dataset: read pre-generated buckets (e.g., t8/t16/.../t256).
"""

import argparse
import inspect
import json
import logging
import math
import os
import re
import sys
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
from datasets import load_dataset
from transformers import AutoTokenizer

from batch_scaling_experiment import (
    GenerationStage,
    QueryEmbeddingStage,
    RetrievalStage,
    extract_question,
    load_yaml_config,
    resolve_path,
    set_random_seed,
    split_queries_for_warmup,
    validate_local_file,
)


@dataclass
class CategoryTimingResult:
    category_name: str
    num_queries: int
    avg_token_len: float
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


def compute_token_lengths(queries: List[str], tokenizer: AutoTokenizer) -> List[int]:
    token_lengths = []
    for query in queries:
        encoded = tokenizer(query, truncation=False, add_special_tokens=True)
        token_lengths.append(len(encoded["input_ids"]))
    return token_lengths


def print_length_distribution(queries: List[str], lengths: List[int], logger: logging.Logger) -> None:
    logger.info("=== Query Token Length Distribution ===")
    logger.info("Total queries: %s", len(queries))
    logger.info(
        "Min tokens: %s, Max tokens: %s, Mean tokens: %.1f",
        min(lengths),
        max(lengths),
        np.mean(lengths),
    )

    ranges = [(0, 5), (6, 10), (11, 15), (16, 20), (21, 30), (31, 100), (101, 1000)]
    for start, end in ranges:
        count = sum(1 for value in lengths if start <= value <= end)
        pct = 100.0 * count / len(lengths) if lengths else 0
        logger.info("  [%3d, %3d]: %4d queries (%5.1f%%)", start, end, count, pct)


def bin_queries_by_length(
    queries: List[str],
    lengths: List[int],
    config: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, Tuple[List[str], List[int]]]:
    categories_config = config["query_length_analysis"]["categories"]
    result: Dict[str, Tuple[List[str], List[int]]] = {}

    for category in categories_config.keys():
        cat_config = categories_config[category]
        min_tokens = cat_config.get("min_tokens", 0)
        max_tokens = cat_config.get("max_tokens", float("inf"))

        selected_indices = [idx for idx, token_len in enumerate(lengths) if min_tokens <= token_len <= max_tokens]
        cat_queries = [queries[idx] for idx in selected_indices]
        cat_lengths = [lengths[idx] for idx in selected_indices]

        result[category] = (cat_queries, cat_lengths)
        logger.info(
            "Category '%s': %s queries, token range [%s, %s], mean tokens: %.1f",
            category,
            len(cat_queries),
            min_tokens,
            max_tokens,
            np.mean(cat_lengths) if cat_lengths else 0.0,
        )

    return result


def sample_category(
    queries: List[str],
    lengths: List[int],
    sample_size: int,
    random_seed: int,
) -> Tuple[List[str], List[int]]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    if len(queries) <= sample_size:
        return queries, lengths

    rng = np.random.RandomState(random_seed)
    indices = rng.choice(len(queries), sample_size, replace=False)
    picked = sorted(indices.tolist())
    sampled_queries = [queries[idx] for idx in picked]
    sampled_lengths = [lengths[idx] for idx in picked]
    return sampled_queries, sampled_lengths


def _supports_backend_arg(stage_cls: Any) -> bool:
    signature = inspect.signature(stage_cls.__init__)
    return "backend" in signature.parameters


def _extract_generation_elapsed(generation_output: Tuple[Any, ...]) -> float:
    if len(generation_output) >= 2:
        return float(generation_output[1])
    raise ValueError("Unexpected generation stage output format.")


def _category_sort_key(category: str) -> Tuple[int, str]:
    match = re.search(r"(\d+)", category)
    if match:
        return int(match.group(1)), category
    return 10**9, category


class QueryLengthExperiment:
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
        self.config_path = config_path
        self.config = load_yaml_config(config_path)

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

        self.output_dir = self._resolve_output_dir(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = self._build_logger()
        self.logger.info("Experiment config path: %s", config_path)
        self.logger.info("nprobe=%s llm_model=%s test_mode=%s", nprobe, llm_model, test_mode)

        set_random_seed(self.config["dataset"]["random_seed"])
        torch.set_num_threads(self.config["system"]["num_workers"])

        self.query_length_mode = self.config["query_length_analysis"].get("mode", "natural_binning")
        self.stage_backend = self.config["query_length_analysis"].get("stage_backend", "cpu")
        self.logger.info("query_length_mode=%s stage_backend=%s", self.query_length_mode, self.stage_backend)

        self.embedding_stage = self._init_embedding_stage()
        self.retrieval_stage = self._init_retrieval_stage()
        self.retrieval_stage.set_nprobe(nprobe)
        self.generation_stage = GenerationStage(self.config, llm_model, self.logger)

        if self.query_length_mode == "controlled_dataset":
            self.warmup_queries = self._load_warmup_queries_for_controlled_mode()
            self.eval_queries: List[str] = []
            self.length_categories = self._load_controlled_length_categories()
        else:
            all_queries = self._load_queries_from_hf()
            self.warmup_queries, self.eval_queries = split_queries_for_warmup(
                all_queries,
                self.config["timing"],
            )
            self.logger.info(
                "Warmup queries: %s | Evaluation queries: %s",
                len(self.warmup_queries),
                len(self.eval_queries),
            )
            self.length_categories = self._bin_and_sample_queries()

        self.results: List[CategoryTimingResult] = []

    def _resolve_output_dir(self, output_dir: Optional[Path]) -> Path:
        if output_dir is not None:
            return output_dir
        configured = Path(self.config["output"]["output_dir"])
        if configured.is_absolute():
            return configured
        return (self.config_path.parent / configured).resolve()

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

    def _init_embedding_stage(self) -> Any:
        if _supports_backend_arg(QueryEmbeddingStage):
            return QueryEmbeddingStage(self.config, self.logger, self.stage_backend)
        return QueryEmbeddingStage(self.config, self.logger)

    def _init_retrieval_stage(self) -> Any:
        if _supports_backend_arg(RetrievalStage):
            return RetrievalStage(self.config, self.logger, self.stage_backend)
        return RetrievalStage(self.config, self.logger)

    def _load_queries_from_hf(self) -> List[str]:
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
        self.logger.info("Computing token lengths for all eval queries...")
        lengths = compute_token_lengths(self.eval_queries, self.embedding_stage.tokenizer)

        if self.config["query_length_analysis"].get("print_distribution", False):
            print_length_distribution(self.eval_queries, lengths, self.logger)

        self.logger.info("Binning queries by length...")
        binned = bin_queries_by_length(self.eval_queries, lengths, self.config, self.logger)

        sample_size = int(self.config["query_length_analysis"]["sample_size"])
        random_seed = int(self.config["dataset"]["random_seed"])

        sampled: Dict[str, Tuple[List[str], List[int]]] = {}
        for category in self.config["query_length_analysis"]["categories"].keys():
            cat_queries, cat_lengths = binned[category]
            sampled_queries, sampled_lengths = sample_category(
                cat_queries,
                cat_lengths,
                sample_size,
                random_seed + ord(category[0]),
            )
            sampled[category] = (sampled_queries, sampled_lengths)
            self.logger.info("Sampled %s queries for category '%s'", len(sampled_queries), category)

        return sampled

    def _load_warmup_queries_for_controlled_mode(self) -> List[str]:
        timing_cfg = self.config.get("timing", {})
        use_warmup = bool(timing_cfg.get("skip_first_batch", False))
        warmup_count = int(timing_cfg.get("warmup_query_count", 0) or 0)
        if not use_warmup or warmup_count <= 0:
            return []

        controlled_cfg = self.config["query_length_analysis"].get("controlled_dataset", {})
        if not controlled_cfg.get("warmup_from_hf", True):
            self.logger.info("Controlled mode warmup_from_hf=false, skipping warmup.")
            return []

        warmup_source_queries = self._load_queries_from_hf()
        if len(warmup_source_queries) <= warmup_count:
            raise ValueError("warmup_query_count is larger than available warmup queries.")
        warmup = warmup_source_queries[:warmup_count]
        self.logger.info("Warmup queries (controlled mode): %s", len(warmup))
        return warmup

    def _resolve_controlled_dataset_path(self) -> Path:
        ql_cfg = self.config["query_length_analysis"]
        controlled_cfg = ql_cfg.get("controlled_dataset", {})
        raw_path = controlled_cfg.get("path", ql_cfg.get("controlled_dataset_path"))
        if raw_path is None:
            raise ValueError(
                "controlled_dataset mode requires query_length_analysis.controlled_dataset.path "
                "or query_length_analysis.controlled_dataset_path."
            )
        return resolve_path(str(raw_path), self.config_path.parent)

    def _load_controlled_length_categories(self) -> Dict[str, Tuple[List[str], List[int]]]:
        dataset_path = self._resolve_controlled_dataset_path()
        if not dataset_path.is_file():
            raise FileNotFoundError(f"Controlled dataset not found: {dataset_path}")

        ql_cfg = self.config["query_length_analysis"]
        controlled_cfg = ql_cfg.get("controlled_dataset", {})
        text_field = controlled_cfg.get("query_field", "query")
        category_field = controlled_cfg.get("category_field", "category")
        length_field = controlled_cfg.get("length_field", "actual_tokens")
        target_field = controlled_cfg.get("target_field", "target_tokens")

        buckets: Dict[str, Tuple[List[str], List[int]]] = {}
        with dataset_path.open("r", encoding="utf-8") as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                record = json.loads(line)

                query = record.get(text_field) or record.get("rewrite_query") or record.get("query")
                if not isinstance(query, str) or not query.strip():
                    raise ValueError(f"Invalid query text at line {line_no} in {dataset_path}")
                query = query.strip()

                category = record.get(category_field)
                if category is None:
                    target = record.get(target_field)
                    if target is None:
                        raise ValueError(
                            f"Missing category/target at line {line_no} in {dataset_path}. "
                            f"Expected fields: '{category_field}' or '{target_field}'."
                        )
                    category = f"t{int(target)}"
                else:
                    category = str(category)

                length_value = record.get(length_field)
                if isinstance(length_value, (int, float)):
                    token_len = int(length_value)
                else:
                    token_len = len(
                        self.embedding_stage.tokenizer(
                            query,
                            truncation=False,
                            add_special_tokens=True,
                        )["input_ids"]
                    )

                bucket_queries, bucket_lengths = buckets.get(category, ([], []))
                bucket_queries.append(query)
                bucket_lengths.append(token_len)
                buckets[category] = (bucket_queries, bucket_lengths)

        if not buckets:
            raise ValueError(f"No valid records loaded from controlled dataset: {dataset_path}")

        if "categories" in ql_cfg and isinstance(ql_cfg["categories"], dict) and ql_cfg["categories"]:
            ordered_categories = [name for name in ql_cfg["categories"].keys() if name in buckets]
        else:
            category_order = controlled_cfg.get("category_order")
            if category_order:
                ordered_categories = [name for name in category_order if name in buckets]
            else:
                ordered_categories = sorted(buckets.keys(), key=_category_sort_key)

        if not ordered_categories:
            ordered_categories = sorted(buckets.keys(), key=_category_sort_key)

        random_seed = int(self.config["dataset"]["random_seed"])
        sample_size_cfg = ql_cfg.get("sample_size")
        if self.test_mode:
            sample_size_cfg = min(
                int(self.config["debug"].get("test_sample_size", 100)),
                min(len(buckets[name][0]) for name in ordered_categories),
            )

        sampled: Dict[str, Tuple[List[str], List[int]]] = {}
        for idx, category in enumerate(ordered_categories):
            queries, lengths = buckets[category]
            if sample_size_cfg is not None:
                sampled_queries, sampled_lengths = sample_category(
                    queries,
                    lengths,
                    int(sample_size_cfg),
                    random_seed + idx,
                )
            else:
                sampled_queries, sampled_lengths = queries, lengths

            sampled[category] = (sampled_queries, sampled_lengths)
            self.logger.info(
                "Controlled category '%s': %s queries, avg tokens=%.2f",
                category,
                len(sampled_queries),
                float(np.mean(sampled_lengths)) if sampled_lengths else 0.0,
            )

        self.logger.info("Loaded controlled dataset from %s", dataset_path)
        return sampled

    def run_category_experiment(
        self,
        category_name: str,
        queries: List[str],
        lengths: List[int],
    ) -> CategoryTimingResult:
        batch_size = int(self.config["query_length_analysis"]["batch_size"])
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

        if self.warmup_queries:
            self.logger.info("Running warmup for category '%s'...", category_name)
            for start_idx in range(0, len(self.warmup_queries), batch_size):
                warmup_batch = self.warmup_queries[start_idx : start_idx + batch_size]
                embeddings, _ = self.embedding_stage(warmup_batch)
                retrieved_docs, _ = self.retrieval_stage(embeddings)
                _ = self.generation_stage(warmup_batch, retrieved_docs)

        for batch_index, start_idx in enumerate(range(0, len(queries), batch_size), start=1):
            batch_queries = queries[start_idx : start_idx + batch_size]
            embeddings, embedding_sec = self.embedding_stage(batch_queries)
            retrieved_docs, retrieval_sec = self.retrieval_stage(embeddings)
            generation_output = self.generation_stage(batch_queries, retrieved_docs)
            generation_sec = _extract_generation_elapsed(generation_output)

            total_embedding_sec += embedding_sec
            total_retrieval_sec += retrieval_sec
            total_generation_sec += generation_sec
            effective_queries += len(batch_queries)

            if batch_index % self.config["debug"].get("print_interval", 10) == 0:
                self.logger.info("  batch %s: %s queries processed", batch_index, len(batch_queries))

        avg_token_len = float(np.mean(lengths)) if lengths else 0.0
        result = CategoryTimingResult(
            category_name=category_name,
            num_queries=effective_queries,
            avg_token_len=avg_token_len,
            embedding_total_sec=total_embedding_sec,
            retrieval_total_sec=total_retrieval_sec,
            generation_total_sec=total_generation_sec,
        )

        self.logger.info(
            "Category '%s': embedding=%.3fms, retrieval=%.3fms, generation=%.3fms, throughput=%.3fqps, avg_tokens=%.1f",
            category_name,
            result.embedding_avg_ms,
            result.retrieval_avg_ms,
            result.generation_avg_ms,
            result.throughput_qps,
            avg_token_len,
        )
        return result

    def run(self) -> None:
        for category, (queries, lengths) in self.length_categories.items():
            result = self.run_category_experiment(category, queries, lengths)
            self.results.append(result)

    def save_results(self) -> None:
        rows = [
            {
                "llm_model": self.llm_model,
                "nprobe": self.nprobe,
                "mode": self.query_length_mode,
                "stage_backend": self.stage_backend,
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
                "mode": self.query_length_mode,
                "stage_backend": self.stage_backend,
                "batch_size": self.config["query_length_analysis"]["batch_size"],
                "sample_size_per_category": self.config["query_length_analysis"].get("sample_size"),
                "categories_config": list(self.length_categories.keys()),
                "controlled_dataset_path": (
                    str(self._resolve_controlled_dataset_path())
                    if self.query_length_mode == "controlled_dataset"
                    else None
                ),
            },
            "results": [asdict(result) for result in self.results],
        }
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        self.logger.info("Saved JSON to %s", json_path)

    def generate_plots(self) -> None:
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

        throughputs = [result.throughput_qps for result in self.results]
        plt.figure(figsize=(10, 6))
        palette = ["steelblue", "green", "orange", "tomato", "red", "purple", "brown"]
        colors = palette[: len(categories)]
        plt.bar(categories, throughputs, color=colors)
        plt.xlabel("Query Length Category")
        plt.ylabel("Throughput (queries/sec)")
        plt.title(f"Throughput by Query Length | nprobe={self.nprobe} | llm={self.llm_model}")
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(plots_dir / f"throughput_by_query_length_{stem}.{plot_format}", dpi=dpi)
        plt.close()

        self.logger.info("Saved plots to %s", plots_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query length impact analysis for one-shot RAG.")
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
