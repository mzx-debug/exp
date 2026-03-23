#!/usr/bin/env python3

import argparse
import json
import logging
import math
import os
import random
import string
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
from heterag.retriever.utils import load_corpus, pooling
from transformers import AutoModel, AutoTokenizer


def synchronize_cuda_if_needed(device_index: Optional[int] = None) -> None:
    if not torch.cuda.is_available():
        return
    if device_index is None:
        torch.cuda.synchronize()
        return
    torch.cuda.synchronize(device=device_index)


def parse_primary_gpu_id(raw_gpu_id: Optional[Any]) -> int:
    if raw_gpu_id is None:
        return 0
    if isinstance(raw_gpu_id, int):
        return raw_gpu_id

    text = str(raw_gpu_id).strip()
    if not text:
        return 0

    first_token = text.split(",")[0].strip()
    if not first_token:
        return 0
    return int(first_token)


def format_output_path(template: str, output_dir: Path, **kwargs: Any) -> Path:
    formatter = string.Formatter()
    field_names = {field_name for _, field_name, _, _ in formatter.parse(template) if field_name}
    if "stage_backend" not in field_names:
        stem, ext = os.path.splitext(template)
        template = f"{stem}_{{stage_backend}}{ext}"
    filename = template.format(**kwargs)
    return output_dir / filename


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(raw_path: str, base_dir: Path) -> Path:
    candidate = Path(os.path.expandvars(os.path.expanduser(str(raw_path))))
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def validate_local_file(path: Path, label: str, config_path: Path) -> None:
    if path.is_file():
        return
    raise FileNotFoundError(
        f"{label} was not found: {path}. "
        f"Update retrieval.index_path in {config_path} or pass --index_path /path/to/ivf.index."
    )


def resolve_output_dir(output_dir: Optional[Path], config_path: Path, config: Dict[str, Any]) -> Path:
    if output_dir is not None:
        return output_dir

    configured = Path(config["output"]["output_dir"])
    if configured.is_absolute():
        return configured
    return (config_path.parent / configured).resolve()


def extract_question(record: Dict[str, Any]) -> Optional[str]:
    question = record.get("question")
    if isinstance(question, str) and question.strip():
        return question.strip()
    if isinstance(question, dict):
        for key in ("text", "question"):
            value = question.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    for key in ("question_text", "query"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def extract_doc_text(doc: Dict[str, Any]) -> str:
    if "contents" in doc and doc["contents"] is not None:
        return str(doc["contents"])
    if "text" in doc and doc["text"] is not None:
        if "title" in doc and doc["title"]:
            return f"{doc['title']}\n{doc['text']}"
        return str(doc["text"])
    if "title" in doc and doc["title"] is not None:
        return str(doc["title"])
    return ""


def build_prompt(query: str, context: str, config: Dict[str, Any]) -> str:
    template = config["rag"].get(
        "prompt_template",
        "Question: {query}\nContext: {context}\nAnswer:",
    )
    return template.format(query=query, context=context)


def filter_batch_sizes(
    config: Dict[str, Any],
    max_batch_size: Optional[int],
    test_mode: bool,
) -> List[int]:
    if test_mode:
        batch_sizes = list(config["debug"]["test_batch_sizes"])
    else:
        batch_sizes = list(config["batch_size_scaling"]["batch_sizes"])

    limit = max_batch_size
    if limit is None:
        limit = config["batch_size_scaling"].get("max_batch_size")

    if limit is not None:
        batch_sizes = [batch_size for batch_size in batch_sizes if batch_size <= limit]

    if not batch_sizes:
        raise ValueError("No batch sizes remain after applying the max batch size filter.")

    return batch_sizes


def split_queries_for_warmup(
    queries: Sequence[str],
    timing_config: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    use_warmup = timing_config.get("skip_first_batch", False)
    warmup_query_count = int(timing_config.get("warmup_query_count", 0) or 0)

    if not use_warmup or warmup_query_count <= 0:
        return [], list(queries)

    if len(queries) <= warmup_query_count:
        raise ValueError(
            "warmup_query_count must be smaller than the number of loaded queries."
        )

    warmup_queries = list(queries[:warmup_query_count])
    eval_queries = list(queries[warmup_query_count:])
    return warmup_queries, eval_queries


@dataclass
class BatchRecord:
    batch_index: int
    batch_size: int
    actual_batch_size: int
    warmup: bool
    embedding_sec: float
    retrieval_sec: float
    generation_sec: float
    generation_tokens: int
    generation_time_per_token: Optional[float]


@dataclass
class TimingResult:
    stage_backend: str
    batch_size: int
    num_queries: int
    embedding_total_sec: float
    retrieval_total_sec: float
    generation_total_sec: float
    generation_tokens_total: int
    embedding_avg_ms: float = field(init=False)
    retrieval_avg_ms: float = field(init=False)
    generation_avg_ms: float = field(init=False)
    avg_generation_tokens: float = field(init=False)
    generation_time_per_token: float = field(init=False)
    generation_time_per_token_ms: float = field(init=False)
    total_sec: float = field(init=False)
    total_ms: float = field(init=False)
    throughput_qps: float = field(init=False)

    def __post_init__(self) -> None:
        if self.num_queries <= 0:
            raise ValueError("num_queries must be positive.")

        self.embedding_avg_ms = (self.embedding_total_sec * 1000.0) / self.num_queries
        self.retrieval_avg_ms = (self.retrieval_total_sec * 1000.0) / self.num_queries
        self.generation_avg_ms = (self.generation_total_sec * 1000.0) / self.num_queries
        self.avg_generation_tokens = self.generation_tokens_total / self.num_queries
        if self.generation_tokens_total > 0:
            self.generation_time_per_token = self.generation_total_sec / self.generation_tokens_total
            self.generation_time_per_token_ms = self.generation_time_per_token * 1000.0
        else:
            self.generation_time_per_token = math.inf
            self.generation_time_per_token_ms = math.inf
        self.total_sec = (
            self.embedding_total_sec + self.retrieval_total_sec + self.generation_total_sec
        )
        self.total_ms = self.total_sec * 1000.0
        self.throughput_qps = self.num_queries / self.total_sec if self.total_sec > 0 else math.inf


class QueryEmbeddingStage:
    def __init__(self, config: Dict[str, Any], logger: logging.Logger, backend: str) -> None:
        retrieval_config = config["models"]["retrieval"]
        self.logger = logger
        self.backend = backend
        self.cuda_device_index: Optional[int] = None
        self.pooling_method = retrieval_config["pooling_method"]
        self.max_length = retrieval_config["max_length"]

        if backend == "gpu":
            if not torch.cuda.is_available():
                raise RuntimeError("Embedding backend is gpu but CUDA is not available.")
            self.device = torch.device("cuda")
            self.cuda_device_index = parse_primary_gpu_id(config["system"].get("gpu_id"))
        elif backend == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"Unknown embedding backend: {backend}")

        model_path = retrieval_config["model_path"]
        self.logger.info("Loading embedding model: %s (backend=%s)", model_path, backend)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

        self.model = self.model.to(self.device)
        if self.device.type == "cuda" and retrieval_config.get("use_fp16", False):
            self.model = self.model.half()
        self.model.eval()

    @torch.inference_mode()
    def __call__(self, queries: Sequence[str]) -> Tuple[np.ndarray, float]:
        synchronize_cuda_if_needed(self.cuda_device_index)
        start = time.perf_counter()

        inputs = self.tokenizer(
            list(queries),
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        outputs = self.model(**inputs, return_dict=True)
        pooler_output = getattr(outputs, "pooler_output", None)
        embeddings = pooling(
            pooler_output,
            outputs.last_hidden_state,
            inputs["attention_mask"],
            self.pooling_method,
        )
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        embeddings = embeddings.detach().cpu().numpy().astype(np.float32, order="C")

        synchronize_cuda_if_needed(self.cuda_device_index)
        elapsed = time.perf_counter() - start
        return embeddings, elapsed


class RetrievalStage:
    def __init__(self, config: Dict[str, Any], logger: logging.Logger, backend: str) -> None:
        import faiss

        retrieval_config = config["retrieval"]
        self.logger = logger
        self.faiss = faiss
        self.backend = backend
        self.cuda_device_index: Optional[int] = None
        self.gpu_resources = None
        self.topk = retrieval_config["topk"]
        self.index_path = retrieval_config["index_path"]
        self.corpus_path = retrieval_config["corpus_path"]

        self.logger.info("Loading FAISS index: %s", self.index_path)
        cpu_index = self.faiss.read_index(self.index_path)

        if backend == "gpu":
            if not torch.cuda.is_available():
                raise RuntimeError("Retrieval backend is gpu but CUDA is not available.")
            if not hasattr(self.faiss, "StandardGpuResources"):
                raise RuntimeError(
                    "FAISS GPU resources are unavailable. Install faiss-gpu to run retrieval on GPU."
                )
            self.cuda_device_index = parse_primary_gpu_id(config["system"].get("gpu_id"))
            self.gpu_resources = self.faiss.StandardGpuResources()
            self.index = self.faiss.index_cpu_to_gpu(self.gpu_resources, self.cuda_device_index, cpu_index)
            self.logger.info("FAISS retrieval backend: gpu (device=%s)", self.cuda_device_index)
        elif backend == "cpu":
            self.index = cpu_index
            self.logger.info("FAISS retrieval backend: cpu")
        else:
            raise ValueError(f"Unknown retrieval backend: {backend}")

        self.logger.info("Loading corpus: %s", self.corpus_path)
        corpus_is_hf = not Path(self.corpus_path).exists()
        self.corpus = load_corpus(self.corpus_path, hf=corpus_is_hf)

    def set_nprobe(self, nprobe: int) -> None:
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = nprobe

    def __call__(self, embeddings: np.ndarray) -> Tuple[List[List[str]], float]:
        synchronize_cuda_if_needed(self.cuda_device_index)
        start = time.perf_counter()
        _, indices = self.index.search(embeddings, self.topk)

        retrieved_docs: List[List[str]] = []
        for idx_row in indices:
            docs: List[str] = []
            for idx in idx_row:
                if idx < 0:
                    continue
                doc = self.corpus[int(idx)]
                docs.append(extract_doc_text(doc))
            retrieved_docs.append(docs)

        synchronize_cuda_if_needed(self.cuda_device_index)
        elapsed = time.perf_counter() - start
        return retrieved_docs, elapsed


class GenerationStage:
    def __init__(self, config: Dict[str, Any], llm_model_name: str, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self.llm_model_name = llm_model_name

        llm_config = None
        for generator in config["models"]["generators"]:
            if generator["name"] == llm_model_name:
                llm_config = generator
                break

        if llm_config is None:
            raise ValueError(f"Unknown llm_model: {llm_model_name}")

        self.llm_config = llm_config
        self.default_max_tokens = int(llm_config["max_output_len"])
        self.active_max_tokens = self.default_max_tokens
        self.logger.info("Loading generation model via vLLM: %s", llm_config["model_path"])

        from vllm import LLM, SamplingParams

        self.SamplingParams = SamplingParams
        self.llm = LLM(
            model=llm_config["model_path"],
            tensor_parallel_size=config["optimization"].get("vllm_tensor_parallel_size", 1),
            gpu_memory_utilization=config["optimization"]["vllm_gpu_memory_utilization"],
            enforce_eager=config["optimization"]["vllm_enforce_eager"],
        )
        self._fallback_tokenizer = self.llm.get_tokenizer()

    def set_max_tokens(self, max_tokens: int) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")
        self.active_max_tokens = int(max_tokens)
        self.logger.info(
            "Generation max_tokens updated: %s -> %s",
            self.default_max_tokens,
            self.active_max_tokens,
        )

    def __call__(
        self,
        queries: Sequence[str],
        retrieved_docs: Sequence[Sequence[str]],
    ) -> Tuple[List[str], float, List[int]]:
        prompts = [
            build_prompt(query=query, context="\n".join(docs), config=self.config)
            for query, docs in zip(queries, retrieved_docs)
        ]

        generation_config = self.config["rag"]["generation"]
        sampling_kwargs = {
            "temperature": generation_config["temperature"],
            "top_p": generation_config["top_p"],
            "max_tokens": self.active_max_tokens,
        }
        if generation_config.get("top_k") is not None:
            sampling_kwargs["top_k"] = generation_config["top_k"]

        sampling_params = self.SamplingParams(**sampling_kwargs)

        synchronize_cuda_if_needed()
        start = time.perf_counter()
        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)
        answers: List[str] = []
        generated_tokens: List[int] = []
        for output in outputs:
            if not output.outputs:
                answers.append("")
                generated_tokens.append(0)
                continue
            first_output = output.outputs[0]
            answers.append(first_output.text)
            token_ids = getattr(first_output, "token_ids", None)
            if token_ids is not None:
                generated_tokens.append(len(token_ids))
            else:
                generated_tokens.append(len(self._fallback_tokenizer.encode(first_output.text)))

        synchronize_cuda_if_needed()
        elapsed = time.perf_counter() - start
        return answers, elapsed, generated_tokens


class BatchScalingExperiment:
    def __init__(
        self,
        config_path: Path,
        nprobe: int,
        llm_model: str,
        stage_backend: str,
        output_dir: Optional[Path],
        max_batch_size: Optional[int],
        test_mode: bool,
        index_path_override: Optional[Path],
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
        self.stage_backend = stage_backend
        self.test_mode = test_mode
        self.config["debug"]["test_mode"] = test_mode
        self.batch_sizes = filter_batch_sizes(self.config, max_batch_size, test_mode)
        self.generation_length_probe: Optional[Dict[str, Any]] = None

        gpu_id = self.config["system"].get("gpu_id")
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        self.output_dir = resolve_output_dir(output_dir, config_path, self.config)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = self._build_logger()
        self.logger.info("Experiment config path: %s", config_path)
        self.logger.info(
            "nprobe=%s llm_model=%s stage_backend=%s test_mode=%s",
            nprobe,
            llm_model,
            stage_backend,
            test_mode,
        )
        self.logger.info("Batch sizes: %s", self.batch_sizes)

        set_random_seed(self.config["dataset"]["random_seed"])
        torch.set_num_threads(self.config["system"]["num_workers"])

        self.embedding_stage = QueryEmbeddingStage(self.config, self.logger, backend=stage_backend)
        self.retrieval_stage = RetrievalStage(self.config, self.logger, backend=stage_backend)
        self.retrieval_stage.set_nprobe(nprobe)
        self.generation_stage = GenerationStage(self.config, llm_model, self.logger)

        all_queries = self._load_queries()
        self.warmup_queries, self.eval_queries = split_queries_for_warmup(
            all_queries,
            self.config["timing"],
        )
        self.queries = self.eval_queries
        self.results: List[TimingResult] = []
        self.batch_records: Dict[int, List[BatchRecord]] = {}

        self.logger.info(
            "Warmup queries: %s | Evaluation queries: %s",
            len(self.warmup_queries),
            len(self.eval_queries),
        )
        self._configure_generation_token_limit()

    def _build_logger(self) -> logging.Logger:
        logger = logging.getLogger(
            f"batch_scaling.{self.nprobe}.{self.llm_model}.{self.stage_backend}"
        )
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.handlers.clear()

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        file_handler = logging.FileHandler(
            self.output_dir
            / f"experiment_log_{self.nprobe}_{self.llm_model}_{self.stage_backend}.log",
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
            raise NotImplementedError("Only HuggingFace datasets are supported in this experiment.")

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

    def _configure_generation_token_limit(self) -> None:
        generation_config = self.config.get("rag", {}).get("generation", {})
        probe_config = generation_config.get("length_probe", {})
        if probe_config is None:
            probe_config = {}

        if not probe_config.get("enabled", True):
            configured_cap = int(
                generation_config.get("max_tokens_override", self.generation_stage.default_max_tokens)
            )
            self.generation_stage.set_max_tokens(configured_cap)
            self.logger.info("Token-length probing disabled; using max_tokens=%s", configured_cap)
            return

        sample_size = int(probe_config.get("sample_size", 128))
        probe_batch_size = int(probe_config.get("probe_batch_size", 16))
        target_truncation_ratio = float(probe_config.get("target_truncation_ratio", 0.7))
        min_max_tokens = int(probe_config.get("min_max_tokens", 16))
        max_max_tokens_cfg = probe_config.get("max_max_tokens")
        max_allowed = (
            int(max_max_tokens_cfg)
            if max_max_tokens_cfg is not None
            else self.generation_stage.default_max_tokens
        )

        sample_size = max(1, min(sample_size, len(self.eval_queries)))
        probe_batch_size = max(1, probe_batch_size)
        target_truncation_ratio = min(max(target_truncation_ratio, 0.0), 0.95)

        if sample_size <= 0:
            raise ValueError("No evaluation queries available for generation length probing.")

        rng = np.random.RandomState(self.config["dataset"]["random_seed"])
        if sample_size < len(self.eval_queries):
            sampled_indices = rng.choice(len(self.eval_queries), size=sample_size, replace=False)
            probe_queries = [self.eval_queries[idx] for idx in sorted(sampled_indices.tolist())]
        else:
            probe_queries = list(self.eval_queries)

        self.logger.info(
            "Running generation length probe over %s queries (batch_size=%s, default_max_tokens=%s)",
            len(probe_queries),
            probe_batch_size,
            self.generation_stage.default_max_tokens,
        )

        token_lengths: List[int] = []
        for start_idx in range(0, len(probe_queries), probe_batch_size):
            query_batch = probe_queries[start_idx : start_idx + probe_batch_size]
            embeddings, _ = self.embedding_stage(query_batch)
            retrieved_docs, _ = self.retrieval_stage(embeddings)
            _, _, generated_tokens = self.generation_stage(query_batch, retrieved_docs)
            token_lengths.extend(generated_tokens)

        if not token_lengths:
            selected_max_tokens = min_max_tokens
            self.logger.warning("Length probe generated zero tokens; fallback max_tokens=%s", selected_max_tokens)
        else:
            percentile = (1.0 - target_truncation_ratio) * 100.0
            candidate = int(np.floor(np.percentile(token_lengths, percentile)))
            selected_max_tokens = max(min_max_tokens, min(candidate, max_allowed))

        self.generation_stage.set_max_tokens(selected_max_tokens)

        if token_lengths:
            probe_np = np.array(token_lengths, dtype=np.int64)
            truncation_rate = float(np.mean(probe_np >= selected_max_tokens))
            self.generation_length_probe = {
                "sample_size": len(token_lengths),
                "probe_batch_size": probe_batch_size,
                "target_truncation_ratio": target_truncation_ratio,
                "selected_max_tokens": selected_max_tokens,
                "min_tokens": int(probe_np.min()),
                "max_tokens": int(probe_np.max()),
                "mean_tokens": float(np.mean(probe_np)),
                "p50_tokens": float(np.percentile(probe_np, 50)),
                "p75_tokens": float(np.percentile(probe_np, 75)),
                "p90_tokens": float(np.percentile(probe_np, 90)),
                "p95_tokens": float(np.percentile(probe_np, 95)),
                "estimated_truncation_rate": truncation_rate,
            }
            self.logger.info(
                (
                    "Length probe selected max_tokens=%s | mean=%.2f p90=%.2f p95=%.2f "
                    "estimated_truncation_rate=%.3f"
                ),
                selected_max_tokens,
                self.generation_length_probe["mean_tokens"],
                self.generation_length_probe["p90_tokens"],
                self.generation_length_probe["p95_tokens"],
                self.generation_length_probe["estimated_truncation_rate"],
            )
        else:
            self.generation_length_probe = {
                "sample_size": 0,
                "probe_batch_size": probe_batch_size,
                "target_truncation_ratio": target_truncation_ratio,
                "selected_max_tokens": selected_max_tokens,
                "estimated_truncation_rate": None,
            }

    def run_batch_experiment(self, batch_size: int) -> TimingResult:
        total_embedding_sec = 0.0
        total_retrieval_sec = 0.0
        total_generation_sec = 0.0
        total_generation_tokens = 0
        records: List[BatchRecord] = []

        effective_queries = 0

        self.logger.info(
            "Running batch_size=%s over %s evaluation queries (backend=%s)",
            batch_size,
            len(self.eval_queries),
            self.stage_backend,
        )

        if self.warmup_queries:
            self.logger.info(
                "Running warmup for batch_size=%s over %s fixed warmup queries",
                batch_size,
                len(self.warmup_queries),
            )
            for start_idx in range(0, len(self.warmup_queries), batch_size):
                warmup_batch = self.warmup_queries[start_idx : start_idx + batch_size]
                embeddings, _ = self.embedding_stage(warmup_batch)
                retrieved_docs, _ = self.retrieval_stage(embeddings)
                self.generation_stage(warmup_batch, retrieved_docs)

        for batch_index, start_idx in enumerate(range(0, len(self.eval_queries), batch_size), start=1):
            batch_queries = self.eval_queries[start_idx : start_idx + batch_size]
            is_warmup = False

            embeddings, embedding_sec = self.embedding_stage(batch_queries)
            retrieved_docs, retrieval_sec = self.retrieval_stage(embeddings)
            _, generation_sec, generated_tokens = self.generation_stage(batch_queries, retrieved_docs)
            batch_generation_tokens = int(sum(generated_tokens))
            batch_generation_time_per_token = (
                generation_sec / batch_generation_tokens if batch_generation_tokens > 0 else None
            )

            records.append(
                BatchRecord(
                    batch_index=batch_index,
                    batch_size=batch_size,
                    actual_batch_size=len(batch_queries),
                    warmup=is_warmup,
                    embedding_sec=embedding_sec,
                    retrieval_sec=retrieval_sec,
                    generation_sec=generation_sec,
                    generation_tokens=batch_generation_tokens,
                    generation_time_per_token=batch_generation_time_per_token,
                )
            )

            total_embedding_sec += embedding_sec
            total_retrieval_sec += retrieval_sec
            total_generation_sec += generation_sec
            total_generation_tokens += batch_generation_tokens
            effective_queries += len(batch_queries)

        if effective_queries == 0:
            raise ValueError(
                f"Effective query count is zero for batch_size={batch_size}. "
                "Disable warmup skipping or use more than one batch."
            )

        result = TimingResult(
            stage_backend=self.stage_backend,
            batch_size=batch_size,
            num_queries=effective_queries,
            embedding_total_sec=total_embedding_sec,
            retrieval_total_sec=total_retrieval_sec,
            generation_total_sec=total_generation_sec,
            generation_tokens_total=total_generation_tokens,
        )
        self.batch_records[batch_size] = records

        self.logger.info(
            (
                "batch_size=%s backend=%s avg_embedding=%.3fms avg_retrieval=%.3fms "
                "avg_generation=%.3fms generation_time_per_token=%.6fs throughput=%.3fqps"
            ),
            batch_size,
            self.stage_backend,
            result.embedding_avg_ms,
            result.retrieval_avg_ms,
            result.generation_avg_ms,
            result.generation_time_per_token,
            result.throughput_qps,
        )
        return result

    def run(self) -> None:
        for batch_size in self.batch_sizes:
            result = self.run_batch_experiment(batch_size)
            self.results.append(result)

    def save_results(self) -> None:
        rows = [
            {
                "llm_model": self.llm_model,
                "nprobe": self.nprobe,
                "stage_backend": self.stage_backend,
                "batch_size": result.batch_size,
                "num_queries": result.num_queries,
                "avg_embedding_time_ms": round(result.embedding_avg_ms, 4),
                "avg_retrieval_time_ms": round(result.retrieval_avg_ms, 4),
                "avg_generation_time_ms": round(result.generation_avg_ms, 4),
                "generation_tokens_total": result.generation_tokens_total,
                "avg_generation_tokens": round(result.avg_generation_tokens, 4),
                "generation_time_per_token": round(result.generation_time_per_token, 8),
                "generation_time_per_token_ms": round(result.generation_time_per_token_ms, 6),
                "total_time_ms": round(result.total_ms, 4),
                "throughput_qps": round(result.throughput_qps, 6),
            }
            for result in self.results
        ]

        csv_path = format_output_path(
            self.config["output"]["csv_filename"],
            self.output_dir,
            nprobe=self.nprobe,
            llm_model=self.llm_model,
            stage_backend=self.stage_backend,
        )
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        json_path = format_output_path(
            self.config["output"]["json_filename"],
            self.output_dir,
            nprobe=self.nprobe,
            llm_model=self.llm_model,
            stage_backend=self.stage_backend,
        )
        payload = {
            "experiment_config": {
                "config_path": str(self.config_path),
                "dataset": self.config["dataset"]["name"],
                "dataset_split": self.config["dataset"]["split"],
                "total_queries_loaded": len(self.warmup_queries) + len(self.eval_queries),
                "warmup_queries": len(self.warmup_queries),
                "evaluation_queries": len(self.eval_queries),
                "test_mode": self.test_mode,
                "stage_backend": self.stage_backend,
                "workflow_type": self.config["rag"]["workflow_type"],
                "topk": self.config["retrieval"]["topk"],
                "nprobe": self.nprobe,
                "llm_model": self.llm_model,
                "embedding_model": self.config["models"]["retrieval"]["model_path"],
                "index_path": self.config["retrieval"]["index_path"],
                "corpus_path": self.config["retrieval"]["corpus_path"],
                "batch_sizes": self.batch_sizes,
                "skip_first_batch": self.config["timing"]["skip_first_batch"],
                "warmup_query_count": self.config["timing"].get("warmup_query_count", 0),
                "generation_max_tokens_selected": self.generation_stage.active_max_tokens,
                "generation_default_max_tokens": self.generation_stage.default_max_tokens,
            },
            "generation_length_probe": self.generation_length_probe,
            "results": [asdict(result) for result in self.results],
            "per_batch_records": {
                str(batch_size): [asdict(record) for record in records]
                for batch_size, records in self.batch_records.items()
            },
        }
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

        self.logger.info("Saved CSV to %s", csv_path)
        self.logger.info("Saved JSON to %s", json_path)

    def generate_plots(self) -> None:
        if not self.config["output"].get("generate_plots", False):
            return

        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(
            {
                "batch_size": [result.batch_size for result in self.results],
                "embedding_ms": [result.embedding_avg_ms for result in self.results],
                "retrieval_ms": [result.retrieval_avg_ms for result in self.results],
                "generation_ms": [result.generation_avg_ms for result in self.results],
                "generation_time_per_token_ms": [
                    result.generation_time_per_token_ms for result in self.results
                ],
                "total_ms": [result.total_ms for result in self.results],
                "throughput_qps": [result.throughput_qps for result in self.results],
            }
        )

        plot_format = self.config["output"].get("plot_format", "png")
        dpi = self.config["output"].get("plot_dpi", 300)
        stem = f"{self.nprobe}_{self.llm_model}_{self.stage_backend}"

        plt.figure(figsize=(10, 6))
        plt.plot(df["batch_size"], df["embedding_ms"], marker="o", label="Embedding")
        plt.plot(df["batch_size"], df["retrieval_ms"], marker="o", label="Retrieval")
        plt.plot(df["batch_size"], df["generation_ms"], marker="o", label="Generation")
        plt.plot(df["batch_size"], df["total_ms"], marker="o", label="Total")
        plt.xscale("log", base=2)
        plt.xlabel("Query batch size")
        plt.ylabel("Average time per query (ms)")
        plt.title(f"Latency breakdown | nprobe={self.nprobe} | llm={self.llm_model}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / f"latency_breakdown_{stem}.{plot_format}", dpi=dpi)
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(df["batch_size"], df["throughput_qps"], marker="o")
        plt.xscale("log", base=2)
        plt.xlabel("Query batch size")
        plt.ylabel("Throughput (queries/sec)")
        plt.title(f"Throughput | nprobe={self.nprobe} | llm={self.llm_model}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / f"throughput_{stem}.{plot_format}", dpi=dpi)
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(df["batch_size"], df["generation_time_per_token_ms"], marker="o")
        plt.xscale("log", base=2)
        plt.xlabel("Query batch size")
        plt.ylabel("Generation time per token (ms)")
        plt.title(
            f"Generation efficiency | nprobe={self.nprobe} | llm={self.llm_model} | backend={self.stage_backend}"
        )
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / f"generation_per_token_{stem}.{plot_format}", dpi=dpi)
        plt.close()

        self.logger.info("Saved plots to %s", plots_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch size scaling benchmark for one-shot RAG.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the YAML config file.")
    parser.add_argument("--nprobe", type=int, required=True, help="FAISS nprobe value.")
    parser.add_argument("--llm_model", type=str, required=True, help="Generator model name from config.")
    parser.add_argument(
        "--stage_backend",
        type=str,
        default="both",
        choices=["cpu", "gpu", "both"],
        help="Where to run embedding+retrieval stages: cpu, gpu, or both (default).",
    )
    parser.add_argument(
        "--index_path",
        type=Path,
        default=None,
        help="Override retrieval.index_path for the FAISS index file.",
    )
    parser.add_argument("--output_dir", type=Path, default=None, help="Directory for CSV, JSON and plots.")
    parser.add_argument("--max_batch_size", type=int, default=None, help="Only run batch sizes up to this value.")
    parser.add_argument("--test_mode", action="store_true", help="Run the reduced quick-test configuration.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backends = ["cpu", "gpu"] if args.stage_backend == "both" else [args.stage_backend]
    for backend in backends:
        experiment = BatchScalingExperiment(
            config_path=args.config,
            nprobe=args.nprobe,
            llm_model=args.llm_model,
            stage_backend=backend,
            output_dir=args.output_dir,
            max_batch_size=args.max_batch_size,
            test_mode=args.test_mode,
            index_path_override=args.index_path,
        )
        experiment.run()
        experiment.save_results()
        experiment.generate_plots()


if __name__ == "__main__":
    main()
