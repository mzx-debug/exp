#!/usr/bin/env python3
"""
Build a controlled query-length dataset for RAG latency experiments.

Pipeline:
1) Sample N seed questions from Natural Questions.
2) Rewrite each seed into multiple target token lengths (e5 tokenizer).
3) Optionally run semantic-consistency filtering.
4) Export one combined JSONL/CSV and one JSONL per target length bucket.
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SYSTEM_PROMPT = (
    "You rewrite user questions while preserving meaning exactly. "
    "Output a single standalone question only, with no explanation."
)

DEFAULT_PROVIDER_CONFIG: Dict[str, Dict[str, Optional[str]]] = {
    "dashscope": {
        "api_key_env": "DASHSCOPE_API_KEY",
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_mode": "chat",
    },
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "api_base": "https://api.deepseek.com",
        "api_mode": "chat",
    },
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "api_base": None,
        "api_mode": "responses",
    },
}


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked_embeddings = last_hidden_state * mask
    sum_embeddings = masked_embeddings.sum(dim=1)
    sum_mask = mask.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / sum_mask


@dataclass
class RewriteRecord:
    seed_id: int
    target_tokens: int
    category: str
    source_query: str
    query: str
    actual_tokens: int
    token_error: int
    rewrite_attempts: int
    semantic_similarity: Optional[float]
    semantic_passed: Optional[bool]


class SemanticChecker:
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        device: str,
        logger: logging.Logger,
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.logger = logger

        self.logger.info(
            "Loading semantic checker model: %s (tokenizer=%s, device=%s)",
            model_path,
            tokenizer_path,
            self.device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def embed(self, texts: Sequence[str], max_length: int = 512) -> np.ndarray:
        prefixed = [f"query: {text}" for text in texts]
        inputs = self.tokenizer(
            prefixed,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        outputs = self.model(**inputs, return_dict=True)
        vectors = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
        vectors = torch.nn.functional.normalize(vectors, dim=-1)
        return vectors.detach().cpu().numpy().astype(np.float32, order="C")

    def cosine_similarity(self, text_a: str, text_b: str) -> float:
        embeddings = self.embed([text_a, text_b])
        return float(np.dot(embeddings[0], embeddings[1]))


def build_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("controlled_query_length_dataset")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(output_dir / "build_dataset.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def normalize_query(query: str) -> str:
    return " ".join(query.strip().split()).lower()


def count_tokens(text: str, tokenizer: AutoTokenizer) -> int:
    return len(tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"])


def sanitize_model_output(text: str) -> str:
    line = " ".join(text.strip().splitlines()).strip()
    prefixes = ("question:", "query:")
    lowered = line.lower()
    for prefix in prefixes:
        if lowered.startswith(prefix):
            line = line[len(prefix) :].strip()
            break
    if line.startswith('"') and line.endswith('"') and len(line) >= 2:
        line = line[1:-1].strip()
    return line


def parse_responses_api_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = getattr(response, "output", None)
    if output:
        chunks: List[str] = []
        for item in output:
            content = getattr(item, "content", None) or []
            for block in content:
                block_text = getattr(block, "text", None)
                if isinstance(block_text, str) and block_text.strip():
                    chunks.append(block_text.strip())
        if chunks:
            return "\n".join(chunks)
    raise ValueError("Could not parse text from model response.")


def parse_chat_completions_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError("Chat completion returned no choices.")

    first = choices[0]
    message = getattr(first, "message", None)
    if message is None and isinstance(first, dict):
        message = first.get("message")
    if message is None:
        raise ValueError("Chat completion missing message.")

    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = getattr(message, "content", None)

    if isinstance(content, str) and content.strip():
        return content.strip()

    if isinstance(content, list):
        chunks: List[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text") or block.get("content")
            else:
                text = getattr(block, "text", None)
            if isinstance(text, str) and text.strip():
                chunks.append(text.strip())
        if chunks:
            return "\n".join(chunks)

    raise ValueError("Could not parse content from chat completion.")


def call_llm_and_get_text(
    *,
    client: Any,
    model: str,
    api_mode: str,
    prompt: str,
    request_timeout_sec: int,
) -> str:
    if api_mode == "chat":
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            timeout=request_timeout_sec,
        )
        return parse_chat_completions_text(response)

    if api_mode == "responses":
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            timeout=request_timeout_sec,
        )
        return parse_responses_api_text(response)

    raise ValueError(f"Unsupported api_mode: {api_mode}")


def resolve_provider_settings(
    *,
    provider: str,
    api_key_env_override: Optional[str],
    api_base_override: Optional[str],
    api_mode_override: Optional[str],
) -> Tuple[str, Optional[str], str]:
    if provider not in DEFAULT_PROVIDER_CONFIG:
        raise ValueError(f"Unsupported provider: {provider}")

    default_cfg = DEFAULT_PROVIDER_CONFIG[provider]
    api_key_env = api_key_env_override or str(default_cfg["api_key_env"])
    api_base = api_base_override if api_base_override is not None else default_cfg["api_base"]
    default_mode = str(default_cfg["api_mode"])

    if api_mode_override is None or api_mode_override == "auto":
        api_mode = default_mode
    else:
        api_mode = api_mode_override

    return api_key_env, api_base, api_mode


def build_rewrite_prompt(
    source_query: str,
    target_tokens: int,
    tolerance: int,
    attempt: int,
    last_candidate: Optional[str],
    last_tokens: Optional[int],
    extra_constraint: Optional[str],
) -> str:
    lines = [
        "Rewrite the question while preserving meaning exactly.",
        f"Target token length (e5 tokenizer): {target_tokens}",
        f"Allowed range: [{target_tokens - tolerance}, {target_tokens + tolerance}]",
        "Output one question only. No answer. No explanation. No prefix.",
        f"Original question: {source_query}",
    ]
    if attempt > 1 and last_candidate and last_tokens is not None:
        lines.append(f"Previous rewrite ({last_tokens} tokens): {last_candidate}")
        if last_tokens > target_tokens + tolerance:
            lines.append("Previous rewrite was too long. Make it shorter.")
        elif last_tokens < target_tokens - tolerance:
            lines.append("Previous rewrite was too short. Make it longer.")
    if extra_constraint:
        lines.append(extra_constraint)
    return "\n".join(lines)


def rewrite_to_target_length(
    *,
    client: Any,
    model: str,
    api_mode: str,
    source_query: str,
    target_tokens: int,
    tokenizer: AutoTokenizer,
    tolerance: int,
    max_attempts: int,
    request_timeout_sec: int,
    api_retries: int,
    sleep_sec: float,
    logger: logging.Logger,
    extra_constraint: Optional[str] = None,
) -> Tuple[str, int, int]:
    best_query: Optional[str] = None
    best_error = 10**9
    best_tokens = -1
    last_candidate: Optional[str] = None
    last_tokens: Optional[int] = None

    for attempt in range(1, max_attempts + 1):
        prompt = build_rewrite_prompt(
            source_query=source_query,
            target_tokens=target_tokens,
            tolerance=tolerance,
            attempt=attempt,
            last_candidate=last_candidate,
            last_tokens=last_tokens,
            extra_constraint=extra_constraint,
        )
        response = None
        response_text: Optional[str] = None
        for api_try in range(api_retries):
            try:
                response_text = call_llm_and_get_text(
                    client=client,
                    model=model,
                    api_mode=api_mode,
                    prompt=prompt,
                    request_timeout_sec=request_timeout_sec,
                )
                break
            except Exception as exc:
                wait_sec = min(2 ** api_try, 10)
                logger.warning(
                    "API error on attempt=%s api_try=%s/%s: %s. Retry in %ss.",
                    attempt,
                    api_try + 1,
                    api_retries,
                    exc,
                    wait_sec,
                )
                time.sleep(wait_sec)
        if response_text is None:
            raise RuntimeError(f"LLM API failed after {api_retries} retries on attempt={attempt}")

        candidate = sanitize_model_output(response_text)
        if not candidate:
            continue

        token_len = count_tokens(candidate, tokenizer)
        token_error = abs(token_len - target_tokens)
        if token_error < best_error:
            best_query = candidate
            best_error = token_error
            best_tokens = token_len

        if token_error <= tolerance:
            return candidate, token_len, attempt

        last_candidate = candidate
        last_tokens = token_len
        if sleep_sec > 0:
            time.sleep(sleep_sec)

    if best_query is None:
        raise RuntimeError(
            f"Failed to rewrite query after {max_attempts} attempts: '{source_query[:120]}'"
        )
    logger.warning(
        "Could not hit target=%s exactly. Using best candidate with %s tokens (error=%s).",
        target_tokens,
        best_tokens,
        best_error,
    )
    return best_query, best_tokens, max_attempts


def load_seed_queries(
    *,
    dataset_name: str,
    split: str,
    seed_count: int,
    tokenizer: AutoTokenizer,
    min_seed_tokens: int,
    max_seed_tokens: int,
    random_seed: int,
    logger: logging.Logger,
) -> List[str]:
    logger.info("Loading source dataset: %s split=%s", dataset_name, split)
    dataset = load_dataset(dataset_name, split=split)
    extracted = [extract_question(record) for record in dataset]
    extracted = [query.strip() for query in extracted if isinstance(query, str) and query.strip()]

    deduped: List[str] = []
    seen = set()
    for query in extracted:
        key = normalize_query(query)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(query)

    filtered: List[str] = []
    for query in deduped:
        token_len = count_tokens(query, tokenizer)
        if min_seed_tokens <= token_len <= max_seed_tokens:
            filtered.append(query)

    if len(filtered) < seed_count:
        raise ValueError(
            f"Not enough seed queries after filtering: need {seed_count}, got {len(filtered)}. "
            f"Relax min/max seed tokens [{min_seed_tokens}, {max_seed_tokens}]."
        )

    rng = np.random.RandomState(random_seed)
    indices = rng.choice(len(filtered), size=seed_count, replace=False)
    selected = [filtered[idx] for idx in sorted(indices.tolist())]
    logger.info(
        "Selected %s seed queries from %s candidates (token range [%s, %s]).",
        len(selected),
        len(filtered),
        min_seed_tokens,
        max_seed_tokens,
    )
    return selected


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_openai_client(api_key: str, api_base: Optional[str]) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("openai package is required. Install with: pip install openai") from exc

    kwargs: Dict[str, Any] = {"api_key": api_key}
    if api_base:
        kwargs["base_url"] = api_base
    return OpenAI(**kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build controlled query-length buckets from Natural Questions using an LLM.",
    )
    parser.add_argument("--dataset_name", type=str, default="natural_questions")
    parser.add_argument("--dataset_split", type=str, default="validation")
    parser.add_argument("--seed_count", type=int, default=64)
    parser.add_argument("--target_lengths", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256])
    parser.add_argument("--min_seed_tokens", type=int, default=8)
    parser.add_argument("--max_seed_tokens", type=int, default=32)
    parser.add_argument("--tokenizer_model", type=str, default="intfloat/e5-large-v2")

    parser.add_argument(
        "--provider",
        type=str,
        choices=["dashscope", "deepseek", "openai"],
        default="dashscope",
    )
    parser.add_argument("--rewrite_model", type=str, default="qwen-plus-latest")
    parser.add_argument("--api_key_env", type=str, default=None)
    parser.add_argument("--api_base", type=str, default=None)
    parser.add_argument(
        "--api_mode",
        type=str,
        choices=["auto", "chat", "responses"],
        default="auto",
    )
    parser.add_argument("--max_rewrite_attempts", type=int, default=8)
    parser.add_argument("--length_tolerance", type=int, default=2)
    parser.add_argument("--request_timeout_sec", type=int, default=120)
    parser.add_argument("--api_retries", type=int, default=5)
    parser.add_argument("--sleep_sec", type=float, default=0.0)

    parser.add_argument("--semantic_check", action="store_true")
    parser.add_argument("--semantic_model", type=str, default="intfloat/e5-large-v2")
    parser.add_argument("--semantic_threshold", type=float, default=0.82)
    parser.add_argument("--semantic_device", type=str, default="auto")
    parser.add_argument("--semantic_retry", type=int, default=2)

    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--random_seed", type=int, default=2026)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger(args.output_dir)

    set_random_seed(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model, use_fast=True)
    seed_queries = load_seed_queries(
        dataset_name=args.dataset_name,
        split=args.dataset_split,
        seed_count=args.seed_count,
        tokenizer=tokenizer,
        min_seed_tokens=args.min_seed_tokens,
        max_seed_tokens=args.max_seed_tokens,
        random_seed=args.random_seed,
        logger=logger,
    )

    api_key_env, api_base, api_mode = resolve_provider_settings(
        provider=args.provider,
        api_key_env_override=args.api_key_env,
        api_base_override=args.api_base,
        api_mode_override=args.api_mode,
    )
    logger.info(
        "Provider=%s model=%s api_mode=%s api_base=%s api_key_env=%s",
        args.provider,
        args.rewrite_model,
        api_mode,
        api_base,
        api_key_env,
    )

    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(f"Missing API key. Set environment variable: {api_key_env}")
    client = build_openai_client(api_key, api_base)

    semantic_checker: Optional[SemanticChecker] = None
    if args.semantic_check:
        semantic_checker = SemanticChecker(
            model_path=args.semantic_model,
            tokenizer_path=args.tokenizer_model,
            device=args.semantic_device,
            logger=logger,
        )

    all_records: List[RewriteRecord] = []
    target_lengths = sorted(set(int(v) for v in args.target_lengths))
    logger.info("Building dataset for target lengths: %s", target_lengths)

    for seed_id, source_query in enumerate(seed_queries):
        source_tokens = count_tokens(source_query, tokenizer)
        logger.info("Seed %s/%s | source_tokens=%s", seed_id + 1, len(seed_queries), source_tokens)

        for target in target_lengths:
            category = f"t{target}"
            semantic_similarity: Optional[float] = None
            semantic_passed: Optional[bool] = None
            rewrite_query: Optional[str] = None
            rewrite_tokens = -1
            used_attempts = 0
            extra_constraint: Optional[str] = None

            for semantic_try in range(args.semantic_retry + 1):
                rewrite_query, rewrite_tokens, used_attempts = rewrite_to_target_length(
                    client=client,
                    model=args.rewrite_model,
                    api_mode=api_mode,
                    source_query=source_query,
                    target_tokens=target,
                    tokenizer=tokenizer,
                    tolerance=args.length_tolerance,
                    max_attempts=args.max_rewrite_attempts,
                    request_timeout_sec=args.request_timeout_sec,
                    api_retries=args.api_retries,
                    sleep_sec=args.sleep_sec,
                    logger=logger,
                    extra_constraint=extra_constraint,
                )

                if semantic_checker is None:
                    semantic_passed = None
                    break

                semantic_similarity = semantic_checker.cosine_similarity(source_query, rewrite_query)
                semantic_passed = semantic_similarity >= args.semantic_threshold
                if semantic_passed:
                    break

                extra_constraint = (
                    "The rewrite changed meaning too much. Keep entities, relation, and answer target unchanged."
                )
                logger.warning(
                    (
                        "Semantic check failed: seed=%s target=%s similarity=%.4f (< %.2f), "
                        "retry=%s/%s"
                    ),
                    seed_id,
                    target,
                    semantic_similarity,
                    args.semantic_threshold,
                    semantic_try + 1,
                    args.semantic_retry,
                )

            if rewrite_query is None:
                raise RuntimeError(f"Failed to generate rewrite for seed_id={seed_id} target={target}")

            record = RewriteRecord(
                seed_id=seed_id,
                target_tokens=target,
                category=category,
                source_query=source_query,
                query=rewrite_query,
                actual_tokens=rewrite_tokens,
                token_error=abs(rewrite_tokens - target),
                rewrite_attempts=used_attempts,
                semantic_similarity=semantic_similarity,
                semantic_passed=semantic_passed,
            )
            all_records.append(record)
            logger.info(
                "  -> target=%s actual=%s error=%s semantic=%s",
                target,
                rewrite_tokens,
                record.token_error,
                "n/a" if semantic_similarity is None else f"{semantic_similarity:.4f}",
            )

    if args.strict:
        failed_length = [r for r in all_records if r.token_error > args.length_tolerance]
        failed_semantic = [
            r for r in all_records if (r.semantic_passed is not None and not r.semantic_passed)
        ]
        if failed_length:
            raise RuntimeError(f"Strict mode failed: {len(failed_length)} records out of length tolerance.")
        if failed_semantic:
            raise RuntimeError(f"Strict mode failed: {len(failed_semantic)} records failed semantic check.")

    rows = [asdict(record) for record in all_records]
    jsonl_path = args.output_dir / "controlled_query_length_dataset.jsonl"
    csv_path = args.output_dir / "controlled_query_length_dataset.csv"
    write_jsonl(jsonl_path, rows)
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    buckets_dir = args.output_dir / "buckets"
    buckets_dir.mkdir(parents=True, exist_ok=True)
    by_category: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_category.setdefault(row["category"], []).append(row)

    for category, items in sorted(by_category.items(), key=lambda kv: int(kv[0][1:])):
        bucket_path = buckets_dir / f"{category}.jsonl"
        write_jsonl(bucket_path, items)

    summary = {
        "dataset_name": args.dataset_name,
        "dataset_split": args.dataset_split,
        "seed_count": args.seed_count,
        "target_lengths": target_lengths,
        "records": len(rows),
        "provider": args.provider,
        "rewrite_model": args.rewrite_model,
        "api_mode": api_mode,
        "api_base": api_base,
        "api_key_env": api_key_env,
        "tokenizer_model": args.tokenizer_model,
        "length_tolerance": args.length_tolerance,
        "semantic_check": args.semantic_check,
        "semantic_threshold": args.semantic_threshold if args.semantic_check else None,
        "random_seed": args.random_seed,
        "avg_token_error": float(np.mean([r["token_error"] for r in rows])),
        "max_token_error": int(np.max([r["token_error"] for r in rows])),
    }
    if args.semantic_check:
        semantic_values = [r["semantic_similarity"] for r in rows if r["semantic_similarity"] is not None]
        if semantic_values:
            summary["avg_semantic_similarity"] = float(np.mean(semantic_values))
            summary["min_semantic_similarity"] = float(np.min(semantic_values))
            summary["semantic_failures"] = int(
                sum(1 for row in rows if row["semantic_passed"] is False)
            )

    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    logger.info("Saved combined JSONL: %s", jsonl_path)
    logger.info("Saved combined CSV: %s", csv_path)
    logger.info("Saved per-category buckets: %s", buckets_dir)
    logger.info("Saved summary: %s", summary_path)
    logger.info("Dataset generation completed.")


if __name__ == "__main__":
    main()
