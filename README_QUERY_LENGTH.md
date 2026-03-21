# Query Length Experiment Framework

A comprehensive experimental framework for analyzing the impact of query length on RAG system performance.

## Quick Start

### 3-Level Classification (Fast)
```bash
python query_length_experiment.py \
  --config query_length_config.yaml \
  --nprobe 128 \
  --llm_model llama3-8B \
  --test_mode
```

### 5-Level Classification (Recommended)
```bash
python query_length_experiment.py \
  --config query_length_config_5level.yaml \
  --nprobe 128 \
  --llm_model llama3-8B \
  --test_mode
```

## Project Structure

### Core Scripts
- **query_length_experiment.py** - Main experiment framework
  - Loads dataset and models
  - Classifies queries by token length
  - Measures embedding/retrieval/generation performance
  - Generates CSV/JSON results and plots

- **analyze_query_length_results.py** - Results analysis
  - Loads and prints summary statistics
  - Generates comparison plots
  - Exports markdown reports

- **analyze_token_distribution.py** - Token analysis
  - Analyzes token length distribution
  - Suggests optimal classification thresholds
  - Generates visualization plots

### Configuration Files
- **query_length_config.yaml** - 3-level classification config
  - Categories: short/medium/long
  - Sample size: 600 queries
  - Runtime: 30-60 minutes

- **query_length_config_5level.yaml** - 5-level classification config
  - Categories: very_short/short/medium/long/very_long
  - Sample size: 1000 queries
  - Runtime: 60-90 minutes
  - Recommended for detailed analysis

## Classification Methods

### 3-Level (Fast Verification)
| Level | Token Range | Share |
|-------|-------------|-------|
| short | 0-10 | ~40% |
| medium | 11-20 | ~40% |
| long | 21+ | ~20% |

### 5-Level (Detailed Analysis)
| Level | Token Range | Share | Description |
|-------|-------------|-------|-------------|
| very_short | 0-6 | ~20% | Ultra-short queries |
| short | 7-10 | ~25% | Short queries |
| medium | 11-15 | ~25% | Medium queries |
| long | 16-25 | ~20% | Long queries |
| very_long | 26+ | ~10% | Very long queries |

## Output Files

Results are saved to:
- `../output/query_length_experiment/` (3-level)
- `../output/query_length_experiment_5level/` (5-level)

Each includes:
- `results_qlength_*.csv` - Performance metrics by category
- `detailed_log_qlength_*.json` - Detailed configuration and results
- `experiment_log_qlength_*.log` - Execution logs
- `plots/*.svg` - Visualization graphs

## Performance Metrics

For each query length category:
- **avg_token_len**: Average token count
- **avg_embedding_time_ms**: Query embedding latency
- **avg_retrieval_time_ms**: Document retrieval latency
- **avg_generation_time_ms**: Answer generation latency
- **total_time_ms**: Total end-to-end latency
- **throughput_qps**: Queries per second

## Expected Results

Typical performance curve (NQ dataset):
- Embedding time: +150% (very_short → very_long)
- Retrieval time: ±2% (stable)
- Generation time: ±1% (stable)
- Total latency: +7% (very_short → very_long)
- Throughput: -7% (very_short → very_long)

## Configuration Parameters

Key settings in YAML files:

```yaml
query_length_analysis:
  batch_size: 128              # Fixed batch size
  sample_size: 200             # Samples per category
  categories: {...}            # Classification thresholds
  print_distribution: true     # Show token distribution
```

## Usage Examples

### Full experiment (90 minutes)
```bash
python query_length_experiment.py \
  --config query_length_config_5level.yaml \
  --nprobe 128 \
  --llm_model llama3-8B
```

### Multiple nprobe values
```bash
for nprobe in 128 256 512; do
  python query_length_experiment.py \
    --config query_length_config_5level.yaml \
    --nprobe $nprobe \
    --llm_model llama3-8B
done
```

### Analyze results
```bash
python analyze_query_length_results.py
# Generates: report.md, comparison.png
```

### Analyze token distribution
```bash
python analyze_token_distribution.py \
  --dataset natural_questions \
  --sample_size 10000
# Outputs: optimal thresholds and distribution plots
```

## Dependencies

Required packages:
- torch, transformers
- faiss
- vllm
- pandas, numpy
- pyyaml, matplotlib

See `batch_scaling_experiment.py` for complete imports.

## Notes

- All experiments use fixed random seed (2026) for reproducibility
- Token counting uses e5-tokenizer
- Results are deterministic within the same system
- GPU memory requirements: ~40GB (2 GPUs recommended)

## Related Work

This framework extends `batch_scaling_experiment.py` to study query length effects
rather than batch size effects on RAG system performance.

## Version

- 2.0 - 5-level classification support
- 1.0 - Initial 3-level classification
