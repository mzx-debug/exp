基于 HedraRAG 流程的批大小扩展实验，测量 query batch size 对三个阶段平均每条 query 延迟的影响：

1. Embedding
2. Retrieval
3. Generation

使用 one-shot RAG，对不同 `batch_size`、`nprobe`、`llm_model` 组合。


- `batch_scaling_experiment.py`：主实验脚本
- `batch_scaling_config.yaml`：实验配置
- `run_full_experiment.sh`：批量多配置

    配置内容：
- 数据集：`natural_questions` 的 `test` split
- 检索编码模型：`intfloat/e5-large-v2`
- 索引：预构建 FAISS IVF 索引，默认路径 `/data/index_0319/msacro/IVF4096/ivf.index`
- 语料：`Tevatron/msmarco-passage-corpus`
- `nprobe`：`128, 256, 512`
- 生成模型：
  - `llama3-8B` -> `meta-llama/Llama-3.1-8B-Instruct`
  - `llama2-13B` -> `meta-llama/Llama-2-13b-hf`
  - `opt-30B` -> `facebook/opt-30b`
- `batch_size`：`1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048`
- 预热：`32` 条 query 做 warmup，不计入结果
- 输出目录： `../output/batch_scaling_experiment`



安装依赖：

```bash
pip install -r requirement.txt
```

## 运行示例

创建index：
 python -m heterag.retriever.index_builder \
    --retrieval_method e5 \
    --model_path intfloat/e5-large-v2 \
    --corpus_path Tevatron/msmarco-passage-corpus \
    --corpus_source huggingface \
    --save_dir ../data/index_0319/msacro/IVF4096 \
    --max_length 512 \
    --batch_size 512 \
    --use_fp16 \
    --pooling_method mean \
    --faiss_type "IVF4096,Flat"

先跑一个小测试：

```bash
python batch_scaling_experiment.py \
  --config batch_scaling_config.yaml \
  --index_path /data/home/mazhenxiang/Hedra-RAG-EXP/HedraRAG/data/index_0319/msacro/IVF4096/ivf.index \
  --nprobe 128 \
  --llm_model llama3-8B \
  --test_mode
```

如果 `retrieval.index_path` 是机器相关的绝对路径，可以用 `--index_path` 覆盖，不需要改 YAML。

跑单个完整配置：

```bash
python batch_scaling_experiment.py \
  --config batch_scaling_config.yaml \
  --nprobe 128 \
  --llm_model llama3-8B
```

限制最大批大小：

```bash
python batch_scaling_experiment.py \
  --config batch_scaling_config.yaml \
  --nprobe 128 \
  --llm_model llama3-8B \
  --max_batch_size 512
```

批量跑默认全部组合：

```bash
bash run_full_experiment.sh
```

只跑部分组合：

```bash
bash run_full_experiment.sh \
  --nprobe 128,256 \
  --llm llama3-8B,llama2-13B \
  --max_batch_size 512
```
简单实验：

```bash
bash run_full_experiment.sh \
  --nprobe 128,256,512 \
  --llm llama3-8B \
  --max_batch_size 1024
```

```bash
bash run_full_experiment.sh \
  --nprobe 128,256,512 \
  --llm llama2-13B \
  --max_batch_size 1024
```

```bash
bash run_full_experiment.sh \
  --nprobe 128,256,512 \
  --llm opt-30B \
  --max_batch_size 1024
```

"# Hedra-EXP" 
"# exp" 
