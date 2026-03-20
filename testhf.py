import os, json
from datasets import load_dataset

out = "/data/home/mazhenxiang/Hedra-RAG-EXP/HedraRAG/data/msmarco_2k.jsonl"
os.makedirs(os.path.dirname(out), exist_ok=True)

ds = load_dataset("Tevatron/msmarco-passage-corpus", split="train")
with open(out, "w", encoding="utf-8") as f:
      for x in ds:
          f.write(json.dumps({
              "title": x.get("title", ""),
              "text": x.get("text", "")
          }, ensure_ascii=False) + "\n")

print("saved_to:", os.path.abspath(out))
print("exists:", os.path.exists(out), "size_bytes:", os.path.getsize(out))