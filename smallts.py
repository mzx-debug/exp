import json
from datasets import load_dataset

ds = load_dataset("Tevatron/msmarco-passage-corpus", split="train[:2000]")
out = "data/msmarco_2k.jsonl"
with open(out, "w", encoding="utf-8") as f:
      for x in ds:
          f.write(json.dumps({
              "title": x.get("title", ""),
              "text": x.get("text", "")
          }, ensure_ascii=False) + "\n")
print(out)
