import json, os

DS = r"F:\aiwork\llm-jepa-ref\datasets"
FILES = ["spider_train.jsonl", "gsm8k_train.jsonl", "hellaswag_train.jsonl",
         "nq_open_train.jsonl", "yelp_train.jsonl", "turk_train.jsonl",
         "rotten_tomatoes_train.jsonl"]

for f in FILES:
    p = os.path.join(DS, f)
    if not os.path.exists(p):
        continue
    rows = [json.loads(l) for l in open(p, encoding="utf-8")][:200]
    nmsg = [len(r["messages"]) for r in rows]
    ans = [len(r["messages"][-1]["content"]) for r in rows]
    first = rows[0]["messages"][-1]["content"][:70].replace("\n", "\\n")
    print("%-26s msgs=%d-%d  ans_len mean=%.0f p90=%d  first=%r"
          % (f, min(nmsg), max(nmsg), sum(ans) / len(ans),
             sorted(ans)[int(len(ans) * 0.9)], first))
