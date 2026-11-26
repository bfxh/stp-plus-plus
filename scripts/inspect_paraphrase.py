import json, os, statistics

DS = r"F:\aiwork\llm-jepa-ref\datasets"
p = os.path.join(DS, "paraphrase_train.jsonl")
n = 0
lens = []
roles = set()
sample = None
with open(p, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        n += 1
        lens.append(len(" ".join(m["content"] for m in r["messages"])))
        roles.add(tuple(m["role"] for m in r["messages"]))
        if sample is None:
            sample = r
        if n >= 3000:
            break
print("rows scanned:", n)
print("roles:", list(roles)[:3])
print("chars per row: mean=%.0f median=%d p10=%d max=%d"
      % (statistics.mean(lens), statistics.median(lens),
         sorted(lens)[int(len(lens) * .1)], max(lens)))
for m in sample["messages"]:
    print("  [%s] %s" % (m["role"], m["content"][:120].replace("\n", " ")))
