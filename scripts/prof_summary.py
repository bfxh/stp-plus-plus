import re, sys, collections

path = sys.argv[1]
rows = []
for line in open(path, encoding="utf-8", errors="ignore"):
    if "ms" not in line and "us" not in line:
        continue
    parts = [p for p in re.split(r"  +", line.strip()) if p]
    if len(parts) < 6 or not re.match(r"^[-+]?\d", parts[-1]):
        continue
    # columns: name, selfCPU%, selfCPU, CPUtotal%, CPUtotal, CPUavg, selfCUDA, selfCUDA%, CUDAtotal, CUDAavg, calls
    # name may itself contain spaces -> rejoin from the right
    tail = parts[-11:]
    if len(tail) < 11:
        continue
    name = " ".join(parts[:len(parts) - 11]) if len(parts) > 11 else tail[0]
    num = tail[1:]
    def to_ms(s):
        s = s.replace(",", "")
        if s.endswith("ms"):
            return float(s[:-2])
        if s.endswith("us"):
            return float(s[:-2]) / 1000.0
        if s.endswith("s"):
            return float(s[:-1]) * 1000.0
        return 0.0
    try:
        self_cuda = to_ms(num[5])
        calls = int(num[9].replace(",", ""))
    except Exception:
        continue
    rows.append((name, self_cuda, calls))

cats = collections.defaultdict(lambda: [0.0, 0])
def classify(n):
    ln = n.lower()
    if "gemm" in ln or "cutlass" in ln: return "GEMM"
    if "softmax" in ln: return "softmax"
    if "dropout" in ln: return "dropout"
    if "elementwise" in ln or "vectorized" in ln: return "elementwise"
    if "reduce" in ln or "norm" in ln: return "reduce/norm"
    if "copy" in ln or "memcpy" in ln or "fill" in ln: return "copy/fill"
    if "sdpa" in ln or "attention" in ln or "flash" in ln: return "attention"
    return "other"

for name, ms, calls in rows:
    c = cats[classify(name)]
    c[0] += ms
    c[1] += calls

total = sum(v[0] for v in cats.values())
print(f"profiled window: 3 steps, total self-CUDA = {total:.0f} ms  -> {total/3:.0f} ms/step\n")
print(f"{'category':<14}{'ms/step':>9}{'%':>7}{'calls/step':>12}")
for k, v in sorted(cats.items(), key=lambda kv: -kv[1][0]):
    print(f"{k:<14}{v[0]/3:9.1f}{v[0]/total*100:6.1f}%{v[1]/3:12.0f}")
print("\ntop 12 individual kernels (ms/step):")
for name, ms, calls in sorted(rows, key=lambda r: -r[1])[:12]:
    print(f"  {ms/3:8.1f}  {calls/3:5.0f}x  {name[:70]}")
