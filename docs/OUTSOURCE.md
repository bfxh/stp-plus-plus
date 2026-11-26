# 外包说明：跑完「缺口测试」的正例对照（9 个 run，约 1 小时）

这份文档可以整段发给接活的人或云端服务商，不需要额外背景。

---

## 0. 一句话

在一台有 NVIDIA GPU 的机器上跑 **9 个训练**（3 个臂 × 3 个 seed）就能回答核心问题；顺手把另外 9 个补齐则为 18 个。本机（RTX 4060 Ti 8GB）实测每个 run **约 3 分钟**（含加载与评测），**9 个约 30 分钟，18 个约 55 分钟**；加首次下载模型与切块，总占用 **1 小时上下**，云上成本 **< 3 美元**。

这份工作是回答一个特定的科学问题，**不是调模型、不是追指标**：判定标准见第 5 节。数值好坏都接受，要的是按协议跑完并如实回报。

---

## 1. 前置条件

| 项 | 要求 |
|---|---|
| GPU | **≥ 12GB 显存最省事**；8GB 也能跑但很容易 OOM（见第 6 节）；CPU 不行 |
| Python | 3.10–3.12，`torch` 带 CUDA（本项目在 2.11.0+cu128 上验证） |
| 磁盘 | ≥ 12GB（模型 2.4GB + 数据 + 缓存 + 9 个 adapter 各约 44MB） |
| 网络 | 需要能下载 HuggingFace 模型；在中国大陆可设 `STPP_HF_ENDPOINT=https://hf-mirror.com` |
| 时间 | 首次运行要下载模型 + 切语料块（约 5–10 分钟），之后每个 run 2–4 分钟 |

---

## 2. 照抄就能跑

```bash
git clone https://github.com/bfxh/stp-plus-plus
cd stp-plus-plus

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 官方仓库自带数据集（只需 datasets/ 下的 jsonl）
git clone --depth 1 https://github.com/galilai-group/llm-jepa data/llm-jepa

# 跑 9 个 run（决定性三臂）
STPP_CORE_ONLY=1 bash scripts/run_control.sh

# 或 18 个（把 MTP 剂量曲线一起补完）
bash scripts/run_control.sh

# 出表
python scripts/agg_pretrain.py
```

Windows（PowerShell）：

```powershell
git clone https://github.com/bfxh/stp-plus-plus; cd stp-plus-plus
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
git clone --depth 1 https://github.com/galilai-group/llm-jepa data/llm-jepa
$env:STPP_PYTHON = ".\.venv\Scripts\python.exe"
.\scripts\retry_mtp.ps1
python scripts\agg_pretrain.py
```

路径全部可覆盖（不用改代码）：

| 环境变量 | 默认 | 含义 |
|---|---|---|
| `STPP_MODEL` | `unsloth/Llama-3.2-1B-Instruct` | 模型路径或 HF id |
| `STPP_DATA_DIR` | `data/llm-jepa/datasets` | `*_train.jsonl` / `*_test.jsonl` 所在目录 |
| `STPP_CACHE_DIR` | `cache` | 切块缓存 |
| `STPP_HF_HOME` | `~/.cache/hf` | 模型缓存 |
| `STPP_HF_ENDPOINT` | 空（= 官方源） | 走镜像时填 `https://hf-mirror.com` |
| `STPP_PYTHON` | `python` | 跑 `.ps1` 助手脚本时用的解释器 |

---

## 3. 要跑的臂（其余臂已经在仓库里）

**决定性三臂（必须跑，9 个 run）**

| arm | 含义 |
|---|---|
| `ce33` | 基线：**CE 只监督每 3 个 token 中的一个**（其余标签置 −100），没有别的目标 |
| `ce33_mtp2` | 同样的 1/3 CE，**加上 multi-token prediction**：`h_t` 额外预测 `t+2`，权重 0.05 |
| `mtp2` | 全量 CE + 同一个 MTP 项，用来量"MTP 在监督充足时是不是纯成本" |

**加分项（想一次做完就一起跑，9 个 run）**

`mtp3_s37`、`ce25_mtp3_s82`、`ce25_mtp3_s23`（补齐已有的两个臂到 3 seed），
`mtp3_l` ×3、`ce25_mtp3_l` ×3（λ 从 0.15 降到 0.05 的剂量点）。
名字都在 `scripts/run_pretrain.py` 的 `ARMS` 里，脚本会自动跳过已经存在的 run。

背景一句话：仓库里的 `RESULTS.md §11` 已证明 STP 的几何目标在缺口测试里只补回 3%。但那个测法本身需要**正例对照**才能成立 —— 如果连一个确定携带信息的 MTP 都补不回来，说明测法测的是"学习成本"而不是"信息量"。这三条臂就是来分这个岔的。

---

## 4. 协议（**不要修改**）

```
steps=120  bs=4  block=512  lr=2e-4 cosine  warmup=3%
LoRA r=16 alpha=32 dropout=0.05（q,k,v,o,gate,up,down）
bf16 + gradient checkpointing，tap 第 15 层
seeds=82 23 37；评测 = 200 个 held-out 512-token block 的困惑度
```

这些值已写进 `scripts/run_pretrain.py` 的 `ARMS` 与 `scripts/run_control.sh` 的参数里，**直接跑脚本即可**。如果要改任何一项（步数、bs、λ、block），请在交付时单独标注，否则结果不能和仓库里的臂对比。

---

## 5. 验收标准（这就是全部）

跑完之后应该能拿到这样的表（数字会不一样，格式一致即可）：

```
arm           n        ppl
reg           3       3.36      # 仓库里已有，不需重跑
ce25          3       3.48      # 仓库里已有，不需重跑
ce33          3       3.4x      # 新增
ce33_mtp2     3       3.4x      # 新增
mtp2          3       3.3x      # 新增
```

**交付要求：**

1. 九个目录：`runs/pretrain/ce33_s{82,23,37}/`、`ce33_mtp2_s{82,23,37}/`、`mtp2_s{82,23,37}/`（若跑了加分项，再加对应的九个），每个里面至少有 `run.json`（含 `eval.ppl`）和 `train.jsonl`（≥ 5 行）；
2. `python scripts/agg_pretrain.py` 的完整 stdout；
3. 一句话结论：**`ce33_mtp2` 的 ppl 比 `ce33` 低了多少**（正数=更差，负数=更好）。

**判定（由我方来做，接活人只需如实回报）：** 若 `ce33_mtp2 − ce33 ≤ −0.05`，说明缺口测试有分辨力，STP 的零分是它自己的问题；若差值在 ±0.05 内或为正，则 `RESULTS.md §11` 的结论要降级为"净贡献≈0"，测法测的是成本不是信息。

---

## 6. 已知坑（都是实测踩过的）

1. **8GB 卡容易被 OOM 打死**：bf16 权重 2.5GB，MTP 的几个 CE 项各留一份 logits。若只有 8GB，用 `STPP_LOAD_8BIT=1 bash scripts/run_control.sh`（等价于给 `run_pretrain.py` 加 `--load_8bit`，需先 `pip install bitsandbytes`）。8-bit 结果只能作为子协议参考，**不能和 bf16 的臂直接比**。最好直接要 ≥12GB 的卡，省掉所有麻烦。
2. **不要设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`**：在 Windows/WDDM 上会让 OOM 来得更早；Linux 上可留空。
3. 首次运行慢是正常的（下载模型 + 切 15565 个 block），别以为卡住了。
4. 失败重试是安全的：`run_pretrain.py` 见到 `run.json` 就跳过，所以重跑只会补缺的。
5. 不要动 `runs/` 里已有的目录 —— 那是现成的对照组。

---

## 7. 交接物

- 代码与全部历史结果：<https://github.com/bfxh/stp-plus-plus>（公开）
- 全部结论与边界：仓库 `RESULTS.md`（454 行），先读 §11 与 §13 就够理解这件事
- 6.4GB 的 per-run 模型权重**不在仓库里**（`.gitignore` 排除），也不需要：每个 adapter 都能用上面的命令重新生成

---

## 8. 若外包给云端

任何按小时计费的 GPU 实例都够（T4 16GB / L4 24GB / A10 24GB / 4090 24GB）。整个任务 1 小时以内，按需开机即可，不需要常驻。交付前记得 `shutdown`，别让实例空转计费。
