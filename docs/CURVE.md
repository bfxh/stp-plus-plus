# L0 曲线（定稿）：ppl 随步数的衰减

设置：Llama-3.2-1B-Instruct + LoRA r16/α32（可训练 0.90%），bs4 × block512 = 2048 tok/步，三分域混合语料（code_search_net-python / ultrachat_200k / wikipedia-zh），lr 2e-4，eval 200 blocks；训练在 Linux + `torch.compile`（曲线配置 0.598 s/步），三个 seed（82 / 23 / 37）。

## 1. 测量值

| 步数 | seed 82 | seed 23 | seed 37 | 均值 | 相对上一档 |
|---|---|---|---|---|---|
| 240 | 6.397 | 6.356 | 6.387 | 6.380 | — |
| 480 | 6.171 | 6.133 | 6.169 | 6.158 | −3.48% |
| 960 | 5.9889 | 5.9773 | 6.0017 | 5.989 | −2.74% |
| 1920 | 5.849 | 5.837 | 5.850 | **5.845** | −2.40% |

三个 seed 在 1920 档落在 5.837–5.850，一致到小数点后两位。

## 2. 拟合

`ppl = 5.2972 + 6.4591 · N^(−0.326)`（四点最小二乘，网格搜 b，sse=1.9e−5）

| N | 三 seed 均值 | 拟合 | 残差 |
|---|---|---|---|
| 240 | 6.380 | 6.379 | −0.001 |
| 480 | 6.158 | 6.160 | +0.003 |
| 960 | 5.989 | 5.986 | −0.003 |
| 1920 | 5.845 | 5.847 | +0.001 |
| 3840 | 待测 | **5.735（外推）** | — |

三点版（`5.451 + 8.051·N^-0.394`）的 1920 预测是 5.860、实测 5.845；把 1920 加进去后底从 5.45 降到 5.30，说明三点拟合把衰减估快了。

## 3. 怎么读

- 每翻一倍的收益：−3.48% → −2.74% → −2.40%，在收窄但没停。
- 底是 **5.297**：这条 1B + LoRA 在这份混合语料上，再怎么加步数只能逼近 ~5.30，不会到 0。
- 加步数仍然有效，但边际递减；要更大的收益只能动数据侧。§16.8 已经证明：难数据选择本身不赚，**多给该域 token（配比）才赚**。

## 4. 同样墙钟能换到多少 loss

| T | 配置 | 步数 | 预测 ppl |
|---|---|---|---|
| 11 min | Windows（曲线配置 1.486 s/步） | 444 | 6.180 |
| 11 min | Linux+compile（曲线配置 0.691 s/步） | 955 | 5.990 |
| 11 min | Linux+compile（最优配置 0.525 s/步） | 1257 | **5.935** |
| 22 min | Windows | 888 | 6.006 |
| 22 min | Linux+compile（曲线配置） | 1910 | 5.861 |
| 22 min | Linux+compile（最优配置） | 2514 | **5.819** |

同样 11 分钟：6.180 → 5.935，**−0.245 ppl（−4.0%）**。这是 2.6× 的真实兑现方式。

## 5. 复现

```bash
export PATH=/home/lbx13/stpenv/bin:/usr/local/bin:/usr/bin:/bin
cd /mnt/c/Users/lbx13/Doubao/chats/2026-11-24/new-chat-7/stp-plus-plus
STPP_MODEL=/mnt/f/aiwork/models/Llama-3.2-1B-Instruct \
STPP_CACHE_DIR=/mnt/f/aiwork/mixed_cache \
STPP_SKIP_SAVE=1 STPP_LOGIT_CHUNK=512 STPP_COMPILE=1 \
python -m stpp.pretrain --out_dir runs/curve1920/selall_s82 --steps 1920 \
  --bs 4 --seed 82 --log_every 40 --lora_dropout 0.05
```

- 三 seed 一键：`bash scripts/run_curve1920.sh`
- 结果：`runs/curve1920/selall_s{82,23,37}/run.json`（`eval.ppl`）
- 拟合：`python scripts/fit_curve.py`
- 步时间/瓶颈/加速的全部实测：`docs/MFU.md` §24.11–24.13
