# STP++

> **待认领的任务 → [Issues #1](https://github.com/bfxh/stp-plus-plus/issues/1)**：跑 9 个 run（约 30 分钟，需 ≥12GB 显卡），把 §11 的正例对照补完。任务书：[docs/OUTSOURCE.md](docs/OUTSOURCE.md)。

把 **Semantic Tube Prediction**（Huang / LeCun / Balestriero, [arXiv:2602.22617](https://arxiv.org/abs/2602.22617)）复现出来，并在它最脆的几处换成更硬的假设，做成可一键消融的单卡实验台。

> 官方实现（[galilai-group/llm-jepa](https://github.com/galilai-group/llm-jepa)）是 8 卡 FSDP + HF Trainer 的 1700 行脚本。
> 这里是单卡 8GB 显存可跑的精简版，`stpp.geometry.stp_random_span` 与官方 `linear="random_span"` 的算子逐项对齐。

---

## 1. 它到底在优化什么

官方 STP（`linear="random_span"`）的算子：

```
v_patch = h[p1-1] - h[p0-1]                       # 随机切出的一段
v_out   = (h[p0-1] - h[start-1]) + (h[end-1] - h[p1-1])
L_STP   = 1 - cos(v_out, v_patch)                 # 段内位移应和段外位移共线
L       = L_NTP + λ · L_STP
```

论文版（三点式）也实现了：

```
L = 1 - cos(h_t - h_r, h_r - h_s),  s < r < t
```

两者的共同假设是 **"无误差的隐状态轨迹是局部直线"**。这就是我们要撬的地方。

---

## 2. 九条手臂（每条对应一个可证伪的论点）

| arm | 论点 | 实现 |
|---|---|---|
| `reg` | 基线：只用 NTP | — |
| `stp` | 复现官方 16× | `stp_random_span`, λ=0.02 |
| `stp_whiten` | 原始 cos 被残差流里几个超大 outlier 维度绑架；白化后才是在量"语义夹角" | 逐维 z-score（运行统计）后再算余弦 |
| `curv` | 官方自带的曲率基线：惩罚所有转角 | `curvature_angle` |
| `curv_budget` | **直线假设是错的**：真实轨迹是平滑曲线，不该把正常弯曲也罚掉 | 无量纲离散曲率 `κ = ‖perp‖/‖chord‖`，只罚超出预算 τ 的部分，τ 自动跟到观测中位数 |
| `stpp` | 全套：曲率预算 + 白化 + 跨样本分离 + 零调参 λ | 上面三项 + VICReg 式 repulsion + `loss_ratio` 自适应 λ |
| `stpp_pred` | **别再硬编码轨迹形状，让一个头把它学出来**：残差 2 层 MLP + stop-grad 目标（JEPA 式） | `LatentPredictor`，`L = 1 − cos(g(h_t), sg(h_{t+k}))`, k∈{1,2,3} |
| `stp_share3` | **同样的 STP 几何，只换控制器**：λ 不取常数，而是把辅助项钉在总损失的 3% | `adaptive_lambda=loss_ratio`, target_ratio=0.03 |
| `curv_tau50` | **剂量反应**：同样是曲率，把 τ 放到自然中位数（0.77）以下，让有界力真正发力 | `curvature_budget`, τ=0.5 固定 |
| `goal_pred` | **替换式**：不问"下一步往哪拐"，问"这个答案要走到哪"——从段内任意位置预测答案末尾表征（stop-grad） | `goal_prediction` + `LatentPredictor` |

关键区别在 `curv` 和 `curv_budget`：前者（官方仓库里就有）把轨迹往死里掰直；后者承认弯曲、只挤掉超出预算的噪声。**如果后继工作说的"3 层 MLP 比线性外推好 3–12×"成立，`curv_budget` 就该赢 `curv` 和 `stp`。** 实测：它没赢，但它**没输**；而 `curv` 输了 2.4 点、`stp` 输了 4.3 点 —— "有界 vs 无界"这个区分是被证实了的，"能赢"这个预期是被证伪了的。

---

## 3. 安装

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

模型默认 `unsloth/Llama-3.2-1B-Instruct`（免 gated 的镜像，显存约 5.5 GB），首次运行自动下载。
在中国大陆可 `export STPP_HF_ENDPOINT=https://hf-mirror.com` 走镜像。

**所有本机路径都是可覆盖的环境变量**，代码里不再有绝对路径：

| 变量 | 默认 | 用途 |
|---|---|---|
| `STPP_MODEL` | `unsloth/Llama-3.2-1B-Instruct` | 模型目录或 HF id |
| `STPP_DATA_DIR` | `data/llm-jepa/datasets` | 官方仓库的 `*_{train,test}.jsonl` |
| `STPP_CACHE_DIR` | `cache` | 预训练语料的 512-token 切块缓存 |
| `STPP_HF_HOME` | `~/.cache/hf` | 模型缓存目录 |
| `STPP_HF_ENDPOINT` | 空（官方源） | 镜像地址 |
| `STPP_PYTHON` | `python` | `.ps1` 助手脚本用的解释器 |

> 这台开发机若要复跑，先设 `STPP_MODEL=F:\aiwork\models\Llama-3.2-1B-Instruct`、`STPP_DATA_DIR=F:\aiwork\llm-jepa-ref\datasets`、`STPP_HF_HOME=F:\aiwork\hf`，否则会把 2.4GB 权重重新下到 C:（C: 只剩 5.9GB）。

**只想跑某一组实验、不想读全文？** 见 `docs/OUTSOURCE.md`：那是给外人的一页任务书（含协议常数、验收标准、已知坑、时间与成本）。

---

## 4. 跑

```bash
# 任意平台（Linux 示例；Windows 把 python 换成 .\.venv\Scripts\python.exe）
export PYTHONPATH=$(pwd)

# 单次训练
python -m stpp.train --geom_kind stp_random_span --lbd 0.02 \
    --train_file data/llm-jepa/datasets/synth_train.jsonl \
    --out_dir runs/stp_seed82 --epochs 4 --seed 82 --bs 8 --lr 2e-4

# 评测（exact-match + 碰撞率 + 轨迹 dump）
python -m stpp.evaluate --adapter runs/stp_seed82/adapter \
    --test_file data/llm-jepa/datasets/synth_test.jsonl \
    --out runs/stp_seed82/eval.json --limit 300 --dump_traj

# 直线 vs 曲线检验
python -m stpp.probe --traj runs/stp_seed82/eval.traj.npz --out runs/stp_seed82/probe.json

# 矩阵（冒烟 / 正式）
python scripts/run_matrix.py --smoke
python scripts/run_matrix.py --fractions 1.0 0.25 0.0625 --seeds 82 23 37 --epochs 4

# 预训练区制 + 缺口测试（§10–§13）
python scripts/run_pretrain.py --arms reg stp goal_pred --seeds 82 23 37
```

---

## 5. 实测（五 seed 配对，Llama-3.2-1B / LoRA / 250 步 / 1-16 数据 / 300 条测试）

| arm | seq acc | 配对差 vs reg | t (df=4) | 判定 |
|---|---|---|---|---|
| `reg` | 0.4520 ± 0.0347 | — | — | 基线 |
| `stp` | 0.4087 ± 0.0534 | **−4.33** | **−3.66** | **显著变差** |
| `curv_budget` | 0.4493 ± 0.0348 | −0.27 | −0.25 | 打平 |
| `stpp` | 0.4447 ± 0.0370 | −0.73 | −0.55 | 打平 |
| `stpp_pred` | 0.4067 ± 0.0359 | −4.53 | −2.34 | 掉点（未达显著） |
| `stp_share3` | 0.4140 ± 0.0590 | −3.80 | −2.48 | 掉点（未达显著） |
| `curv_tau50` | **0.4547 ± 0.0313** | **+0.27** | +0.20 | 打平（全部实验的最好成绩） |
| `curv` | 0.4280 ± 0.0341 | −2.40 | **−3.54** | **显著变差** |
| `goal_pred` | 0.3973 ± 0.0611 | −5.47 | −1.99 | 掉点（替换式，目标 cos 0.96 仍输） |

三条结论：

1. **原版 STP 稳定掉 4.3 点，配对检验显著**；九个臂里没有一个超过朴素微调，最好的是 +0.27 点（等于没做）。
2. **分界线不是"力多大"，而是"是否让表征额外携带信息"**：`goal_pred`（预测答案终点，cos 0.96 几乎完美达成、只占损失 7.4%）掉 **5.47** 点；`curv_tau50`（只要求别比自然弯曲更弯，占损失 17.5%）**+0.27** 点。同样的量级、相反的符号。而把 STP 的 λ 压 26 倍只从 −4.33 挪到 −3.80 —— 权重也不是病根。
3. **STP 那个"轨迹变直"的目标，在两个区制下都从未达成**（`aux` 停在 1.65，cos ≈ −0.65），且 probe 显示 3 步跨度上线性外推的误差是 MLP 的 19.5–22 倍。自然语料（nq_open）在 250 步下全部贴 2–4% 地板，效应不可测。

完整判定与机制在 **`RESULTS.md`**。上一版单 seed 的数字保留在 `runs/pilot/PILOT.md`，作为"单 seed 不可当真"的实物证据。

### 5b. 换到预训练区制才把两者分开（全 token 监督 / 困惑度 / CE 未收敛）

九条臂全在记忆区跑（CE 塌到 0.008），而 STP 的叙事是预训练。补一组真像预训练的：15565 个 512-token block、120 步、bs4、3 seed，CE 从 4.29 只走到 1.21。

| arm | λ | mean ppl | Δ vs reg | 相对 | t (df=2) | 判定 |
|---|---|---|---|---|---|---|
| `reg` | 0 | 3.3616 | — | — | — | 基线 |
| `stp` | 0.02 | 3.3613 | −0.0003 | −0.0% | −1.62 | 空转（无效剂量） |
| `goal_pred` | 0.02 | 3.3608 | −0.0008 | −0.0% | −2.78 | 空转（无效剂量） |
| `stp_l0.5` | 0.5 | **3.4520** | **+0.0904** | **+2.7%** | **+14.90** | **显著变差** |
| `goal_l0.5` | 0.5 | 3.3631 | +0.0015 | +0.0% | +1.38 | 噪声级（几乎免费） |

**同一个 λ=0.5、同一批 seed、同一份语料：STP 稳定伤害语言建模（+2.7% ppl，t≈15），替换版被完全吸收（+0.04%，不显著）。** 这是全项目唯一一处能明确区分"我这条"和"它那条"的结果 —— 但注意是**免费**、不是**有益**：没有任何 seed 真变好。

### 5c. 正面测试"它到底携带不携带信息"（§11）

不叠加了，改成**抽掉信息**：CE 只监督每 4 个 token 中的一个，剩下 3/4 的位置没有任何 token 级目标，用来量出"缺口"有多大；再看 latent 目标能不能补。

| arm | mean ppl | vs `reg` | vs `ce25` | t | 判定 |
|---|---|---|---|---|---|
| `reg`（全 CE） | 3.3616 | — | — | — | 参照 |
| `ce25`（CE→1/4） | 3.4830 | +0.1214（+3.6%） | — | +18.4 | 缺口是真的 |
| `ce25_goal` | 3.4868 | +0.1252 | **+0.0038（填回 3%）** | +1.90 | 无信息 |
| `ce25_stp` | 3.5540 | +0.1924 | **+0.0710（挖大 58%）** | +18.7 | 显著更差 |

**结论（已经校准过）：latent 几何目标补不回 CE 拿不到的信息，而且 STP 是负号。** 更强的读法是 §13 的正例对照：真正携带新信息的 MTP（w=0.05，3 seed）不但没填回缺口，反而把缺口从 +0.121 挖到 **+0.157（t=+13.66，显著）**；全监督下也净掉 +0.165。**这个预算里任何额外目标都是净损失，STP 的"只填回 3%"已经是最好成绩** —— 所以这一节证明的是"净贡献≈0（成本主导）"，而不是"信息量为零"。STP 的 16× 不可能来自"补上 CE 拿不到的信息"——在它唯一能兑现的场景里它让结果更差。

| 追加（§13） | mean ppl | 配对 Δ | t |
|---|---|---|---|
| `ce33`（CE→1/3） | 3.4523 | vs `ce25` −0.0308 | −19.0 |
| `ce33_mtp2`（1/3 CE + MTP t+2, w0.05） | 3.6095 | **vs `ce33` +0.1572** | +13.7 |
| `mtp2`（全 CE + MTP t+2, w0.05） | 3.5273 | vs `reg` +0.1654 | +19.2 |


### 5d. 论文主图本身：数据效率曲线（§12）

论文的主张是"1/16 数据 + STP ≈ 全量数据 + 朴素微调"。补齐中间点后的三点曲线（等算力 250 步、300 条测试）：

| fraction | `reg` | `stp` | `goal_pred` |
|---|---|---|---|
| 1/16（500 条 × 8 epoch） | 0.444 ± 0.037 | **0.401 ± 0.051** | 0.397 ± 0.061 |
| 1/4（2000 条 × 2 epoch） | 0.478 ± 0.029 | 0.451 ± 0.066 | 0.493 ± 0.064 |
| 全量（8000 条 × 0.5 epoch） | 0.530 ± 0.081 | 0.517 ± 0.087 | 0.513 ± 0.103 |

- **16× 的直接检验：`stp@1/16 = 0.401` vs `reg@全量 = 0.530`，差 12.9 点。**
- 配对差随数据量单调收缩：**−4.33（显著）→ −2.00 → −1.33**。**它伤得最重的地方，正是它声称最擅长的地方。**
- 这个任务的数据跨度本身只有 8.6 点（0.444→0.530），而 STP 的效应是 −4.3：方向与幅度都不对。
- 外推的等效数据倍数是 **1.47× 全量**（需要更多数据，不是更少）；`reg` 自身标定为 1.10×，故只能读方向。

---

## 6. 指标怎么读

| 指标 | 含义 | 期望 |
|---|---|---|
| `exact_match` | 贪心解码后正则完全匹配 | 数据效率曲线的主轴 |
| `unique_ratio` | 生成结果去重率 | 掉下去 = 模式坍塌 |
| `collision_rate` | 去中心化后端点余弦 > 0.99 的样本对占比 | STP 声称能防，但没实测过 |
| `curvature_evidence` | `mse_linear / mse_mlp`（m=1..3） | **>1 说明轨迹是曲线**，直线先验站不住 |

`curvature_evidence` 是这套代码里最有信息量的一个数：它把你对 STP 核心假设的判断变成一个可以打印出来的比值。

---

## 7. 已知局限（别自欺）

- 单卡 LoRA（r=16）不是论文的全量微调，绝对数值会低；结论只看**同预算下 arm 之间的相对差**。
- 合成数据集 NL-RX-SYNTH 本身几何感很强，`1/16` 在这里容易出漂亮数字；自然语料的迁移需要换 `--train_file`。
- `probe.py` 的锚点是按 token 均匀取的，不是论文那种推理步边界；对 synth 够用，对数学推理链要换成 `<|step|>`（`--anchors step`）。
- 曲率预算的 τ 是 EMA 自适应的，前 50 步不稳定；写论文需要给 τ 收敛曲线（`train.jsonl` 里已经记了 `tau`）。
