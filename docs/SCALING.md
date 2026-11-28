# 25. 数据到顶之后：从哪儿再挤（论文清单 + 可执行队列）

## 25.1 三面墙，先分清

| 墙 | 实测现状 | 还有多少 |
|---|---|---|
| 数据 | 地板 3.800，当前 3.879（11.8M token） | 0.079，且 ×4 数据只买 0.06 |
| 步时间 | 0.525 s/步（最优配置）、0.66（曲线配置） | GEMM 406 ms/83.2%，32.5 TFLOPS = 实测可达 48.5 的 67% |
| 容量 | LoRA r16 = 11.3M 参数 = 全模型 0.90% | 未知，但 Biderman 那篇说这是 code 上的硬约束 |

数据轴已排除（§7.1）。剩下两条：**优化器几何** 和 **容量**，外加一条纯速度的 **16F GEMM**。

## 25.2 论文清单（只留能在这张 4060 Ti 8GB 上兑现的）

| 方法 | 论文 | 报告的倍数 | 我们的适配性 | 判定 |
|---|---|---|---|---|
| **Muon** | 2502.16982（Moonshot） | compute-optimal 下只要 AdamW 的 **52% FLOPs**；Moonlight 3B/16B MoE 5.7T token 验证 | LoRA 的 A/B 都是 2D 矩阵，NS 迭代作用在 16×2048 上，微秒级；RMS 对齐后 `--lr` 含义不变 | **P0** |
| Muon（补充） | 2505.02222 | 在 compute-time Pareto 前沿上严格外扩；大 batch 下数据效率不塌（超过 critical batch size 仍有效） | 我们 bs 只能 4，吃不到大 batch 那一半收益 | 参考 |
| **AdEMAMix** | 2409.03137（ICLR'25） | 1.3B 模型 **101B token ≈ AdamW 197B（+95%）**，开销 <3%，drop-in | 纯 optimizer-state 改动，零结构改动；但 β3=0.9999 的窗口 ~10k 步，我们只跑 1920–3840 步，**只能覆盖 1/5–2/5 窗口** | **P1** |
| SOAP | 2409.11321 | 360M/660M 上比 AdamW 少 **≥40% 迭代 / ≥35% 墙钟**；还能抬高 critical batch size | 要维护预条件特征分解，矩阵小可行但超参多、易踩坑 | P3（备选） |
| Conda | 2509.24218 | 列归一化 Adam，LLaMA-**1B** 上 **2.7× 步数 / 2.5× 时间**，并压过 Muon 1.69× | 实现极简（对 Adam 更新做列归一）；但只有一篇论文、复现证据薄 | P2（待复核） |
| **Fantastic Pretraining Optimizers…** | ICLR 2026 | 1.2B 上 Muon / NAdamW / SOAP 的加速**全部衰减到 ≈1.1×**，下游指标不再提升 | **这是对我们最不利的一条**：1B 规模别指望 2×，能拿到 1.1–1.3× 就算兑现 | 预期校准 |
| LoRA Learns Less and Forgets Less | 2405.09673（Biderman） | code CPT 上 r=256 才追平全参 20B token 的效果；r16/r64 从第 1 个 epoch 就落后；全参的 ΔW 远离低秩 | 我们 r16、11.3M(0.90%)、评的正是 code+chat+zh 的 ppl —— **3.80 的地板很可能就是容量地板** | **P4** |
| Expressive power of LoRA | 2310.17513 | Transformer 要表达同尺寸目标需要 rank ≥ d/2（我们 d=2048 → 1024） | 理论上界，实际任务远小于此 | 参考 |
| Procedural Knowledge Is Not Low-Rank | 2607.21612 | 全参 ΔW 有效秩 761–1026；r128 只覆盖 43–51% 的 Frobenius 能量；r 从 32→128 提升微弱 | 说明"加秩"收益递减很快，别指望 r16→r64 解决一切 | 预期校准 |
| MiniCPM / WSD | 2404.06395 | WSD：decay 段陡降、可无限续训、可随时取 checkpoint；但另一篇（Index SLM, 2607.09885）对照说 cosine/linear/WSD **收敛值基本一致** | 对我们的价值是"续训不用重跑"，不是"同 token 更低 loss" | P5 |

## 25.3 为什么论文倍数不能直接搬过来（三条硬约束）

1. **只训 11.3M 参数**。Muon 的正交化作用在 LoRA 的 A/B 因子上，不是全权重矩阵；论文里的 2× 是全参预训练。二阶几何在低秩子空间里是否还成立，没人测过。
2. **1B + 短程**。AdEMAMix 的收益来自"几十万步之前"的梯度（β3=0.9999），我们 1920 步的总长度只有它记忆窗口的 1/5；Muon 的 2× 也是 compute-optimal（~20 token/参数）下的，我们是 11.8M/1B ≈ 0.012 token/参数，处在完全不同的区间。
3. **判定阈值**。旧集三 seed 极差 0.013 → 只有 **ppl 差 > 0.02** 才算真信号；单 seed 的 0.01 级差异一律当噪声。

## 25.4 队列（代码已就位）

| 序 | 实验 | 命令要点 | 成本 | 判据 |
|---|---|---|---|---|
| P0 | Muon | `--optim muon` (RMS 对齐, momentum 0.95, NS 5 步) | 21 min | ppl < 4.004 |
| P1 | AdEMAMix | `--optim ademamix --ademamix_alpha 5` | 21 min | ppl < 4.004 |
| P2 | Conda | 待复核公式后实现 | — | — |
| P3 | 16F GEMM | `--fp16`（编译后 GEMM 占 83%，与早期 elementwise 主导时的结论不同，需重测） | 21 min | 步时间 < 0.53 s 且 ppl 不掉 |
| P4 | 容量 | `--lora_r 32 --rslora`（α/√r 缩放） | 21 min | ppl 明显 < 4.004 |
| P5 | WSD 续训 | 从 3840 的 checkpoint 续到 5760，只跑 decay 段 | 20 min | 与一次性 5760 比 |

全部在 8.85M 大语料、1920 步、seed 82、bs4、block512、compile 上跑，与 `runs/big/steps1920_s82`（**ppl 4.004**）对照。

## 25.5 结果回填

| 臂 | ppl | Δ vs 4.004 | 步时间 | 备注 |
|---|---|---|---|---|
| AdamW（基线） | 4.004 | — | 0.657 s | `runs/big/steps1920_s82` |
| Muon | 待填 | | | |
| AdEMAMix | 待填 | | | |
| fp16/16F | 待填 | | | |
| r32+rsLoRA | 待填 | | | |
