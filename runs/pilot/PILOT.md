# STP++ pilot (Llama-3.2-1B, LoRA, 250 steps, 1 seed)

Single-GPU LoRA pilot. Absolute numbers are low by construction — read the *differences between arms* at the same budget, not the absolute level.

| arm | data fraction | exact match | TF seq | TF token | unique | collision | curv evidence (m=1) |
|---|---|---|---|---|---|---|---|
| `curv_budget` | 0.0625 | 0.425 | 0.427 | 0.931 | 0.99 | 0.00 | 8.96 |
| `curv_budget` | 0.25 | 0.480 | 0.453 | 0.942 | 0.99 | 0.00 | 8.59 |
| `reg` | 0.0625 | 0.400 | 0.403 | 0.925 | 0.99 | 0.00 | 7.58 |
| `reg` | 0.25 | 0.475 | 0.453 | 0.943 | 0.99 | 0.00 | 9.46 |
| `stp` | 0.0625 | 0.345 | 0.363 | 0.923 | 0.99 | 0.00 | 8.55 |
| `stp` | 0.25 | 0.425 | 0.403 | 0.938 | 0.99 | 0.00 | 8.88 |

## 怎么读

- **exact match** 是数据效率曲线的主轴；同分数下用更少数据的 arm 赢。
- **curvature evidence** = `mse_linear / mse_mlp`。>1 说明隐状态轨迹是曲线，STP 的"局部直线"假设只是零阶近似。
- **collision** 是去中心化端点余弦 > 0.99 的样本对占比；STP 声称能防模式坍塌，这里是实测。
- 单 seed、LoRA、300 条测试集 —— 只能当方向性证据，不能当论文数字。
