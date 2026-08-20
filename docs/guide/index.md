# 功能介绍 · 安装与总览

SynOmega 对外提供六项能力，本章每个功能一节，讲**使用方法**（命令行与 Python API）；每项的模型 /
算法原理见对应的[研究报告](../research/index.md)章节，两章一一对应：

| 功能 | 用法（本章） | 原理（研究报告） |
|---|---|---|
| 单步正向反应预测 | [↗](forward.md) | [↗](../research/forward.md) |
| 多组分演化预测 | [↗](evolution.md) | [↗](../research/evolution.md) |
| 单步逆向反应预测 | [↗](retro.md) | [↗](../research/retro.md) |
| 多步反应路径规划 | [↗](planning.md) | [↗](../research/planning.md) |
| 反应合理性评分 | [↗](plausibility.md) | [↗](../research/plausibility.md) |
| 可合成性评分 SynScore | [↗](synscore.md) | [↗](../research/synscore.md) |

## 安装

```bash
pip install synomega           # 核心：rdkit + numpy（纯模板后端可直接用）
pip install "synomega[gnn]"    # 追加 D-MPNN 神经单步后端（torch），神经能力推荐
```

神经后端是**可选 extra**：不装 torch 也能用纯模板规则后端；需要神经模板分类精度（正向 / 逆向 /
演化 / 合理性都基于它）时再装 `[gnn]`。默认模型权重与 ZINC 可购砌块集在**首次调用时按需下载**到
本地缓存（`~/.cache/synomega`，可用 `SYNOMEGA_CACHE` 覆盖；`SYNOMEGA_MIRROR=ustc|github` 选源），
不随 wheel 分发。可显式预取：`synomega download`。要求 Python ≥ 3.10。
