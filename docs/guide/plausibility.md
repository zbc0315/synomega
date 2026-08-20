# 反应合理性评分

**做什么**：给一条候选反应（一组反应物 → 一个产物）打一个 0–1 的**合理性**分——"这些反应物到底
能不能真的给出这个产物"。主要用来在单步拆分里**过滤**明显不合理的候选（只删、不重排存活项）。
模型（双塔 D-MPNN，无需原子映射）与评测见[研究报告 · 反应合理性评分](../research/plausibility.md)；
安装见[安装与总览](index.md)。

**默认关闭**：评测显示它对单步 top-k 召回是净负收益、还加延迟，所以软件里默认不启用，需要时显式开。

## 用法一：挂进规划器，过滤每一步

```python
import synomega

planner = synomega.load_default_planner(plausibility=True,
                                        plausibility_threshold=0.4)
# 之后 plan / score 的每个单步候选都会被筛：reactants → target 合理性低于 0.4 的拆分被丢弃
```

## 用法二：直接给一批反应打分

```python
from synomega.plausibility import PlausibilityScorer

scorer = PlausibilityScorer.default()               # 首用下载合理性模型
scores = scorer.score_reactions([
    ("CC(=O)O.NCc1ccccc1", "CC(=O)NCc1ccccc1"),      # 每项是 (反应物, 产物) 元组
    ("CCO.CC(=O)O",        "CC(=O)OCC"),
])
print(scores)   # -> [0.99, 0.95] 之类；每条一个 [0,1] 分，越高越可信；无法解析记 0.0
```

!!! warning "入参是元组，不是反应 SMILES"
    `score_reactions` 收的是 `(反应物_smiles, 产物_smiles)` 元组的可迭代对象，**不是** `"A.B>>C"`
    这种反应 SMILES 串。同一目标的多个拆分共享反应物 / 产物图缓存，打分很快。

## 参数与注意

- `plausibility_threshold`（默认 0.4）：越高筛得越狠；过滤器**只删**候选、不重排存活项。
- 需要在过滤器里保底留几条 / 过取候选 / 重排，传
  `plausibility_kwargs={"min_keep": ..., "overfetch": ..., "rerank": ...}` 给
  `load_default_planner` 或 `Planner`。
- `PlausibilityScorer.default(device="cuda:0")` 可指定设备；`scorer.meta["val_auc"]` 是验证集 AUC。
