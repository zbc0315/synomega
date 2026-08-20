# 多步反应路径规划

**做什么**：给一个目标分子，在 AND-OR 图上反复调用单步逆合成，搜索一条把目标一路拆到**可购砌块**
的完整合成路线。搜索算法与评测见[研究报告 · 多步反应路径规划](../research/planning.md)；安装见
[安装与总览](index.md)。

## 命令行

```bash
synomega plan --target "CC(=O)Nc1ccccc1O" --max-steps 5 --simplify
```

## Python

```python
import synomega

planner = synomega.load_default_planner()          # 默认原始模型 + retrostar；首用下载模型 + 砌块集
result = planner.plan("CC(=O)Nc1ccccc1O", max_depth=5)

print(result.solved)                                # 是否找到全叶可购的路线
print(result.best_route.describe())                 # 最优路线，逐步打印
for r in result.routes[:3]:                          # 前几条候选路线
    print(r.num_steps, r.depth, r.bb_coverage)
print(result.stats.expansions, result.stats.terminated_by)   # 搜索代价与终止原因
```

`best_route.describe()` 示例输出（数字随模型 / 库存而变）：

```
target: CC(=O)Nc1ccccc1O
solved: True  steps: 2  depth: 2  bb_coverage: 1.00
  [1] ...>>CC(=O)Nc1ccccc1O   (score=0.43)
  [2] ...                      (score=0.22)
```

`solved=True` 表示所有叶子都在砌块库里；`bb_coverage` 是可购叶子比例（近失时看它，1.00 = 完全解出）。

## 参数

| 参数（CLI / Python） | 默认 | 含义 |
|---|---|---|
| `--algorithm` | retrostar | 搜索算法：`retrostar`（默认）/ `mcts`（单步弱时更稳）/ `bfs`（基线） |
| `--max-steps` / `max_depth=` | 5 | 路线深度上限 |
| `--expansion-width` | 50 | 每个分子节点取单步 top-k 候选反应物集 |
| `--time-limit` / `--max-expansions` | 60 s / 500 | 搜索预算（时限 / 节点扩展数） |
| `--exclude-target` | 关 | 把目标本身当不可购，避免"目标可买 → 0 步 trivially solved" |
| `--simplify` | 关 | 用简化约束单步模型（搜索更省，见研究报告） |
| `--stock` / `--stock-is-keys` | 默认下载 ZINC | 自定义砌块库（`.keys` 或 SMILES 目录） |

## 注意

- 默认开缓存（同一分子只扩展一次）；`Planner(cache_path="x.sqlite")` 可持久化到 SQLite 跨进程复用。
- 三种算法共享同一 AND-OR 图、预算与路线抽取，结果可直接比较。
- 想同时拿路线树和搜索统计、又只搜一次，用 `SynthesizabilityScorer(planner).score_detailed(smiles)`。
