# 多步反应路径规划

给一个目标分子，在 AND-OR 图上搜索到可购砌块的完整合成路径。搜索算法与评测见
[研究报告 · 多步反应路径规划](../research/planning.md)；安装见[安装与总览](index.md)。

```bash
synomega plan --target "CC(=O)Nc1ccccc1O" --max-steps 5 --simplify
```

```python
import synomega

planner = synomega.load_default_planner()            # 默认原始模型 + retrostar
result = planner.plan("CC(=O)Nc1ccccc1O")
print(result.solved)
print(result.best_route.describe())
```

常用旋钮：`--algorithm {retrostar,mcts,bfs}`、`--expansion-width`（每节点取 top-k 候选）、
`--max-steps`（深度上限）、`--time-limit`、`--max-expansions`、`--exclude-target`（把目标本身
当作不可购，避免 0 步 trivially solved）、`--simplify`（用简化约束单步模型，搜索更省）。
