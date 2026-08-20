# 多组分演化预测

从一组起始反应物出发，反复两两反应、把产物加回分子池，长出一张正向**合成网络**。每个分子带
**总合成分数**（`min(两父总分) × 单步概率`，起始物=1.0）与**合成树深度**。算法与验证见
[研究报告 · 多组分演化预测](../research/evolution.md)；安装见[安装与总览](index.md)。

```bash
synomega evolve --reactants "CC(=O)c1ccccc1.C=O.CNC" \
                --max-depth 3 --score-threshold 0.01 --out network.json
```

```python
from synomega.forward import ForwardTemplateGNN, MultiComponentEvolution

evo = MultiComponentEvolution(ForwardTemplateGNN.default(),
                              max_depth=3, score_threshold=0.01)
result = evo.evolve(["CC(=O)c1ccccc1", "C=O", "CNC"])   # 三组分 Mannich 起始物
print(result.describe())
for m in result.top(10, min_depth=1):
    print(m.total_score, f"d{m.depth}", m.smiles)
result.close()
```

常用参数：`--mode {memory,disk,auto}`（大量起始物用 `disk` 落 SQLite，需 `--work-dir`）、
`--forward-top-k`（每对取几个产物）、`--frontier-width`（每代配对数封顶，控扇出）、
`--no-self-pair`（禁 A+A 自反应）。
