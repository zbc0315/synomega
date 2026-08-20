# 可合成性评分 SynScore

对一个目标分子给出连续的**可合成性分数** SynScore = \(1/(U+1)^U\)（`U` = 最优路径中不可购起始物
数；全部可购→1，缺得越多分越低），用于对一批分子排序。分数定义与操作点见
[研究报告 · 可合成性评分 SynScore](../research/synscore.md)；安装见[安装与总览](index.md)。

```bash
# 默认用简化约束模型 @ 扩展宽度 k=10（评分推荐配置），--original 换原始模型
synomega score --targets targets.smi --out scores.jsonl
```

```python
import synomega

scorer = synomega.load_default_scorer()              # 默认 simplify=True, k=10
report = scorer.score("CC(=O)Nc1ccccc1O")
print(report.as_dict())                              # score / solved / min_steps ...

batch = scorer.score_batch(open("targets.smi").read().split())
print(batch.solve_rate)
```
