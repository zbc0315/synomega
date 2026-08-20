# 单步逆向反应预测

给一个产物，预测并排序可能的**反应物**（拆分）。无独立命令行子命令——它是 `plan` / `score` 的
底层引擎；要单独取一步逆合成候选，用 Python。模型与评测见
[研究报告 · 单步逆向反应预测](../research/retro.md)；安装见[安装与总览](index.md)。

```python
from synomega.singlestep import TemplateGNN

for p in TemplateGNN.default().predict("CC(=O)Nc1ccccc1O", top_k=5):
    print(p.score, p.reactants)          # p.reactants 是排序好的 canonical SMILES 元组
```

`TemplateGNN.simplify()` 是**简化约束变体**（只输出把目标拆成两个及以上前体的"简化型"拆分），
是可合成性评分的推荐后端。
