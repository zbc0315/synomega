# 单步正向反应预测

给一组反应物，预测并排序最可能的**产物**。模型与评测见[研究报告 · 单步正向反应预测](../research/forward.md)；
安装见[安装与总览](index.md)。

```bash
synomega forward "CC(=O)O.NCc1ccccc1" --top-k 5     # 多分子用 . 连接
```

```python
from synomega.forward import ForwardTemplateGNN

fwd = ForwardTemplateGNN.default()                   # 首次调用自动下载正向模型
for pred in fwd.predict("CC(=O)O.NCc1ccccc1", top_k=5):
    print(pred.score, pred.product, pred.template_id)
```

输出为排序产物，每个带 `product`（SMILES）、`score`（正向预测概率）、`template_id`。
`--topk-templates`（默认 10）控制每次搜索的模板数。
